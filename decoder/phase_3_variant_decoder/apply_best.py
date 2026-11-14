"""
Take a sweep's best trial config, train it on the requested family and save it.

Design: phase_3_variant_decoder/DESIGN.md §5.3

It does two jobs:

1. **Persist the sweep result.** Sweep trials do not write checkpoints (400 trials
   x a checkpoint each is pointless); the best config is looked up and retrained
   into `runs/` in the standard format, so `evaluate.py --compare` can list them
   side by side.

2. **Cross-family control.** Because the two families are optimised separately, the
   easy-vs-hard difference is no longer pure "split effect" — it also contains a
   hyperparameter difference. Running hard's best config on easy separates them:
       easy(own config) - easy(hard's config)  = hyperparameter effect
       easy(hard's config) - hard              = pure split effect

Usage:
    # persist the sweep result
    python -m phase_3_variant_decoder.apply_best \\
        --sweep_id abc123 --family hard --embedding raw_esm --cuda 3

    # run hard's config on easy (cross-check)
    python -m phase_3_variant_decoder.apply_best \\
        --sweep_id <hard_sweep> --family easy --embedding raw_esm \\
        --run_name easy_raw_esm_hardcfg --cuda 3
"""

from __future__ import annotations

import argparse
import json

from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, INPUT_MODES, SECONDARY_REQUIRED, WANDB_ENTITY,
    WANDB_PROJECT, VariantDecoderConfig,
)
from phase_3_variant_decoder.make_sweeps import PRIMARY_OF
from phase_3_variant_decoder import results
from phase_3_variant_decoder.sweep_train import parse_hidden_dims
from phase_3_variant_decoder.train import fit, prepare_run

HPARAM_KEYS = ("margin", "w_gof", "w_neutral",
               "lr", "weight_decay", "dropout", "batch_size", "hidden_dims")


def fetch_best_config_from_results(results_csv: str, family: str, embedding_type: str,
                                    input_mode: str, sweep_id: str | None = None,
                                    metric: str = "val_macro_f1") -> tuple[dict, str]:
    """Find the sweep's best trial directly from results.csv.

    BUG FOUND & FIXED HERE (replaces the old wandb-based fetch_best_config()):
    `wandb.Api().sweep(...).best_run()` ranks runs by the LAST-logged value of
    the sweep's target metric — a run's `summary` holds "last value logged"
    unless the run calls `run.define_metric(..., summary="max")`, which this
    codebase never did. Early stopping keeps training `patience` epochs past
    the true peak, so the last-logged epoch's value is *always* worse than the
    peak — best_run() was landing on an essentially arbitrary trial, not the
    best one. Confirmed empirically: a "collect" retrain of what wandb called
    the best trial used hyperparameters with no relation to the trial that
    actually scored highest in results.csv (different margin, lr, dropout,
    batch_size entirely). results.csv already stores each trial's own
    historical-best row (fit()'s own tracking, not wandb's) — read it from
    there instead; wandb is not consulted at all for this.

    (There was a second, compounding bug in the same area: the sweep YAMLs'
    declared metric name, "val_macro_f1", was never actually logged anywhere —
    build_wandb_log() namespaces it as "val/macro_f1". Fixed in make_sweeps.py;
    this function does not depend on that name either way.)
    """
    rows = results.load(results_csv)
    cands = [r for r in rows if r.get("family") == family
             and r.get("embedding_type") == embedding_type
             and r.get("input_mode") == input_mode
             and r.get("source") == "sweep"]
    if sweep_id:
        bare = sweep_id.split("/")[-1]
        scoped = [r for r in cands if r.get("sweep_id") == bare]
        if scoped:
            cands = scoped
        else:
            print(f"[WARN] no results.csv rows tagged sweep_id={bare!r} for "
                  f"{family}/{embedding_type}/{input_mode} — using all sweep "
                  "rows for this architecture instead")
    if not cands:
        raise SystemExit(f"[ERROR] no sweep rows in {results_csv} for "
                          f"{family}/{embedding_type}/{input_mode}")

    def _score(r):
        try:
            return float(r.get(metric) or "nan")
        except (TypeError, ValueError):
            return float("-inf")

    best = max(cands, key=_score)
    cfg = {
        "margin": best["margin"], "w_gof": best["w_gof"], "w_neutral": best["w_neutral"],
        "lr": best["lr"], "weight_decay": best["weight_decay"], "dropout": best["dropout"],
        "batch_size": best["batch_size"],
        "hidden_dims": best["hidden_dims"].replace("-", ","),
        "family": family, "embedding_type": embedding_type,
    }
    print(f"[INFO] best trial (from {results_csv}): {metric}={_score(best):.4f}  "
          f"wandb_run_id={best.get('wandb_run_id')}  sweep_id={best.get('sweep_id')}")
    return cfg, best.get("wandb_url", ""), best.get("sweep_id", "")


def main():
    p = argparse.ArgumentParser(description="Apply a sweep's best config")
    p.add_argument("--sweep_id", default=None,
                   help="optional filter — restrict to this sweep's own trials in "
                        "results.csv (bare id or entity/project/id, either works). "
                        "Without it, the best trial across ALL sweep rows for this "
                        "architecture is used.")
    p.add_argument("--family", choices=FAMILIES, required=True,
                   help="family to TRAIN on — may differ from the sweep's family (cross-check)")
    p.add_argument("--embedding", choices=EMBEDDING_TYPES, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    p.add_argument("--project", default=WANDB_PROJECT,
                   help="wandb project for this run — overridden automatically if the "
                        "best trial's own results.csv row names one")
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--h5_prefix", default="data/processed/phase3")
    p.add_argument("--cache_dir", default="data/processed/phase3_cache")
    p.add_argument("--out_dir", default="phase_3_variant_decoder/runs")
    p.add_argument("--results_csv", default=str(results.DEFAULT_PATH),
                   help="also where the best trial's config is READ from (see "
                        "fetch_best_config_from_results) — not just written to")
    p.add_argument("--metric", default="val_macro_f1",
                   help="results.csv column used to pick the best trial")
    p.add_argument("--run_name", default=None,
                   help="checkpoint name; defaults to {family}_{type}")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--config_json", default=None,
                   help="read the config from a file instead of results.csv (offline use)")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--cuda", default=None)
    args = p.parse_args()

    retrain_project = args.project   # overridden below if the best row names one
    resolved_sweep_id = args.sweep_id or ""
    if args.config_json:
        best_cfg = json.loads(open(args.config_json, encoding="utf-8").read())
        print(f"[INFO] config read from file: {args.config_json}")
    else:
        best_cfg, wandb_url, resolved_sweep_id = fetch_best_config_from_results(
            args.results_csv, args.family, args.embedding, args.input_mode,
            sweep_id=args.sweep_id, metric=args.metric,
        )
        # wandb_url looks like ".../<entity>/<project>/runs/<id>" — retrain
        # lands next to the trial it came from by default (same fix as before,
        # now sourced from results.csv instead of a wandb API round-trip).
        parts = wandb_url.split("/")
        if len(parts) >= 3 and "runs" in parts:
            retrain_project = parts[parts.index("runs") - 1]

    cfg = VariantDecoderConfig(family=args.family, embedding_type=args.embedding,
                               input_mode=args.input_mode)
    cfg.wandb_project = retrain_project
    cfg.source = "apply_best"
    cfg.sweep_id = resolved_sweep_id
    primary = PRIMARY_OF.get(args.embedding, args.embedding)
    cfg.global_h5 = f"{args.h5_prefix}_{primary}_global.h5"
    cfg.local_h5 = f"{args.h5_prefix}_{primary}_local.h5"
    if args.embedding in SECONDARY_REQUIRED:
        cfg.global_h5_secondary = f"{args.h5_prefix}_raw_esm_global.h5"
        cfg.local_h5_secondary = f"{args.h5_prefix}_raw_esm_local.h5"

    cfg.margin = float(best_cfg["margin"])
    cfg.w_lof = 1.0
    cfg.w_gof = float(best_cfg["w_gof"])
    cfg.w_neutral = float(best_cfg["w_neutral"])
    cfg.lr = float(best_cfg["lr"])
    cfg.weight_decay = float(best_cfg["weight_decay"])
    cfg.dropout = float(best_cfg["dropout"])
    cfg.batch_size = int(best_cfg["batch_size"])
    cfg.hidden_dims = parse_hidden_dims(best_cfg["hidden_dims"])
    cfg.epochs = args.epochs
    cfg.early_stop_patience = args.patience
    cfg.cuda = args.cuda
    cfg.wandb_enabled = not args.no_wandb
    if args.run_name:
        cfg.wandb_run_name = f"phase3-{args.run_name}"
    cfg.wandb_tags = ["phase3", "apply_best"]
    cfg.preset = "collect"   # a longer retrain of a sweep's winning config, not part of the search itself

    print(f"[INFO] applied hyperparameters: m={cfg.margin:.4f} "
          f"w_gof={cfg.w_gof:.2f} w_neutral={cfg.w_neutral:.2f} lr={cfg.lr:.2e} "
          f"dropout={cfg.dropout:.2f} bs={cfg.batch_size} hidden={cfg.hidden_dims}")

    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=args.cache_dir)

    run = None
    if cfg.wandb_enabled:
        try:
            import wandb
            tags, meta = cfg.wandb_metadata()
            run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                             tags=tags, config={**best_cfg, **meta})
        except Exception as exc:
            print(f"[WARN] WandB disabled: {exc}")

    # If run_name is given, the checkpoint name follows it, so cross-check runs do
    # not overwrite the family's own primary run.
    if args.run_name:
        cfg.family, real_family = args.run_name, cfg.family
        try:
            best = fit(cfg, tensors, views, go_vocab, device, out_dir=args.out_dir,
                       run=run, results_csv=args.results_csv)
        finally:
            cfg.family = real_family
    else:
        best = fit(cfg, tensors, views, go_vocab, device, out_dir=args.out_dir,
                   run=run, results_csv=args.results_csv)

    if run is not None:
        run.summary.update({f"best_{k}": v for k, v in best.items()
                            if isinstance(v, (int, float, str))})
        run.finish()


if __name__ == "__main__":
    main()
