"""
Phase 3 training: two-stage GO-conditioned scalar cascade (CLI default).

--architecture legacy retains the original per-GO hinge experiment below.
Cascade defaults to hinge; squared hinge and BCE are explicit alternatives.
Cascade details: docs/CASCADE.md. Existing sweep callers use legacy config.

Design: phase_3_variant_decoder/DESIGN.md §4, §5, §6.5

Selection metric: val macro-F1 (thresholds searched on val).
Test metrics are computed every epoch but DO NOT DRIVE SELECTION — they are logged
so that afterwards you can read whether the val curve tracked the test curve.

Usage:
    python -m phase_3_variant_decoder.train \\
        --family hard --embedding raw_esm \\
        --global_h5 data/processed/phase3_raw_esm_global.h5 \\
        --local_h5  data/processed/phase3_raw_esm_local.h5 \\
        --cuda 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from phase_2_decoder.config import pick_device
from phase_2_decoder.models.ffn_decoder import FFNDecoder
from phase_3_variant_decoder.config import (
    CASCADE_LOSS_TYPES, EFFECT_THRESHOLD_STRATEGIES, EMBEDDING_TYPES, FAMILIES, INPUT_MODES, LABELS,
    SECONDARY_REQUIRED, VariantDecoderConfig, resolve_default_go_embeddings,
)
from phase_3_variant_decoder.data.pair_dataset import (
    build_pair_tensors, iterate_batches,
)
from phase_3_variant_decoder.data.variant_dataset import prepare
from phase_3_variant_decoder.losses import (
    IDX_GOF, IDX_LOF, IDX_NEUTRAL, margin_ranking_loss, pair_delta,
)
from phase_3_variant_decoder import results
from phase_3_variant_decoder.metrics import (
    breakdown, compute_metrics, margin_diagnostics, search_thresholds,
)


@torch.no_grad()
def compute_deltas(model, tensors, view, batch_size: int = 4096) -> np.ndarray:
    """The d = s_wt - s_var vector for one split."""
    model.eval()
    out = torch.empty(view["n"], dtype=torch.float32, device=tensors.wt_feat.device)
    for idx in iterate_batches(view["n"], batch_size, shuffle=False):
        idx = idx.to(out.device)
        x_wt, x_var = tensors.gather(view["wt_row"][idx], view["var_row"][idx])
        out[idx] = pair_delta(model(x_wt), model(x_var), view["go_idx"][idx]).float()
    return out.cpu().numpy()


@torch.no_grad()
def compute_pair_scores(model, tensors, view, batch_size: int = 4096):
    """(s_wt, s_var) for one split — used by the per-row score dump (R5).

    compute_deltas() returns only the difference; error analysis also needs the
    absolute level of both sides.
    """
    model.eval()
    dev = tensors.wt_feat.device
    s_wt = torch.empty(view["n"], dtype=torch.float32, device=dev)
    s_var = torch.empty(view["n"], dtype=torch.float32, device=dev)
    for idx in iterate_batches(view["n"], batch_size, shuffle=False):
        idx = idx.to(dev)
        x_wt, x_var = tensors.gather(view["wt_row"][idx], view["var_row"][idx])
        rows = torch.arange(idx.shape[0], device=dev)
        go = view["go_idx"][idx]
        s_wt[idx] = torch.sigmoid(model(x_wt)[rows, go]).float()
        s_var[idx] = torch.sigmoid(model(x_var)[rows, go]).float()
    return s_wt.cpu().numpy(), s_var.cpu().numpy()


# Headline numbers shown in the wandb "val"/"test" tabs — the set you'd actually
# watch live to judge whether a run is doing well or has collapsed into the
# degenerate solution (DESIGN §8.6). Everything else computed by evaluate_split()
# still goes to wandb, just under "val_detail"/"test_detail" instead, and every key
# is still written in full to results.csv regardless of this split (report.py needs
# all of it — see DESIGN §5.2, §8.7).
WANDB_SUMMARY_KEYS = frozenset({
    "macro_f1", "macro_auprc", "accuracy", "lof_gof_auroc",
    "lof_recall", "gof_recall", "neutral_recall",
    "lof_f1", "gof_f1", "neutral_f1",
    "center", "width",
})


def evaluate_split(model, tensors, view, center, width, cfg) -> dict:
    """Full metric set plus the two diagnostic breakdowns (DESIGN §5.2).

    Returned unprefixed; callers decide how to namespace it (flat "val_x" for
    results.csv/best-tracking, "val/x" vs "val_detail/x" for wandb — see
    build_wandb_log()).
    """
    d = compute_deltas(model, tensors, view)
    y = view["label_idx"].cpu().numpy()

    m = compute_metrics(d, y, center, width)
    m.update(margin_diagnostics(d, y, cfg.margin))
    m.update(breakdown(d, y, view["seen_in_train"], center, width, "seen"))
    m.update(breakdown(d, y, view["is_alanine"], center, width, "ala"))
    return m


def build_wandb_log(epoch: int, train_loss: float, train_m: dict,
                    val_m: dict, test_m: dict, prefix: str = "") -> dict:
    """Namespace metrics into wandb's train/val/test tabs (+ a "_detail" tab each).

    wandb groups panels by the segment before the first "/", so this alone is what
    splits the dashboard into separate train/val/test/train_detail/val_detail/
    test_detail sections instead of one flat wall of ~96 charts.

    `train_m` is evaluate_split() run on the TRAIN split itself (using the same
    (center, width) just found on val) — it exists purely to watch, live, what
    fraction of each class's TRAIN rows already sit inside/beyond their margin
    each epoch (train_detail/{lof,gof}_beyond_margin, train_detail/neutral_within_
    _margin): once a row crosses that line the hinge sends it zero loss/gradient,
    i.e. it has effectively dropped out of training. Everything else evaluate_split
    computes (per-class precision/recall/f1/auprc/auroc, seen/ala breakdowns) comes
    along for the same reason it already does for val/test — so "_detail" carries
    the full metric set, not just the headline numbers in WANDB_SUMMARY_KEYS.
    """
    log = {"epoch": epoch, "train/loss": round(train_loss, 5)}
    for prefix, m in (("train", train_m), ("val", val_m), ("test", test_m)):
        for k, v in m.items():
            if k in ("confusion", "n"):
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            ns = prefix if k in WANDB_SUMMARY_KEYS else f"{prefix}_detail"
            log[f"{ns}/{k}"] = v
    if prefix:
        return {f"{prefix}/{key}": value for key, value in log.items()}
    return log


def prepare_run(cfg: VariantDecoderConfig, cache_dir: str | None = None, quiet: bool = False):
    """Prepare data and build tensors. Separate from fit() so sweeps can reuse it."""
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(pick_device(cfg.cuda))

    df, go_vocab, report = prepare(cfg)
    go_table = None
    if cfg.architecture == "cascade":
        from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable
        go_table = GOEmbeddingTable(cfg.go_embeddings, allow_proxies=cfg.allow_go_proxies)
        go_features = go_table.aligned_to_vocab(go_vocab)  # fail before H5 loading if coverage is incomplete
    if not quiet:
        print(f"[INFO] Device: {device}")
        print(report.render())
        print(f"[INFO] vocabulary size: {len(go_vocab):,}")

    tensors = build_pair_tensors(df, cfg, cache_dir=cache_dir, verbose=not quiet).to(device)
    if go_table is not None:
        tensors.go_features = go_features.to(device)
        tensors.go_artifact_sha256 = go_table.artifact_sha256
        tensors.go_metadata = go_table.metadata
    views = {s: tensors.split_view(s) for s in ("train", "val", "test")}
    if not quiet:
        print(f"[INFO] input dim: {tensors.input_dim} | train={views['train']['n']:,} "
              f"val={views['val']['n']:,} test={views['test']['n']:,}")
        if tensors.go_features is not None:
            print(f"[INFO] GO input: {len(go_vocab):,} run terms <- "
                  f"{len(go_table.go_ids):,} x {go_table.dimension} artifact | "
                  f"sha256={go_table.artifact_sha256}")
    return tensors, views, go_vocab, device


def fit(cfg: VariantDecoderConfig, tensors, views, go_vocab: dict, device,
        out_dir: str | None = None, run=None, quiet: bool = False,
        results_csv: str | None = str(results.DEFAULT_PATH),
        log_prefix: str = "") -> dict:
    """The training loop. Shared by train.py and sweep_train.py.

    Thresholds are searched on VAL every epoch; test metrics are computed with the
    same thresholds and logged, but DO NOT DRIVE SELECTION (DESIGN §5.2).
    """
    if cfg.architecture == "cascade":
        from phase_3_variant_decoder.cascade import fit_cascade
        return fit_cascade(cfg, tensors, views, go_vocab, device, out_dir, run, quiet)
    model = FFNDecoder(
        input_dim=tensors.input_dim, n_classes=len(go_vocab),
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    ).to(device)
    if not quiet:
        print(f"[INFO] decoder parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tv = views["train"]
    # The legacy FFN uses BatchNorm and cannot train on a final singleton batch.
    # Grow the batch by one only when that would otherwise happen; no row is
    # dropped and ordinary batch layouts remain unchanged.
    train_batch_size = cfg.batch_size
    if tv["n"] > 1 and tv["n"] % train_batch_size == 1:
        train_batch_size += 1
    best = {"macro_f1": -1.0, "epoch": -1}
    best_state = None
    patience = 0
    gen = torch.Generator().manual_seed(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        cls_loss_sum = {IDX_LOF: 0.0, IDX_GOF: 0.0, IDX_NEUTRAL: 0.0}
        cls_loss_n = {IDX_LOF: 0, IDX_GOF: 0, IDX_NEUTRAL: 0}
        for idx in iterate_batches(tv["n"], train_batch_size, shuffle=True, generator=gen):
            idx = idx.to(device)
            x_wt, x_var = tensors.gather(tv["wt_row"][idx], tv["var_row"][idx])
            d = pair_delta(model(x_wt), model(x_var), tv["go_idx"][idx])
            per_row = margin_ranking_loss(
                d, tv["label_idx"][idx], cfg.margin,
                cfg.w_lof, cfg.w_gof, cfg.w_neutral,
                reduction="none",
            )
            loss = per_row.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
            # Per-class mean loss this batch — cheap (per_row is already computed
            # for the backward pass, this just groups it), and answers directly
            # what fraction of the epoch's gradient each class is still supplying:
            # a class whose mean approaches 0 has (mostly) stopped contributing.
            with torch.no_grad():
                labels_batch = tv["label_idx"][idx]
                for cls_idx in (IDX_LOF, IDX_GOF, IDX_NEUTRAL):
                    mask = labels_batch == cls_idx
                    n_cls = int(mask.sum())
                    if n_cls:
                        cls_loss_sum[cls_idx] += float(per_row[mask].sum())
                        cls_loss_n[cls_idx] += n_cls
        train_loss = total / max(nb, 1)
        cls_loss_mean = {
            f"{LABELS[cls_idx].lower()}_loss_mean":
                (cls_loss_sum[cls_idx] / cls_loss_n[cls_idx] if cls_loss_n[cls_idx] else float("nan"))
            for cls_idx in (IDX_LOF, IDX_GOF, IDX_NEUTRAL)
        }

        d_val = compute_deltas(model, tensors, views["val"])
        y_val = views["val"]["label_idx"].cpu().numpy()
        center, width, val_f1 = search_thresholds(d_val, y_val)

        # Train-split diagnostics: same (center, width) just found on val, applied
        # to train's own rows. Not used for anything but visibility — in particular
        # margin_diagnostics() here shows, per epoch, what fraction of LOF/GOF/
        # NEUTRAL train rows already sit inside/beyond their margin — i.e. the
        # hinge already sends them zero loss/gradient this epoch (see
        # build_wandb_log's docstring).
        train_m = evaluate_split(model, tensors, tv, center, width, cfg)
        train_m.update(cls_loss_mean)   # {lof,gof,neutral}_loss_mean -> train_detail/
        val_m = evaluate_split(model, tensors, views["val"], center, width, cfg)
        test_m = evaluate_split(model, tensors, views["test"], center, width, cfg)

        # Flat dict — unchanged shape, this is what results.csv/report.py and
        # best-checkpoint tracking rely on (DESIGN §5.2's full metric set).
        log = {"epoch": epoch, "train_loss": round(train_loss, 5)}
        log.update({f"val_{k}": v for k, v in val_m.items()})
        log.update({f"test_{k}": v for k, v in test_m.items()})

        if not quiet:
            print(f"[{epoch:03d}] loss={train_loss:.4f}  val_macroF1={log['val_macro_f1']:.4f} "
                  f"(c={center:+.3f} w={width:.3f})  "
                  f"val_neutral_recall={log['val_neutral_recall']:.3f} "
                  f"| test_macroF1={log['test_macro_f1']:.4f}")
        if run is not None:
            run.log(build_wandb_log(
                epoch, train_loss, train_m, val_m, test_m, prefix=log_prefix
            ))

        if val_f1 > best["macro_f1"]:
            best = {"macro_f1": val_f1, "epoch": epoch, "center": center,
                    "width": width, **log}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                if not quiet:
                    print(f"[INFO] Early stop (epoch {epoch}, best was {best['epoch']})")
                break

    ckpt_path = ""
    if out_dir is not None:
        saved = _save(cfg, best, best_state, tensors, go_vocab, out_dir, quiet)
        ckpt_path = str(saved) if saved is not None else ""
        best["checkpoint"] = ckpt_path

    # Persistent log — written the moment the run finishes, crash-resistant (R2).
    if results_csv:
        row = results.build_row(
            cfg, best,
            n_rows={s: views[s]["n"] for s in views},
            n_classes=len(go_vocab), checkpoint=ckpt_path, wandb_run=run,
        )
        path = results.append(row, results_csv)
        if run is not None:
            run.summary["results_run_id"] = row["run_id"]
        if not quiet:
            print(f"[DONE] result row -> {path}")

    if not quiet:
        print(f"[DONE] best epoch {best['epoch']} · val macro-F1 {best['macro_f1']:.4f}")
    return best


def _save(cfg, best, best_state, tensors, go_vocab, out_dir, quiet=False):
    """Save the checkpoint for this (family, embedding, input_mode) architecture —
    but only if this run's val macro-F1 beats whatever is already saved there.

    Sweep trials share the same stem across many trials of the same sweep (and
    `run_sweeps.py run` runs them strictly sequentially, so there is no race).
    The result is that `decoder_{stem}.pt` always holds the single best checkpoint
    found for that architecture so far, without a separate collection step.

    Returns the checkpoint path if it was (over)written, or None if this run's
    result did not improve on the existing one.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}"
    decoder_path = out / f"decoder_{stem}.pt"
    best_json_path = out / f"best_{stem}.json"

    prev_score = -1.0
    if best_json_path.exists():
        try:
            prev_score = float(json.loads(best_json_path.read_text(encoding="utf-8"))
                               .get("macro_f1", -1.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            prev_score = -1.0

    if best["macro_f1"] <= prev_score:
        if not quiet:
            print(f"[INFO] not saving: val macro-F1 {best['macro_f1']:.4f} does not beat "
                  f"the current best {prev_score:.4f} for {stem}")
        return None

    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": tensors.input_dim,
            "n_classes": len(go_vocab),
            "hidden_dims": cfg.hidden_dims,
            "center": best.get("center"), "width": best.get("width"),
            "family": cfg.family, "embedding_type": cfg.embedding_type,
            "input_mode": cfg.input_mode,
            "best_epoch": best["epoch"],
            "hparams": {
                "margin": cfg.margin,
                "w_lof": cfg.w_lof, "w_gof": cfg.w_gof, "w_neutral": cfg.w_neutral,
                "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                "dropout": cfg.dropout, "batch_size": cfg.batch_size,
                "hidden_dims": cfg.hidden_dims,
            },
        },
        decoder_path,
    )
    (out / f"go_vocab_{stem}.json").write_text(json.dumps(go_vocab), encoding="utf-8")
    best_json_path.write_text(
        json.dumps({k: v for k, v in best.items() if not isinstance(v, torch.Tensor)}, indent=2),
        encoding="utf-8",
    )
    if not quiet:
        print(f"[DONE] new best for {stem} (val macro-F1 {best['macro_f1']:.4f}) -> {decoder_path}")
    return decoder_path
    return out / f"decoder_{stem}.pt"


def train(cfg: VariantDecoderConfig, args) -> dict:
    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=getattr(args, "cache_dir", None))

    run = None
    if cfg.wandb_enabled:
        try:
            import wandb
            tags, meta = cfg.wandb_metadata()
            run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity,
                name=cfg.wandb_run_name, tags=tags,
                config={
                    **meta, "n_go_terms": len(go_vocab),
                    **({"n_outputs_per_branch": 1, "n_binary_stages": 2,
                        "loss": cfg.cascade_loss, "cascade_margin": cfg.cascade_margin,
                        "margin_space": "scaled_signed_logit", "balance_classes": cfg.cascade_balance_classes,
                        "effect_threshold_strategy": cfg.effect_threshold_strategy,
                        "go_artifact_sha256": tensors.go_artifact_sha256}
                       if cfg.architecture == "cascade" else
                       {"n_classes": len(go_vocab), "margin": cfg.margin,
                        "w_lof": cfg.w_lof, "w_gof": cfg.w_gof, "w_neutral": cfg.w_neutral}),
                    "lr": cfg.lr, "weight_decay": cfg.weight_decay, "dropout": cfg.dropout,
                    "batch_size": cfg.batch_size, "hidden_dims": cfg.hidden_dims,
                    "stage1": cfg.stage1,
                },
            )
            from phase_3_variant_decoder.presentation_terms import log_reporting_guide
            log_reporting_guide(run, wandb, [vars(cfg)])
        except Exception as exc:
            raise RuntimeError(
                "WandB is enabled but the run could not be initialized. "
                "Run `wandb login`, fix the connection/configuration, or explicitly "
                "use --no_wandb for a local-only run."
            ) from exc

    best = fit(cfg, tensors, views, go_vocab, device, out_dir=args.out_dir,
               run=run, results_csv=args.results_csv)

    if run is not None:
        run.summary.update({f"best_{k}": v for k, v in best.items()
                            if isinstance(v, (int, float, str))})
        run.finish()
    return best


def main():
    p = argparse.ArgumentParser(description="Phase 3 variant decoder training")
    p.add_argument("--architecture", choices=["cascade", "legacy"], default="cascade")
    p.add_argument("--go_embeddings", default=resolve_default_go_embeddings(),
                   help="GO input NPZ; defaults to go_terms_embeddings_anc2vec.npz or PHASE3_GO_EMBEDDINGS")
    p.add_argument("--allow_go_proxies", action="store_true")
    p.add_argument("--unweighted_cascade", action="store_true", help="Disable train-only binary class balancing")
    p.add_argument("--cascade_loss", choices=CASCADE_LOSS_TYPES, default="hinge",
                   help="Cascade objective in both stages (default: hinge; BCE retained explicitly)")
    p.add_argument("--cascade_margin", type=float, default=1.0,
                   help="Positive margin in scaled decision-logit space; unrelated to legacy --margin")
    p.add_argument(
        "--effect_threshold_strategy", choices=EFFECT_THRESHOLD_STRATEGIES,
        default="validation",
        help="Effect gate: validation=tune probability threshold; "
             "learned_radius=fix p=0.5 and use the trainable radius as the boundary",
    )
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--sequences_json", default=None)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--embedding", choices=EMBEDDING_TYPES, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="both",
                   help="both=[global|local] / global=no local (ablation control)")
    p.add_argument("--global_h5", required=True)
    p.add_argument("--local_h5", required=True)
    p.add_argument("--global_h5_secondary", default=None, help="concat_* only")
    p.add_argument("--local_h5_secondary", default=None, help="concat_* only")
    p.add_argument("--out_dir", default="phase_3_variant_decoder/runs")
    p.add_argument("--results_csv", default=str(results.DEFAULT_PATH),
                   help="Legacy only. Cascade writes metrics.json/history.json in a unique run directory")
    p.add_argument("--cache_dir", default=None,
                   help="pair-tensor cache directory (strongly recommended for sweeps)")
    p.add_argument("--margin", type=float, default=None)
    p.add_argument("--w_lof", type=float, default=None)
    p.add_argument("--w_gof", type=float, default=None)
    p.add_argument("--w_neutral", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--drop_contradictory", action="store_true")
    p.add_argument("--stage1", choices=["none", "goa"], default="none",
                   help="GOA pre-training (DESIGN §7) — not implemented yet; anything but 'none' errors")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--cuda", default=None)
    args = p.parse_args()

    if args.architecture == "cascade":
        if not args.go_embeddings:
            p.error("--architecture cascade requires --go_embeddings; see docs/CASCADE.md")
        if any(getattr(args, name) is not None for name in ("margin", "w_lof", "w_gof", "w_neutral")):
            p.error("--margin/--w_* belong to legacy hinge; use --cascade_loss/--cascade_margin for cascade")

    if args.stage1 != "none":
        raise SystemExit(
            "--stage1 goa is not implemented yet (DESIGN §7, deferred work). "
            "See phase_3_variant_decoder/docs/GOA_INTEGRATION_DEFERRED.md"
        )
    if args.embedding in SECONDARY_REQUIRED and not (
        args.global_h5_secondary and args.local_h5_secondary
    ):
        raise SystemExit(
            f"embedding={args.embedding} requires --global_h5_secondary and --local_h5_secondary"
        )

    cfg = VariantDecoderConfig(family=args.family, embedding_type=args.embedding,
                               input_mode=args.input_mode)
    cfg.architecture = args.architecture
    cfg.go_embeddings = args.go_embeddings
    cfg.allow_go_proxies = args.allow_go_proxies
    cfg.cascade_balance_classes = not args.unweighted_cascade
    cfg.cascade_loss, cfg.cascade_margin = args.cascade_loss, args.cascade_margin
    cfg.effect_threshold_strategy = args.effect_threshold_strategy
    cfg.wandb_run_name = (f"phase3-cascade-{cfg.cascade_loss}-{args.family}-{args.embedding}-{args.input_mode}"
                         if cfg.architecture == "cascade" else
                         f"phase3-legacy-{args.family}-{args.embedding}-{args.input_mode}")
    cfg.preset = "manual"   # distinguishes ad-hoc single runs from a sweep search
    cfg.global_h5 = args.global_h5
    cfg.local_h5 = args.local_h5
    cfg.global_h5_secondary = args.global_h5_secondary
    cfg.local_h5_secondary = args.local_h5_secondary
    cfg.drop_contradictory = args.drop_contradictory
    cfg.stage1 = args.stage1
    cfg.cuda = args.cuda
    cfg.wandb_enabled = not args.no_wandb
    for name in ("margin", "w_lof", "w_gof", "w_neutral",
                 "lr", "batch_size", "epochs", "dataset_dir", "sequences_json",
                 "hidden_dims", "dropout", "early_stop_patience", "seed"):
        val = getattr(args, name)
        if val is not None:
            setattr(cfg, name, val)

    train(cfg, args)


if __name__ == "__main__":
    main()
