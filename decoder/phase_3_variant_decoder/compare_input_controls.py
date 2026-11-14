"""Compare GO-only, WT+GO, Mutation+GO and Pair-MLP against train-only GO lookup.

This is an isolated frozen-embedding experiment, not a change to the cascade.
Use --family both for separate EASY and HARD W&B runs and local reports.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from phase_3_variant_decoder.config import EMBEDDING_TYPES, INPUT_MODES, VariantDecoderConfig
from phase_3_variant_decoder.go_lookup_rules import fit_lookup
from phase_3_variant_decoder.input_controls import (
    LEARNED, METHODS, attach_mutation_features, class_weights, classification_metrics, fit_normalization,
    save_predictions, train_control,
)
from phase_3_variant_decoder.input_control_reports import (
    SECTIONS, generate_reports, log_results,
)
from phase_3_variant_decoder.presentation_terms import configuration_metadata
from phase_3_variant_decoder.train import prepare_run

FORMAT = "contvar_input_control_comparison_v1"
CONFIG_KEYS = (
    "family", "embedding_type", "input_mode", "dataset_dir", "sequences_json",
    "go_embeddings", "allow_go_proxies", "drop_contradictory", "max_seq_len",
    "global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary",
    "hidden_dims", "dropout", "lr", "weight_decay", "batch_size", "epochs",
    "early_stop_patience", "seed", "cuda", "wandb_project", "wandb_entity", "wandb_run_name",
)


def start_wandb(config, provenance):
    if not config["wandb_enabled"]:
        return None, None
    import wandb
    run = wandb.init(project=config["wandb_project"], entity=config["wandb_entity"],
                     name=config["wandb_run_name"], mode=config["wandb_mode"],
                     group="input-control-comparison", job_type="input-ablation",
                     tags=["phase3", "input-controls", f"fam:{config['family']}", f"seed:{config['seed']}"],
                     config={**config, **configuration_metadata(config), **provenance})
    for method in LEARNED:
        prefix = f"{SECTIONS[method]}/Training"
        run.define_metric(f"{prefix}/epoch")
        run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
    return run, wandb


def package(run, wandb, root):
    artifact = wandb.Artifact(f"phase3-input-controls-{root.name}", type="evaluation")
    artifact.add_dir(str(root))
    run.log_artifact(artifact)


def validate_training(cfg):
    for name in ("epochs", "batch_size", "early_stop_patience"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not cfg.hidden_dims or any(width < 1 for width in cfg.hidden_dims):
        raise ValueError("hidden_dims must contain positive widths")
    if not math.isfinite(cfg.dropout) or not 0 <= cfg.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0 or not math.isfinite(cfg.weight_decay) or cfg.weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative")
    if not 0 <= cfg.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")


def run_experiment(cfg, *, cache_dir=None, out_dir="runs/phase3_input_controls", quiet=False):
    cfg = copy.deepcopy(cfg)
    validate_training(cfg)
    cfg.input_balance_classes = getattr(cfg, "input_balance_classes", True)
    cfg.architecture = "cascade"  # Reuse only data preparation + GO-artifact coverage validation.
    tensors, views, _, device = prepare_run(cfg, cache_dir=cache_dir, quiet=quiet)
    if any(view["n"] == 0 for view in views.values()):
        raise ValueError("Input controls require nonempty train, val and test partitions")
    class_weights(views["train"]["label_idx"], cfg.input_balance_classes)
    attach_mutation_features(tensors, views, cfg.max_seq_len)
    normalization = fit_normalization(tensors, views["train"])
    config = {key: getattr(cfg, key) for key in CONFIG_KEYS}
    config.update(architecture="input_control_mlp", input_balance_classes=cfg.input_balance_classes,
                  loss="train-balanced sample-mean cross-entropy" if cfg.input_balance_classes else "sample-mean cross-entropy",
                  decision="three-class argmax; no threshold tuning", encoder_frozen=True,
                  normalizer="unique train WT rows; unique train GO rows; shared WT/VAR transform",
                  wandb_enabled=cfg.wandb_enabled, wandb_mode=getattr(cfg, "wandb_mode", "online"))
    if not config["wandb_run_name"] or config["wandb_run_name"] == f"phase3-{cfg.family}-{cfg.embedding_type}-{cfg.input_mode}":
        config["wandb_run_name"] = f"phase3-input-controls-{cfg.family}-{cfg.embedding_type}-{cfg.input_mode}-s{cfg.seed}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"inputs_{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}_", dir=Path(out_dir).resolve()))
    identities = ["split", "protein_id", "snp", "matched_go_id", "label"]
    provenance = dict(filtered_rows_sha256=hashlib.sha256(tensors.df[identities].to_csv(index=False).encode()).hexdigest(),
                      split_rows={s: v["n"] for s, v in views.items()}, go_artifact_sha256=tensors.go_artifact_sha256,
                      protein_input_dim=tensors.input_dim, go_input_dim=tensors.go_features.shape[1])
    (root / "configuration.json").write_text(json.dumps(dict(config=config, provenance=provenance), indent=2) + "\n")
    run, wandb = start_wandb(config, provenance)
    try:
        # This reference is refit on the exact same filtered train rows as the MLPs.
        train = tensors.df.loc[tensors.df.split == "train"]
        rules, fallback, counts = fit_lookup(train.matched_go_id.to_numpy(), train.label_idx.to_numpy())
        lookup = dict(parameter_count=0, best_epoch=None, epochs_run=0, metrics={})
        lookup_root = root / "go_lookup"
        lookup_root.mkdir()
        (lookup_root / "rules.json").write_text(json.dumps(dict(rules=rules, fallback=fallback, train_counts=counts), indent=2) + "\n")
        for split in ("val", "test"):
            frame = tensors.df.loc[tensors.df.split == split]
            pred = np.asarray([rules.get(go, fallback) for go in frame.matched_go_id], dtype=int)
            lookup["metrics"][split] = classification_metrics(frame.label_idx.to_numpy(), pred)
            save_predictions(tensors, split, pred, lookup_root / f"{split}_predictions.csv")
        results, histories = {"go_lookup": lookup}, {}

        def log_epoch(method, row):
            if run is not None:
                run.log({f"{SECTIONS[method]}/Training/{key}": value for key, value in row.items()})

        for method in LEARNED:
            if not quiet:
                print(f"\n[INPUT CONTROL] {METHODS[method]} -> {root / method}", flush=True)
            results[method], histories[method] = train_control(method, cfg, tensors, views, normalization,
                                                               root / method, on_epoch=log_epoch, quiet=quiet)
            results[method]["checkpoint"] = str(Path(results[method]["checkpoint"]).relative_to(root))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        artifacts = generate_reports(results, histories, root, config)
        manifest = dict(format=FORMAT, config=config, provenance=provenance, runs=results, histories=histories,
                        controlled_variable="available input blocks; hidden widths matched, parameter counts not matched",
                        test_used_for_selection=False,
                        artifacts={k: str(Path(p).relative_to(root)) for k, p in artifacts.items()},
                        wandb_url=run.url if run else None)
        source = root / "input_comparison.json"
        source.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if run is not None:
            log_results(run, wandb, manifest, artifacts)
            package(run, wandb, root)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    if run is not None:
        run.finish()
    if not quiet:
        for method, result in results.items():
            print(f"[TEST] {METHODS[method]} macro-F1={result['metrics']['test']['macro_f1']:.4f}")
        print(f"[DONE] input comparison -> {source}")
        if run is not None:
            print(f"  wandb: {manifest['wandb_url']}")
    return {**manifest, "manifest_path": str(source)}


def upload_saved(path, *, project=None, entity=None, name=None, mode="online"):
    source = Path(path).expanduser().resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Expected input_comparison.json from compare_input_controls")
    paths = {}
    for key, relative in manifest["artifacts"].items():
        target = (source.parent / relative).resolve()
        if not target.is_relative_to(source.parent) or not target.is_file():
            raise ValueError(f"Missing or external report artifact: {relative}")
        paths[key] = str(target)
    config = {**manifest["config"], "wandb_enabled": True, "wandb_mode": mode}
    for key, value in (("wandb_project", project), ("wandb_entity", entity), ("wandb_run_name", name)):
        if value is not None:
            config[key] = value
    run, wandb = start_wandb(config, manifest["provenance"])
    try:
        for method, history in manifest["histories"].items():
            for row in history:
                run.log({f"{SECTIONS[method]}/Training/{k}": v for k, v in row.items()})
        log_results(run, wandb, manifest, paths)
        package(run, wandb, source.parent)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()
    return run.url


def main(argv=None):
    defaults = VariantDecoderConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("easy", "hard", "both"), default="hard")
    parser.add_argument("--embedding", choices=EMBEDDING_TYPES, default="seq_phase1")
    parser.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    for key in ("global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "cache_dir", "cuda"):
        parser.add_argument(f"--{key}")
    for key in ("dataset_dir", "sequences_json", "go_embeddings"):
        parser.add_argument(f"--{key}", default=getattr(defaults, key))
    for key in ("allow_go_proxies", "drop_contradictory", "unweighted", "no_wandb"):
        parser.add_argument(f"--{key}", action="store_true")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--dropout", type=float, default=.2)
    for key, dtype in (("lr", float), ("weight_decay", float), ("batch_size", int), ("epochs", int), ("early_stop_patience", int), ("seed", int)):
        parser.add_argument(f"--{key}", type=dtype, default=getattr(defaults, key))
    parser.add_argument("--out_dir", default="runs/phase3_input_controls")
    for key in ("wandb_project", "wandb_entity", "wandb_run_name"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--upload_existing", metavar="INPUT_COMPARISON_JSON")
    args = parser.parse_args(argv)
    if args.upload_existing:
        if args.no_wandb:
            parser.error("--upload_existing cannot be combined with --no_wandb")
        print("[DONE] saved input comparison:", upload_saved(args.upload_existing, project=args.wandb_project,
              entity=args.wandb_entity, name=args.wandb_run_name, mode=args.wandb_mode))
        return
    if not all((args.global_h5, args.local_h5, args.dataset_dir, args.sequences_json, args.go_embeddings)):
        parser.error("Provide both H5 paths and nonempty dataset/sequence/GO paths; check exported variables")
    for family in (("easy", "hard") if args.family == "both" else (args.family,)):
        cfg = VariantDecoderConfig(family, args.embedding, args.input_mode)
        for key in CONFIG_KEYS:
            if key in ("family", "embedding_type", "max_seq_len", "wandb_run_name"):
                continue
            value = getattr(args, key, None)
            if value is not None:
                setattr(cfg, key, value)
        cfg.input_balance_classes = not args.unweighted
        cfg.wandb_enabled, cfg.wandb_mode = not args.no_wandb, args.wandb_mode
        cfg.wandb_run_name = (args.wandb_run_name + (f"-{family}" if args.family == "both" else "")) if args.wandb_run_name else None
        run_experiment(cfg, cache_dir=args.cache_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
