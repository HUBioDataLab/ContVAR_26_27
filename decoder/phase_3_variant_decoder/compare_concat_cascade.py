"""Concat vs saved cascade, with matched EASY/HARD mutation diagnostics.

--direct_manifest reuses existing Concat checkpoints without any training.
--train_concat explicitly trains Concat only. Compare is not part of this run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import torch

from phase_3_variant_decoder.cascade import load_cascade
from phase_3_variant_decoder.concat_cascade_evaluation import evaluate_models, prepare_cohorts, validate_cascade_config
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable
from phase_3_variant_decoder.direct_pair_decoder import load_decoder, train_decoder
from phase_3_variant_decoder.direct_pair_reports import create_workspace, generate_reports, log_results
from phase_3_variant_decoder.evaluate_mutation_sensitivity import checkpoints_from_suite, sha256
from phase_3_variant_decoder.input_controls import fit_normalization
from phase_3_variant_decoder.train import prepare_run

FORMAT = "contvar_concat_cascade_comparison_v1"
PATH_OVERRIDES = ("dataset_dir", "sequences_json", "global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "cuda", "batch_size")


def direct_sources(path, families):
    source = Path(path).expanduser().resolve()
    manifest = json.loads(source.read_text())
    historical = manifest.get("format") == "contvar_direct_pair_comparison_v1"
    if not historical and manifest.get("format") != FORMAT:
        raise ValueError("--direct_manifest must be direct_pair_comparison.json or concat_cascade_comparison.json")
    selected = {}
    for family in families:
        result = manifest.get("families", {}).get(family)
        if result is None or "concat" not in result.get("arms", {}):
            raise ValueError(f"No saved Concat checkpoint for {family}")
        checkpoint = (source.parent / result["arms"]["concat"]["checkpoint"]).resolve()
        if not checkpoint.is_file() or historical and not checkpoint.is_relative_to(source.parent):
            raise ValueError(f"Concat checkpoint missing or outside original run: {checkpoint}")
        if not historical and sha256(checkpoint) != result["arms"]["concat"]["checkpoint_sha256"]:
            raise ValueError("Referenced Concat checkpoint no longer matches its saved hash")
        selected[family] = dict(checkpoint=checkpoint, config=result["config"],
                                provenance=result["provenance"], history=result.get("histories", {}).get("concat", result.get("concat_history", [])),
                                source_manifest=str(source))
    return selected


def cascade_sources(args, families, configs):
    paths = args.cascade_checkpoints or [checkpoints_from_suite(args.suite_json, family=f,
        embedding=configs[f].embedding_type, input_mode=configs[f].input_mode, seed=configs[f].seed)[0] for f in families]
    selected = {}
    for path in paths:
        if not Path(path).is_file():
            raise ValueError(f"Cascade checkpoint not found: {path}")
        _, checkpoint = load_cascade(path, "cpu")
        family = checkpoint["config"]["family"]
        if family in selected:
            raise ValueError(f"More than one cascade checkpoint for {family}")
        selected[family] = Path(path).resolve()
    if set(selected) != set(families):
        raise ValueError("Supply exactly one cascade checkpoint per requested family")
    return selected


def start_wandb(config):
    import wandb
    run = wandb.init(project=config["wandb_project"], entity=config.get("wandb_entity"),
        name=config["wandb_run_name"], mode=config["wandb_mode"], group="concat-vs-cascade",
        job_type="checkpoint-comparison", tags=["phase3", "concat-vs-cascade", "mutation-sensitivity"], config=config)
    run.config.update({"concat_cascade_comparison_id": run.id})
    return run, wandb


def publish(run, wandb, report, source, make_workspace=False):
    log_results(run, wandb, report, source.parent)
    if make_workspace:
        try:
            report["workspace_url"] = create_workspace(run, report["artifacts"])
            report.pop("workspace_error", None)
            print(f"[WORKSPACE] {report['workspace_url']}", flush=True)
        except Exception as exc:
            report["workspace_error"] = str(exc)
            print(f"[WARN] Saved view failed: {exc}; retry with --upload_existing.", flush=True)
    source.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    artifact = wandb.Artifact(f"concat-cascade-{run.id}", type="evaluation")
    artifact.add_dir(str(source.parent))
    run.log_artifact(artifact)


def run_comparison(args, quiet=False):
    families = ("easy", "hard") if args.family == "both" else (args.family,)
    direct = direct_sources(args.direct_manifest, families) if args.direct_manifest else {}
    configs = {}
    for family in families:
        cfg = VariantDecoderConfig(family, args.embedding or "seq_phase1", args.input_mode or "both")
        if direct:
            for key, value in direct[family]["config"].items():
                if hasattr(cfg, key) or key in ("projection_dim", "balance_classes"):
                    setattr(cfg, key, value)
            for key, requested in (("embedding_type", args.embedding), ("input_mode", args.input_mode), ("seed", args.seed)):
                if requested is not None and requested != getattr(cfg, key):
                    raise ValueError(f"Requested {key} differs from saved Concat run")
        else:
            cfg.projection_dim, cfg.hidden_dims, cfg.balance_classes = args.projection_dim, args.hidden_dims, not args.unweighted
            for key in ("epochs", "early_stop_patience", "lr", "dropout", "weight_decay", "seed", "go_embeddings"):
                value = getattr(args, key)
                if value is not None:
                    setattr(cfg, key, value)
        for key in PATH_OVERRIDES:
            value = getattr(args, key)
            if value is not None:
                if value == "":
                    raise ValueError(f"Empty --{key}")
                setattr(cfg, key, value)
        cfg.family, cfg.architecture = family, "legacy"  # Only input preparation, no legacy training.
        cfg.validate()
        configs[family] = cfg
    cascades = cascade_sources(args, families, configs)
    if args.create_workspace:
        if args.no_wandb or args.wandb_mode != "online":
            raise ValueError("--create_workspace requires online W&B")
        if importlib.util.find_spec("wandb_workspaces") is None:
            raise ValueError("Install saved-view support: python -m pip install wandb-workspaces")
    default = next(iter(configs.values()))
    config = dict(experiment=FORMAT, family=args.family, families=list(families),
        embedding_type=default.embedding_type, input_mode=default.input_mode, seed=default.seed,
        trained_in_this_experiment=bool(args.train_concat), test_used_for_selection=False,
        wandb_project=args.wandb_project or default.wandb_project, wandb_entity=args.wandb_entity or default.wandb_entity,
        wandb_run_name=args.wandb_run_name or f"phase3-concat-vs-cascade-{'-'.join(families)}-s{default.seed}",
        wandb_mode=args.wandb_mode, tie_tolerance=args.tie_tolerance)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="concat_cascade_", dir=Path(args.out_dir).resolve()))
    source = root / "concat_cascade_comparison.json"
    report = dict(format=FORMAT, config=config, families={}, artifacts={}, wandb_url=None, workspace_url=None,
                  test_used_for_selection=False, trained_in_this_experiment=bool(args.train_concat))
    run, wandb = start_wandb(config) if not args.no_wandb else (None, None)
    if run is not None:
        report["wandb_url"] = run.url
    try:
        for family in families:
            cfg, destination = configs[family], root / family
            destination.mkdir()
            cascade_model, cascade_ckpt = load_cascade(cascades[family], "cpu")
            validate_cascade_config(cfg, cascade_ckpt["config"])
            tensors, views, vocab, device = prepare_run(cfg, cache_dir=args.cache_dir, quiet=quiet)
            identity = ["split", "protein_id", "snp", "matched_go_id", "label"]
            rows_hash = hashlib.sha256(tensors.df[identity].to_csv(index=False).encode()).hexdigest()
            if direct:
                if rows_hash != direct[family]["provenance"]["filtered_rows_sha256"]:
                    raise ValueError("Current filtered rows/order/labels differ from original Concat run")
                concat_path, history = direct[family]["checkpoint"], direct[family]["history"]
            else:
                if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.early_stop_patience < 1:
                    raise ValueError("Training epochs, batch size and patience must be positive")
                table = GOEmbeddingTable(cfg.go_embeddings, allow_proxies=cfg.allow_go_proxies)
                tensors.go_features = table.aligned_to_vocab(vocab).to(device)
                tensors.go_artifact_sha256 = table.artifact_sha256
                result, history = train_decoder("concat", cfg, tensors, views, fit_normalization(tensors, views["train"]),
                                                destination / "trained_concat", quiet=quiet)
                concat_path = Path(result["checkpoint"])
            concat_model, concat_ckpt = load_decoder(concat_path, "cpu")
            context = prepare_cohorts(cfg, tensors.df, destination)
            arms = evaluate_models(concat_model, concat_ckpt, cascade_model, cascade_ckpt,
                                   tensors, views, vocab, cfg, context, destination, args.tie_tolerance)
            for method, path in (("concat", concat_path), ("cascade", cascades[family])):
                arms[method].update(checkpoint=str(path), checkpoint_sha256=sha256(path))
            report["families"][family] = dict(config={**vars(cfg), "architecture": "concat_vs_cascade"}, arms=arms,
                concat_history=history, cohort_audit=context["audit"], provenance=dict(filtered_rows_sha256=rows_hash,
                split_rows={s: v["n"] for s, v in views.items()}, aligned_go_vectors_identical=True,
                direct_source_manifest=direct[family]["source_manifest"] if direct else None,
                source_tsv_sha256={str(p): sha256(p) for p in sorted(Path(cfg.dataset_dir).glob(f"*{family}_*.tsv"))},
                sequences_sha256=sha256(cfg.sequences_json), cascade_training_data_hash_verified=False))
            source.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            del tensors, views, concat_model, concat_ckpt, cascade_model, cascade_ckpt
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        artifacts = generate_reports(report["families"], root)
        report["artifacts"] = {k: str(Path(p).relative_to(root)) for k, p in artifacts.items()}
        source.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        if run is not None:
            publish(run, wandb, report, source, args.create_workspace)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    if run is not None:
        run.finish()
    for family, result in report["families"].items():
        for method, arm in result["arms"].items():
            metric, rank = arm["metrics"]["test"], arm["mutation"]["pairs"]["ranking"][0]
            print(f"[TEST] {family.upper()} / {method}: macro-F1={metric['macro_f1']:.4f}; "
                  f"pair accuracy={rank['pair_accuracy']}; group macro accuracy={rank['group_macro_accuracy']}")
            if "easy" in arm["mutation"]:
                challenge = next(r for r in arm["mutation"]["easy"]["classification"] if r["cohort"] == "Anti-majority")
                print(f"[EASY CHALLENGE] {method}: n={challenge['n']}; correct={challenge['correct_n']}; "
                      f"accuracy={challenge['accuracy']}; macro-F1={challenge['macro_f1']}")
    print(f"[DONE] {source}")
    if run is not None:
        print(f"[WANDB] {run.url}")
    return {**report, "manifest_path": str(source)}


def upload_saved(path, *, project=None, entity=None, name=None, mode="online", make_workspace=False):
    source = Path(path).expanduser().resolve()
    report = json.loads(source.read_text())
    if report.get("format") != FORMAT or not report.get("artifacts"):
        raise ValueError("Expected completed concat_cascade_comparison.json")
    for relative in report["artifacts"].values():
        target = (source.parent / relative).resolve()
        if not target.is_relative_to(source.parent) or not target.is_file():
            raise ValueError(f"Missing or external artifact: {relative}")
    if make_workspace and mode != "online":
        raise ValueError("Saved workspace requires online W&B")
    config = {**report["config"], "wandb_mode": mode}
    for key, value in (("wandb_project", project), ("wandb_entity", entity), ("wandb_run_name", name)):
        if value is not None:
            config[key] = value
    run, wandb = start_wandb(config)
    report.update(config=config, wandb_url=run.url, workspace_url=None)
    try:
        publish(run, wandb, report, source, make_workspace)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()
    return run.url


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    direct = parser.add_mutually_exclusive_group()
    direct.add_argument("--direct_manifest")
    direct.add_argument("--train_concat", action="store_true")
    cascade = parser.add_mutually_exclusive_group()
    cascade.add_argument("--suite_json")
    cascade.add_argument("--cascade_checkpoints", nargs="+")
    parser.add_argument("--family", choices=("easy", "hard", "both"), default="both")
    for key in (*PATH_OVERRIDES, "cache_dir", "go_embeddings", "embedding", "input_mode"):
        parser.add_argument(f"--{key}", type=int if key == "batch_size" else str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64])
    for key, kind in (("epochs", int), ("early_stop_patience", int), ("lr", float), ("dropout", float), ("weight_decay", float)):
        parser.add_argument(f"--{key}", type=kind)
    parser.add_argument("--unweighted", action="store_true")
    parser.add_argument("--tie_tolerance", type=float, default=1e-8)
    parser.add_argument("--out_dir", default="runs/concat_vs_cascade")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--create_workspace", action="store_true")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    for key in ("wandb_project", "wandb_entity", "wandb_run_name"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--upload_existing")
    args = parser.parse_args(argv)
    if not args.tie_tolerance >= 0 or not args.tie_tolerance < float("inf"):
        parser.error("--tie_tolerance must be finite and nonnegative")
    if args.upload_existing:
        if args.no_wandb:
            parser.error("--upload_existing requires W&B")
        print("[WANDB]", upload_saved(args.upload_existing, project=args.wandb_project, entity=args.wandb_entity,
              name=args.wandb_run_name, mode=args.wandb_mode, make_workspace=args.create_workspace))
        return
    if not (args.direct_manifest or args.train_concat) or not (args.suite_json or args.cascade_checkpoints):
        parser.error("Provide --direct_manifest (or --train_concat), and --suite_json (or --cascade_checkpoints)")
    return run_comparison(args)


if __name__ == "__main__":
    main()
