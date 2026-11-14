"""Evaluate frozen EASY/HARD cascade checkpoints; no retraining or test tuning.

python -m phase_3_variant_decoder.evaluate_mutation_sensitivity --help
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase_2_decoder.config import pick_device
from phase_3_variant_decoder.cascade import attach_checkpoint_go, load_cascade, stage_outputs
from phase_3_variant_decoder.config import LABELS, VariantDecoderConfig
from phase_3_variant_decoder.data.pair_dataset import build_pair_tensors
from phase_3_variant_decoder.data.variant_dataset import prepare, load_sequences
from phase_3_variant_decoder.audit_mutation_cohorts import audit_family
from phase_3_variant_decoder.mutation_sensitivity import (
    class_metrics, dataset_audit, easy_diagnostic, pair_diagnostic, signed_index,
)
from phase_3_variant_decoder.mutation_sensitivity_reports import (
    generate_report, log_wandb, create_workspace,
)

FORMAT = "contvar_mutation_sensitivity_v2"
LEGACY_FORMAT = "contvar_mutation_sensitivity_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoints_from_suite(path, *, family="both", embedding="seq_phase1", input_mode="both", seed=42):
    """Select by a prespecified configuration, NEVER by best test performance."""
    suite = json.loads(Path(path).read_text())
    if suite.get("format") != "contvar_cascade_diagnostic_suite_v1":
        raise ValueError("Expected diagnostic_suite.json from the cascade diagnostic suite")
    checkpoints = []
    for wanted in (("easy", "hard") if family == "both" else (family,)):
        key = f"{wanted}_{embedding}_{input_mode}_seed{seed}"
        result = suite.get("results", {}).get(key)
        if result is None or not result.get("checkpoint"):
            raise ValueError(f"Suite has no saved checkpoint for {key}; pass --checkpoints explicitly")
        checkpoints.append(result["checkpoint"])
    return checkpoints


def evaluate_one(checkpoint, output: Path, *, overrides=None, cache_dir=None, tolerance=1e-8, quiet=False):
    output.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_cascade(checkpoint, "cpu")
    saved = ckpt["config"]
    cfg = VariantDecoderConfig(saved["family"], saved["embedding_type"], saved["input_mode"])
    for key, value in saved.items():
        setattr(cfg, key, value)
    for key, value in (overrides or {}).items():
        if value is not None:
            if value == "":
                raise ValueError(f"Empty --{key}; check exported shell variables")
            setattr(cfg, key, value)
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    threshold = float(ckpt["effect_threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved effect threshold")
    # GO vectors are self-contained. Do not require the old NPZ path to exist.
    cfg.architecture = "legacy"
    cfg.validate()
    cfg.architecture = "cascade"
    frame, vocab, filter_report = prepare(cfg)
    train = frame.loc[frame.split == "train"].copy()
    test = frame.loc[frame.split == "test"].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"{cfg.family}: empty filtered train or test partition")
    cohort_audit, source_test, raw_pairs, raw_frame = audit_family(
        cfg.family, cfg.dataset_dir, load_sequences(cfg.sequences_json),
        cfg.max_seq_len, cfg.drop_contradictory)
    source_layout = cohort_audit["source_layout"]
    raw_train, raw_test = [raw_frame.loc[raw_frame.split == split] for split in ("train", "test")]
    covered = source_test.loc[source_test.exclusion_reason.eq("")].reset_index(drop=True)
    identity = ["protein_id", "matched_go_id", "snp", "label"]
    pd.testing.assert_frame_equal(test[identity], covered[identity], check_dtype=False)
    test["source_row"] = covered.source_row.to_numpy()
    # The manifests include unscorable rows/pairs; never invent their predictions.
    source_test.to_csv(output / "source_test_row_audit.csv", index=False)
    raw_pairs.to_csv(output / "raw_pair_manifest.csv", index=False)
    (output / "cohort_audit.json").write_text(json.dumps(cohort_audit, indent=2, ensure_ascii=False) + "\n")
    if cfg.family == "hard":
        # Fail loudly if a changed data directory violates the claimed hard split.
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = set(raw_frame.loc[raw_frame.split == a, "protein_id"]) & set(raw_frame.loc[raw_frame.split == b, "protein_id"])
            if overlap:
                raise ValueError(f"HARD split is not protein-disjoint: {a}/{b} overlap ({len(overlap)} proteins)")
    data_audits = {"Raw TSV": dataset_audit(raw_train, raw_test),
                   "Model-filtered": dataset_audit(train, test)}
    if not quiet:
        print(f"[EVALUATE] {cfg.family.upper()} <- {checkpoint}", flush=True)
        print(filter_report.render(), flush=True)
    # Only test representations are needed, reducing memory versus loading all
    # train/validation H5 data. Train labels are used solely for the lookup rule.
    device = torch.device(pick_device(cfg.cuda))
    tensors = build_pair_tensors(test, cfg, cache_dir=cache_dir, verbose=not quiet).to(device)
    attach_checkpoint_go(tensors, vocab, ckpt)
    if tensors.input_dim != model.spec["protein_dim"]:
        raise ValueError("Input embedding dimension differs from checkpoint")
    model.to(device).eval()
    view = tensors.split_view("test")
    # Do NOT use predict(): it leaves direction outputs missing on neutral-routed
    # rows. Both heads are scored here for diagnostics; hard routing is unchanged.
    effect = stage_outputs(model.effect_stage, tensors, view, batch_size=cfg.batch_size)["probability"]
    direction = stage_outputs(model.direction_stage, tensors, view, batch_size=cfg.batch_size)["probability"]
    test["effect_score"], test["direction_score"] = effect, direction
    test["signed_index"] = signed_index(effect, direction)
    test["prediction"] = np.where(effect < threshold, "NEUTRAL", np.where(direction >= .5, "LOF", "GOF"))
    test["diagnostic_row_id"] = np.arange(len(test))
    result = {
        "config": vars(cfg), "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "source_layout": source_layout, "filter_report": asdict(filter_report),
        "cohort_audit": cohort_audit,
        "protocol": "Raw-train EASY cohort frozen before filtering; covered original HARD row pairs primary; clean subset supplementary",
        "sequences_sha256": sha256(cfg.sequences_json),
        "data_audits": data_audits, "effect_threshold": threshold, "direction_threshold": .5,
        "tie_tolerance": tolerance, "score_formula": "effect_score * (1 - 2 * direction_score)",
        "score_interpretation": "Uncalibrated signed decision index, not measured function or fitness",
        "full_test": class_metrics(test, "prediction"),
        "test_used_for_selection": False, "trained_in_this_experiment": False,
        "data_files_sha256": {str(path.resolve()): sha256(path) for path in sorted(Path(cfg.dataset_dir).glob(f"*{cfg.family}_*.tsv"))},
    }
    if cfg.family == "easy":
        result["easy"], test = easy_diagnostic(train, test, reference_train=raw_train)
        if result["easy"]["audit"]["anti_majority_rows"] != cohort_audit["easy"]["raw_cohort_covered"]:
            raise ValueError("EASY raw-cohort membership drifted after input preparation")
    result["pairs"], pairs, groups = pair_diagnostic(test, tolerance, clean=False)
    result["clean_pairs"], clean_pairs, clean_groups = pair_diagnostic(test, tolerance)
    if len(pairs) != int(raw_pairs.model_covered.sum()):
        raise ValueError("Covered raw pair count differs from prediction pair count")
    pair_identity = ["protein_id", "matched_go_id", "lower_source_row", "higher_source_row", "lower_label", "higher_label"]
    expected_pairs = raw_pairs.loc[raw_pairs.model_covered, pair_identity].sort_values(pair_identity).reset_index(drop=True)
    scored_pairs = pairs[pair_identity].sort_values(pair_identity).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_pairs, scored_pairs, check_dtype=False)
    test.to_csv(output / "test_predictions.csv", index=False)
    pairs.to_csv(output / "variant_pairs.csv", index=False)
    groups.to_csv(output / "protein_GO_groups.csv", index=False)
    clean_pairs.to_csv(output / "clean_variant_pairs.csv", index=False)
    clean_groups.to_csv(output / "clean_protein_GO_groups.csv", index=False)
    result["figures"] = generate_report(result, output)
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if not quiet:
        r = result["pairs"]["ranking"][0]
        print(f"[{cfg.family.upper()}] {r['n_groups']} covered groups / {r['n_pairs']} original row pairs | "
              f"pair accuracy={r['pair_accuracy']} | group macro={r['group_macro_accuracy']}", flush=True)
        c = result["clean_pairs"]["ranking"][0]
        print(f"[{cfg.family.upper()} supplementary clean] {c['n_groups']} groups / {c['n_pairs']} pairs | pair accuracy={c['pair_accuracy']}", flush=True)
        if "easy" in result:
            r = next(x for x in result["easy"]["classification"] if x["cohort"] == "Anti-majority" and x["model"] == "Cascade")
            print(f"[EASY challenge] n={r['n']} correct={r['correct_n']} macro-F1={r['macro_f1']}", flush=True)
    del model, tensors, view
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def upload_report(report, root, args):
    import wandb
    cfg = next(iter(report["results"].values()))["config"]
    run = wandb.init(project=args.wandb_project or cfg["wandb_project"],
        entity=args.wandb_entity or cfg.get("wandb_entity"), name=args.name,
        job_type="mutation-sensitivity-evaluation", group="mutation-sensitivity",
        tags=["phase3", "cascade", "mutation-sensitivity", "posthoc", *report["results"]],
        mode=args.wandb_mode,
        config={"experiment": report["format"], "trained_in_this_experiment": False,
                "test_used_for_selection": False, "tie_tolerance": report["tie_tolerance"],
                "checkpoints": {family: r["checkpoint"] for family, r in report["results"].items()}})
    if run is None:
        raise RuntimeError("W&B did not create a run; use --no_wandb for local-only evaluation")
    try:
        run.config.update({"mutation_diagnostic_id": run.id})
        log_wandb(run, wandb, report["results"], root)
        report["wandb_url"] = run.url if args.wandb_mode == "online" else None
        if args.create_workspace:
            report["workspace_url"] = create_workspace(run, report["results"])
            run.summary["mutation_sensitivity_workspace"] = report["workspace_url"]
        run.summary["mutation_sensitivity_output_dir"] = str(root)
        run.finish()
    except Exception:
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoints", "--checkpoint", nargs="+", help="One EASY and/or one HARD cascade decoder.pt")
    source.add_argument("--suite_json", help="Use saved checkpoints from a diagnostic_suite.json")
    source.add_argument("--upload_existing", help="Upload existing mutation_sensitivity.json without inference")
    parser.add_argument("--family", choices=["easy", "hard", "both"], default="both", help="Suite selection only")
    parser.add_argument("--embedding", default="seq_phase1", help="Suite selection only")
    parser.add_argument("--input_mode", choices=["global", "local", "both"], default="both", help="Suite selection only")
    parser.add_argument("--seed", type=int, default=42, help="Suite selection only; never selects by test score")
    for name in ("dataset_dir", "sequences_json", "global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "cuda", "cache_dir"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--out_dir", default="runs/mutation_sensitivity")
    parser.add_argument("--tie_tolerance", type=float, default=1e-8)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_mode", choices=["online", "offline"], default="online")
    parser.add_argument("--wandb_project")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--name", default="phase3-mutation-sensitivity-easy-hard")
    parser.add_argument("--create_workspace", action="store_true", help="Create a dedicated saved view with the section at the top (requires wandb-workspaces)")
    args = parser.parse_args()
    if not np.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("--tie_tolerance must be finite and nonnegative")
    if args.create_workspace:
        if args.no_wandb or args.wandb_mode != "online":
            parser.error("--create_workspace requires online W&B")
        try:
            import wandb_workspaces.workspaces  # noqa: F401
        except ImportError:
            parser.error("--create_workspace requires: python -m pip install wandb-workspaces")
    if args.upload_existing:
        path = Path(args.upload_existing).resolve()
        report = json.loads(path.read_text())
        if report.get("format") not in (FORMAT, LEGACY_FORMAT):
            parser.error("Expected a mutation_sensitivity.json")
        root = path.parent
    else:
        checkpoints = args.checkpoints or checkpoints_from_suite(args.suite_json,
            family=args.family, embedding=args.embedding, input_mode=args.input_mode, seed=args.seed)
        families = []
        for path in checkpoints:
            if not Path(path).is_file():
                parser.error(f"Checkpoint not found: {path}; pass existing --checkpoints")
            _, ckpt = load_cascade(path, "cpu")
            families.append(ckpt["config"]["family"])
        if len(set(families)) != len(families):
            parser.error("Use at most one checkpoint per family in a run; evaluate other seeds in separate runs")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="mutation_sensitivity_", dir=Path(args.out_dir).resolve()))
        overrides = {name: getattr(args, name) for name in ("dataset_dir", "sequences_json", "global_h5", "local_h5",
                     "global_h5_secondary", "local_h5_secondary", "cuda", "batch_size")}
        report = {"format": FORMAT, "tie_tolerance": args.tie_tolerance,
                  "trained_in_this_experiment": False, "test_used_for_selection": False, "results": {}}
        for family, checkpoint in zip(families, checkpoints):
            cache = str(Path(args.cache_dir) / family) if args.cache_dir else None
            report["results"][family] = evaluate_one(checkpoint, root / family,
                overrides=overrides, cache_dir=cache, tolerance=args.tie_tolerance)
        path = root / "mutation_sensitivity.json"
    # Persist BEFORE any network request. Upload can be retried without inference.
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_wandb:
        try:
            upload_report(report, root, args)
        except Exception:
            print(f"[SAVED] Local evaluation is complete: {path}\nRetry with --upload_existing {path}", flush=True)
            raise
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[DONE] {path}")
    for name in ("wandb_url", "workspace_url"):
        if report.get(name):
            print(f"  {name}: {report[name]}")


if __name__ == "__main__":
    main()
