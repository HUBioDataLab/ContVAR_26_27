"""Matched old/full x window none/mean x four models, with delayed test access."""
import argparse
from collections import Counter, defaultdict
from copy import copy
import json
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, WORKSPACE, data_root
from .model import LABELS, load_decoder
from .prepare import sha256_file
from .train import evaluate, load_datasets, metrics, train, write_json
from .window_prepare import find_window


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_predictions(train_frame, frame, kind):
    """Fit lookup rules on train only. Ties use the fixed LOF/GOF/NEUTRAL order."""
    order = {label: i for i, label in enumerate(LABELS)}

    def majority(labels):
        counts = Counter(labels)
        return max(LABELS, key=lambda label: (counts[label], -order[label]))

    global_label = majority(train_frame.label)
    go = {key: majority(group.label) for key, group in train_frame.groupby("matched_go_id")}
    protein_go = {key: majority(group.label) for key, group in train_frame.groupby(["protein_id", "matched_go_id"])}
    pred = []
    for row in frame.itertuples(index=False):
        label = global_label
        if kind in ("go", "protein_go"):
            label = go.get(row.matched_go_id, global_label)
        if kind == "protein_go":
            label = protein_go.get((row.protein_id, row.matched_go_id), label)
        pred.append(order[label])
    return np.asarray(pred, dtype=np.int64)


def classification_summary(frame, prediction):
    labels = frame.label_idx.to_numpy(dtype=np.int64)
    scores = np.eye(3, dtype=np.float32)[prediction]
    zeros = np.zeros(len(frame), dtype=np.float32)
    result = metrics(labels, prediction, scores, zeros, zeros, zeros, zeros, scores, "baseline")
    for key in ("reverse_mse", "reverse_mae", "direction_score_rms", "probability_direction_reverse_mse", "class_swap_mse"):
        result.pop(key)
    return result


def common_evaluation(frame, model, dataset, device, batch_size):
    selected = dataset.frame.sample_id.isin(frame.sample_id)
    indexes = np.flatnonzero(selected.to_numpy())
    if len(indexes) != len(frame):
        raise ValueError("Old test cohort is not a subset of full test")
    from torch.utils.data import Subset
    subset = Subset(dataset, indexes.tolist())
    # evaluate() selects rows using the original PairDataset row IDs.
    subset.frame = dataset.frame
    return evaluate(model, DataLoader(subset, batch_size=batch_size), device)


def pin_inputs(args):
    manifests = {}
    root = data_root(args.package)
    expected = None
    for cohort in args.cohorts:
        manifest = read_json(args.data_dir / cohort / "manifest.json")
        if expected is not None and manifest["feature_sha256"] != expected:
            raise ValueError("Cohorts were prepared from different feature versions")
        expected = manifest["feature_sha256"]
        for family in args.families:
            for split in ("train", "val", "test"):
                for head in ("parallel", "multiclass"):
                    path = args.data_dir / cohort / head / f"{family}_{split}.tsv"
                    if sha256_file(path) != manifest["prepared"][f"{family}/{split}"]["sha256"][head]:
                        raise ValueError(f"Prepared split modified: {path}")
        manifests[cohort] = sha256_file(args.data_dir / cohort / "manifest.json")
    paths = {"global": root / "embeddings/phase3_seq_phase1_global.h5",
             "local": root / "embeddings/phase3_seq_phase1_local.h5",
             "window_mean": args.window_cache}
    # The mean cache records the checked raw-window hash. File bytes/mtime for the
    # raw input are captured below; preprocessing performs the full raw hash check.
    for name, path in paths.items():
        if sha256_file(path) != expected[name]:
            raise ValueError(f"Frozen feature differs from preparation: {path}")
    with h5py.File(args.window_cache, "r") as h:
        if not h.attrs.get("complete", False) or h.attrs.get("source_window_sha256") != expected["window_raw"]:
            raise ValueError("Mean cache provenance mismatch")
    return manifests, expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=WORKSPACE.parent / "New_Decoder_Data_with_local_Context")
    p.add_argument("--data-dir", type=Path, default=WORKSPACE / "prepared_window")
    p.add_argument("--window-cache", type=Path, default=WORKSPACE / "cache/window_mean.h5")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cohorts", nargs="+", choices=("old", "full"), default=["old", "full"])
    p.add_argument("--families", nargs="+", choices=("easy", "hard"), default=["easy", "hard"])
    p.add_argument("--window-modes", nargs="+", choices=("none", "mean"), default=["none", "mean"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse completed model runs with the exact pinned plan")
    args = p.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        p.error("Training limits must be positive")
    if len(set(args.cohorts)) != len(args.cohorts) or len(set(args.families)) != len(args.families) or len(set(args.window_modes)) != len(args.window_modes):
        p.error("Duplicate experiment dimensions")
    if args.out.exists() and not args.resume:
        p.error("Choose a new output or pass --resume for a completed-run continuation")
    if args.smoke:
        args.epochs = 1
    args.balance = False
    args.exclude_augmented_train = False
    args.lambda_reverse = 0.
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    print("Pinning tables and feature inputs...", flush=True)
    manifests, feature_hashes = pin_inputs(args)
    total = len(args.cohorts) * len(args.families) * len(args.window_modes) * 4
    plan = dict(smoke=args.smoke, seed=args.seed, max_epochs=args.epochs, patience=args.patience,
                batch_size=args.batch_size, width=args.width, attention_heads=args.attention_heads,
                dropout=args.dropout, lr=args.lr, lambda_reverse=0., device=args.device,
                cohorts=args.cohorts, families=args.families, window_modes=args.window_modes,
                distinct_training_runs=total, prepared_manifest_sha256=manifests, feature_sha256=feature_hashes,
                go_sha256=read_json(args.data_dir / args.cohorts[0] / "manifest.json")["go_sha256"],
                window_definition="WT and VAR separately: float32 mean over valid residue vectors; shared window adapter; no comparison reader",
                model_selection="best validation Macro-F1 for every fixed architecture; no test-based selection",
                test_access="only after all requested training runs complete and checkpoint selection is frozen",
                evaluation_context="Original source test partitions have historical exposure; these are controlled follow-up experiments, not an unseen new benchmark",
                source_code_sha256={path.name: sha256_file(path) for path in Path(__file__).parent.glob("*.py")},
                environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                                 pandas=pd.__version__, h5py=h5py.__version__, sklearn=sklearn.__version__,
                                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "plan.json").exists() and read_json(args.out / "plan.json") != plan:
        raise ValueError("Resume plan/code/input mismatch; choose a new output directory")
    write_json(args.out / "plan.json", plan)
    started = time.monotonic()
    fitted = {}
    for cohort in args.cohorts:
        for family in args.families:
            for pool in args.window_modes:
                store = FeatureStore(args.package, window_pool=pool, window_cache=args.window_cache)
                try:
                    cfg = copy(args)
                    cfg.family, cfg.window_pool, cfg.data_dir = family, pool, args.data_dir / cohort
                    ds = {h: load_datasets(cfg, h, store) for h in ("parallel", "multiclass")}
                    store.preload([d for sets in ds.values() for d in sets.values()])
                    for head in ("parallel", "multiclass"):
                        for mode in ("score_difference", "joint"):
                            name = f"{cohort}/{family}/{pool}/{head}_{mode}"
                            dest = args.out / name
                            if args.resume and (dest / "results.json").is_file():
                                result = read_json(dest / "results.json")
                                if not (dest / "best.pt").is_file():
                                    raise ValueError(f"Completed result has no checkpoint: {dest}")
                                print(f"RESUME {name}", flush=True)
                            else:
                                print(f"TRAIN {len(fitted)+1}/{total} {name}", flush=True)
                                result = train(cfg, head, mode, ds[head], torch.device(args.device), dest)
                            result["cohort"] = cohort
                            fitted[name] = result
                            write_json(args.out / "training_progress.json", dict(completed=len(fitted), total=total, models=fitted))
                finally:
                    store.close()
    selection = {name: dict(best_epoch=r["best_epoch"], validation_macro_f1=r["val"]["macro_f1"],
                           checkpoint_sha256=sha256_file(args.out / name / "best.pt")) for name, r in fitted.items()}
    write_json(args.out / "selection_before_test.json", selection)
    baselines = {}
    for cohort in args.cohorts:
        for family in args.families:
            train_frame = pd.read_csv(args.data_dir / cohort / "multiclass" / f"{family}_train.tsv", sep="\t")
            baseline_key = f"{cohort}/{family}"
            baselines[baseline_key] = {}
            for split in ("val", "test"):
                frame = pd.read_csv(args.data_dir / cohort / "multiclass" / f"{family}_{split}.tsv", sep="\t")
                if args.smoke:
                    frame = frame.sample(min(64, len(frame)), random_state=args.seed).reset_index(drop=True)
                predictions = frame[["sample_id", "protein_id", "snp", "matched_go_id", "label", "label_idx"]].copy()
                for kind in ("majority", "go", "protein_go"):
                    pred = baseline_predictions(train_frame, frame, kind)
                    baselines[baseline_key].setdefault(kind, {})[split] = classification_summary(frame, pred)
                    predictions[kind] = [LABELS[i] for i in pred]
                predictions.to_csv(args.out / cohort / family / f"{split}_baseline_predictions.tsv", sep="\t", index=False)
            if cohort == "full" and "old" in args.cohorts and not args.smoke:
                common = pd.read_csv(args.data_dir / "old/multiclass" / f"{family}_test.tsv", sep="\t")
                for kind in ("majority", "go", "protein_go"):
                    pred = baseline_predictions(train_frame, common, kind)
                    baselines[baseline_key][kind]["common_old_test"] = classification_summary(common, pred)
            for pool in args.window_modes:
                store = FeatureStore(args.package, window_pool=pool, window_cache=args.window_cache)
                try:
                    cfg = copy(args)
                    cfg.family, cfg.data_dir = family, args.data_dir / cohort
                    datasets = {h: load_datasets(cfg, h, store, ("test",))["test"] for h in ("parallel", "multiclass")}
                    store.preload(datasets.values())
                    for head in ("parallel", "multiclass"):
                        for mode in ("score_difference", "joint"):
                            name = f"{cohort}/{family}/{pool}/{head}_{mode}"
                            dest = args.out / name
                            model, checkpoint = load_decoder(dest / "best.pt", args.device)
                            if checkpoint["go_sha256"] != store.go.artifact_sha256:
                                raise ValueError("GO checkpoint mismatch")
                            result = fitted[name]
                            dataset = datasets[head]
                            result["test"], predictions = evaluate(model, DataLoader(dataset, batch_size=args.batch_size), args.device)
                            predictions.to_csv(dest / "test_predictions.tsv", sep="\t", index=False)
                            if cohort == "full" and "old" in args.cohorts and not args.smoke:
                                old_frame = pd.read_csv(args.data_dir / "old" / head / f"{family}_test.tsv", sep="\t")
                                result["common_old_test"], common = common_evaluation(old_frame, model, dataset, args.device, args.batch_size)
                                common.to_csv(dest / "common_old_test_predictions.tsv", sep="\t", index=False)
                            write_json(dest / "results.json", result)
                            print(f"TEST {name} n={result['test']['n']} F1={result['test']['macro_f1']:.4f}", flush=True)
                finally:
                    store.close()
    report = dict(plan=plan, selection_before_test=selection, models=fitted, baselines=baselines,
                  distinct_training_runs=len(fitted), seconds_this_invocation=time.monotonic()-started,
                  test_used_for_selection=False)
    write_json(args.out / "summary.json", report)
    from .window_report import report_suite
    report_suite(args.out)
    print(f"COMPLETED {len(fitted)} training runs; report: {args.out / 'RAPOR.md'}", flush=True)


if __name__ == "__main__":
    main()
