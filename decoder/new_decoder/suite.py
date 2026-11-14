"""Full four-model comparison on easy/hard with validation-only lambda selection.

Fit 6 distinct models per family: four baselines and lambda=.1 joint controls.
Score-difference lambda controls are mathematically redundant and not retrained.
All training completes before any test metrics are evaluated.
"""
from __future__ import annotations

import argparse
from copy import copy
import json
import inspect
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNS_DIR,
    add_training_arguments,
    validate_training_arguments,
)
from .data import FeatureStore
from .evaluation import evaluate
from .go_embeddings import GOEmbeddingTable
from .model import load_decoder
from .prepare import DEFAULT_OUT, sha256_file
from .train import load_datasets, train, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR / "full_seed42")
    p.add_argument("--smoke", action="store_true")
    add_training_arguments(p)
    args = p.parse_args()
    if args.out.exists():
        p.error("Choose a new --out directory")
    try:
        validate_training_arguments(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.smoke:
        args.epochs = 1
    args.balance = False
    args.exclude_augmented_train = False
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    started = time.monotonic()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    # Pin source files before the first optimizer step; refuse modified tables.
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            for head in ("parallel", "multiclass"):
                path = args.data_dir / head / f"{family}_{split}.tsv"
                if sha256_file(path) != manifest["prepared"][f"{family}/{split}"]["sha256"][head]:
                    raise ValueError(f"Prepared table changed: {path}")
    plan = dict(
        smoke=args.smoke,
        seed=args.seed,
        max_epochs=args.epochs,
        patience=args.patience,
        source_layouts={
            "parallel": "native parallel layout",
            "multiclass": "native multiclass layout",
        },
        evaluation_augmented_excluded=True,
        multiple_label_policy="retain source labels and report groups",
        balance_classes=False,
        lambda_candidates=[0.0, 0.1],
        model_selection="best validation macro-F1; ties prefer lambda=0",
        test_access="only after all 12 distinct model training runs complete",
        prepared_manifest_sha256=sha256_file(args.data_dir / "manifest.json"),
        source_code_sha256={p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        shared_go_reader_sha256=sha256_file(Path(inspect.getfile(GOEmbeddingTable))),
    )
    write_json(args.out / "plan.json", plan)
    fitted = {}
    for family in ("easy", "hard"):
        store = FeatureStore(args.data_root)
        try:
            args.family = family
            ds = {head: load_datasets(args, head, store) for head in ("parallel", "multiclass")}
            store.preload([d for sets in ds.values() for d in sets.values()])
            fitted[family] = {}
            for head in ("parallel", "multiclass"):
                for mode in ("score_difference", "joint"):
                    for lam in [0.0, 0.1] if mode == "joint" else [0.0]:
                        cfg = copy(args)
                        cfg.head = head
                        cfg.pair_mode = mode
                        cfg.lambda_reverse = lam
                        name = f"{head}_{mode}_lambda{lam:g}"
                        dest = args.out / family / name
                        result = train(cfg, head, mode, ds[head], torch.device(args.device), dest)
                        fitted[family][name] = result
                        write_json(args.out / "training_progress.json", fitted)
        finally:
            store.close()
    selection = {}
    for family, models in fitted.items():
        selection[family] = {}
        for head in ("parallel", "multiclass"):
            for mode in ("score_difference", "joint"):
                names = [
                    f"{head}_{mode}_lambda{lam:g}"
                    for lam in ([0.0, 0.1] if mode == "joint" else [0.0])
                ]
                selection[family][f"{head}_{mode}"] = max(
                    names, key=lambda n: models[n]["val"]["macro_f1"]
                )
    write_json(args.out / "selection_before_test.json", selection)
    # The test dataset is loaded for the first time after selection is frozen.
    for family, models in fitted.items():
        args.family = family
        store = FeatureStore(args.data_root)
        try:
            ds = {
                h: load_datasets(args, h, store, ("test",))["test"]
                for h in ("parallel", "multiclass")
            }
            store.preload(ds.values())
            for name, result in models.items():
                dest = args.out / family / name
                model, checkpoint = load_decoder(dest / "best.pt", args.device)
                if checkpoint["go_sha256"] != store.go.artifact_sha256:
                    raise ValueError("GO input differs from training")
                test, predictions = evaluate(
                    model, DataLoader(ds[result["head"]], batch_size=args.batch_size), args.device
                )
                result["test"] = test
                predictions.to_csv(dest / "test_predictions.tsv", sep="\t", index=False)
                write_json(dest / "results.json", result)
                print(
                    f"TEST {family}/{name}: n={test['n']} macro-F1={test['macro_f1']:.4f} reverse={test['reverse_mse']:.6g}",
                    flush=True,
                )
        finally:
            store.close()
    report = dict(
        plan=plan,
        selection_before_test=selection,
        models=fitted,
        distinct_training_runs=sum(len(m) for m in fitted.values()),
        total_seconds=time.monotonic() - started,
        score_difference_lambda_control="No retraining: reversal penalty is structurally zero; baseline is the control",
        test_used_for_selection=False,
    )
    write_json(args.out / "summary.json", report)
    print(f"Completed suite in {report['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
