"""Evaluate a saved starter checkpoint against a prepared cached-feature split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_DATA_ROOT
from .data import FeatureStore, PairDataset
from .evaluation import evaluate
from .model import load_decoder
from .prepare import DEFAULT_OUT
from .train import write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--family", choices=("easy", "hard"), default="hard")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.batch_size < 1:
        p.error("Choose a new --out directory and a positive batch size")
    torch.set_num_threads(4)
    model, checkpoint = load_decoder(args.checkpoint, args.device)
    if checkpoint.get("single_mutation_only") is not True:
        raise ValueError(
            "Checkpoint predates the single-mutation evidence filter; use a newly trained checkpoint"
        )
    if checkpoint.get("family") != args.family:
        raise ValueError("Checkpoint family differs from the selected easy/hard protocol")
    store = FeatureStore(args.data_root)
    try:
        if store.go.artifact_sha256 != checkpoint["go_sha256"]:
            raise ValueError("GO embedding artifact differs from training")
        head = checkpoint["model_spec"]["head"]
        dataset = PairDataset(
            args.data_dir / head / f"{args.family}_{args.split}.tsv",
            store,
            exclude_augmented=args.split != "train",
            head=head,
        )
        store.preload([dataset])
        report, predictions = evaluate(
            model, DataLoader(dataset, batch_size=args.batch_size), args.device
        )
        args.out.mkdir(parents=True)
        write_json(args.out / "metrics.json", report)
        predictions.to_csv(args.out / "predictions.tsv", sep="\t", index=False)
        print(
            f"{len(dataset)} rows; macro-F1={report['macro_f1']:.4f}; "
            f"reverse-MSE={report['reverse_mse']:.6g}"
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
