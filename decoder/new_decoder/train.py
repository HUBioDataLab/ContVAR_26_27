"""Train one or all four decoder experiments without updating the encoder."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    CHECKPOINT_FORMAT,
    DEFAULT_DATA_ROOT,
    DEFAULT_RUNS_DIR,
    add_training_arguments,
    validate_training_arguments,
)
from .data import FeatureStore, PairDataset, move_batch
from .evaluation import evaluate
from .go_embeddings import GOEmbeddingTable
from .model import (
    LABELS,
    PairedGODecoder,
    classification_loss,
    deterministic,
    load_decoder,
    reverse_loss,
)
from .prepare import DEFAULT_OUT


def write_json(path: str | Path, value: Any) -> None:
    """Serialize finite experiment metadata without non-standard NaN values."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch for a recorded training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    args: argparse.Namespace,
    head: str,
    pair_mode: str,
    datasets: dict[str, PairDataset],
    device: str | torch.device,
    output: Path,
) -> dict[str, Any]:
    """Fit a decoder and select its checkpoint using validation Macro-F1 only."""
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    seed_everything(args.seed)
    model = PairedGODecoder(
        go_dim=datasets["train"].store.go.dimension,
        width=args.width,
        heads=args.attention_heads,
        head=head,
        pair_mode=pair_mode,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            shuffle=(s == "train"),
            generator=generator if s == "train" else None,
            num_workers=0,
        )
        for s, d in datasets.items()
    }
    counts = np.bincount(datasets["train"].frame.label_idx, minlength=3)
    class_weights = pos_weight = None
    if args.balance:
        if (counts == 0).any():
            raise ValueError("Class balancing requires all three classes in train")
        class_weights = torch.tensor(
            len(datasets["train"]) / (3 * counts), dtype=torch.float32, device=device
        )
        pos_weight = torch.tensor(
            (counts.sum() - counts[:2]) / counts[:2], dtype=torch.float32, device=device
        )
    record = dict(
        model_spec=model.spec,
        labels=list(LABELS),
        seed=args.seed,
        device=str(device),
        encoder_frozen=True,
        smoke=args.smoke,
        lambda_reverse=args.lambda_reverse,
        single_mutation_only=True,
        parameter_count=sum(p.numel() for p in model.parameters()),
        class_counts=counts.tolist(),
        balance=args.balance,
        class_weights=class_weights.tolist() if class_weights is not None else None,
        pos_weight=pos_weight.tolist() if pos_weight is not None else None,
        training_rows=len(datasets["train"]),
        validation_rows=len(datasets["val"]),
        selection="validation macro-F1 on supplied available-cache cohort",
        reverse_definition="d = logit_GOF - logit_LOF; E[(d_forward+d_reverse)^2]",
        reverse_is_structural=(pair_mode == "score_difference"),
        parallel_decision="threshold=.5; both positive -> larger head; neither -> NEUTRAL",
        source_layout=head,
        evaluation_augmented_excluded=True,
        multiple_label_policy="retain source labels; report groups separately",
        data_dir=str(datasets["train"].path.parent.resolve()),
        prepared_audit_sha256=hashlib.sha256(
            (datasets["train"].path.parent / "audit.json").read_bytes()
        ).hexdigest(),
        split_sha256={
            s: hashlib.sha256(d.path.read_bytes()).hexdigest() for s, d in datasets.items()
        },
        train_table_sha256=hashlib.sha256(datasets["train"].path.read_bytes()).hexdigest(),
        go_sha256=datasets["train"].store.go.artifact_sha256,
        shared_go_reader_sha256=hashlib.sha256(
            Path(inspect.getfile(GOEmbeddingTable)).read_bytes()
        ).hexdigest(),
        args={
            k: str(v) if isinstance(v, Path) else v
            for k, v in {**vars(args), "head": head, "pair_mode": pair_mode}.items()
        },
    )
    write_json(output / "config.json", record)
    history, best, stale = [], -1.0, 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3)
        for batch in loaders["train"]:
            wt, var, go = move_batch(batch, device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            forward = model(wt, var, go)
            cls = classification_loss(
                forward,
                labels,
                head,
                class_weights,
                pos_weight,
                batch["binary_target"].to(device) if head == "parallel" else None,
            )
            penalty = cls.new_zeros(())
            if args.lambda_reverse and pair_mode == "joint":
                # A separate deterministic pass isolates reversal from dropout noise.
                with deterministic(model):
                    penalty = reverse_loss(model(wt, var, go), model(var, wt, go))
            loss = cls + args.lambda_reverse * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += np.array([loss.item(), cls.item(), penalty.item()]) * len(labels)
        validation, _ = evaluate(model, loaders["val"], device)
        row = dict(
            epoch=epoch,
            train_loss=totals[0] / len(datasets["train"]),
            train_classification_loss=totals[1] / len(datasets["train"]),
            train_reverse_loss=(totals[2] / len(datasets["train"]))
            if args.lambda_reverse or pair_mode == "score_difference"
            else None,
            validation=validation,
        )
        history.append(row)
        write_json(output / "history.json", history)
        if validation["macro_f1"] > best:
            best, stale = validation["macro_f1"], 0
            torch.save(
                dict(
                    format=CHECKPOINT_FORMAT,
                    model_spec=model.spec,
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    labels=list(LABELS),
                    epoch=epoch,
                    validation_macro_f1=best,
                    single_mutation_only=True,
                    smoke=args.smoke,
                    family=args.family,
                    source_layout=head,
                    evaluation_augmented_excluded=True,
                    seed=args.seed,
                    lambda_reverse=args.lambda_reverse,
                    go_sha256=record["go_sha256"],
                ),
                output / "best.pt",
            )
        else:
            stale += 1
        print(
            f"{args.family}/{head}/{pair_mode}/lambda={args.lambda_reverse} epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_F1={validation['macro_f1']:.4f} reverse_MSE={validation['reverse_mse']:.6g}",
            flush=True,
        )
        if stale >= args.patience:
            break
    model, checkpoint = load_decoder(output / "best.pt", device)
    final = dict(
        smoke=args.smoke,
        best_epoch=checkpoint["epoch"],
        epochs_run=len(history),
        family=args.family,
        head=head,
        pair_mode=pair_mode,
        lambda_reverse=args.lambda_reverse,
        single_mutation_only=True,
        source_layout=head,
        evaluation_augmented_excluded=True,
        n_train=len(datasets["train"]),
        n_val=len(datasets["val"]),
        seconds=time.monotonic() - started,
        parameter_count=record["parameter_count"],
    )
    for split in ("val", "test"):
        if split in loaders:
            final[split], predictions = evaluate(model, loaders[split], device)
            predictions.to_csv(output / f"{split}_predictions.tsv", sep="\t", index=False)
    write_json(output / "results.json", final)
    return final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--family", choices=("easy", "hard"), default="hard")
    p.add_argument("--head", choices=("parallel", "multiclass"), default="multiclass")
    p.add_argument("--pair-mode", choices=("joint", "score_difference"), default="joint")
    p.add_argument("--four-experiments", action="store_true")
    p.add_argument("--lambda-reverse", type=float, default=0.0)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--exclude-augmented-train", action="store_true")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test after validation checkpoint selection",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch, at most 128 train/64 val or test; not a scientific result",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_RUNS_DIR / "run")
    add_training_arguments(p)
    args = p.parse_args()
    try:
        validate_training_arguments(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.out.exists():
        p.error("--out must be a new directory to preserve previous runs")
    if args.smoke:
        args.epochs = 1
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    store = FeatureStore(args.data_root)
    try:
        splits = ("train", "val", "test") if args.evaluate_test else ("train", "val")
        experiments = (
            [(h, m) for h in ("parallel", "multiclass") for m in ("score_difference", "joint")]
            if args.four_experiments
            else [(args.head, args.pair_mode)]
        )
        summary = {}
        datasets_by_head = {
            head: load_datasets(args, head, store, splits)
            for head in sorted({h for h, m in experiments})
        }
        store.preload([d for datasets in datasets_by_head.values() for d in datasets.values()])
        for head, mode in experiments:
            name = f"{head}_{mode}"
            out = args.out / name if args.four_experiments else args.out
            summary[name] = train(args, head, mode, datasets_by_head[head], device, out)
        if args.four_experiments:
            write_json(args.out / "summary.json", summary)
    finally:
        store.close()


def load_datasets(
    args: argparse.Namespace,
    head: str,
    store: FeatureStore,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, PairDataset]:
    """Load only requested native partitions; keep augmented data out of evaluation."""
    folder = args.data_dir / head
    datasets = {}
    for split in splits:
        dataset = PairDataset(
            folder / f"{args.family}_{split}.tsv",
            store,
            limit=(128 if split == "train" else 64) if args.smoke else None,
            seed=args.seed,
            exclude_augmented=(split != "train" or args.exclude_augmented_train),
            head=head,
        )
        if split != "train" and dataset.frame.augmented.astype(str).str.lower().eq("true").any():
            raise ValueError("Augmented evaluation records are not allowed")
        datasets[split] = dataset
    return datasets


if __name__ == "__main__":
    main()
