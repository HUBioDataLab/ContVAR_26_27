"""Train one or all four decoder experiments without updating the encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset, PACKAGE, move_batch
from .model import LABELS, PairedGODecoder, classification_loss, deterministic, load_decoder, reverse_loss
from .prepare import DEFAULT_OUT


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(labels, prediction, scores, d, reverse_d, index, reverse_index, reverse_scores, head):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1, 2], zero_division=0)
    out = dict(n=len(labels), macro_f1=float(f1.mean()), accuracy=float((labels == prediction).mean()),
               balanced_accuracy=float(recall.mean()),
               mcc=float(matthews_corrcoef(labels, prediction))
               if np.unique(np.concatenate([labels,prediction])).size > 1 else 0.,
               reverse_mse=float(np.mean((d + reverse_d) ** 2)),
               reverse_mae=float(np.mean(np.abs(d + reverse_d))),
               direction_score_rms=float(np.sqrt(np.mean(d ** 2))),
               probability_direction_reverse_mse=float(np.mean((index + reverse_index) ** 2)),
               class_swap_mse=float(np.mean((scores - reverse_scores[:, [1, 0, 2]]) ** 2)),
               confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist())
    for i, label in enumerate(LABELS):
        target = labels == i
        out[label.lower()] = dict(precision=float(precision[i]), recall=float(recall[i]),
                                 f1=float(f1[i]), support=int(support[i]),
                                 auprc=float(average_precision_score(target, scores[:, i]))
                                 if target.any() else None,
                                 auroc=float(roc_auc_score(target, scores[:, i]))
                                 if target.any() and (~target).any() else None)
    out["macro_auprc"] = float(np.mean([out[label.lower()]["auprc"] for label in LABELS
                                        if out[label.lower()]["auprc"] is not None]))
    if head == "parallel":
        out["both_heads_positive_rate"] = float(((scores[:, 0] >= .5) & (scores[:, 1] >= .5)).mean())
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    collected = {k: [] for k in ("label", "prediction", "scores", "d", "reverse_d", "index", "reverse_index", "reverse_scores", "row", "loss")}
    for batch in loader:
        wt, var, go = move_batch(batch, device)
        forward, reverse = model(wt, var, go), model(var, wt, go)
        labels = batch["label"].to(device)
        if model.head_kind == "multiclass":
            per_row_loss = F.cross_entropy(forward["logits"], labels, reduction="none")
        else:
            target = torch.stack([labels == 0, labels == 1], dim=1).float()
            per_row_loss = F.binary_cross_entropy_with_logits(forward["logits"], target, reduction="none").mean(1)
        for name, value in dict(label=batch["label"], prediction=forward["prediction"],
                                scores=forward["scores"], d=forward["d"], reverse_d=reverse["d"],
                                index=forward["direction_index"], reverse_index=reverse["direction_index"],
                                reverse_scores=reverse["scores"], row=batch["row"], loss=per_row_loss).items():
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite evaluation {name}")
            collected[name].append(array)
    c = {k: np.concatenate(v) for k, v in collected.items()}

    def summarize(mask):
        result = metrics(c["label"][mask], c["prediction"][mask], c["scores"][mask],
                         c["d"][mask], c["reverse_d"][mask], c["index"][mask],
                         c["reverse_index"][mask], c["reverse_scores"][mask], model.head_kind)
        result["unweighted_classification_loss"] = float(c["loss"][mask].mean())
        return result

    result = summarize(np.ones(len(c["label"]), dtype=bool))
    frame = loader.dataset.frame.iloc[c["row"]].copy().reset_index(drop=True)
    if "augmented" in frame:
        mask = ~frame.augmented.astype(str).str.lower().eq("true").to_numpy()
        if mask.any():
            result["nonaugmented_subset"] = summarize(mask)
    for column in ("seen_in_train", "multiple_labels_for_input"):
        if column in frame:
            mask = frame[column].astype(str).str.lower().eq("true").to_numpy()
            for name, subset in ((column, mask), ("not_" + column, ~mask)):
                if subset.any():
                    result[name] = summarize(subset)
    for i, label in enumerate(LABELS):
        frame[f"score_{label.lower()}"] = c["scores"][:, i]
    frame["prediction"] = [LABELS[i] for i in c["prediction"]]
    frame["d_forward"], frame["d_reverse"] = c["d"], c["reverse_d"]
    return result, frame


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, head, pair_mode, datasets, device, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    seed_everything(args.seed)
    model = PairedGODecoder(scale_dims=datasets["train"].store.scale_dims,
                            go_dim=datasets["train"].store.go.dimension, width=args.width,
                            heads=args.attention_heads, head=head, pair_mode=pair_mode,
                            dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=(s == "train"),
                             generator=generator if s == "train" else None, num_workers=0)
               for s, d in datasets.items()}
    counts = np.bincount(datasets["train"].frame.label_idx, minlength=3)
    class_weights = pos_weight = None
    if args.balance:
        if (counts == 0).any():
            raise ValueError("Class balancing requires all three classes in train")
        class_weights = torch.tensor(len(datasets["train"]) / (3 * counts), dtype=torch.float32, device=device)
        pos_weight = torch.tensor((counts.sum() - counts[:2]) / counts[:2], dtype=torch.float32, device=device)
    record = dict(model_spec=model.spec, labels=list(LABELS), seed=args.seed, device=str(device),
                  encoder_frozen=True, smoke=args.smoke, lambda_reverse=args.lambda_reverse,
                  window_pool=datasets["train"].store.window_pool,
                  feature_package=str(datasets["train"].store.package.resolve()),
                  window_cache=(str(datasets["train"].store.window_cache.resolve())
                                if datasets["train"].store.window_cache is not None else None),
                  single_mutation_only=True,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  class_counts=counts.tolist(), balance=args.balance,
                  class_weights=class_weights.tolist() if class_weights is not None else None,
                  pos_weight=pos_weight.tolist() if pos_weight is not None else None,
                  training_rows=len(datasets["train"]), validation_rows=len(datasets["val"]),
                  selection="validation macro-F1 on supplied available-cache cohort",
                  reverse_definition="d = logit_GOF - logit_LOF; E[(d_forward+d_reverse)^2]",
                  reverse_is_structural=(pair_mode == "score_difference"),
                  parallel_decision="threshold=.5; both positive -> larger head; neither -> NEUTRAL",
                  source_layout=head, evaluation_augmented_excluded=True,
                  multiple_label_policy="retain source labels; report groups separately",
                  data_dir=str(datasets["train"].path.parent.resolve()),
                  prepared_audit_sha256=hashlib.sha256((datasets["train"].path.parent / "audit.json").read_bytes()).hexdigest(),
                  split_sha256={s: hashlib.sha256(d.path.read_bytes()).hexdigest() for s,d in datasets.items()},
                  train_table_sha256=hashlib.sha256(datasets["train"].path.read_bytes()).hexdigest(),
                  go_sha256=datasets["train"].store.go.artifact_sha256,
                  args={k: str(v) if isinstance(v, Path) else v
                        for k, v in {**vars(args), "head": head, "pair_mode": pair_mode}.items()})
    audit = json.loads((datasets["train"].path.parent / "audit.json").read_text(encoding="utf-8"))
    record["feature_sha256"] = audit.get("feature_sha256", {})
    write_json(output / "config.json", record)
    history, best, stale = [], -1., 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3)
        for batch in loaders["train"]:
            wt, var, go = move_batch(batch, device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            forward = model(wt, var, go)
            cls = classification_loss(forward, labels, head, class_weights, pos_weight,
                                      batch["binary_target"].to(device) if head == "parallel" else None)
            penalty = cls.new_zeros(())
            if args.lambda_reverse and pair_mode == "joint":
                # A separate deterministic pass isolates reversal from dropout noise.
                with deterministic(model):
                    penalty = reverse_loss(model(wt, var, go), model(var, wt, go))
            loss = cls + args.lambda_reverse * penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals += np.array([loss.item(), cls.item(), penalty.item()]) * len(labels)
        validation, _ = evaluate(model, loaders["val"], device)
        row = dict(epoch=epoch, train_loss=totals[0] / len(datasets["train"]),
                   train_classification_loss=totals[1] / len(datasets["train"]),
                   train_reverse_loss=(totals[2] / len(datasets["train"]))
                   if args.lambda_reverse or pair_mode == "score_difference" else None,
                   validation=validation)
        history.append(row)
        write_json(output / "history.json", history)
        if validation["macro_f1"] > best:
            best, stale = validation["macro_f1"], 0
            torch.save(dict(format="contvar_go_decoder_starter_v1", model_spec=model.spec,
                            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            labels=list(LABELS), epoch=epoch, validation_macro_f1=best,
                            single_mutation_only=True, smoke=args.smoke, family=args.family,
                            source_layout=head, evaluation_augmented_excluded=True,
                            seed=args.seed, lambda_reverse=args.lambda_reverse,
                            go_sha256=record["go_sha256"], feature_sha256=record["feature_sha256"],
                            window_pool=record["window_pool"]), output / "best.pt")
        else:
            stale += 1
        print(f"{args.family}/{head}/{pair_mode}/lambda={args.lambda_reverse} epoch={epoch} loss={row['train_loss']:.4f} "
              f"val_F1={validation['macro_f1']:.4f} reverse_MSE={validation['reverse_mse']:.6g}", flush=True)
        if stale >= args.patience:
            break
    model, checkpoint = load_decoder(output / "best.pt", device)
    final = dict(smoke=args.smoke, best_epoch=checkpoint["epoch"],
                 epochs_run=len(history), family=args.family, head=head, pair_mode=pair_mode,
                 lambda_reverse=args.lambda_reverse, single_mutation_only=True,
                 window_pool=datasets["train"].store.window_pool,
                 source_layout=head, evaluation_augmented_excluded=True,
                 n_train=len(datasets["train"]), n_val=len(datasets["val"]),
                 seconds=time.monotonic() - started, parameter_count=record["parameter_count"])
    for split in ("val", "test"):
        if split in loaders:
            final[split], predictions = evaluate(model, loaders[split], device)
            predictions.to_csv(output / f"{split}_predictions.tsv", sep="\t", index=False)
    write_json(output / "results.json", final)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, default=PACKAGE)
    p.add_argument("--window-pool", choices=("none", "mean"), default="none")
    p.add_argument("--window-cache", type=Path)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--family", choices=("easy", "hard"), default="hard")
    p.add_argument("--head", choices=("parallel", "multiclass"), default="multiclass")
    p.add_argument("--pair-mode", choices=("joint", "score_difference"), default="joint")
    p.add_argument("--four-experiments", action="store_true")
    p.add_argument("--lambda-reverse", type=float, default=0.)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--balance", action="store_true")
    p.add_argument("--exclude-augmented-train", action="store_true")
    p.add_argument("--evaluate-test", action="store_true", help="Evaluate test after validation checkpoint selection")
    p.add_argument("--smoke", action="store_true", help="One epoch, at most 128 train/64 val or test; not a scientific result")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "results/run")
    args = p.parse_args()
    if (args.lambda_reverse < 0 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1
            or args.lr <= 0 or args.threads < 1):
        p.error("Invalid loss or training settings")
    if args.out.exists():
        p.error("--out must be a new directory to preserve previous runs")
    if args.smoke:
        args.epochs = 1
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    store = FeatureStore(args.package, window_pool=args.window_pool, window_cache=args.window_cache)
    try:
        splits = ("train", "val", "test") if args.evaluate_test else ("train", "val")
        experiments = ([(h, m) for h in ("parallel", "multiclass")
                        for m in ("score_difference", "joint")]
                       if args.four_experiments else [(args.head, args.pair_mode)])
        summary = {}
        datasets_by_head = {head: load_datasets(args, head, store, splits)
                            for head in sorted({h for h,m in experiments})}
        store.preload([d for datasets in datasets_by_head.values() for d in datasets.values()])
        for head, mode in experiments:
            name = f"{head}_{mode}"
            out = args.out / name if args.four_experiments else args.out
            summary[name] = train(args, head, mode, datasets_by_head[head], device, out)
        if args.four_experiments:
            write_json(args.out / "summary.json", summary)
    finally:
        store.close()


def load_datasets(args, head, store, splits=("train", "val")):
    folder = args.data_dir / head
    datasets = {}
    for split in splits:
        dataset = PairDataset(folder / f"{args.family}_{split}.tsv", store,
                              limit=(128 if split == "train" else 64) if args.smoke else None,
                              seed=args.seed, exclude_augmented=(split != "train" or args.exclude_augmented_train),
                              head=head)
        if split != "train" and dataset.frame.augmented.astype(str).str.lower().eq("true").any():
            raise ValueError("Augmented evaluation records are not allowed")
        datasets[split] = dataset
    return datasets


if __name__ == "__main__":
    main()
