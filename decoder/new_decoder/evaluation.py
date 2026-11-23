"""Class metrics and deterministic evaluation, separate from optimizer state."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)

from .config import LABELS
from .data import move_batch
from .model import PairedGODecoder


def metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    d: np.ndarray,
    reverse_d: np.ndarray,
    index: np.ndarray,
    reverse_index: np.ndarray,
    reverse_scores: np.ndarray,
    head: str,
) -> dict[str, Any]:
    """Summarize class performance, signed reversal and head coactivation."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1, 2], zero_division=0
    )
    out = dict(
        n=len(labels),
        macro_f1=float(f1.mean()),
        accuracy=float((labels == prediction).mean()),
        balanced_accuracy=float(recall.mean()),
        mcc=float(matthews_corrcoef(labels, prediction))
        if np.unique(np.concatenate([labels, prediction])).size > 1
        else 0.0,
        reverse_mse=float(np.mean((d + reverse_d) ** 2)),
        reverse_mae=float(np.mean(np.abs(d + reverse_d))),
        direction_score_rms=float(np.sqrt(np.mean(d**2))),
        probability_direction_reverse_mse=float(np.mean((index + reverse_index) ** 2)),
        class_swap_mse=float(np.mean((scores - reverse_scores[:, [1, 0, 2]]) ** 2)),
        confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    )
    for i, label in enumerate(LABELS):
        target = labels == i
        out[label.lower()] = dict(
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
            auprc=float(average_precision_score(target, scores[:, i])) if target.any() else None,
            auroc=float(roc_auc_score(target, scores[:, i]))
            if target.any() and (~target).any()
            else None,
        )
    out["macro_auprc"] = float(
        np.mean(
            [
                out[label.lower()]["auprc"]
                for label in LABELS
                if out[label.lower()]["auprc"] is not None
            ]
        )
    )
    if head == "parallel":
        out["both_heads_positive_rate"] = float(
            ((scores[:, 0] >= 0.5) & (scores[:, 1] >= 0.5)).mean()
        )
    return out


@torch.no_grad()
def evaluate(
    model: PairedGODecoder,
    loader: DataLoader,
    device: str | torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate deterministic forward/reverse passes and preserve sample metadata."""
    model.eval()
    collected = {
        k: []
        for k in (
            "label",
            "prediction",
            "scores",
            "d",
            "reverse_d",
            "index",
            "reverse_index",
            "reverse_scores",
            "row",
            "loss",
        )
    }
    for batch in loader:
        wt, var, go = move_batch(batch, device)
        forward, reverse = model(wt, var, go), model(var, wt, go)
        labels = batch["label"].to(device)
        if model.head_kind == "multiclass":
            per_row_loss = F.cross_entropy(forward["logits"], labels, reduction="none")
        else:
            target = torch.stack([labels == 0, labels == 1], dim=1).float()
            per_row_loss = F.binary_cross_entropy_with_logits(
                forward["logits"], target, reduction="none"
            ).mean(1)
        for name, value in dict(
            label=batch["label"],
            prediction=forward["prediction"],
            scores=forward["scores"],
            d=forward["d"],
            reverse_d=reverse["d"],
            index=forward["direction_index"],
            reverse_index=reverse["direction_index"],
            reverse_scores=reverse["scores"],
            row=batch["row"],
            loss=per_row_loss,
        ).items():
            array = value.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError(f"Non-finite evaluation {name}")
            collected[name].append(array)
    c = {k: np.concatenate(v) for k, v in collected.items()}

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        result = metrics(
            c["label"][mask],
            c["prediction"][mask],
            c["scores"][mask],
            c["d"][mask],
            c["reverse_d"][mask],
            c["index"][mask],
            c["reverse_index"][mask],
            c["reverse_scores"][mask],
            model.head_kind,
        )
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
