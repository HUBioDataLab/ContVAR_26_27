"""
Phase 3 metrics: threshold search, three-class decision rule, breakdowns.

Design: phase_3_variant_decoder/DESIGN.md §5

Decision rule (d = s_wt - s_var):
    d > c + w/2    -> LOF
    |d - c| <= w/2 -> NEUTRAL
    d < c - w/2    -> GOF

Thresholds (centre c, width w) are searched on VAL and applied unchanged to TEST.
Phase 2's bug in benchmark_eval.py:66-72 — searching the threshold on the very
split being evaluated — is not repeated here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from phase_3_variant_decoder.config import LABELS, LABEL_TO_IDX

IDX_LOF = LABEL_TO_IDX["LOF"]
IDX_GOF = LABEL_TO_IDX["GOF"]
IDX_NEUTRAL = LABEL_TO_IDX["NEUTRAL"]


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Fixed three-class macro average, including absent/prediction-free classes."""
    cm = np.asarray(confusion, dtype=float)
    if cm.shape != (3, 3) or not np.isfinite(cm).all() or (cm < 0).any() or cm.sum() <= 0:
        raise ValueError("Expected a nonempty, finite, nonnegative 3 x 3 confusion matrix")
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    precision, recall = divide(tp, predicted), divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    out = {
        "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / cm.sum()),
        "weighted_f1": float(np.dot(f1, support) / cm.sum()),
    }
    for i, label in enumerate(LABELS):
        for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            out[f"{label.lower()}_{name}"] = float(values[i])
    return out


def predict_three_class(d: np.ndarray, center: float, width: float) -> np.ndarray:
    """d -> {LOF, GOF, NEUTRAL} indices."""
    half = width / 2.0
    out = np.full(d.shape, IDX_NEUTRAL, dtype=np.int64)
    out[d > center + half] = IDX_LOF
    out[d < center - half] = IDX_GOF
    return out


def search_thresholds(
    d: np.ndarray,
    y: np.ndarray,
    n_grid: int = 81,
) -> tuple[float, float, float]:
    """The (centre, width) pair that maximises macro-F1 on VAL.

    The (c, w) parameterisation has the same two degrees of freedom as
    (tau_LOF, tau_GOF), but "how far the band has shifted" is read directly — and
    class imbalance can shift the validation-optimal band away from zero.

    Vectorised: the decision rule depends only on two cut points
    (lo = c - w/2, hi = c + w/2), so the confusion matrix for every candidate pair
    can be built in O(log n) with searchsorted. The naive version called sklearn's
    f1_score once per candidate (~4.6 s); this one takes milliseconds — decisive
    once multiplied by epochs x trials in a sweep.

    Returns: (centre, width, best_macro_f1)
    """
    n_classes = len(LABELS)
    cuts = np.unique(np.percentile(d, np.linspace(0.5, 99.5, n_grid)))
    n_cuts = len(cuts)
    if n_cuts < 2:
        return 0.0, 0.0, float(f1_score(y, predict_three_class(d, 0.0, 0.0),
                                        average="macro", zero_division=0))

    per_class = [np.sort(d[y == k]) for k in range(n_classes)]
    n_true = np.array([len(s) for s in per_class], dtype=np.float64)

    # A[k, i] = class-k samples predicted GOF when lo=cuts[i]  (d < lo)
    # B[k, j] = class-k samples predicted LOF when hi=cuts[j]  (d > hi)
    A = np.stack([np.searchsorted(s, cuts, side="left") for s in per_class]).astype(np.float64)
    B = np.stack([len(s) - np.searchsorted(s, cuts, side="right") for s in per_class]).astype(np.float64)

    lo = np.arange(n_cuts)[:, None]   # i axis
    hi = np.arange(n_cuts)[None, :]   # j axis
    valid = cuts[lo] <= cuts[hi]

    a_sum, b_sum = A.sum(0), B.sum(0)
    # predicted-class totals
    pred_gof = a_sum[lo] + np.zeros_like(hi, dtype=np.float64)
    pred_lof = b_sum[hi] + np.zeros_like(lo, dtype=np.float64)
    pred_neu = float(len(y)) - pred_gof - pred_lof

    # diagonal (correct predictions)
    tp_lof = B[IDX_LOF][hi] + np.zeros_like(lo, dtype=np.float64)
    tp_gof = A[IDX_GOF][lo] + np.zeros_like(hi, dtype=np.float64)
    tp_neu = n_true[IDX_NEUTRAL] - A[IDX_NEUTRAL][lo] - B[IDX_NEUTRAL][hi]

    def _f1(tp, pred_total, true_total):
        den = pred_total + true_total
        return np.divide(2.0 * tp, den, out=np.zeros_like(tp), where=den > 0)

    macro = (
        _f1(tp_lof, pred_lof, n_true[IDX_LOF])
        + _f1(tp_gof, pred_gof, n_true[IDX_GOF])
        + _f1(tp_neu, pred_neu, n_true[IDX_NEUTRAL])
    ) / n_classes
    macro = np.where(valid, macro, -1.0)

    i, j = np.unravel_index(int(np.argmax(macro)), macro.shape)
    lo_v, hi_v = float(cuts[i]), float(cuts[j])
    return (lo_v + hi_v) / 2.0, hi_v - lo_v, float(macro[i, j])


def _ovr_scores(d: np.ndarray) -> dict[int, np.ndarray]:
    """Threshold-free one-vs-rest scores (DESIGN §5.2).

    +d for LOF, -d for GOF, -|d| for NEUTRAL — in all three "higher = that class".
    """
    return {IDX_LOF: d, IDX_GOF: -d, IDX_NEUTRAL: -np.abs(d)}


def compute_metrics(d: np.ndarray, y: np.ndarray, center: float, width: float) -> dict:
    """The full metric set. The selection metric is macro_f1."""
    if len(y) == 0:
        return {"n": 0}

    pred = predict_three_class(d, center, width)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(LABELS))), zero_division=0
    )

    out = {
        "n": int(len(y)),
        "center": round(float(center), 4),
        "width": round(float(width), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "confusion": confusion_matrix(y, pred, labels=list(range(len(LABELS)))).tolist(),
    }

    ovr = _ovr_scores(d)
    auprc_list = []
    auroc_list = []
    for i, name in enumerate(LABELS):
        key = name.lower()
        out[f"{key}_precision"] = round(float(prec[i]), 4)
        out[f"{key}_recall"] = round(float(rec[i]), 4)
        out[f"{key}_f1"] = round(float(f1[i]), 4)
        out[f"{key}_support"] = int(support[i])

        binary = (y == i).astype(int)
        if 0 < binary.sum() < len(binary):
            ap = float(average_precision_score(binary, ovr[i]))
            auc = float(roc_auc_score(binary, ovr[i]))
        else:
            ap, auc = float("nan"), float("nan")
        out[f"{key}_auprc"] = round(ap, 4)
        out[f"{key}_auroc"] = round(auc, 4)
        auprc_list.append(ap)
        auroc_list.append(auc)

    out["macro_auprc"] = round(float(np.nanmean(auprc_list)), 4)
    # Macro-averaged one-vs-rest AUROC — the "overall AUROC" the per-class
    # lof/gof/neutral_auroc fields don't directly give you (mean of the three,
    # NOT the same statistic as lof_gof_auroc below, which is a different
    # subset/label setup — see metrics.py's module docstring / DESIGN §8.5).
    out["macro_auroc"] = round(float(np.nanmean(auroc_list)), 4)

    # DESIGN §8.5 — the one number comparable to phase 2's baseline (0.54):
    # LOF vs GOF only (NEUTRAL excluded), score = -d (higher = GOF).
    mask = (y == IDX_LOF) | (y == IDX_GOF)
    if mask.sum() and 0 < (y[mask] == IDX_GOF).sum() < mask.sum():
        out["lof_gof_auroc"] = round(
            float(roc_auc_score((y[mask] == IDX_GOF).astype(int), -d[mask])), 4
        )
    else:
        out["lof_gof_auroc"] = float("nan")

    return out


def margin_diagnostics(d: np.ndarray, y: np.ndarray, margin: float) -> dict:
    """DESIGN §8.7 — was the (single, shared) hinge margin actually met?

    Mass is expected beyond m for LOF/GOF and within m for NEUTRAL — same m for
    all three classes (the earlier separate `margin_neutral` was removed). If the
    hinge is satisfied but the metric is poor, the problem is the representation,
    not the thresholds.
    """
    out = {}
    for i, name in enumerate(LABELS):
        m = y == i
        if not m.any():
            continue
        key = name.lower()
        out[f"{key}_abs_d_median"] = round(float(np.median(np.abs(d[m]))), 4)
        if i == IDX_NEUTRAL:
            out[f"{key}_within_margin"] = round(float((np.abs(d[m]) <= margin).mean()), 4)
        else:
            signed = d[m] if i == IDX_LOF else -d[m]
            out[f"{key}_beyond_margin"] = round(float((signed >= margin).mean()), 4)
    return out


def breakdown(d: np.ndarray, y: np.ndarray, mask: np.ndarray,
              center: float, width: float, prefix: str) -> dict:
    """DESIGN §5.2 subset breakdown — both sides of the mask, separately."""
    out = {}
    for suffix, m in ((f"{prefix}", mask), (f"{prefix}_not", ~mask)):
        if m.sum() == 0:
            continue
        sub = compute_metrics(d[m], y[m], center, width)
        out[f"{suffix}_n"] = sub["n"]
        out[f"{suffix}_macro_f1"] = sub["macro_f1"]
        out[f"{suffix}_macro_auprc"] = sub["macro_auprc"]
        out[f"{suffix}_macro_auroc"] = sub["macro_auroc"]
        out[f"{suffix}_lof_gof_auroc"] = sub["lof_gof_auroc"]
    return out
