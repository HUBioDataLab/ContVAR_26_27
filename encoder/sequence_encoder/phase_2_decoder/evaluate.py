"""
Decoder değerlendirme metrikleri.

mAP, F1_macro, Precision, Recall, Accuracy, MCC
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def _fast_map(labels: np.ndarray, probs: np.ndarray) -> float:
    """Vektörize mean Average Precision."""
    N = labels.shape[0]
    order         = np.argsort(-probs, axis=0)
    sorted_labels = np.take_along_axis(labels, order, axis=0)
    cumpos        = np.cumsum(sorted_labels, axis=0)
    rank          = np.arange(1, N + 1, dtype=np.float32)[:, None]
    precision     = cumpos / rank
    ap            = (precision * sorted_labels).sum(axis=0) / (sorted_labels.sum(axis=0) + 1e-8)
    return float(ap.mean())


def evaluate(model: torch.nn.Module, loader: DataLoader,
             device: torch.device, threshold: float = 0.5,
             criterion: torch.nn.Module = None,
             propagation_mat: np.ndarray = None) -> dict:
    """
    Model değerlendirmesi — tüm metrikleri döndürür.

    propagation_mat: [N_go, N_go] float32 — GOHierarchy.build_propagation_matrix()
        çıktısı. Verilirse binary tahminlere post-processing propagation uygulanır:
        eğer bir GO terimi tahmin edildiyse tüm ancestor'ları da tahmin edilmiş sayılır.
        mAP orijinal olasılıklar üzerinden hesaplanır (değişmez).
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_batches  = 0.0, 0

    with torch.no_grad():
        for emb, label in loader:
            emb_d  = emb.to(device)
            logits = model(emb_d).cpu()
            all_logits.append(logits)
            all_labels.append(label)
            if criterion is not None:
                total_loss += criterion(logits.to(device), label.to(device)).item()
                n_batches  += 1

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))

    valid    = labels.sum(axis=0) > 0
    labels_v = labels[:, valid]
    probs_v  = probs[:, valid]
    N        = labels_v.shape[0]

    # mAP: propagation uygulanmaz (orijinal prob sıralaması korunur)
    map_score = _fast_map(labels_v, probs_v)

    # Binary tahminler — propagation varsa tüm vocab üzerinde uygula, sonra filtrele
    preds_full = (probs > threshold).astype(np.float32)
    if propagation_mat is not None:
        preds_full = (preds_full @ propagation_mat > 0).astype(np.float32)
    preds = preds_full[:, valid]

    tp = (preds * labels_v).sum(axis=0)
    tn = ((1 - preds) * (1 - labels_v)).sum(axis=0)
    fp = (preds * (1 - labels_v)).sum(axis=0)
    fn = ((1 - preds) * labels_v).sum(axis=0)

    f1  = float((2 * tp / (2 * tp + fp + fn + 1e-8)).mean())
    prec = float((tp / (tp + fp + 1e-8)).mean())
    rec  = float((tp / (tp + fn + 1e-8)).mean())
    acc  = float(((tp + tn) / (N + 1e-8)).mean())

    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc     = float((mcc_num / mcc_den).mean())

    results = {
        "mAP": map_score, "F1_macro": f1, "Precision": prec,
        "Recall": rec, "Accuracy": acc, "MCC": mcc,
        "TP": float(tp.sum()), "TN": float(tn.sum()),
        "FP": float(fp.sum()), "FN": float(fn.sum()),
    }
    if criterion is not None:
        results["val_loss"] = total_loss / (n_batches + 1e-8)
    return results


FINAL_METRIC_KEYS = ["threshold", "F1_macro", "Precision", "Recall", "Accuracy", "MCC"]


def get_probs_labels(model: torch.nn.Module, loader: DataLoader,
                      device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Runs `model` over `loader`, returns (probs, labels) as [N, C] numpy
    arrays over ALL classes (no valid-class filtering, no thresholding)."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for emb, label in loader:
            logits = model(emb.to(device)).cpu()
            all_logits.append(logits)
            all_labels.append(label)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return probs, labels


def compute_metrics_at_threshold(probs: np.ndarray, labels: np.ndarray,
                                  threshold: float,
                                  propagation_mat: np.ndarray | None = None) -> dict:
    """probs, labels: [N, C] raw arrays from get_probs_labels().

    Filters to classes with >=1 positive in `labels` (same convention as
    evaluate()), then returns macro-averaged Precision/Recall/F1/Accuracy/MCC
    at `threshold`.

    propagation_mat: [N_go, N_go] float32 — GOHierarchy.build_propagation_matrix()
        çıktısı. Verilirse, thresholded tahminlere valid-class filtrelemeden
        ÖNCE ancestor propagation uygulanır (evaluate()'teki ile aynı sıra):
        preds = (preds @ propagation_mat) > 0.
    """
    valid    = labels.sum(axis=0) > 0
    labels_v = labels[:, valid]
    N        = labels_v.shape[0]

    preds_full = (probs > threshold).astype(np.float32)
    if propagation_mat is not None:
        preds_full = (preds_full @ propagation_mat > 0).astype(np.float32)
    preds = preds_full[:, valid]

    tp = (preds * labels_v).sum(axis=0)
    tn = ((1 - preds) * (1 - labels_v)).sum(axis=0)
    fp = (preds * (1 - labels_v)).sum(axis=0)
    fn = ((1 - preds) * labels_v).sum(axis=0)

    f1   = float((2 * tp / (2 * tp + fp + fn + 1e-8)).mean())
    prec = float((tp / (tp + fp + 1e-8)).mean())
    rec  = float((tp / (tp + fn + 1e-8)).mean())
    acc  = float(((tp + tn) / (N + 1e-8)).mean())

    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    mcc     = float((mcc_num / mcc_den).mean())

    return {
        "threshold": float(threshold), "F1_macro": f1, "Precision": prec,
        "Recall": rec, "Accuracy": acc, "MCC": mcc,
    }


def find_best_f1_threshold(probs: np.ndarray, labels: np.ndarray,
                            propagation_mat: np.ndarray | None = None) -> float:
    """Sweeps thresholds 0.05..0.95 (step 0.05), returns the one maximizing
    macro F1 (see compute_metrics_at_threshold)."""
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 1.00, 0.05):
        f1 = compute_metrics_at_threshold(probs, labels, t, propagation_mat=propagation_mat)["F1_macro"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
