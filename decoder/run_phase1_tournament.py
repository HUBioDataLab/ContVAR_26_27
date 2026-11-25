#!/usr/bin/env python3
"""ContVAR Phase 1 Head-to-Head Tournament Runner.

Compares frozen encoder feature sources:
  1. baseline             (old untrained 33-residue window, center-sliced to 7)
  2. trained_mean         (Omer Taha Phase-1 DMS 7-window, mean pooling)
  3. trained_distance     (Omer Taha Phase-1 DMS 7-window, distance-weighted pooling)

Scientific protocol (18-19 Nov 2026 meeting decisions):
  - 3 fixed seeds: [42, 43, 44] -> reported as mean +/- std
  - Augmentation kept in train, strictly removed from val and test
  - Confidence & calibration tracking (max prob, correct/wrong gap, margin, entropy, ECE)
  - Checkpoint selected strictly by val Macro-F1; test evaluated once on best checkpoint
  - Evaluates both Full Test and exact Common Test intersection across all sources
  - Saves all artifacts neatly under decoder/tournament_results/phase1/ (gitignored)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
from torch.utils.data import Dataset

# Resolve repository paths cleanly
DECODER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DECODER_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_ROOT = DECODER_DIR / "tournament_results" / "phase1"

# Import audited modules from repository
sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(DECODER_DIR / "new_decoder" / "contvar_decoder_mf_window7"))

from contvar_mf.model import PairedGODecoder, classification_loss  # noqa: E402
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402

LABELS = ("LOF", "GOF", "NEUTRAL")
SOURCE_DIRS = {
    "baseline": DATA_DIR / "embeddings" / "baseline",
    "trained_mean": DATA_DIR / "embeddings" / "trained_win7_mean",
    "trained_distance": DATA_DIR / "embeddings" / "trained_win7_distance",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_seconds(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def is_true_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def load_prepared_split(family: str, split: str) -> pd.DataFrame:
    path = DATA_DIR / "prepared" / f"{family}_{split}.tsv"
    df = pd.read_csv(path, sep="\t", low_memory=False)
    # Protocol rule: keep augmentation in train, strictly remove from val/test
    if split != "train" and "augmented" in df.columns:
        before = len(df)
        df = df[~is_true_series(df["augmented"])].reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            print(f"  [Filter] {family}_{split}: removed {removed} augmented rows ({before} -> {len(df)})", flush=True)
    return df


def valid_key_mask(df: pd.DataFrame, embed_dir: Path) -> np.ndarray:
    """Return boolean mask of rows whose 6 required H5 keys exist in embed_dir."""
    wt_w_col = "wt_window_key" if "wt_window_key" in df.columns else "wt_mut_key"
    var_w_col = "var_window_key" if "var_window_key" in df.columns else "var_mut_key"
    with h5py.File(embed_dir / "phase3_seq_phase1_global.h5", "r") as hg, \
         h5py.File(embed_dir / "phase3_seq_phase1_local.h5", "r") as hl, \
         h5py.File(embed_dir / "phase3_seq_phase1_window.h5", "r") as hw:
        g_keys = set(hg.keys())
        l_keys = set(hl.keys())
        w_keys = set(hw.keys())
        m_keys = set(hw["mask"].keys()) if "mask" in hw else w_keys

    mask = (
        df["wt_global_key"].isin(g_keys)
        & df["var_global_key"].isin(g_keys)
        & df["wt_mut_key"].isin(l_keys)
        & df["var_mut_key"].isin(l_keys)
        & df[wt_w_col].isin(w_keys)
        & df[var_w_col].isin(w_keys)
        & df[wt_w_col].isin(m_keys)
        & df[var_w_col].isin(m_keys)
    )
    return mask.to_numpy(dtype=bool)


class PreloadedFeatureStore:
    """Preloads only unique required H5 keys directly into GPU VRAM (or CPU fallback) for maximum speed."""

    def __init__(self, source: str, frames: list[pd.DataFrame], go_table: GOEmbeddingTable, device: torch.device) -> None:
        self.source = source
        self.embed_dir = SOURCE_DIRS[source]
        self.go = go_table
        self.device = device
        t0 = time.perf_counter()

        wt_w_col = "wt_window_key" if "wt_window_key" in frames[0].columns else "wt_mut_key"
        var_w_col = "var_window_key" if "var_window_key" in frames[0].columns else "var_mut_key"

        g_keys = sorted(set().union(*(set(f["wt_global_key"]) | set(f["var_global_key"]) for f in frames)))
        l_keys = sorted(set().union(*(set(f["wt_mut_key"]) | set(f["var_mut_key"]) for f in frames)))
        w_keys = sorted(set().union(*(set(f[wt_w_col]) | set(f[var_w_col]) for f in frames)))

        self.g_map = {k: i for i, k in enumerate(g_keys)}
        self.l_map = {k: i for i, k in enumerate(l_keys)}
        self.w_map = {k: i for i, k in enumerate(w_keys)}

        # Load global vectors (fp32)
        g_arr = np.empty((len(g_keys), 1280), dtype=np.float32)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_global.h5", "r") as hg:
            for i, k in enumerate(g_keys):
                g_arr[i] = hg[k][:]
        self.global_vecs = torch.from_numpy(g_arr)

        # Load local mutation-site vectors (fp32)
        l_arr = np.empty((len(l_keys), 1280), dtype=np.float32)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_local.h5", "r") as hl:
            for i, k in enumerate(l_keys):
                l_arr[i] = hl[k][:]
        self.mut_vecs = torch.from_numpy(l_arr)

        # Load 7-residue windows (fp16 in CPU RAM to save memory, cast to fp32 on GPU)
        w_arr = np.empty((len(w_keys), 7, 1280), dtype=np.float16)
        m_arr = np.ones((len(w_keys), 7), dtype=bool)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_window.h5", "r") as hw:
            sample_shape = hw[w_keys[0]].shape
            has_mask = "mask" in hw
            if len(sample_shape) == 2 and sample_shape[0] == 7:
                sl = slice(None)
            else:
                saved_k = int(hw.attrs.get("window_radius_k", 16))
                center = int(hw.attrs.get("center_index", saved_k))
                sl = slice(center - 3, center + 4)  # exact 7-residue window centered at mutation
            for i, k in enumerate(w_keys):
                w_arr[i] = hw[k][sl].astype(np.float16)
                if has_mask:
                    m_arr[i] = hw["mask"][k][sl].astype(bool)
        self.win_vecs = torch.from_numpy(w_arr)
        self.win_masks = torch.from_numpy(m_arr)

        mem_mb = (
            self.global_vecs.nbytes + self.mut_vecs.nbytes + self.win_vecs.nbytes + self.win_masks.nbytes
        ) / (1024 * 1024)

        loc = "CPU RAM"
        if device.type == "cuda":
            try:
                self.global_vecs = self.global_vecs.to(device)
                self.mut_vecs = self.mut_vecs.to(device)
                self.win_vecs = self.win_vecs.to(device)
                self.win_masks = self.win_masks.to(device)
                loc = f"GPU VRAM ({device})"
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                loc = "CPU RAM (VRAM fallback)"

        dt = time.perf_counter() - t0
        print(
            f"  [Preload] {source}: global={len(g_keys)}, mut={len(l_keys)}, win7={len(w_keys)} "
            f"({mem_mb:.0f} MB in {loc} in {dt:.1f}s)",
            flush=True,
        )


class FastVariantDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, store: PreloadedFeatureStore) -> None:
        self.frame = frame.reset_index(drop=True)
        self.store = store
        dev = store.device
        wt_w_col = "wt_window_key" if "wt_window_key" in self.frame.columns else "wt_mut_key"
        var_w_col = "var_window_key" if "var_window_key" in self.frame.columns else "var_mut_key"

        self.wt_g = torch.tensor([store.g_map[k] for k in self.frame["wt_global_key"]], dtype=torch.long, device=dev)
        self.var_g = torch.tensor([store.g_map[k] for k in self.frame["var_global_key"]], dtype=torch.long, device=dev)
        self.wt_l = torch.tensor([store.l_map[k] for k in self.frame["wt_mut_key"]], dtype=torch.long, device=dev)
        self.var_l = torch.tensor([store.l_map[k] for k in self.frame["var_mut_key"]], dtype=torch.long, device=dev)
        self.wt_w = torch.tensor([store.w_map[k] for k in self.frame[wt_w_col]], dtype=torch.long, device=dev)
        self.var_w = torch.tensor([store.w_map[k] for k in self.frame[var_w_col]], dtype=torch.long, device=dev)
        self.labels = torch.tensor(self.frame["label_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=dev)
        self.go = store.go.lookup(self.frame["matched_go_id"].tolist()).to(dev)

    def __len__(self) -> int:
        return len(self.frame)

    def get_batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        wt_w_idx = self.wt_w[idx]
        var_w_idx = self.var_w[idx]
        return {
            "wt_g": self.store.global_vecs[self.wt_g[idx]],
            "var_g": self.store.global_vecs[self.var_g[idx]],
            "wt_l": self.store.mut_vecs[self.wt_l[idx]],
            "var_l": self.store.mut_vecs[self.var_l[idx]],
            "wt_w": self.store.win_vecs[wt_w_idx],
            "var_w": self.store.win_vecs[var_w_idx],
            "wt_m": self.store.win_masks[wt_w_idx],
            "var_m": self.store.win_masks[var_w_idx],
            "go": self.go[idx],
            "label": self.labels[idx],
        }


def iter_batches(ds: FastVariantDataset, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
    n = len(ds)
    dev = ds.store.device
    if shuffle:
        order = torch.randperm(n, generator=generator).to(dev)
    else:
        order = torch.arange(n, device=dev)
    for start in range(0, n, batch_size):
        yield ds.get_batch(order[start : start + batch_size])


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """10-bin Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            ece += (in_bin.sum() / n) * abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    n = len(labels)
    if n == 0:
        return {"n": 0}

    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    mcc = float(matthews_corrcoef(labels, preds)) if len(np.unique(np.concatenate([labels, preds]))) > 1 else 0.0

    # Confidence & uncertainty tracking (18 Nov meeting item 1)
    conf = probs.max(axis=1)
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / math.log(3.0)  # normalized [0, 1]
    correct_mask = preds == labels
    conf_correct = float(conf[correct_mask].mean()) if correct_mask.any() else float("nan")
    conf_wrong = float(conf[~correct_mask].mean()) if (~correct_mask).any() else float("nan")

    res: dict[str, Any] = {
        "n": int(n),
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct_mask.mean()),
        "balanced_accuracy": float(rec.mean()),
        "mcc": mcc,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "confidence": {
            "mean_confidence": float(conf.mean()),
            "mean_confidence_correct": conf_correct,
            "mean_confidence_wrong": conf_wrong,
            "confidence_gap": float(conf_correct - conf_wrong) if np.isfinite(conf_correct) and np.isfinite(conf_wrong) else float("nan"),
            "mean_margin": float(margin.mean()),
            "mean_norm_entropy": float(entropy.mean()),
            "ece_10bin": compute_ece(probs, labels, n_bins=10),
        },
    }

    aps = []
    for i, name in enumerate(LABELS):
        target = labels == i
        auprc = float(average_precision_score(target, probs[:, i])) if target.any() else None
        auroc = float(roc_auc_score(target, probs[:, i])) if target.any() and (~target).any() else None
        if auprc is not None:
            aps.append(auprc)
        res[name.lower()] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
            "auprc": auprc,
            "auroc": auroc,
        }
    res["macro_auprc"] = float(np.mean(aps)) if aps else None
    return res


def forward_batch(model: PairedGODecoder, batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    wt = {
        "global": batch["wt_g"].to(device, non_blocking=True),
        "mut": batch["wt_l"].to(device, non_blocking=True),
    }
    var = {
        "global": batch["var_g"].to(device, non_blocking=True),
        "mut": batch["var_l"].to(device, non_blocking=True),
    }
    go = batch["go"].to(device, non_blocking=True)
    wt_w = batch["wt_w"].to(device, dtype=torch.float32, non_blocking=True)
    var_w = batch["var_w"].to(device, dtype=torch.float32, non_blocking=True)
    wt_m = batch["wt_m"].to(device, non_blocking=True)
    var_m = batch["var_m"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    # Ensure mutation center (index 3) is always marked valid
    wt_m[:, 3] = True
    var_m[:, 3] = True

    out = model(wt, var, go, wt_window=wt_w, var_window=var_w, wt_mask=wt_m, var_mask=var_m)
    return out, labels


@torch.no_grad()
def evaluate_dataset(
    model: PairedGODecoder,
    ds: FastVariantDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    total_n = 0
    for batch in iter_batches(ds, batch_size=batch_size, shuffle=False):
        out, labels = forward_batch(model, batch, device)
        loss = classification_loss(out, labels, "multiclass", reduction="sum")
        total_loss += float(loss.item())
        total_n += len(labels)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(out["prediction"].cpu().numpy())
        all_probs.append(out["scores"].cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        total_loss / max(1, total_n),
    )


def run_single_experiment(
    *,
    run_idx: int,
    total_runs: int,
    family: str,
    source: str,
    window_mode: str,
    seed: int,
    train_ds: FastVariantDataset,
    val_ds: FastVariantDataset,
    test_ds: FastVariantDataset,
    common_test_ids: set[str],
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    tournament_start: float,
) -> dict[str, Any]:
    run_name = f"{family}__{source}__{window_mode}__seed{seed}"
    run_dir = out_dir / "runs" / run_name
    metrics_path = run_dir / "metrics.json"

    if args.resume and metrics_path.exists():
        print(f"[{run_idx:02d}/{total_runs:02d}] SKIP (resume) {run_name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    eval_bs = args.batch_size * 4

    model = PairedGODecoder(
        input_dim=1280,
        go_dim=train_ds.store.go.dimension,
        width=args.width,
        heads=args.heads,
        head="multiclass",
        pair_mode="joint",
        window_mode=window_mode,
        dropout=args.dropout,
        window_size=7,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    run_t0 = time.perf_counter()

    print(f"\n[{run_idx:02d}/{total_runs:02d}] START {run_name} (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})", flush=True)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for batch in iter_batches(train_ds, batch_size=args.batch_size, shuffle=True, generator=g):
            optimizer.zero_grad(set_to_none=True)
            out, labels = forward_batch(model, batch, device)
            loss = classification_loss(out, labels, "multiclass", reduction="mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(labels)
            train_n += len(labels)

        train_loss = train_loss_sum / max(1, train_n)
        v_labels, v_preds, v_probs, v_loss = evaluate_dataset(model, val_ds, eval_bs, device)
        v_metrics = compute_metrics(v_labels, v_preds, v_probs)
        val_f1 = v_metrics["macro_f1"]
        val_conf = v_metrics["confidence"]["mean_confidence"]
        ep_dt = time.perf_counter() - ep_t0

        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = "*"
        else:
            no_improve += 1
            star = " "

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": v_loss,
            "val_macro_f1": val_f1,
            "val_gof_f1": v_metrics["gof"]["f1"],
            "val_mean_confidence": val_conf,
            "seconds": ep_dt,
        })

        print(
            f"  Ep {epoch:02d}/{args.epochs:02d} {star} | tr_loss={train_loss:.4f} | val_loss={v_loss:.4f} "
            f"| val_F1={val_f1:.4f} (best={best_val_f1:.4f}@{best_epoch}) "
            f"| val_GOF={v_metrics['gof']['f1']:.4f} | conf={val_conf:.3f} | {ep_dt:.1f}s",
            flush=True,
        )

        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    # Save best model & evaluate test ONCE
    assert best_state is not None
    torch.save(
        {
            "format": "contvar_phase1_tournament_v1",
            "family": family,
            "source": source,
            "window_mode": window_mode,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "model_spec": model.spec,
            "state_dict": best_state,
        },
        run_dir / "best_model.pt",
    )

    model.load_state_dict(best_state)
    v_labels, v_preds, v_probs, v_loss = evaluate_dataset(model, val_ds, eval_bs, device)
    t_labels, t_preds, t_probs, t_loss = evaluate_dataset(model, test_ds, eval_bs, device)

    test_frame = test_ds.frame.copy()
    is_common = test_frame["sample_id"].isin(common_test_ids).to_numpy(dtype=bool)
    is_comb = is_true_series(test_frame["is_combinatorial"]).to_numpy(dtype=bool) if "is_combinatorial" in test_frame.columns else np.zeros(len(test_frame), dtype=bool)
    is_single = ~is_comb

    val_metrics = compute_metrics(v_labels, v_preds, v_probs)
    val_metrics["loss"] = v_loss

    full_test_metrics = compute_metrics(t_labels, t_preds, t_probs)
    full_test_metrics["loss"] = t_loss

    common_test_metrics = compute_metrics(t_labels[is_common], t_preds[is_common], t_probs[is_common])
    single_test_metrics = compute_metrics(t_labels[is_single], t_preds[is_single], t_probs[is_single])
    comb_test_metrics = compute_metrics(t_labels[is_comb], t_preds[is_comb], t_probs[is_comb]) if is_comb.any() else {"n": 0}

    # Save row-level test predictions with confidence & entropy
    clipped = np.clip(t_probs, 1e-12, 1.0)
    ent = -(clipped * np.log(clipped)).sum(axis=1) / math.log(3.0)
    pred_df = pd.DataFrame({
        "sample_id": test_frame["sample_id"],
        "protein_id": test_frame["protein_id"],
        "snp": test_frame["snp"],
        "matched_go_id": test_frame["matched_go_id"],
        "is_combinatorial": is_comb,
        "is_common_test": is_common,
        "label": test_frame["label"],
        "prediction": [LABELS[i] for i in t_preds],
        "prob_lof": t_probs[:, 0],
        "prob_gof": t_probs[:, 1],
        "prob_neutral": t_probs[:, 2],
        "confidence": t_probs.max(axis=1),
        "norm_entropy": ent,
    })
    pred_df.to_csv(run_dir / "test_predictions.tsv", sep="\t", index=False)

    run_sec = time.perf_counter() - run_t0
    payload = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "source": source,
        "window_mode": window_mode,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "common_test_rows": int(is_common.sum()),
        "duration_seconds": run_sec,
        "val": val_metrics,
        "full_test": full_test_metrics,
        "common_test": common_test_metrics,
        "single_test": single_test_metrics,
        "combinatorial_test": comb_test_metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - tournament_start
    avg_per_run = elapsed / run_idx
    eta = avg_per_run * (total_runs - run_idx)
    print(
        f"  DONE {run_name} in {run_sec:.1f}s | Full Test F1={full_test_metrics['macro_f1']:.4f} "
        f"(GOF={full_test_metrics['gof']['f1']:.4f}) | Common Test F1={common_test_metrics['macro_f1']:.4f} "
        f"(GOF={common_test_metrics['gof']['f1']:.4f}) | Conf={full_test_metrics['confidence']['mean_confidence']:.3f} "
        f"| Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}",
        flush=True,
    )
    return payload


def build_summaries_and_report(all_results: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for r in all_results:
        ft = r["full_test"]
        ct = r["common_test"]
        rows.append({
            "family": r["family"],
            "source": r["source"],
            "window_mode": r["window_mode"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "val_macro_f1": r["val"]["macro_f1"],
            "full_n": ft["n"],
            "full_macro_f1": ft["macro_f1"],
            "full_lof_f1": ft["lof"]["f1"],
            "full_gof_f1": ft["gof"]["f1"],
            "full_neu_f1": ft["neutral"]["f1"],
            "full_mcc": ft["mcc"],
            "full_auprc": ft["macro_auprc"],
            "full_conf": ft["confidence"]["mean_confidence"],
            "full_conf_gap": ft["confidence"]["confidence_gap"],
            "full_ece": ft["confidence"]["ece_10bin"],
            "common_n": ct["n"],
            "common_macro_f1": ct["macro_f1"],
            "common_lof_f1": ct["lof"]["f1"],
            "common_gof_f1": ct["gof"]["f1"],
            "common_neu_f1": ct["neutral"]["f1"],
            "common_mcc": ct["mcc"],
            "common_conf": ct["confidence"]["mean_confidence"],
            "common_ece": ct["confidence"]["ece_10bin"],
        })

    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_dir / "summary_all_runs.csv", index=False)

    # Aggregate 3-seed mean +/- std
    metric_cols = [
        "val_macro_f1",
        "full_macro_f1", "full_lof_f1", "full_gof_f1", "full_neu_f1", "full_mcc", "full_auprc", "full_conf", "full_conf_gap", "full_ece",
        "common_macro_f1", "common_lof_f1", "common_gof_f1", "common_neu_f1", "common_mcc", "common_conf", "common_ece",
    ]
    agg_rows = []
    for (fam, src, wmode), grp in df_all.groupby(["family", "source", "window_mode"], sort=False):
        rec: dict[str, Any] = {
            "family": fam,
            "source": src,
            "window_mode": wmode,
            "n_seeds": len(grp),
            "full_n": int(grp["full_n"].iloc[0]),
            "common_n": int(grp["common_n"].iloc[0]),
        }
        for col in metric_cols:
            rec[f"{col}_mean"] = float(grp[col].mean())
            rec[f"{col}_std"] = float(grp[col].std(ddof=1)) if len(grp) > 1 else 0.0
        agg_rows.append(rec)

    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(out_dir / "summary_3seed_mean_std.csv", index=False)

    # Generate clean Markdown presentation report
    lines = [
        "# ContVAR Phase 1 Tournament Report (3-Seed Benchmark)",
        "",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Protocol:** `PairedGODecoder` (`multiclass joint`), 3 fixed seeds `[42, 43, 44]`, augmentation kept in `train` only (removed from `val`/`test`), checkpoint selected strictly on `val` Macro-F1.",
        "",
    ]

    def fmt(m: float, s: float) -> str:
        return f"{m:.4f} ± {s:.4f}"

    for fam in df_agg["family"].unique():
        sub = df_agg[df_agg["family"] == fam].sort_values("common_macro_f1_mean", ascending=False)
        lines.append(f"## {fam.upper()} Split — Common Test (Birebir Adil Kıyas, n={sub['common_n'].iloc[0]})")
        lines.append("")
        lines.append("| Source | Window | Common Macro-F1 | LOF-F1 | GOF-F1 | NEUTRAL-F1 | MCC | Mean Conf | ECE ↓ |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['source']}` | `{r['window_mode']}` | **{fmt(r['common_macro_f1_mean'], r['common_macro_f1_std'])}** "
                f"| {fmt(r['common_lof_f1_mean'], r['common_lof_f1_std'])} "
                f"| {fmt(r['common_gof_f1_mean'], r['common_gof_f1_std'])} "
                f"| {fmt(r['common_neu_f1_mean'], r['common_neu_f1_std'])} "
                f"| {fmt(r['common_mcc_mean'], r['common_mcc_std'])} "
                f"| {r['common_conf_mean']:.3f} | {r['common_ece_mean']:.4f} |"
            )
        lines.append("")

        sub_full = df_agg[df_agg["family"] == fam].sort_values("full_macro_f1_mean", ascending=False)
        lines.append(f"## {fam.upper()} Split — Full Test (Kendi Tam Test Kümesi)")
        lines.append("")
        lines.append("| Source | Window | N | Full Macro-F1 | LOF-F1 | GOF-F1 | NEUTRAL-F1 | Conf Gap (C-W) | ECE ↓ |")
        lines.append("| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in sub_full.iterrows():
            lines.append(
                f"| `{r['source']}` | `{r['window_mode']}` | {int(r['full_n'])} "
                f"| **{fmt(r['full_macro_f1_mean'], r['full_macro_f1_std'])}** "
                f"| {fmt(r['full_lof_f1_mean'], r['full_lof_f1_std'])} "
                f"| {fmt(r['full_gof_f1_mean'], r['full_gof_f1_std'])} "
                f"| {fmt(r['full_neu_f1_mean'], r['full_neu_f1_std'])} "
                f"| {r['full_conf_gap_mean']:+.3f} | {r['full_ece_mean']:.4f} |"
            )
        lines.append("")

    report_path = out_dir / "PHASE1_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== REPORTS SAVED TO: {out_dir} ===", flush=True)
    print(f"  - {out_dir / 'summary_all_runs.csv'}", flush=True)
    print(f"  - {out_dir / 'summary_3seed_mean_std.csv'}", flush=True)
    print(f"  - {report_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--families", nargs="+", choices=("hard", "easy"), default=["hard", "easy"])
    p.add_argument("--sources", nargs="+", choices=("baseline", "trained_mean", "trained_distance"),
                   default=["baseline", "trained_mean", "trained_distance"])
    p.add_argument("--window-modes", nargs="+", choices=("pair_mean", "esim"), default=["pair_mean", "esim"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--resume", action="store_true", help="Skip already completed runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gpu_info = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_runs = len(args.families) * len(args.sources) * len(args.window_modes) * len(args.seeds)

    print("=" * 78, flush=True)
    print("ContVAR PHASE 1 TOURNAMENT", flush=True)
    print(f"Device       : {device} ({gpu_info})", flush=True)
    print(f"Families     : {args.families}", flush=True)
    print(f"Sources      : {args.sources}", flush=True)
    print(f"Window Modes : {args.window_modes}", flush=True)
    print(f"Seeds        : {args.seeds}", flush=True)
    print(f"Total Runs   : {total_runs}", flush=True)
    print(f"Output Dir   : {args.out_dir}", flush=True)
    print("=" * 78, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    go_table = GOEmbeddingTable(DATA_DIR / "embeddings" / "go_terms_embeddings_anc2vec.npz")

    all_results: list[dict[str, Any]] = []
    run_idx = 0
    tournament_start = time.perf_counter()

    for family in args.families:
        print(f"\n{'#' * 78}\n# FAMILY: {family.upper()}\n{'#' * 78}", flush=True)
        raw_train = load_prepared_split(family, "train")
        raw_val = load_prepared_split(family, "val")
        raw_test = load_prepared_split(family, "test")

        # Compute exact common test IDs across all selected sources
        common_test_ids = set(raw_test["sample_id"])
        for src in args.sources:
            m = valid_key_mask(raw_test, SOURCE_DIRS[src])
            common_test_ids &= set(raw_test.loc[m, "sample_id"])
        print(f"  [Common Test] {family}: {len(common_test_ids)} / {len(raw_test)} test rows shared by all sources {args.sources}", flush=True)

        for source in args.sources:
            embed_dir = SOURCE_DIRS[source]
            m_tr = valid_key_mask(raw_train, embed_dir)
            m_va = valid_key_mask(raw_val, embed_dir)
            m_te = valid_key_mask(raw_test, embed_dir)

            f_tr = raw_train.loc[m_tr].reset_index(drop=True)
            f_va = raw_val.loc[m_va].reset_index(drop=True)
            f_te = raw_test.loc[m_te].reset_index(drop=True)
            print(
                f"\n--- Source: {source} | train={len(f_tr)}/{len(raw_train)}, "
                f"val={len(f_va)}/{len(raw_val)}, test={len(f_te)}/{len(raw_test)} ---",
                flush=True,
            )

            store = PreloadedFeatureStore(source, [f_tr, f_va, f_te], go_table, device)
            train_ds = FastVariantDataset(f_tr, store)
            val_ds = FastVariantDataset(f_va, store)
            test_ds = FastVariantDataset(f_te, store)

            for window_mode in args.window_modes:
                for seed in args.seeds:
                    run_idx += 1
                    res = run_single_experiment(
                        run_idx=run_idx,
                        total_runs=total_runs,
                        family=family,
                        source=source,
                        window_mode=window_mode,
                        seed=seed,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        test_ds=test_ds,
                        common_test_ids=common_test_ids,
                        device=device,
                        out_dir=args.out_dir,
                        args=args,
                        tournament_start=tournament_start,
                    )
                    all_results.append(res)

            # Free RAM before loading next source
            del store, train_ds, val_ds, test_ds
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    build_summaries_and_report(all_results, args.out_dir)
    print(f"\nALL {total_runs} RUNS COMPLETED IN {format_seconds(time.perf_counter() - tournament_start)}!", flush=True)


if __name__ == "__main__":
    main()
