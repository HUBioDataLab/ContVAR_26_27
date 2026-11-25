#!/usr/bin/env python3
"""ContVAR MF-Only Master Experiment Suite (Round 1 Screening -> Round 2 3-Seed Finalists).

100% Official Molecular Function (MF-Only):
  - HARD: train=33,852, val=3,772, test=3,875 (LOF=2,892, GOF=255, NEUTRAL=728)
  - EASY: train=34,050, val=3,584, test=3,313 (LOF=2,543, GOF=123, NEUTRAL=647)
  - Val/Test augmentation strictly removed.
  - Precomputed 18-feature Encoder-2 cache (15 biophysical + 3 ESM-2 650M LLR, 100% coverage).

Round 1 (--round 1):
  - 24 controlled experiments on HARD MF-Only, Seed 42 (V5 Distance)
  - Screens B1-B6, C1-C5, D (no_aux, biophys15, concat18, aux_film, dual_film),
    E1-E5 (CE, Reverse, Reverse+Identity-Neutral, 1/sqrt(N), Focal, Logit-Adjusted CE),
    F1-F4 (Light Reg, Embedding Noise, Temperature Scaling, Weight EMA), + 2 Top Combos.
  - Ranks strictly by VALIDATION Macro-F1 (zero test leakage!).

Round 2 (--round 2, or --round all):
  - PDF Page 9 Input Modality Ablations (GO-only, WT+GO, VAR+GO, WT+VAR+GO) x 3 seeds (HARD & EASY)
  - Top-5 Round 1 Finalists + Raw Reference x 3 seeds [42, 43, 44] on HARD & EASY (V5 Distance)
  - Top-2 Finalists cross-checked on V5 Mean x 3 seeds (HARD & EASY)
  - Multi-Architecture Temperature-Scaled Ensemble on HARD & EASY!

Outputs saved neatly under decoder/master_results/ (gitignored).
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
from torch.nn import functional as F

DECODER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DECODER_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_ROOT = DECODER_DIR / "master_results"

sys.path.insert(0, str(DECODER_DIR))
from master_models import (  # noqa: E402
    MasterGODecoder,
    ModelEMA,
    fit_temperature,
    focal_loss,
    identity_neutral_loss,
    logit_adjusted_ce,
    reverse_consistency_loss,
)
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402

LABELS = ("LOF", "GOF", "NEUTRAL")
SOURCE_DIRS = {
    "trained_distance": DATA_DIR / "embeddings" / "trained_win7_distance",
    "trained_mean": DATA_DIR / "embeddings" / "trained_win7_mean",
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
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"


def load_mf_split(family: str, split: str, mf_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "prepared" / f"{family}_{split}.tsv", sep="\t", low_memory=False)
    if split != "train" and "augmented" in df.columns:
        df = df[~df["augmented"].astype(str).str.strip().str.lower().eq("true")]
    df = df[df["matched_go_id"].isin(mf_ids)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Fast VRAM Feature Store (with optional Per-Site Combinatorial Windows for C5)
# ---------------------------------------------------------------------------
class MasterFeatureStore:
    def __init__(self, source: str, frames: list[pd.DataFrame], go_table: GOEmbeddingTable, device: torch.device) -> None:
        self.source = source
        self.embed_dir = SOURCE_DIRS[source]
        self.go = go_table
        self.device = device
        t0 = time.perf_counter()

        g_keys = sorted(set().union(*(set(f["wt_global_key"]) | set(f["var_global_key"]) for f in frames)))
        l_keys = sorted(set().union(*(set(f["wt_mut_key"]) | set(f["var_mut_key"]) for f in frames)))
        w_keys = sorted(set().union(*(set(f["wt_window_key"]) | set(f["var_window_key"]) for f in frames)))

        self.g_map = {k: i for i, k in enumerate(g_keys)}
        self.l_map = {k: i for i, k in enumerate(l_keys)}
        self.w_map = {k: i for i, k in enumerate(w_keys)}

        g_arr = np.empty((len(g_keys), 1280), dtype=np.float32)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_global.h5", "r") as hg:
            for i, k in enumerate(g_keys):
                g_arr[i] = hg[k][:]

        l_arr = np.empty((len(l_keys), 1280), dtype=np.float32)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_local.h5", "r") as hl:
            for i, k in enumerate(l_keys):
                l_arr[i] = hl[k][:]

        w_arr = np.empty((len(w_keys), 7, 1280), dtype=np.float16)
        m_arr = np.ones((len(w_keys), 7), dtype=bool)
        with h5py.File(self.embed_dir / "phase3_seq_phase1_window.h5", "r") as hw:
            has_mask = "mask" in hw
            for i, k in enumerate(w_keys):
                w_arr[i] = hw[k][:].astype(np.float16)
                if has_mask:
                    m_arr[i] = hw["mask"][k][:].astype(bool)

        self.global_vecs = torch.from_numpy(g_arr)
        self.mut_vecs = torch.from_numpy(l_arr)
        self.win_vecs = torch.from_numpy(w_arr)
        self.win_masks = torch.from_numpy(m_arr)

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

        mem_mb = (self.global_vecs.nbytes + self.mut_vecs.nbytes + self.win_vecs.nbytes + self.win_masks.nbytes) / (1024 ** 2)
        print(f"  [Preload] {source}: g={len(g_keys)}, l={len(l_keys)}, w={len(w_keys)} ({mem_mb:.0f} MB in {loc}, {time.perf_counter()-t0:.1f}s)", flush=True)


class PerSiteCache:
    """Lazy-loaded per-site combinatorial windows [M<=4, 7, 1280] for C5 (persite_mil)."""

    def __init__(self, embed_dir: Path, frames: list[pd.DataFrame], max_sites: int = 4) -> None:
        self.max_sites = max_sites
        comb_keys = set()
        for f in frames:
            if "is_combinatorial" in f.columns:
                cm = f["is_combinatorial"].astype(str).str.lower().eq("true")
                comb_keys |= set(f.loc[cm, "wt_window_key"]) | set(f.loc[cm, "var_window_key"])

        self.ps_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with h5py.File(embed_dir / "phase3_seq_phase1_window.h5", "r") as hw:
            if "per_site" in hw:
                ps_grp = hw["per_site"]
                pm_grp = hw["per_site_mask"] if "per_site_mask" in hw else None
                for k in comb_keys:
                    if k in ps_grp:
                        arr = ps_grp[k][:max_sites].astype(np.float16)  # [m, 7, 1280]
                        msk = pm_grp[k][:max_sites].astype(bool) if (pm_grp is not None and k in pm_grp) else np.ones(arr.shape[:2], dtype=bool)
                        self.ps_data[k] = (arr, msk)
        print(f"  [PerSite] Loaded {len(self.ps_data)} combinatorial per-site window tensors (max_sites={max_sites})", flush=True)


class MasterDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        store: MasterFeatureStore,
        aux_18: np.ndarray,
        persite_cache: PerSiteCache | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.store = store
        self.persite_cache = persite_cache
        dev = store.device

        self.wt_g = torch.tensor([store.g_map[k] for k in self.frame["wt_global_key"]], dtype=torch.long, device=dev)
        self.var_g = torch.tensor([store.g_map[k] for k in self.frame["var_global_key"]], dtype=torch.long, device=dev)
        self.wt_l = torch.tensor([store.l_map[k] for k in self.frame["wt_mut_key"]], dtype=torch.long, device=dev)
        self.var_l = torch.tensor([store.l_map[k] for k in self.frame["var_mut_key"]], dtype=torch.long, device=dev)
        self.wt_w = torch.tensor([store.w_map[k] for k in self.frame["wt_window_key"]], dtype=torch.long, device=dev)
        self.var_w = torch.tensor([store.w_map[k] for k in self.frame["var_window_key"]], dtype=torch.long, device=dev)
        self.labels = torch.tensor(self.frame["label_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=dev)
        self.go = store.go.lookup(self.frame["matched_go_id"].tolist()).to(dev)
        self.aux = torch.tensor(aux_18, dtype=torch.float32, device=dev)
        self.wt_w_keys = self.frame["wt_window_key"].tolist()
        self.var_w_keys = self.frame["var_window_key"].tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def get_batch(self, idx: torch.Tensor, need_persite: bool = False) -> dict[str, Any]:
        wt_wi, var_wi = self.wt_w[idx], self.var_w[idx]
        batch: dict[str, Any] = {
            "wt_g": self.store.global_vecs[self.wt_g[idx]],
            "var_g": self.store.global_vecs[self.var_g[idx]],
            "wt_l": self.store.mut_vecs[self.wt_l[idx]],
            "var_l": self.store.mut_vecs[self.var_l[idx]],
            "wt_w": self.store.win_vecs[wt_wi],
            "var_w": self.store.win_vecs[var_wi],
            "wt_m": self.store.win_masks[wt_wi],
            "var_m": self.store.win_masks[var_wi],
            "go": self.go[idx],
            "aux": self.aux[idx],
            "label": self.labels[idx],
        }
        if need_persite and self.persite_cache is not None:
            idx_cpu = idx.cpu().tolist()
            B = len(idx_cpu)
            M = self.persite_cache.max_sites
            ps_wt = np.zeros((B, M, 7, 1280), dtype=np.float16)
            ps_var = np.zeros((B, M, 7, 1280), dtype=np.float16)
            ps_m = np.zeros((B, M, 7), dtype=bool)
            ps_sv = np.zeros((B, M), dtype=bool)

            wt_w_np = batch["wt_w"].detach().cpu().numpy()
            var_w_np = batch["var_w"].detach().cpu().numpy()
            wt_m_np = (batch["wt_m"] & batch["var_m"]).detach().cpu().numpy()

            for b, row_i in enumerate(idx_cpu):
                wk = self.wt_w_keys[row_i]
                vk = self.var_w_keys[row_i]
                if wk in self.persite_cache.ps_data and vk in self.persite_cache.ps_data:
                    wa, wm = self.persite_cache.ps_data[wk]
                    va, vm = self.persite_cache.ps_data[vk]
                    m_cnt = min(len(wa), len(va), M)
                    ps_wt[b, :m_cnt] = wa[:m_cnt]
                    ps_var[b, :m_cnt] = va[:m_cnt]
                    ps_m[b, :m_cnt] = wm[:m_cnt] & vm[:m_cnt]
                    ps_sv[b, :m_cnt] = True
                else:
                    ps_wt[b, 0] = wt_w_np[b]
                    ps_var[b, 0] = var_w_np[b]
                    ps_m[b, 0] = wt_m_np[b]
                    ps_sv[b, 0] = True

            batch["ps_wt_w"] = torch.from_numpy(ps_wt)
            batch["ps_var_w"] = torch.from_numpy(ps_var)
            batch["ps_mask"] = torch.from_numpy(ps_m)
            batch["ps_site_valid"] = torch.from_numpy(ps_sv)
        return batch


def iter_batches(ds: MasterDataset, batch_size: int, shuffle: bool, generator: torch.Generator | None = None, need_persite: bool = False):
    n = len(ds)
    dev = ds.store.device
    order = torch.randperm(n, generator=generator).to(dev) if shuffle else torch.arange(n, device=dev)
    for start in range(0, n, batch_size):
        yield ds.get_batch(order[start : start + batch_size], need_persite=need_persite)


# ---------------------------------------------------------------------------
# Metrics, Calibration (ECE + Temperature Scaling + Val Logit Bias)
# ---------------------------------------------------------------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels, preds, probs = np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64), np.asarray(probs, dtype=np.float64)
    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    mcc = float(matthews_corrcoef(labels, preds)) if len(np.unique(np.concatenate([labels, preds]))) > 1 else 0.0
    conf = probs.max(axis=1)
    correct = preds == labels
    c_ok = float(conf[correct].mean()) if correct.any() else float("nan")
    c_no = float(conf[~correct].mean()) if (~correct).any() else float("nan")
    res: dict[str, Any] = {
        "n": int(len(labels)),
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.mean()),
        "mcc": mcc,
        "ece": compute_ece(probs, labels),
        "mean_conf": float(conf.mean()),
        "conf_gap": float(c_ok - c_no) if np.isfinite(c_ok) and np.isfinite(c_no) else float("nan"),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
    }
    for i, name in enumerate(LABELS):
        tgt = labels == i
        res[name.lower()] = {
            "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i]),
            "auprc": float(average_precision_score(tgt, probs[:, i])) if tgt.any() else None,
            "auroc": float(roc_auc_score(tgt, probs[:, i])) if tgt.any() and (~tgt).any() else None,
        }
    return res


def tune_val_logit_bias(val_logits: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    best_bias = np.zeros(3, dtype=np.float64)
    best_f1 = compute_metrics(val_labels, val_logits.argmax(1), F.softmax(torch.tensor(val_logits), dim=-1).numpy())["macro_f1"]
    for b_gof in np.linspace(-0.4, 1.4, 19):
        for b_neu in np.linspace(-0.4, 0.8, 13):
            bias = np.array([0.0, b_gof, b_neu], dtype=np.float64)
            preds = (val_logits + bias).argmax(axis=1)
            f1s = []
            for c in range(3):
                tp = np.sum((preds == c) & (val_labels == c))
                fp = np.sum((preds == c) & (val_labels != c))
                fn = np.sum((preds != c) & (val_labels == c))
                d = 2 * tp + fp + fn
                f1s.append((2 * tp / d) if d > 0 else 0.0)
            mf1 = float(np.mean(f1s))
            if mf1 > best_f1 + 1e-6:
                best_f1 = mf1
                best_bias = bias
    return best_bias


def call_model(
    model: MasterGODecoder,
    batch: dict[str, Any],
    device: torch.device,
    aux_slice: int,
    mode: str = "fwd",  # fwd, rev, id_wt
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    wt_g = batch["wt_g"].to(device)
    var_g = batch["var_g"].to(device)
    wt_l = batch["wt_l"].to(device)
    var_l = batch["var_l"].to(device)
    wt_w = batch["wt_w"].to(device, dtype=torch.float32)
    var_w = batch["var_w"].to(device, dtype=torch.float32)
    wt_m = batch["wt_m"].to(device).clone()
    var_m = batch["var_m"].to(device).clone()
    wt_m[:, 3] = True
    var_m[:, 3] = True
    go = batch["go"].to(device)
    labels = batch["label"].to(device)
    aux = batch["aux"][:, :aux_slice].to(device) if aux_slice > 0 else None

    ps_wt = batch["ps_wt_w"].to(device, dtype=torch.float32) if "ps_wt_w" in batch else None
    ps_var = batch["ps_var_w"].to(device, dtype=torch.float32) if "ps_var_w" in batch else None
    ps_m = batch["ps_mask"].to(device) if "ps_mask" in batch else None
    ps_sv = batch["ps_site_valid"].to(device) if "ps_site_valid" in batch else None

    if mode == "rev":
        if aux is not None:
            aux = aux.clone()
            # Negate signed directional deltas (charge, hydro, vol, polarity; plus LLR if present)
            neg_cols = [2, 4, 6, 8] + ([15, 16] if aux_slice >= 18 else [])
            aux[:, neg_cols] = -aux[:, neg_cols]
        out = model(var_g, wt_g, var_l, wt_l, var_w, wt_w, var_m, wt_m, go, aux=aux,
                    ps_wt_w=ps_var, ps_var_w=ps_wt, ps_mask=ps_m, ps_site_valid=ps_sv)
    elif mode == "id_wt":
        # Identity-Neutral: WT == VAR => zero physical change
        aux_id = torch.zeros_like(aux) if aux is not None else None
        out = model(wt_g, wt_g, wt_l, wt_l, wt_w, wt_w, wt_m, wt_m, go, aux=aux_id,
                    ps_wt_w=ps_wt, ps_var_w=ps_wt, ps_mask=ps_m, ps_site_valid=ps_sv)
    else:
        out = model(wt_g, var_g, wt_l, var_l, wt_w, var_w, wt_m, var_m, go, aux=aux,
                    ps_wt_w=ps_wt, ps_var_w=ps_var, ps_mask=ps_m, ps_site_valid=ps_sv)
    return out, labels


@torch.no_grad()
def evaluate_ds(
    model: MasterGODecoder,
    ds: MasterDataset,
    batch_size: int,
    device: torch.device,
    aux_slice: int,
    need_persite: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    ys, ps, pr, lg = [], [], [], []
    loss_sum, n_tot = 0.0, 0
    for batch in iter_batches(ds, batch_size=batch_size, shuffle=False, need_persite=need_persite):
        out, labels = call_model(model, batch, device, aux_slice, mode="fwd")
        loss_sum += float(F.cross_entropy(out["logits"], labels, reduction="sum").item())
        n_tot += len(labels)
        ys.append(labels.cpu().numpy())
        ps.append(out["prediction"].cpu().numpy())
        pr.append(out["scores"].cpu().numpy())
        lg.append(out["logits"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(pr), np.concatenate(lg), loss_sum / max(1, n_tot)


# ---------------------------------------------------------------------------
# Experiment Definitions: Round 1 (24 Focused Configs Covering ALL B, C, D, E, F)
# ---------------------------------------------------------------------------
def base_cfg(name: str, group: str, **overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "name": name, "group": group,
        "arch": "joint", "window_reader": "esim",
        "aux_mode": "concat", "aux_slice": 18,  # 15 biophys + 3 ESM-2 650M LLR
        "input_mode": "full",
        "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "embed_noise": 0.0, "use_ema": False,
        "loss_type": "ce",  # ce, focal, logit_adj
        "class_weight": "none",  # none, sqrt
        "lambda_reverse": 0.0, "lambda_identity": 0.0,
    }
    cfg.update(overrides)
    return cfg


ROUND1_EXPERIMENTS: list[dict[str, Any]] = [
    # --- F0: Unregularized raw reference (matches ~0.484-0.502 baseline) ---
    base_cfg("F0_raw_baseline", "Ref", aux_mode="none", aux_slice=0,
             adapter_dropout=0.0, attn_dropout=0.0, dropout=0.1, weight_decay=1e-4, label_smoothing=0.0, cosine_lr=False),

    # --- Group B: Main Decoder Architectures (B1-B6) ---
    base_cfg("B1_joint", "Arch"),
    base_cfg("B2_go_film", "Arch", arch="go_film"),
    base_cfg("B3_mag_dir", "Arch", arch="mag_dir"),
    base_cfg("B4_bilinear", "Arch", arch="bilinear"),
    base_cfg("B5_res_mlp", "Arch", arch="res_mlp"),
    base_cfg("B6_delta_only", "Arch", arch="delta_only"),

    # --- Group C: Local Window Readers (C2-C5; C1=B1_joint) ---
    base_cfg("C2_bi_xattn", "Window", window_reader="bi_xattn"),
    base_cfg("C3_cnn1d", "Window", window_reader="cnn1d"),
    base_cfg("C4_hybrid_cnn_attn", "Window", window_reader="hybrid_cnn_attn"),
    base_cfg("C5_persite_mil", "Window", window_reader="persite_mil"),

    # --- Group D: Auxiliary Features & FiLM Conditioning ---
    base_cfg("D0_no_aux", "Aux", aux_mode="none", aux_slice=0),
    base_cfg("D1_biophys15_only", "Aux", aux_mode="concat", aux_slice=15),
    base_cfg("D_aux_film", "Aux", aux_mode="aux_film", aux_slice=18),
    base_cfg("D_dual_film", "Aux", arch="go_film", aux_mode="dual_film", aux_slice=18),

    # --- Group E: Loss Functions, Symmetry & Class Imbalance ---
    base_cfg("E2a_reverse", "Loss", lambda_reverse=0.15),
    base_cfg("E2b_reverse_id_neutral", "Loss", lambda_reverse=0.15, lambda_identity=0.15),
    base_cfg("E3_sqrt_weight", "Loss", class_weight="sqrt"),
    base_cfg("E4_focal", "Loss", loss_type="focal"),
    base_cfg("E5_logit_adj", "Loss", loss_type="logit_adj"),

    # --- Group F: Regularization & Stability ---
    base_cfg("F2_embed_noise", "Reg", embed_noise=0.05),
    base_cfg("F4_weight_ema", "Reg", use_ema=True),

    # --- Strategic Combos (Testing Synergy Early in Round 1) ---
    base_cfg("X1_dual_film_rev_id_sqrt_ema", "Combo", arch="go_film", aux_mode="dual_film",
             lambda_reverse=0.15, lambda_identity=0.10, class_weight="sqrt", use_ema=True),
    base_cfg("X2_magdir_auxfilm_rev_id_sqrt", "Combo", arch="mag_dir", aux_mode="aux_film",
             lambda_reverse=0.15, lambda_identity=0.10, class_weight="sqrt", use_ema=True),
]

INPUT_ABLATION_EXPERIMENTS: list[dict[str, Any]] = [
    base_cfg("S1_go_only", "Sanity", input_mode="go_only", aux_mode="none", aux_slice=0),
    base_cfg("S2_wt_go", "Sanity", input_mode="wt_go", aux_mode="none", aux_slice=0),
    base_cfg("S3_var_go", "Sanity", input_mode="var_go", aux_mode="none", aux_slice=0),
    base_cfg("S4_wt_var_go", "Sanity", input_mode="full", aux_mode="none", aux_slice=0),
]


# ---------------------------------------------------------------------------
# Single Run Trainer
# ---------------------------------------------------------------------------
def run_single(
    *,
    run_idx: int,
    total_runs: int,
    cfg: dict[str, Any],
    family: str,
    source: str,
    seed: int,
    train_ds: MasterDataset,
    val_ds: MasterDataset,
    test_ds: MasterDataset,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    start_time: float,
) -> dict[str, Any]:
    run_name = f"{family}__{source}__{cfg['name']}__seed{seed}"
    run_dir = out_dir / "runs" / run_name
    metrics_path = run_dir / "metrics.json"

    if args.resume and metrics_path.exists():
        print(f"[{run_idx:02d}/{total_runs:02d}] SKIP (resume) {run_name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    need_persite = cfg["window_reader"] == "persite_mil"
    aux_slice = cfg["aux_slice"] if cfg["aux_mode"] != "none" else 0
    g = torch.Generator()
    g.manual_seed(seed)
    eval_bs = args.batch_size * 4

    counts = np.bincount(train_ds.frame["label_idx"].to_numpy(dtype=np.int64), minlength=3).astype(np.float64)
    priors = counts / counts.sum()
    log_priors = torch.tensor(np.log(np.clip(priors, 1e-6, 1.0)), dtype=torch.float32, device=device)

    class_weights = None
    if cfg["class_weight"] == "sqrt":
        w = np.sqrt(counts.sum() / np.maximum(1.0, counts))
        w = w / w.mean()
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    model = MasterGODecoder(
        input_dim=1280,
        go_dim=train_ds.store.go.dimension,
        width=args.width,
        heads=args.heads,
        arch=cfg["arch"],
        window_reader=cfg["window_reader"],
        aux_mode=cfg["aux_mode"],
        aux_dim=aux_slice,
        input_mode=cfg["input_mode"],
        adapter_dropout=cfg["adapter_dropout"],
        attn_dropout=cfg["attn_dropout"],
        dropout=cfg["dropout"],
        embed_noise=cfg["embed_noise"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) if cfg["cosine_lr"] else None
    ema = ModelEMA(model, decay=0.992) if cfg["use_ema"] else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    run_t0 = time.perf_counter()

    print(f"\n[{run_idx:02d}/{total_runs:02d}] START {run_name}", flush=True)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for batch in iter_batches(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, need_persite=need_persite):
            optimizer.zero_grad(set_to_none=True)
            out_fwd, labels = call_model(model, batch, device, aux_slice, mode="fwd")

            if cfg["loss_type"] == "focal":
                loss = focal_loss(out_fwd["logits"], labels, gamma=1.5, weight=class_weights)
            elif cfg["loss_type"] == "logit_adj":
                loss = logit_adjusted_ce(out_fwd["logits"], labels, log_priors, tau=0.75, label_smoothing=cfg["label_smoothing"])
            else:
                loss = F.cross_entropy(out_fwd["logits"], labels, weight=class_weights, label_smoothing=cfg["label_smoothing"])

            if cfg["lambda_reverse"] > 0.0 and cfg["input_mode"] == "full":
                out_rev, _ = call_model(model, batch, device, aux_slice, mode="rev")
                loss = loss + cfg["lambda_reverse"] * reverse_consistency_loss(out_fwd["logits"], out_rev["logits"])

            if cfg["lambda_identity"] > 0.0 and cfg["input_mode"] == "full":
                out_id, _ = call_model(model, batch, device, aux_slice, mode="id_wt")
                loss = loss + cfg["lambda_identity"] * identity_neutral_loss(out_id["logits"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            tr_loss_sum += float(loss.item()) * len(labels)
            tr_n += len(labels)

        if scheduler is not None:
            scheduler.step()

        # Evaluate (using EMA weights if enabled)
        if ema is not None:
            orig_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.shadow)
        vy, vp, vpr, _, v_loss = evaluate_ds(model, val_ds, eval_bs, device, aux_slice, need_persite)
        vm = compute_metrics(vy, vp, vpr)
        val_f1 = vm["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = "*"
        else:
            no_improve += 1
            star = " "

        if ema is not None:
            model.load_state_dict(orig_sd)

        ep_dt = time.perf_counter() - ep_t0
        history.append({
            "epoch": epoch, "tr_loss": tr_loss_sum / max(1, tr_n), "val_loss": v_loss,
            "val_f1": val_f1, "val_gof": vm["gof"]["f1"], "val_ece": vm["ece"], "sec": ep_dt,
        })
        print(
            f"  Ep {epoch:02d}/{args.epochs:02d} {star} | tr={tr_loss_sum/max(1,tr_n):.4f} | val={v_loss:.4f} "
            f"| val_F1={val_f1:.4f} (best={best_val_f1:.4f}@{best_epoch}) | GOF={vm['gof']['f1']:.4f} | ECE={vm['ece']:.3f} | {ep_dt:.1f}s",
            flush=True,
        )
        if no_improve >= args.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    vy, vp, vpr, vlg, _ = evaluate_ds(model, val_ds, eval_bs, device, aux_slice, need_persite)
    ty, tp, tpr, tlg, _ = evaluate_ds(model, test_ds, eval_bs, device, aux_slice, need_persite)

    # Post-hoc F3 Temperature Scaling (strictly fit on val)
    temp = fit_temperature(vlg, vy)
    t_probs_ts = F.softmax(torch.tensor(tlg / temp, dtype=torch.float32), dim=-1).numpy()
    test_ts_m = compute_metrics(ty, t_probs_ts.argmax(1), t_probs_ts)

    # Post-hoc Val-Tuned Logit Bias (strictly fit on val)
    val_bias = tune_val_logit_bias(vlg, vy)
    t_probs_adj = F.softmax(torch.tensor(tlg + val_bias, dtype=torch.float32), dim=-1).numpy()
    test_adj_m = compute_metrics(ty, (tlg + val_bias).argmax(1), t_probs_adj)

    # Save test logits for ensembling in Round 2
    np.save(run_dir / "test_logits.npy", tlg.astype(np.float32))
    np.save(run_dir / "val_logits.npy", vlg.astype(np.float32))

    val_m = compute_metrics(vy, vp, vpr)
    test_m = compute_metrics(ty, tp, tpr)

    run_sec = time.perf_counter() - run_t0
    payload = {
        "run_name": run_name, "exp_name": cfg["name"], "group": cfg["group"],
        "family": family, "source": source, "seed": seed, "best_epoch": best_epoch,
        "duration_seconds": run_sec, "temperature": temp, "val_logit_bias": val_bias.tolist(),
        "cfg": cfg, "val": val_m, "test": test_m, "test_temp_scaled": test_ts_m, "test_logit_adj": test_adj_m,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    eta = (elapsed / run_idx) * (total_runs - run_idx)
    print(
        f"  DONE {run_name} ({run_sec:.1f}s) | Val F1={val_m['macro_f1']:.4f} | Test F1={test_m['macro_f1']:.4f} "
        f"(GOF={test_m['gof']['f1']:.4f}, ECE={test_m['ece']:.4f} -> TS_ECE={test_ts_m['ece']:.4f}) "
        f"| +Adj={test_adj_m['macro_f1']:.4f} | Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}",
        flush=True,
    )
    if args.cool_sleep > 0:
        time.sleep(args.cool_sleep)
    return payload


def summarize_results(results: list[dict[str, Any]], out_dir: Path, title: str, filename: str) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "family": r["family"], "source": r["source"], "group": r["group"], "exp_name": r["exp_name"],
            "seed": r["seed"], "best_epoch": r["best_epoch"],
            "val_f1": r["val"]["macro_f1"], "val_ece": r["val"]["ece"],
            "test_f1": r["test"]["macro_f1"], "test_lof": r["test"]["lof"]["f1"],
            "test_gof": r["test"]["gof"]["f1"], "test_neu": r["test"]["neutral"]["f1"],
            "test_mcc": r["test"]["mcc"], "test_ece": r["test"]["ece"],
            "ts_ece": r["test_temp_scaled"]["ece"], "adj_f1": r["test_logit_adj"]["macro_f1"],
            "adj_gof": r["test_logit_adj"]["gof"]["f1"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{filename}_all.csv", index=False)

    mcols = ["val_f1", "val_ece", "test_f1", "test_lof", "test_gof", "test_neu", "test_mcc", "test_ece", "ts_ece", "adj_f1", "adj_gof"]
    agg_rows = []
    for (fam, src, grp_name, ename), g in df.groupby(["family", "source", "group", "exp_name"], sort=False):
        rec: dict[str, Any] = {"family": fam, "source": src, "group": grp_name, "exp_name": ename, "n_seeds": len(g)}
        for c in mcols:
            rec[f"{c}_mean"] = float(g[c].mean())
            rec[f"{c}_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(rec)
    df_agg = pd.DataFrame(agg_rows).sort_values(["family", "source", "val_f1_mean"], ascending=[True, True, False])
    df_agg.to_csv(out_dir / f"{filename}_summary.csv", index=False)

    lines = [f"# {title}", "", f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`", ""]
    for (fam, src), sub in df_agg.groupby(["family", "source"], sort=False):
        lines.append(f"## {fam.upper()} | {src} (Sorted by Validation Macro-F1)")
        lines.append("")
        lines.append("| Group | Experiment | Val F1 | Test F1 | +Adj F1 | LOF | GOF | NEU | MCC | ECE ↓ | TS ECE ↓ |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for _, r in sub.iterrows():
            fmt = (lambda m, s: f"{m:.4f} ± {s:.4f}") if r["n_seeds"] > 1 else (lambda m, s: f"{m:.4f}")
            lines.append(
                f"| {r['group']} | `{r['exp_name']}` | {fmt(r['val_f1_mean'], r['val_f1_std'])} "
                f"| **{fmt(r['test_f1_mean'], r['test_f1_std'])}** | {fmt(r['adj_f1_mean'], r['adj_f1_std'])} "
                f"| {r['test_lof_mean']:.4f} | {r['test_gof_mean']:.4f} | {r['test_neu_mean']:.4f} "
                f"| {r['test_mcc_mean']:.4f} | {r['test_ece_mean']:.4f} | {r['ts_ece_mean']:.4f} |"
            )
        lines.append("")
    (out_dir / f"{filename}.md").write_text("\n".join(lines), encoding="utf-8")
    return df_agg


def evaluate_top3_ensemble(
    finalist_names: list[str],
    family: str,
    source: str,
    seeds: list[int],
    test_labels: np.ndarray,
    val_labels: np.ndarray,
    r2_dir: Path,
) -> dict[str, Any]:
    """Combine temperature-scaled probabilities of Top-3 distinct finalists per seed."""
    top3 = finalist_names[:3]
    ens_f1s, ens_gofs, ens_eces, ens_mccs = [], [], [], []
    for s in seeds:
        probs_list = []
        for ename in top3:
            rdir = r2_dir / "runs" / f"{family}__{source}__{ename}__seed{s}"
            tlg = np.load(rdir / "test_logits.npy")
            vlg = np.load(rdir / "val_logits.npy")
            t = fit_temperature(vlg, val_labels)
            probs_list.append(F.softmax(torch.tensor(tlg / t, dtype=torch.float32), dim=-1).numpy())
        avg_p = np.mean(probs_list, axis=0)
        m = compute_metrics(test_labels, avg_p.argmax(1), avg_p)
        ens_f1s.append(m["macro_f1"])
        ens_gofs.append(m["gof"]["f1"])
        ens_eces.append(m["ece"])
        ens_mccs.append(m["mcc"])
    return {
        "family": family, "source": source, "members": top3,
        "macro_f1_mean": float(np.mean(ens_f1s)), "macro_f1_std": float(np.std(ens_f1s, ddof=1)),
        "gof_f1_mean": float(np.mean(ens_gofs)), "mcc_mean": float(np.mean(ens_mccs)),
        "ece_mean": float(np.mean(ens_eces)),
    }


def load_standardized_aux(bundle: Any, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use the 18 100%-covered features (15 biophysical + 3 ESM-2 650M LLR)
    tr = bundle[f"{family}_train"][:, :18].astype(np.float32)
    va = bundle[f"{family}_val"][:, :18].astype(np.float32)
    te = bundle[f"{family}_test"][:, :18].astype(np.float32)
    mu = tr.mean(axis=0, keepdims=True)
    sd = np.where(tr.std(axis=0, keepdims=True) > 1e-6, tr.std(axis=0, keepdims=True), 1.0)
    return (tr - mu) / sd, (va - mu) / sd, (te - mu) / sd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", choices=("1", "2", "all"), default="all")
    p.add_argument("--top-k", type=int, default=5, help="Number of Round-1 winners to promote to Round 2 (selected by val F1)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cool-sleep", type=float, default=2.0, help="Seconds to pause between runs for quiet laptop thermals")
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mf_ids = set(json.loads((DATA_DIR / "mf_go_ids.json").read_text(encoding="utf-8")))
    go_table = GOEmbeddingTable(DATA_DIR / "embeddings" / "go_terms_embeddings_anc2vec.npz")
    aux_bundle = np.load(DATA_DIR / "auxiliary" / "mf_auxiliary_bundle.npz", allow_pickle=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    r1_dir = args.out_dir / "round1_screening"
    r2_dir = args.out_dir / "round2_finalists"

    cfg_by_name = {c["name"]: c for c in ROUND1_EXPERIMENTS}

    # =========================================================================
    # ROUND 1: 1-Seed Screening on HARD MF-Only (V5 Distance, Seed 42)
    # =========================================================================
    r1_agg: pd.DataFrame | None = None
    if args.round in ("1", "all"):
        print("=" * 78, flush=True)
        print(f"ROUND 1: 1-SEED SCREENING (HARD MF-Only | V5 Distance | {len(ROUND1_EXPERIMENTS)} runs)", flush=True)
        print("=" * 78, flush=True)

        tr_df = load_mf_split("hard", "train", mf_ids)
        va_df = load_mf_split("hard", "val", mf_ids)
        te_df = load_mf_split("hard", "test", mf_ids)
        aux_tr, aux_va, aux_te = load_standardized_aux(aux_bundle, "hard")

        store = MasterFeatureStore("trained_distance", [tr_df, va_df, te_df], go_table, device)
        ps_cache = PerSiteCache(SOURCE_DIRS["trained_distance"], [tr_df, va_df, te_df], max_sites=4)
        train_ds = MasterDataset(tr_df, store, aux_tr, ps_cache)
        val_ds = MasterDataset(va_df, store, aux_va, ps_cache)
        test_ds = MasterDataset(te_df, store, aux_te, ps_cache)

        r1_results = []
        t0 = time.perf_counter()
        for i, cfg in enumerate(ROUND1_EXPERIMENTS, 1):
            res = run_single(
                run_idx=i, total_runs=len(ROUND1_EXPERIMENTS), cfg=cfg,
                family="hard", source="trained_distance", seed=args.seeds[0],
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                device=device, out_dir=r1_dir, args=args, start_time=t0,
            )
            r1_results.append(res)

        r1_agg = summarize_results(r1_results, r1_dir, "Round 1 — 1-Seed Screening (HARD MF-Only | V5 Distance)", "ROUND1_REPORT")
        print(f"\nROUND 1 COMPLETED IN {format_seconds(time.perf_counter() - t0)}!", flush=True)
        del store, ps_cache, train_ds, val_ds, test_ds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # ROUND 2: 3-Seed Finalists + Input Modality Ablations (HARD & EASY)
    # =========================================================================
    if args.round in ("2", "all"):
        if r1_agg is None:
            r1_csv = r1_dir / "ROUND1_REPORT_summary.csv"
            if not r1_csv.exists():
                raise FileNotFoundError(f"Run --round 1 first! Missing {r1_csv}")
            r1_agg = pd.read_csv(r1_csv)

        # Pick Top-K strictly by VALIDATION Macro-F1 (excluding F0_raw_baseline which is always included as reference)
        cand = r1_agg[r1_agg["exp_name"] != "F0_raw_baseline"].sort_values("val_f1_mean", ascending=False)
        top_names = cand["exp_name"].head(args.top_k).tolist()
        print("\n" + "=" * 78, flush=True)
        print(f"ROUND 2: 3-SEED FINALISTS (Selected strictly by Round 1 Val F1): {top_names}", flush=True)
        print("=" * 78, flush=True)

        finalist_cfgs = [cfg_by_name["F0_raw_baseline"]] + [cfg_by_name[n] for n in top_names]
        v5_mean_cfgs = [cfg_by_name[n] for n in top_names[:2]]  # Cross-check Top-2 on V5 Mean (A2)

        # Count total Round 2 runs
        total_r2 = (
            2 * (len(INPUT_ABLATION_EXPERIMENTS) + len(finalist_cfgs)) * len(args.seeds)
            + 2 * len(v5_mean_cfgs) * len(args.seeds)
        )
        r2_results = []
        ensembles = []
        run_idx = 0
        t0 = time.perf_counter()

        for family in ("hard", "easy"):
            tr_df = load_mf_split(family, "train", mf_ids)
            va_df = load_mf_split(family, "val", mf_ids)
            te_df = load_mf_split(family, "test", mf_ids)
            aux_tr, aux_va, aux_te = load_standardized_aux(aux_bundle, family)

            for source, cfgs_to_run in [
                ("trained_distance", INPUT_ABLATION_EXPERIMENTS + finalist_cfgs),
                ("trained_mean", v5_mean_cfgs),
            ]:
                store = MasterFeatureStore(source, [tr_df, va_df, te_df], go_table, device)
                need_ps_any = any(c["window_reader"] == "persite_mil" for c in cfgs_to_run)
                ps_cache = PerSiteCache(SOURCE_DIRS[source], [tr_df, va_df, te_df], max_sites=4) if need_ps_any else None
                train_ds = MasterDataset(tr_df, store, aux_tr, ps_cache)
                val_ds = MasterDataset(va_df, store, aux_va, ps_cache)
                test_ds = MasterDataset(te_df, store, aux_te, ps_cache)

                for cfg in cfgs_to_run:
                    for seed in args.seeds:
                        run_idx += 1
                        res = run_single(
                            run_idx=run_idx, total_runs=total_r2, cfg=cfg,
                            family=family, source=source, seed=seed,
                            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                            device=device, out_dir=r2_dir, args=args, start_time=t0,
                        )
                        r2_results.append(res)

                if source == "trained_distance" and len(args.seeds) > 1:
                    ens = evaluate_top3_ensemble(
                        top_names, family, source, args.seeds,
                        te_df["label_idx"].to_numpy(dtype=np.int64),
                        va_df["label_idx"].to_numpy(dtype=np.int64),
                        r2_dir,
                    )
                    ensembles.append(ens)
                    print(
                        f"  [ENSEMBLE {family.upper()}] Top-3 {ens['members']} -> "
                        f"Macro-F1={ens['macro_f1_mean']:.4f} ± {ens['macro_f1_std']:.4f} "
                        f"(GOF={ens['gof_f1_mean']:.4f}, MCC={ens['mcc_mean']:.4f}, ECE={ens['ece_mean']:.4f})",
                        flush=True,
                    )

                del store, ps_cache, train_ds, val_ds, test_ds
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        summarize_results(r2_results, r2_dir, "Round 2 — 3-Seed Finalists & Input Ablations (HARD & EASY MF-Only)", "ROUND2_REPORT")
        (r2_dir / "ensembles.json").write_text(json.dumps(ensembles, indent=2), encoding="utf-8")
        print(f"\nALL ROUND 2 RUNS COMPLETED IN {format_seconds(time.perf_counter() - t0)}!", flush=True)


if __name__ == "__main__":
    main()
