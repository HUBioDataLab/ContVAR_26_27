#!/usr/bin/env python3
"""ContVAR Phase 2 & 3 Controlled Ablation & Improvement Suite.

Runs on the Phase 1 Winner (`trained_distance` by default, configurable via --source)
using the exact same 3-seed protocol [42, 43, 44], clean val/test (no augmented rows),
and full confidence/calibration tracking (ECE, conf_gap, entropy, val-tuned logit adj).

7 Controlled Experiments (all 3 seeds):
  1. exp1_esim_relpos       : ESIM + Learned 7-pos Relative Positional Encoding (Meeting #5)
  2. exp2_reg_calib         : ESIM + RelPos + Strong Regularization (Adapter/Attn Dropout, WD=1e-2, Label Smoothing=0.1, Cosine LR)
  3. exp3_reverse_loss      : exp2 + Nihat's Antisymmetry / Reverse-Consistency Loss (WT<->VAR => LOF<->GOF, lambda=0.2)
  4. exp4_imbalance_sqrt    : exp2 + Mild Sqrt Inverse-Frequency Class Weighting (for 7.5% GOF imbalance)
  5. exp5_score_diff        : Score-Difference architecture + Regularization (Meeting #4)
  6. exp6_biophysical_enc2  : exp2 + 15 Protein-Invariant Biophysical Features (Nihat Encoder 2: Grantham, BLOSUM62, dCharge, dHydro, dVol, rel_pos...)
  7. exp7_grand_combo       : ESIM + RelPos + Regularization + Reverse Loss + Mild Sqrt Weighting + 15 Biophysical Features

Outputs saved neatly under decoder/tournament_results/phase2_3/ (gitignored).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

DECODER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DECODER_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_ROOT = DECODER_DIR / "tournament_results" / "phase2_3"

sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(DECODER_DIR / "new_decoder" / "contvar_decoder_mf_window7"))

from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402
from run_phase1_tournament import (  # noqa: E402
    LABELS,
    SOURCE_DIRS,
    PreloadedFeatureStore,
    compute_metrics,
    format_seconds,
    is_true_series,
    load_prepared_split,
    seed_everything,
    valid_key_mask,
)

# ---------------------------------------------------------------------------
# Biophysical lookup tables (Grantham 1974, BLOSUM62, Kyte-Doolittle, Charge, Volume, Polarity)
# ---------------------------------------------------------------------------
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Kyte-Doolittle hydropathy
HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Side-chain charge at physiological pH 7.4
CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0, "Q": 0.0, "E": -1.0,
    "G": 0.0, "H": 0.1, "I": 0.0, "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0,
    "P": 0.0, "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}
# Residue volume (Zamyatnin, A^3)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
    "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
    "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
# Grantham composition (c), polarity (p), molecular volume (v) -> exact Grantham (1974) distance!
GRANTHAM_C = {
    "A": 0.00, "R": 0.65, "N": 1.33, "D": 1.38, "C": 2.75, "Q": 0.89, "E": 0.92,
    "G": 0.74, "H": 0.58, "I": 0.00, "L": 0.00, "K": 0.33, "M": 0.00, "F": 0.00,
    "P": 0.39, "S": 1.42, "T": 0.71, "W": 0.13, "Y": 0.20, "V": 0.00,
}
GRANTHAM_P = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3,
    "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2,
    "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
GRANTHAM_V = {
    "A": 31, "R": 124, "N": 56, "D": 54, "C": 55, "Q": 85, "E": 83,
    "G": 3, "H": 96, "I": 111, "L": 111, "K": 119, "M": 105, "F": 132,
    "P": 32.5, "S": 32, "T": 61, "W": 170, "Y": 136, "V": 84,
}

# Standard BLOSUM62 matrix in AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]

SUB_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


def grantham_distance(a: str, b: str) -> float:
    if a not in GRANTHAM_C or b not in GRANTHAM_C:
        return 80.0
    dc = GRANTHAM_C[a] - GRANTHAM_C[b]
    dp = GRANTHAM_P[a] - GRANTHAM_P[b]
    dv = GRANTHAM_V[a] - GRANTHAM_V[b]
    return float(math.sqrt(1.833 * dc * dc + 0.1018 * dp * dp + 0.000399 * dv * dv) * 50.723 / 50.723)
    # Exact Grantham (1974) formula: D = 50.723 * sqrt(1.833*dc^2 + 0.1018*dp^2 + 0.000399*dv^2)


def exact_grantham(a: str, b: str) -> float:
    if a not in GRANTHAM_C or b not in GRANTHAM_C:
        return 80.0
    dc = GRANTHAM_C[a] - GRANTHAM_C[b]
    dp = GRANTHAM_P[a] - GRANTHAM_P[b]
    dv = GRANTHAM_V[a] - GRANTHAM_V[b]
    return float(50.723 * math.sqrt(1.833 * dc * dc + 0.1018 * dp * dp + 0.000399 * dv * dv))


def blosum62_score(a: str, b: str) -> float:
    if a in AA_TO_IDX and b in AA_TO_IDX:
        return float(_BLOSUM62_ROWS[AA_TO_IDX[a]][AA_TO_IDX[b]])
    return 0.0


def extract_biophysical_features(df: pd.DataFrame, seq_lengths: dict[str, int]) -> np.ndarray:
    """Compute 15 protein-invariant biophysical features for single & combinatorial variants."""
    n = len(df)
    feats = np.zeros((n, 15), dtype=np.float32)

    mut_col = "mutations" if "mutations" in df.columns else "snp"
    pids = df["protein_id"].astype(str).tolist()
    muts = df[mut_col].astype(str).tolist()
    snps = df["snp"].astype(str).tolist()
    is_ala_col = is_true_series(df["is_alanine"]).to_numpy(dtype=np.float32) if "is_alanine" in df.columns else None

    for i in range(n):
        parsed = SUB_RE.findall(muts[i])
        if not parsed:
            parsed = SUB_RE.findall(snps[i])
        if not parsed:
            continue

        seq_len = float(max(1, seq_lengths.get(pids[i], 500)))
        g_list, b_list = [], []
        dc_list, dh_list, dv_list, dp_list = [], [], [], []
        pg_flag, aro_flag, cys_flag = 0.0, 0.0, 0.0
        rel_pos_list = []

        for wt, pos_s, mut in parsed:
            g_list.append(exact_grantham(wt, mut) / 215.0)
            b_list.append(blosum62_score(wt, mut) / 10.0)
            dc_list.append(CHARGE.get(mut, 0.0) - CHARGE.get(wt, 0.0))
            dh_list.append((HYDRO.get(mut, 0.0) - HYDRO.get(wt, 0.0)) / 9.0)
            dv_list.append((VOLUME.get(mut, 130.0) - VOLUME.get(wt, 130.0)) / 170.0)
            dp_list.append((GRANTHAM_P.get(mut, 8.0) - GRANTHAM_P.get(wt, 8.0)) / 8.0)
            if wt in ("P", "G") or mut in ("P", "G"):
                pg_flag = 1.0
            if (wt in ("F", "W", "Y")) != (mut in ("F", "W", "Y")):
                aro_flag = 1.0
            if wt == "C" or mut == "C":
                cys_flag = 1.0
            rel_pos_list.append(min(1.0, float(int(pos_s)) / seq_len))

        feats[i, 0] = float(np.mean(g_list))
        feats[i, 1] = float(np.mean(b_list))
        feats[i, 2] = float(np.mean(dc_list))
        feats[i, 3] = float(np.mean(np.abs(dc_list)))
        feats[i, 4] = float(np.mean(dh_list))
        feats[i, 5] = float(np.mean(np.abs(dh_list)))
        feats[i, 6] = float(np.mean(dv_list))
        feats[i, 7] = float(np.mean(np.abs(dv_list)))
        feats[i, 8] = float(np.mean(dp_list))
        feats[i, 9] = pg_flag
        feats[i, 10] = aro_flag
        feats[i, 11] = cys_flag
        feats[i, 12] = float(np.mean(rel_pos_list))
        feats[i, 13] = float(is_ala_col[i]) if is_ala_col is not None else (1.0 if all(m == "A" for _, _, m in parsed) else 0.0)
        feats[i, 14] = float(math.log1p(len(parsed)))

    return feats


# ---------------------------------------------------------------------------
# Enhanced PairedGODecoder (RelPos ESIM, Regularized Adapters/Attention, ScoreDiff, Aux Biophysical)
# ---------------------------------------------------------------------------
def masked_mean(window: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clean = torch.where(mask.unsqueeze(-1), window, torch.zeros_like(window))
    return clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(window.dtype)


class RegularizedGOReader(nn.Module):
    def __init__(self, width: int, heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.context = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=attn_dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.attention = nn.MultiheadAttention(width, heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(2 * width, width),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = torch.where(valid.unsqueeze(-1), memory, torch.zeros_like(memory))
        memory = self.context(memory, src_key_padding_mask=~valid)
        q = query.unsqueeze(1)
        read, weights = self.attention(q, memory, memory, key_padding_mask=~valid)
        h = self.norm1(q + read)
        return self.norm2(h + self.ffn(h)).squeeze(1), weights.squeeze(1)


class EnhancedPairedGODecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        go_dim: int = 200,
        width: int = 128,
        heads: int = 4,
        pair_mode: str = "joint",
        window_mode: str = "esim",
        use_rel_pos: bool = True,
        adapter_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        aux_dim: int = 0,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        self.spec = dict(
            input_dim=input_dim, go_dim=go_dim, width=width, heads=heads,
            pair_mode=pair_mode, window_mode=window_mode, use_rel_pos=use_rel_pos,
            adapter_dropout=adapter_dropout, attn_dropout=attn_dropout, dropout=dropout,
            aux_dim=aux_dim, window_size=window_size,
        )
        self.pair_mode = pair_mode
        self.window_mode = window_mode
        self.use_rel_pos = use_rel_pos
        self.aux_dim = aux_dim

        def make_adapter(d_in: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.LayerNorm(d_in)]
            if adapter_dropout > 0:
                layers.append(nn.Dropout(adapter_dropout))
            layers.extend([nn.Linear(d_in, width), nn.GELU()])
            if adapter_dropout > 0:
                layers.append(nn.Dropout(adapter_dropout * 0.5))
            return nn.Sequential(*layers)

        self.scales = ("mut", "global") + (("window",) if pair_mode == "score_difference" else ())
        self.adapters = nn.ModuleDict({s: make_adapter(input_dim) for s in self.scales})
        self.types = nn.Parameter(torch.empty(len(self.scales), width))
        nn.init.normal_(self.types, std=0.02)

        self.go_adapter = nn.Sequential(nn.LayerNorm(go_dim), nn.Linear(go_dim, width), nn.GELU())
        self.shared_reader = RegularizedGOReader(width, heads, attn_dropout=attn_dropout)

        if pair_mode == "joint":
            self.window_adapter = make_adapter(input_dim)
            self.comparison = nn.Sequential(
                nn.Linear(4 * width, width),
                nn.GELU(),
                nn.Dropout(attn_dropout),
                nn.Linear(width, width),
                nn.LayerNorm(width),
            )
            if use_rel_pos and window_mode == "esim":
                self.rel_pos = nn.Parameter(torch.empty(window_size, width))
                nn.init.normal_(self.rel_pos, std=0.02)
            else:
                self.rel_pos = None
            self.local_reader = RegularizedGOReader(width, heads, attn_dropout=attn_dropout)

            fusion_in = 4 * width
            if aux_dim > 0:
                aux_width = width // 2
                self.aux_adapter = nn.Sequential(
                    nn.LayerNorm(aux_dim),
                    nn.Linear(aux_dim, aux_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                fusion_in += aux_width
            else:
                self.aux_adapter = None

            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.class_head = nn.Linear(width, 3)
        else:
            self.side_scorer = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
            self.neutral_bias = nn.Parameter(torch.tensor(1.0))
            self.raw_neutral_slope = nn.Parameter(torch.tensor(0.0))

    def compare_window(
        self,
        wt_window: torch.Tensor,
        var_window: torch.Tensor,
        wt_mask: torch.Tensor,
        var_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        valid = wt_mask & var_mask
        w = torch.where(wt_mask.unsqueeze(-1), wt_window, torch.zeros_like(wt_window))
        v = torch.where(var_mask.unsqueeze(-1), var_window, torch.zeros_like(var_window))
        if self.window_mode == "pair_mean":
            w = self.window_adapter(masked_mean(w, wt_mask)).unsqueeze(1)
            v = self.window_adapter(masked_mean(v, var_mask)).unsqueeze(1)
            valid = torch.ones(len(query), 1, dtype=torch.bool, device=query.device)
            tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
        else:
            w = self.window_adapter(w)
            v = self.window_adapter(v)
            tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
            if self.rel_pos is not None:
                tokens = tokens + self.rel_pos.unsqueeze(0)
        pair, _ = self.local_reader(query, tokens, valid)
        return pair

    def forward(
        self,
        wt: dict[str, torch.Tensor],
        var: dict[str, torch.Tensor],
        go: torch.Tensor,
        wt_window: torch.Tensor,
        var_window: torch.Tensor,
        wt_mask: torch.Tensor,
        var_mask: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = len(go)
        w_tokens, v_tokens = [], []
        for i, scale in enumerate(self.scales):
            if scale == "window":
                w_feat = masked_mean(wt_window, wt_mask)
                v_feat = masked_mean(var_window, var_mask)
            else:
                w_feat, v_feat = wt[scale], var[scale]
            w_tokens.append(self.adapters[scale](w_feat) + self.types[i])
            v_tokens.append(self.adapters[scale](v_feat) + self.types[i])

        query = self.go_adapter(go)
        available = torch.ones(batch, len(self.scales), dtype=torch.bool, device=go.device)
        hw, _ = self.shared_reader(query, torch.stack(w_tokens, 1), available)
        hv, _ = self.shared_reader(query, torch.stack(v_tokens, 1), available)

        if self.pair_mode == "joint":
            pair = self.compare_window(wt_window, var_window, wt_mask, var_mask, query)
            parts = [hw, hv, pair, query]
            if self.aux_adapter is not None and aux is not None:
                parts.append(self.aux_adapter(aux))
            h = self.fusion(torch.cat(parts, dim=-1))
            logits = self.class_head(h)
        else:
            sw = self.side_scorer(torch.cat([hw, query], -1))
            sv = self.side_scorer(torch.cat([hv, query], -1))
            d = (sv - sw).squeeze(1)
            neutral = self.neutral_bias - F.softplus(self.raw_neutral_slope) * d.abs()
            logits = torch.stack([-d, d, neutral], 1)

        scores = logits.softmax(-1)
        prediction = scores.argmax(1)
        return {"logits": logits, "scores": scores, "prediction": prediction}


class FastAuxDataset:
    def __init__(self, frame: pd.DataFrame, store: PreloadedFeatureStore, aux_feats: np.ndarray) -> None:
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
        self.aux = torch.tensor(aux_feats, dtype=torch.float32, device=dev)

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
            "aux": self.aux[idx],
            "label": self.labels[idx],
        }


def iter_batches(ds: FastAuxDataset, batch_size: int, shuffle: bool, generator: torch.Generator | None = None):
    n = len(ds)
    dev = ds.store.device
    order = torch.randperm(n, generator=generator).to(dev) if shuffle else torch.arange(n, device=dev)
    for start in range(0, n, batch_size):
        yield ds.get_batch(order[start : start + batch_size])


def forward_model(
    model: EnhancedPairedGODecoder,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    swap_wt_var: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    wt = {"global": batch["wt_g"].to(device), "mut": batch["wt_l"].to(device)}
    var = {"global": batch["var_g"].to(device), "mut": batch["var_l"].to(device)}
    wt_w = batch["wt_w"].to(device, dtype=torch.float32)
    var_w = batch["var_w"].to(device, dtype=torch.float32)
    wt_m = batch["wt_m"].to(device).clone()
    var_m = batch["var_m"].to(device).clone()
    wt_m[:, 3] = True
    var_m[:, 3] = True
    go = batch["go"].to(device)
    aux = batch["aux"].to(device)
    labels = batch["label"].to(device)

    if swap_wt_var:
        # Antisymmetry swap: WT <-> VAR, and negate directional biophysical deltas (indices 2, 4, 6, 8)
        aux_swapped = aux.clone()
        aux_swapped[:, [2, 4, 6, 8]] = -aux_swapped[:, [2, 4, 6, 8]]
        out = model(var, wt, go, var_w, wt_w, var_m, wt_m, aux=aux_swapped)
    else:
        out = model(wt, var, go, wt_w, var_w, wt_m, var_m, aux=aux)
    return out, labels


def reverse_consistency_loss(logits_fwd: torch.Tensor, logits_rev: torch.Tensor) -> torch.Tensor:
    """Symmetric KL enforcing p(VAR, WT) == swap_{LOF<->GOF}(p(WT, VAR))."""
    # Class order: 0=LOF, 1=GOF, 2=NEUTRAL -> swapped order: [1, 0, 2]
    swap_idx = torch.tensor([1, 0, 2], device=logits_fwd.device)
    logits_fwd_swapped = logits_fwd[:, swap_idx]
    p_fwd_swap = F.softmax(logits_fwd_swapped.detach(), dim=-1)
    log_p_rev = F.log_softmax(logits_rev, dim=-1)
    p_rev_swap = F.softmax(logits_rev[:, swap_idx].detach(), dim=-1)
    log_p_fwd = F.log_softmax(logits_fwd, dim=-1)
    kl_1 = F.kl_div(log_p_rev, p_fwd_swap, reduction="batchmean")
    kl_2 = F.kl_div(log_p_fwd, p_rev_swap, reduction="batchmean")
    return 0.5 * (kl_1 + kl_2)


def tune_val_logit_bias(val_logits: np.ndarray, val_labels: np.ndarray) -> np.ndarray:
    """Post-hoc class-imbalance tuning on validation logits (zero test leakage)."""
    best_bias = np.zeros(3, dtype=np.float64)
    best_f1 = compute_metrics(val_labels, val_logits.argmax(1), F.softmax(torch.tensor(val_logits), dim=-1).numpy())["macro_f1"]
    # Coarse grid search over GOF and NEUTRAL logit offsets relative to LOF=0
    for b_gof in np.linspace(-0.5, 1.5, 21):
        for b_neu in np.linspace(-0.5, 0.8, 14):
            bias = np.array([0.0, b_gof, b_neu], dtype=np.float64)
            preds = (val_logits + bias).argmax(axis=1)
            # Fast macro-F1 check
            f1s = []
            for c in range(3):
                tp = np.sum((preds == c) & (val_labels == c))
                fp = np.sum((preds == c) & (val_labels != c))
                fn = np.sum((preds != c) & (val_labels == c))
                denom = 2 * tp + fp + fn
                f1s.append((2 * tp / denom) if denom > 0 else 0.0)
            mf1 = float(np.mean(f1s))
            if mf1 > best_f1 + 1e-6:
                best_f1 = mf1
                best_bias = bias
    return best_bias


@torch.no_grad()
def evaluate_dataset(
    model: EnhancedPairedGODecoder,
    ds: FastAuxDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    all_labels, all_preds, all_probs, all_logits = [], [], [], []
    total_loss, total_n = 0.0, 0
    for batch in iter_batches(ds, batch_size=batch_size, shuffle=False):
        out, labels = forward_model(model, batch, device)
        loss = F.cross_entropy(out["logits"], labels, reduction="sum")
        total_loss += float(loss.item())
        total_n += len(labels)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(out["prediction"].cpu().numpy())
        all_probs.append(out["scores"].cpu().numpy())
        all_logits.append(out["logits"].cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_logits),
        total_loss / max(1, total_n),
    )


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "exp1_esim_relpos",
        "desc": "ESIM + Learned 7-pos Relative Positional Encoding (Meeting #5)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.0, "attn_dropout": 0.0, "dropout": 0.1,
        "weight_decay": 1e-4, "label_smoothing": 0.0, "cosine_lr": False,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": False,
    },
    {
        "name": "exp2_reg_calib",
        "desc": "ESIM + RelPos + Overfitting/Calibration Fix (Drop 0.3/0.2, WD 1e-2, LS 0.1, Cosine)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": False,
    },
    {
        "name": "exp3_reverse_loss",
        "desc": "exp2 + Nihat Antisymmetry / Reverse-Consistency Loss (lambda=0.2)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.2, "class_weight": "none", "use_aux": False,
    },
    {
        "name": "exp4_imbalance_sqrt",
        "desc": "exp2 + Mild Sqrt Class Weighting (1/sqrt(N_c) for 7.5% GOF)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "sqrt", "use_aux": False,
    },
    {
        "name": "exp5_score_diff",
        "desc": "Score-Difference Architecture + Regularization (Meeting #4, Built-in Antisymmetry)",
        "pair_mode": "score_difference", "window_mode": "pair_mean", "use_rel_pos": False,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": False,
    },
    {
        "name": "exp6_biophysical_enc2",
        "desc": "exp2 + 15 Biophysical Features (Nihat Encoder 2: Grantham, BLOSUM62, dCharge, dHydro...)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "exp7_grand_combo",
        "desc": "Grand Combo: RelPos ESIM + Reg/LS + Reverse Loss + Sqrt Weighting + 15 Biophysical",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": True,
        "adapter_dropout": 0.3, "attn_dropout": 0.2, "dropout": 0.3,
        "weight_decay": 1e-2, "label_smoothing": 0.1, "cosine_lr": True,
        "lambda_reverse": 0.2, "class_weight": "sqrt", "use_aux": True,
    },
]


def run_experiment(
    *,
    run_idx: int,
    total_runs: int,
    exp: dict[str, Any],
    family: str,
    source: str,
    seed: int,
    train_ds: FastAuxDataset,
    val_ds: FastAuxDataset,
    test_ds: FastAuxDataset,
    common_test_ids: set[str],
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    start_time: float,
) -> dict[str, Any]:
    run_name = f"{family}__{source}__{exp['name']}__seed{seed}"
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

    # Class weights (computed strictly from train labels)
    class_weights = None
    if exp["class_weight"] == "sqrt":
        counts = np.bincount(train_ds.frame["label_idx"].to_numpy(dtype=np.int64), minlength=3).astype(np.float64)
        w = np.sqrt(counts.sum() / np.maximum(1.0, counts))
        w = w / w.mean()
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    model = EnhancedPairedGODecoder(
        input_dim=1280,
        go_dim=train_ds.store.go.dimension,
        width=args.width,
        heads=args.heads,
        pair_mode=exp["pair_mode"],
        window_mode=exp["window_mode"],
        use_rel_pos=exp["use_rel_pos"],
        adapter_dropout=exp["adapter_dropout"],
        attn_dropout=exp["attn_dropout"],
        dropout=exp["dropout"],
        aux_dim=15 if exp["use_aux"] else 0,
        window_size=7,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=exp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) if exp["cosine_lr"] else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    run_t0 = time.perf_counter()

    print(f"\n[{run_idx:02d}/{total_runs:02d}] START {run_name} ({exp['desc']})", flush=True)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for batch in iter_batches(train_ds, batch_size=args.batch_size, shuffle=True, generator=g):
            optimizer.zero_grad(set_to_none=True)
            out_fwd, labels = forward_model(model, batch, device, swap_wt_var=False)
            loss = F.cross_entropy(
                out_fwd["logits"], labels,
                weight=class_weights,
                label_smoothing=exp["label_smoothing"],
            )
            if exp["lambda_reverse"] > 0.0:
                out_rev, _ = forward_model(model, batch, device, swap_wt_var=True)
                loss = loss + exp["lambda_reverse"] * reverse_consistency_loss(out_fwd["logits"], out_rev["logits"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss_sum += float(loss.item()) * len(labels)
            tr_n += len(labels)

        if scheduler is not None:
            scheduler.step()

        tr_loss = tr_loss_sum / max(1, tr_n)
        v_labels, v_preds, v_probs, _, v_loss = evaluate_dataset(model, val_ds, eval_bs, device)
        v_metrics = compute_metrics(v_labels, v_preds, v_probs)
        val_f1 = v_metrics["macro_f1"]
        val_conf = v_metrics["confidence"]["mean_confidence"]
        val_ece = v_metrics["confidence"]["ece_10bin"]
        ep_dt = time.perf_counter() - ep_t0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = "*"
        else:
            no_improve += 1
            star = " "

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss,
            "val_macro_f1": val_f1, "val_gof_f1": v_metrics["gof"]["f1"],
            "val_conf": val_conf, "val_ece": val_ece, "seconds": ep_dt,
        })
        print(
            f"  Ep {epoch:02d}/{args.epochs:02d} {star} | tr_loss={tr_loss:.4f} | val_loss={v_loss:.4f} "
            f"| val_F1={val_f1:.4f} (best={best_val_f1:.4f}@{best_epoch}) "
            f"| GOF={v_metrics['gof']['f1']:.4f} | conf={val_conf:.3f} | ECE={val_ece:.3f} | {ep_dt:.1f}s",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    assert best_state is not None
    torch.save({"exp": exp, "model_spec": model.spec, "state_dict": best_state, "best_epoch": best_epoch}, run_dir / "best_model.pt")
    model.load_state_dict(best_state)

    v_labels, v_preds, v_probs, v_logits, v_loss = evaluate_dataset(model, val_ds, eval_bs, device)
    t_labels, t_preds, t_probs, t_logits, t_loss = evaluate_dataset(model, test_ds, eval_bs, device)

    # Post-hoc validation-tuned logit bias (for imbalance)
    val_bias = tune_val_logit_bias(v_logits, v_labels)
    t_logits_adj = t_logits + val_bias
    t_probs_adj = F.softmax(torch.tensor(t_logits_adj), dim=-1).numpy()
    t_preds_adj = t_logits_adj.argmax(axis=1)

    test_frame = test_ds.frame.copy()
    is_common = test_frame["sample_id"].isin(common_test_ids).to_numpy(dtype=bool)
    is_comb = is_true_series(test_frame["is_combinatorial"]).to_numpy(dtype=bool) if "is_combinatorial" in test_frame.columns else np.zeros(len(test_frame), dtype=bool)
    is_single = ~is_comb

    val_m = compute_metrics(v_labels, v_preds, v_probs)
    full_m = compute_metrics(t_labels, t_preds, t_probs)
    common_m = compute_metrics(t_labels[is_common], t_preds[is_common], t_probs[is_common])
    single_m = compute_metrics(t_labels[is_single], t_preds[is_single], t_probs[is_single])
    comb_m = compute_metrics(t_labels[is_comb], t_preds[is_comb], t_probs[is_comb]) if is_comb.any() else {"n": 0}
    full_adj_m = compute_metrics(t_labels, t_preds_adj, t_probs_adj)
    common_adj_m = compute_metrics(t_labels[is_common], t_preds_adj[is_common], t_probs_adj[is_common])

    run_sec = time.perf_counter() - run_t0
    payload = {
        "run_name": run_name,
        "exp_name": exp["name"],
        "exp_desc": exp["desc"],
        "family": family,
        "source": source,
        "seed": seed,
        "best_epoch": best_epoch,
        "duration_seconds": run_sec,
        "val_tuned_logit_bias": val_bias.tolist(),
        "val": val_m,
        "full_test": full_m,
        "common_test": common_m,
        "single_test": single_m,
        "combinatorial_test": comb_m,
        "full_test_logit_adj": full_adj_m,
        "common_test_logit_adj": common_adj_m,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    eta = (elapsed / run_idx) * (total_runs - run_idx)
    print(
        f"  DONE {run_name} ({run_sec:.1f}s) | Full F1={full_m['macro_f1']:.4f} (GOF={full_m['gof']['f1']:.4f}, ECE={full_m['confidence']['ece_10bin']:.4f}) "
        f"| Common F1={common_m['macro_f1']:.4f} | +LogitAdj={full_adj_m['macro_f1']:.4f} "
        f"| Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}",
        flush=True,
    )
    return payload


def build_phase2_3_report(all_results: list[dict[str, Any]], out_dir: Path, source: str, family: str) -> None:
    rows = []
    for r in all_results:
        ft, ct, fa = r["full_test"], r["common_test"], r["full_test_logit_adj"]
        rows.append({
            "exp_name": r["exp_name"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "val_macro_f1": r["val"]["macro_f1"],
            "full_macro_f1": ft["macro_f1"],
            "full_lof_f1": ft["lof"]["f1"],
            "full_gof_f1": ft["gof"]["f1"],
            "full_neu_f1": ft["neutral"]["f1"],
            "full_mcc": ft["mcc"],
            "full_conf": ft["confidence"]["mean_confidence"],
            "full_conf_gap": ft["confidence"]["confidence_gap"],
            "full_ece": ft["confidence"]["ece_10bin"],
            "common_macro_f1": ct["macro_f1"],
            "common_gof_f1": ct["gof"]["f1"],
            "full_adj_macro_f1": fa["macro_f1"],
            "full_adj_gof_f1": fa["gof"]["f1"],
            "comb_macro_f1": r["combinatorial_test"].get("macro_f1", float("nan")),
        })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_dir / "summary_all_runs.csv", index=False)

    mcols = [c for c in df_all.columns if c not in ("exp_name", "seed", "best_epoch")]
    agg_rows = []
    for ename, grp in df_all.groupby("exp_name", sort=False):
        rec: dict[str, Any] = {"exp_name": ename, "n_seeds": len(grp)}
        for c in mcols:
            rec[f"{c}_mean"] = float(grp[c].mean())
            rec[f"{c}_std"] = float(grp[c].std(ddof=1)) if len(grp) > 1 else 0.0
        agg_rows.append(rec)
    df_agg = pd.DataFrame(agg_rows).sort_values("full_macro_f1_mean", ascending=False)
    df_agg.to_csv(out_dir / "summary_3seed_mean_std.csv", index=False)

    def fmt(m: float, s: float) -> str:
        return f"{m:.4f} ± {s:.4f}"

    lines = [
        f"# ContVAR Phase 2 & 3 Controlled Experiments Report ({family.upper()} | {source})",
        "",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Phase 1 Champion Reference (`trained_distance` + `esim`):** Full Hard F1 = `0.5826 ± 0.0145`, Common Hard F1 = `0.5844 ± 0.0112`, ECE = `0.1724`",
        "",
        "## 3-Seed Results Summary (Sorted by Full Test Macro-F1)",
        "",
        "| Experiment | Full Macro-F1 | Common Macro-F1 | LOF-F1 | GOF-F1 | NEUTRAL-F1 | +ValLogitAdj F1 | ECE ↓ | Conf Gap |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_agg.iterrows():
        lines.append(
            f"| `{r['exp_name']}` | **{fmt(r['full_macro_f1_mean'], r['full_macro_f1_std'])}** "
            f"| {fmt(r['common_macro_f1_mean'], r['common_macro_f1_std'])} "
            f"| {fmt(r['full_lof_f1_mean'], r['full_lof_f1_std'])} "
            f"| {fmt(r['full_gof_f1_mean'], r['full_gof_f1_std'])} "
            f"| {fmt(r['full_neu_f1_mean'], r['full_neu_f1_std'])} "
            f"| {fmt(r['full_adj_macro_f1_mean'], r['full_adj_macro_f1_std'])} "
            f"| **{r['full_ece_mean']:.4f}** | {r['full_conf_gap_mean']:+.3f} |"
        )
    report_path = out_dir / "PHASE2_3_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== PHASE 2 & 3 REPORT SAVED TO: {report_path} ===", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("hard", "easy"), default="hard")
    p.add_argument("--source", choices=("trained_distance", "trained_mean", "baseline"), default="trained_distance")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_runs = len(EXPERIMENTS) * len(args.seeds)

    print("=" * 78, flush=True)
    print("ContVAR PHASE 2 & 3 CONTROLLED EXPERIMENTS", flush=True)
    print(f"Device      : {device} ({gpu_info})", flush=True)
    print(f"Family      : {args.family} | Source: {args.source}", flush=True)
    print(f"Experiments : {len(EXPERIMENTS)} x {len(args.seeds)} seeds = {total_runs} runs", flush=True)
    print(f"Output Dir  : {args.out_dir}", flush=True)
    print("=" * 78, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    go_table = GOEmbeddingTable(DATA_DIR / "embeddings" / "go_terms_embeddings_anc2vec.npz")

    raw_train = load_prepared_split(args.family, "train")
    raw_val = load_prepared_split(args.family, "val")
    raw_test = load_prepared_split(args.family, "test")

    # Compute exact common test IDs with baseline for apples-to-apples comparison
    common_test_ids = set(raw_test["sample_id"])
    for src in ("baseline", "trained_mean", "trained_distance"):
        m = valid_key_mask(raw_test, SOURCE_DIRS[src])
        common_test_ids &= set(raw_test.loc[m, "sample_id"])

    # Load protein sequence lengths for relative mutation position
    print("  [Biophysical] Loading protein sequence lengths from uniprot_dict.json...", flush=True)
    with open(DATA_DIR / "sequences" / "uniprot_dict.json", "r", encoding="utf-8") as fh:
        seq_data = json.load(fh)
    if isinstance(seq_data, dict):
        seq_lengths = {k: len(v) for k, v in seq_data.items()}
    else:
        seq_lengths = {item["Entry"]: len(item["Sequence"]) for item in seq_data if "Entry" in item and "Sequence" in item}
    del seq_data

    aux_tr = extract_biophysical_features(raw_train, seq_lengths)
    aux_va = extract_biophysical_features(raw_val, seq_lengths)
    aux_te = extract_biophysical_features(raw_test, seq_lengths)
    # Z-score standardize strictly on train split (zero leakage)
    mu = aux_tr.mean(axis=0, keepdims=True)
    sd = np.where(aux_tr.std(axis=0, keepdims=True) > 1e-6, aux_tr.std(axis=0, keepdims=True), 1.0)
    aux_tr = (aux_tr - mu) / sd
    aux_va = (aux_va - mu) / sd
    aux_te = (aux_te - mu) / sd
    print(f"  [Biophysical] Extracted & standardized 15 features (train={aux_tr.shape}, val={aux_va.shape}, test={aux_te.shape})", flush=True)

    store = PreloadedFeatureStore(args.source, [raw_train, raw_val, raw_test], go_table, device)
    train_ds = FastAuxDataset(raw_train, store, aux_tr)
    val_ds = FastAuxDataset(raw_val, store, aux_va)
    test_ds = FastAuxDataset(raw_test, store, aux_te)

    all_results = []
    run_idx = 0
    start_time = time.perf_counter()

    for exp in EXPERIMENTS:
        for seed in args.seeds:
            run_idx += 1
            res = run_experiment(
                run_idx=run_idx, total_runs=total_runs, exp=exp,
                family=args.family, source=args.source, seed=seed,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                common_test_ids=common_test_ids, device=device,
                out_dir=args.out_dir, args=args, start_time=start_time,
            )
            all_results.append(res)

    build_phase2_3_report(all_results, args.out_dir, args.source, args.family)
    print(f"\nALL {total_runs} PHASE 2 & 3 RUNS COMPLETED IN {format_seconds(time.perf_counter() - start_time)}!", flush=True)


if __name__ == "__main__":
    main()
