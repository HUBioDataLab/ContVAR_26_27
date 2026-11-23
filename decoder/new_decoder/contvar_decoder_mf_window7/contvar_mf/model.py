"""GO-conditioned frozen-feature decoders with mean-7 and aligned ESIM adaptation.

The none/mean readers, fusion and pure score-difference equations retain the
handover implementation. Pair branches are additional joint-only alternatives.
No pretrained decoder weights or reverse-consistency loss are used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .util import LABELS


def masked_mean(window, mask):
    if window.ndim != 3 or mask.shape != window.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Expected window [batch, positions, features] and boolean mask")
    if not mask.any(dim=1).all():
        raise ValueError("Every window needs a valid position")
    clean = torch.where(mask.unsqueeze(-1), window, torch.zeros_like(window))
    return clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(window.dtype)


class GOReader(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.context = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=0., batch_first=True,
            norm_first=True, activation="gelu")
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, query, memory, valid):
        if valid.shape != memory.shape[:2] or valid.dtype != torch.bool or not valid.any(1).all():
            raise ValueError("Invalid GO-reader mask")
        memory = torch.where(valid.unsqueeze(-1), memory, torch.zeros_like(memory))
        memory = self.context(memory, src_key_padding_mask=~valid)
        q = query.unsqueeze(1)
        read, weights = self.attention(q, memory, memory, key_padding_mask=~valid)
        h = self.norm1(q + read)
        return self.norm2(h + self.ffn(h)).squeeze(1), weights.squeeze(1)


class PairedGODecoder(nn.Module):
    def __init__(self, input_dim=1280, go_dim=200, width=128, heads=4,
                 head="multiclass", pair_mode="joint", window_mode="none",
                 dropout=.1, window_size=7):
        super().__init__()
        if (head not in ("parallel", "multiclass") or pair_mode not in ("joint", "score_difference")
                or window_mode not in ("none", "mean", "pair_mean", "esim")
                or width < 1 or heads < 1 or width % heads or input_dim < 1 or go_dim < 1
                or not 0 <= dropout < 1 or window_size != 7):
            raise ValueError("Invalid decoder configuration")
        if window_mode in ("pair_mean", "esim") and pair_mode != "joint":
            raise ValueError("Pair-window readers are joint-only in this protocol")
        self.spec = dict(input_dim=input_dim, go_dim=go_dim, width=width, heads=heads,
                         head=head, pair_mode=pair_mode, window_mode=window_mode,
                         dropout=dropout, window_size=window_size)
        self.head_kind, self.pair_mode, self.window_mode = head, pair_mode, window_mode
        self.scales = ("mut", "global") + (("window",) if window_mode == "mean" else ())
        self.adapters = nn.ModuleDict({s: nn.Sequential(nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width), nn.GELU()) for s in self.scales})
        self.types = nn.Parameter(torch.empty(len(self.scales), width))
        nn.init.normal_(self.types, std=.02)
        self.go_adapter = nn.Sequential(nn.LayerNorm(go_dim), nn.Linear(go_dim, width), nn.GELU())
        self.shared_reader = GOReader(width, heads)
        pair_branch = window_mode in ("pair_mean", "esim")
        if pair_branch:
            self.window_adapter = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
            self.comparison = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(),
                                            nn.Linear(width, width), nn.LayerNorm(width))
            self.local_reader = GOReader(width, heads)
        if pair_mode == "joint":
            self.fusion = nn.Sequential(nn.Linear((4 if pair_branch else 3) * width, width),
                                        nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout))
            if head == "parallel":
                self.lof_head, self.gof_head = nn.Linear(width, 1), nn.Linear(width, 1)
            else:
                self.class_head = nn.Linear(width, 3)
        else:
            self.side_scorer = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width),
                nn.GELU(), nn.Linear(width, 2 if head == "parallel" else 1))
            if head == "parallel":
                self.raw_radius = nn.Parameter(torch.tensor(0.))
            else:
                self.neutral_bias = nn.Parameter(torch.tensor(1.))
                self.raw_neutral_slope = nn.Parameter(torch.tensor(0.))

    def compare_window(self, wt_window, var_window, wt_mask, var_mask, query):
        expected = (len(query), self.spec["window_size"], self.spec["input_dim"])
        if wt_window.shape != expected or var_window.shape != expected:
            raise ValueError(f"Expected raw cropped windows {expected}")
        if wt_mask.shape != expected[:2] or var_mask.shape != expected[:2]:
            raise ValueError("Invalid window mask shape")
        if wt_mask.dtype != torch.bool or var_mask.dtype != torch.bool:
            raise ValueError("Window masks must be boolean")
        valid = wt_mask & var_mask
        if not valid[:, 3].all():
            raise ValueError("The mutation center must be present")
        w = torch.where(wt_mask.unsqueeze(-1), wt_window, torch.zeros_like(wt_window))
        v = torch.where(var_mask.unsqueeze(-1), var_window, torch.zeros_like(var_window))
        if self.window_mode == "pair_mean":
            w = self.window_adapter(masked_mean(w, wt_mask)).unsqueeze(1)
            v = self.window_adapter(masked_mean(v, var_mask)).unsqueeze(1)
            valid = torch.ones(len(query), 1, dtype=torch.bool, device=query.device)
        else:
            w, v = self.window_adapter(w), self.window_adapter(v)
        tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
        # ESIM tokens go directly to GO attention: no post-comparison mean.
        pair, weights = self.local_reader(query, tokens, valid)
        return pair, weights

    def forward(self, wt, var, go, wt_window=None, var_window=None, wt_mask=None, var_mask=None):
        if go.ndim != 2 or go.shape[1] != self.spec["go_dim"]:
            raise ValueError("Invalid GO input shape")
        batch = len(go)
        w_tokens, v_tokens = [], []
        for i, scale in enumerate(self.scales):
            expected = (batch, self.spec["input_dim"])
            if wt[scale].shape != expected or var[scale].shape != expected:
                raise ValueError(f"Invalid {scale} feature shape")
            w_tokens.append(self.adapters[scale](wt[scale]) + self.types[i])
            v_tokens.append(self.adapters[scale](var[scale]) + self.types[i])
        query = self.go_adapter(go)
        available = torch.ones(batch, len(self.scales), dtype=torch.bool, device=go.device)
        hw, aw = self.shared_reader(query, torch.stack(w_tokens, 1), available)
        hv, av = self.shared_reader(query, torch.stack(v_tokens, 1), available)
        attention, side_scores = {"wt": aw, "var": av}, None
        if self.pair_mode == "joint":
            if self.window_mode in ("pair_mean", "esim"):
                pair, attention["pair"] = self.compare_window(wt_window, var_window, wt_mask, var_mask, query)
                fused = torch.cat([hw, hv, pair, query], dim=-1)
            else:
                fused = torch.cat([hw, hv, query], dim=-1)
            h = self.fusion(fused)
            logits = torch.cat([self.lof_head(h), self.gof_head(h)], -1) if self.head_kind == "parallel" else self.class_head(h)
        else:
            sw = self.side_scorer(torch.cat([hw, query], -1))
            sv = self.side_scorer(torch.cat([hv, query], -1))
            delta = sv - sw
            side_scores = {"wt": sw, "var": sv, "var_minus_wt": delta}
            if self.head_kind == "parallel":
                logits = torch.stack([-delta[:, 0], delta[:, 1]], 1) - F.softplus(self.raw_radius)
            else:
                d = delta.squeeze(1)
                neutral = self.neutral_bias - F.softplus(self.raw_neutral_slope) * d.abs()
                logits = torch.stack([-d, d, neutral], 1)
        if self.head_kind == "parallel":
            probabilities = logits.sigmoid()
            scores = torch.cat([probabilities, 1 - probabilities.max(1, keepdim=True).values], 1)
            prediction = torch.where(probabilities.max(1).values >= .5, probabilities.argmax(1), 2)
        else:
            scores = logits.softmax(-1)
            prediction = scores.argmax(1)
        return dict(logits=logits, scores=scores, prediction=prediction, attention=attention, side_scores=side_scores)


def classification_loss(output, labels, head, reduction="mean"):
    if head == "multiclass":
        return F.cross_entropy(output["logits"], labels, reduction=reduction)
    targets = torch.stack([labels == 0, labels == 1], 1).to(output["logits"].dtype)
    loss = F.binary_cross_entropy_with_logits(output["logits"], targets, reduction="none").mean(1)
    return loss.mean() if reduction == "mean" else loss


def load_decoder(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "contvar_mf_window7_v1" or checkpoint.get("labels") != list(LABELS):
        raise ValueError("Not an MF/window-7 checkpoint")
    model = PairedGODecoder(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
