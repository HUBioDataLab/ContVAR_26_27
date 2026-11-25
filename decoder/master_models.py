#!/usr/bin/env python3
"""Modular Master Decoder Architectures, Window Readers, Losses & Calibration.

Supports every experimental axis in the MF-Only Master Suite:
  B1-B6 Main Architectures:
    - joint       : Standard concat([hw, hv, pair, query]) -> Linear -> 3 classes
    - go_film     : GO-FiLM modulation: (1 + gamma_GO) * m + beta_GO
    - mag_dir     : Magnitude + Direction 2-head soft-gated (NEUTRAL vs Impact; LOF vs GOF)
    - bilinear    : Low-rank multiplicative interaction U(m) * V(go) + residual
    - res_mlp     : Deep Pre-Norm Residual SwiGLU MLP decoder (2 blocks)
    - delta_only  : Protein-invariant pure-delta decoder (no raw WT_global shortcut)

  C1-C5 Local Window Readers:
    - esim            : Aligned [w, v, v-w, w*v] -> Transformer -> GO Cross-Attention
    - bi_xattn        : True Bidirectional Cross-Attention (WT<->VAR) -> Comparison -> GO Attention
    - cnn1d           : Multi-Scale 1D CNN (k=3, 5) + Residual -> Masked Mean/Center Pooling
    - hybrid_cnn_attn : Multi-Scale 1D CNN (k=3, 5) + Residual -> GO Cross-Attention
    - persite_mil     : Combinatorial Per-Site MIL Attention (GO-queried attention over m sites)

  D Auxiliary Fusion (15 Biophysical + 3 ESM-2 650M LLR = 18 features):
    - none      : No auxiliary features
    - concat    : Concatenate projected auxiliary vector
    - aux_film  : Aux-FiLM modulation: (1 + gamma_aux) * m + beta_aux
    - dual_film : Both GO-FiLM and Aux-FiLM modulation!

  Input Modality Ablations (PDF Page 9 Priorities 1-4):
    - full      : WT + VAR + GO
    - go_only   : GO only
    - wt_go     : WT + GO (no mutant)
    - var_go    : VAR + GO (no WT)
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clean = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
    return clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)


class GOReader(nn.Module):
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
            nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(attn_dropout), nn.Linear(2 * width, width)
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        memory = torch.where(valid.unsqueeze(-1), memory, torch.zeros_like(memory))
        memory = self.context(memory, src_key_padding_mask=~valid)
        q = query.unsqueeze(1)
        read, _ = self.attention(q, memory, memory, key_padding_mask=~valid)
        h = self.norm1(q + read)
        return self.norm2(h + self.ffn(h)).squeeze(1)


class MultiScaleConvBlock(nn.Module):
    """Parallel Conv1d(k=3) + Conv1d(k=5) with residual and LayerNorm."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        half = width // 2
        self.conv3 = nn.Conv1d(width, half, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(width, width - half, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        clean = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        xt = clean.transpose(1, 2)
        out = self.drop(self.act(torch.cat([self.conv3(xt), self.conv5(xt)], dim=1))).transpose(1, 2)
        out = self.norm(clean + out)
        return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


class BidirectionalCrossAttnWindow(nn.Module):
    """True ESIM-style bidirectional cross-attention between WT[7] and VAR[7]."""

    def __init__(self, width: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w2v = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.v2w = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_w = nn.LayerNorm(width)
        self.norm_v = nn.LayerNorm(width)
        self.compare = nn.Sequential(
            nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.LayerNorm(width),
        )
        self.go_reader = GOReader(width, heads, attn_dropout=dropout)

    def forward(
        self,
        w: torch.Tensor, v: torch.Tensor,
        wt_mask: torch.Tensor, var_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        valid = wt_mask & var_mask
        w_clean = torch.where(wt_mask.unsqueeze(-1), w, torch.zeros_like(w))
        v_clean = torch.where(var_mask.unsqueeze(-1), v, torch.zeros_like(v))
        # WT attends to VAR and VAR attends to WT
        w_ctx, _ = self.w2v(w_clean, v_clean, v_clean, key_padding_mask=~var_mask)
        v_ctx, _ = self.v2w(v_clean, w_clean, w_clean, key_padding_mask=~wt_mask)
        w_bar = self.norm_w(w_clean + w_ctx)
        v_bar = self.norm_v(v_clean + v_ctx)
        tokens = self.compare(torch.cat([w_bar, v_bar, v_bar - w_bar, w_bar * v_bar], dim=-1))
        return self.go_reader(query, tokens, valid)


class SwiGLUResBlock(nn.Module):
    """Pre-Norm Residual SwiGLU block for deep MLP decoder."""

    def __init__(self, width: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.w12 = nn.Linear(width, 4 * width)
        self.w3 = nn.Linear(2 * width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        u, v = self.w12(h).chunk(2, dim=-1)
        h = self.drop(F.silu(u) * v)
        return x + self.drop(self.w3(h))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: (1 + gamma(cond)) * x + beta(cond)."""

    def __init__(self, cond_dim: int, feat_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gen = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, 2 * feat_dim),
        )
        # Zero-init so FiLM starts near identity (1 + 0)*x + 0
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)
        self.norm = nn.LayerNorm(feat_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gen(cond).chunk(2, dim=-1)
        out = (1.0 + gamma) * x + beta
        return self.drop(self.norm(out))


class MasterGODecoder(nn.Module):
    """Unified Master Decoder supporting B1-B6, C1-C5, D(none/concat/aux_film/dual_film), and input ablations."""

    def __init__(
        self,
        input_dim: int = 1280,
        go_dim: int = 200,
        width: int = 128,
        heads: int = 4,
        arch: str = "joint",           # B1: joint, B2: go_film, B3: mag_dir, B4: bilinear, B5: res_mlp, B6: delta_only
        window_reader: str = "esim",   # C1: esim, C2: bi_xattn, C3: cnn1d, C4: hybrid_cnn_attn, C5: persite_mil
        aux_mode: str = "none",        # none, concat, aux_film, dual_film
        aux_dim: int = 0,
        input_mode: str = "full",      # full, go_only, wt_go, var_go
        adapter_dropout: float = 0.15,
        attn_dropout: float = 0.10,
        dropout: float = 0.15,
        embed_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = dict(
            input_dim=input_dim, go_dim=go_dim, width=width, heads=heads,
            arch=arch, window_reader=window_reader, aux_mode=aux_mode, aux_dim=aux_dim,
            input_mode=input_mode, adapter_dropout=adapter_dropout, attn_dropout=attn_dropout,
            dropout=dropout, embed_noise=embed_noise,
        )
        self.arch = arch
        self.window_reader = window_reader
        self.aux_mode = aux_mode if aux_dim > 0 else "none"
        self.aux_dim = aux_dim
        self.input_mode = input_mode
        self.embed_noise = embed_noise

        def make_adapter(d_in: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.LayerNorm(d_in)]
            if adapter_dropout > 0:
                layers.append(nn.Dropout(adapter_dropout))
            layers.extend([nn.Linear(d_in, width), nn.GELU()])
            if adapter_dropout > 0:
                layers.append(nn.Dropout(adapter_dropout * 0.5))
            return nn.Sequential(*layers)

        self.go_adapter = nn.Sequential(nn.LayerNorm(go_dim), nn.Linear(go_dim, width), nn.GELU())

        # Input-only ablation: GO-only skips protein encoders completely
        if input_mode == "go_only":
            self.go_head = nn.Sequential(
                nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 3)
            )
            return

        self.adapters = nn.ModuleDict({"mut": make_adapter(input_dim), "global": make_adapter(input_dim)})
        self.types = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.types, std=0.02)
        self.shared_reader = GOReader(width, heads, attn_dropout=attn_dropout)

        # Window branch (active when both WT and VAR are present)
        self.window_adapter = make_adapter(input_dim)
        if window_reader == "esim":
            self.comparison = nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(attn_dropout),
                nn.Linear(width, width), nn.LayerNorm(width),
            )
            self.local_reader = GOReader(width, heads, attn_dropout=attn_dropout)
        elif window_reader == "bi_xattn":
            self.bi_xattn = BidirectionalCrossAttnWindow(width, heads, dropout=attn_dropout)
        elif window_reader == "cnn1d":
            self.comparison = nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(attn_dropout),
                nn.Linear(width, width), nn.LayerNorm(width),
            )
            self.conv_block = MultiScaleConvBlock(width, dropout=attn_dropout)
            self.cnn_proj = nn.Sequential(nn.Linear(2 * width, width), nn.LayerNorm(width), nn.GELU())
        elif window_reader == "hybrid_cnn_attn":
            self.comparison = nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(attn_dropout),
                nn.Linear(width, width), nn.LayerNorm(width),
            )
            self.conv_block = MultiScaleConvBlock(width, dropout=attn_dropout)
            self.hyb_attn = nn.MultiheadAttention(width, heads, dropout=attn_dropout, batch_first=True)
            self.hyb_norm1 = nn.LayerNorm(width)
            self.hyb_norm2 = nn.LayerNorm(width)
            self.hyb_ffn = nn.Sequential(
                nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(attn_dropout), nn.Linear(2 * width, width)
            )
        elif window_reader == "persite_mil":
            # Combinatorial per-site MIL attention over up to M sites (default single site M=1)
            self.comparison = nn.Sequential(
                nn.Linear(4 * width, width), nn.GELU(), nn.Dropout(attn_dropout),
                nn.Linear(width, width), nn.LayerNorm(width),
            )
            self.site_reader = GOReader(width, heads, attn_dropout=attn_dropout)
            self.mil_attn = nn.MultiheadAttention(width, heads, dropout=attn_dropout, batch_first=True)
            self.mil_norm = nn.LayerNorm(width)
        else:
            raise ValueError(f"Unknown window_reader: {window_reader}")

        # Build mutation summary projection m: [B, width]
        if input_mode in ("wt_go", "var_go"):
            mut_in = width
        elif arch == "delta_only":
            # Pure protein-invariant delta: [hv - hw, |hv - hw|, hw * hv, pair] -> no raw WT/VAR identity
            mut_in = 4 * width
        else:
            # Standard rich representation: [hw, hv, hv - hw, pair]
            mut_in = 4 * width

        self.mut_proj = nn.Sequential(
            nn.Linear(mut_in, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
        )

        # Auxiliary encoder
        aux_proj_dim = width // 2
        if self.aux_mode != "none":
            self.aux_encoder = nn.Sequential(
                nn.LayerNorm(aux_dim), nn.Linear(aux_dim, aux_proj_dim), nn.GELU(), nn.Dropout(dropout)
            )
        else:
            self.aux_encoder = None

        # Auxiliary FiLM (if aux_mode in aux_film, dual_film)
        if self.aux_mode in ("aux_film", "dual_film"):
            self.aux_film = FiLMLayer(aux_proj_dim, width, dropout=dropout)
        else:
            self.aux_film = None

        # Main Architecture Fusion & Conditioning (B1-B6)
        concat_aux_extra = aux_proj_dim if self.aux_mode == "concat" else 0

        if arch == "go_film" or self.aux_mode == "dual_film":
            self.go_film = FiLMLayer(width, width, dropout=dropout)
            self.fusion = nn.Sequential(
                nn.Linear(width + concat_aux_extra, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
            )
        elif arch == "bilinear":
            self.bilin_m = nn.Linear(width + concat_aux_extra, width)
            self.bilin_g = nn.Linear(width, width)
            self.fusion = nn.Sequential(
                nn.Linear(3 * width + concat_aux_extra, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
            )
        elif arch == "res_mlp":
            self.fusion_in = nn.Linear(2 * width + concat_aux_extra, width)
            self.res_blocks = nn.Sequential(
                SwiGLUResBlock(width, dropout=dropout),
                SwiGLUResBlock(width, dropout=dropout),
            )
            self.fusion_norm = nn.LayerNorm(width)
        else:
            # joint, mag_dir, delta_only
            self.fusion = nn.Sequential(
                nn.Linear(2 * width + concat_aux_extra, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
            )

        # Output Head (3-class vs Magnitude+Direction 2-head)
        if arch == "mag_dir":
            self.impact_head = nn.Linear(width, 1)  # P(impact | x), P(NEUTRAL) = 1 - P(impact)
            self.dir_head = nn.Linear(width, 1)     # P(GOF | impact, x), P(LOF | impact) = 1 - P(GOF | impact)
        else:
            self.class_head = nn.Linear(width, 3)

    def _add_noise(self, t: torch.Tensor) -> torch.Tensor:
        if self.training and self.embed_noise > 0.0:
            return t + torch.randn_like(t) * self.embed_noise
        return t

    def _read_window(
        self,
        wt_w: torch.Tensor, var_w: torch.Tensor,
        wt_m: torch.Tensor, var_m: torch.Tensor,
        query: torch.Tensor,
        ps_wt_w: torch.Tensor | None = None,
        ps_var_w: torch.Tensor | None = None,
        ps_mask: torch.Tensor | None = None,
        ps_site_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wt_w = self._add_noise(wt_w)
        var_w = self._add_noise(var_w)
        valid = wt_m & var_m

        if self.window_reader == "esim":
            w = self.window_adapter(torch.where(wt_m.unsqueeze(-1), wt_w, torch.zeros_like(wt_w)))
            v = self.window_adapter(torch.where(var_m.unsqueeze(-1), var_w, torch.zeros_like(var_w)))
            tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
            return self.local_reader(query, tokens, valid)

        if self.window_reader == "bi_xattn":
            w = self.window_adapter(wt_w)
            v = self.window_adapter(var_w)
            return self.bi_xattn(w, v, wt_m, var_m, query)

        if self.window_reader == "cnn1d":
            w = self.window_adapter(torch.where(wt_m.unsqueeze(-1), wt_w, torch.zeros_like(wt_w)))
            v = self.window_adapter(torch.where(var_m.unsqueeze(-1), var_w, torch.zeros_like(var_w)))
            tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
            conv_t = self.conv_block(tokens, valid)
            center_feat = conv_t[:, 3, :]
            mean_feat = masked_mean(conv_t, valid)
            return self.cnn_proj(torch.cat([center_feat, mean_feat], dim=-1))

        if self.window_reader == "hybrid_cnn_attn":
            w = self.window_adapter(torch.where(wt_m.unsqueeze(-1), wt_w, torch.zeros_like(wt_w)))
            v = self.window_adapter(torch.where(var_m.unsqueeze(-1), var_w, torch.zeros_like(var_w)))
            tokens = self.comparison(torch.cat([w, v, v - w, w * v], dim=-1))
            conv_t = self.conv_block(tokens, valid)
            q = query.unsqueeze(1)
            read, _ = self.hyb_attn(q, conv_t, conv_t, key_padding_mask=~valid)
            h = self.hyb_norm1(q + read)
            return self.hyb_norm2(h + self.hyb_ffn(h)).squeeze(1)

        if self.window_reader == "persite_mil":
            # ps_wt_w, ps_var_w: [B, M, 7, 1280], ps_mask: [B, M, 7], ps_site_valid: [B, M]
            if ps_wt_w is None:
                ps_wt_w = wt_w.unsqueeze(1)
                ps_var_w = var_w.unsqueeze(1)
                ps_mask = valid.unsqueeze(1)
                ps_site_valid = torch.ones(len(query), 1, dtype=torch.bool, device=query.device)
            B, M, L, D = ps_wt_w.shape
            w_flat = self.window_adapter(self._add_noise(ps_wt_w).reshape(B * M, L, D))
            v_flat = self.window_adapter(self._add_noise(ps_var_w).reshape(B * M, L, D))
            m_flat = ps_mask.reshape(B * M, L).clone()
            m_flat[:, 3] = True
            q_flat = query.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
            tok = self.comparison(torch.cat([w_flat, v_flat, v_flat - w_flat, w_flat * v_flat], dim=-1))
            site_vecs = self.site_reader(q_flat, tok, m_flat).reshape(B, M, -1)
            # GO-queried MIL attention across the M mutation sites
            q = query.unsqueeze(1)
            pooled, _ = self.mil_attn(q, site_vecs, site_vecs, key_padding_mask=~ps_site_valid)
            return self.mil_norm(pooled.squeeze(1))

        raise ValueError(f"Unsupported window_reader: {self.window_reader}")

    def forward(
        self,
        wt_g: torch.Tensor, var_g: torch.Tensor,
        wt_l: torch.Tensor, var_l: torch.Tensor,
        wt_w: torch.Tensor, var_w: torch.Tensor,
        wt_m: torch.Tensor, var_m: torch.Tensor,
        go: torch.Tensor,
        aux: torch.Tensor | None = None,
        ps_wt_w: torch.Tensor | None = None,
        ps_var_w: torch.Tensor | None = None,
        ps_mask: torch.Tensor | None = None,
        ps_site_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        query = self.go_adapter(go)
        if self.input_mode == "go_only":
            logits = self.go_head(query)
            scores = logits.softmax(-1)
            return {"logits": logits, "scores": scores, "prediction": scores.argmax(1)}

        B = len(go)
        avail = torch.ones(B, 2, dtype=torch.bool, device=go.device)

        if self.input_mode == "wt_go":
            w_tok = torch.stack([
                self.adapters["mut"](self._add_noise(wt_l)) + self.types[0],
                self.adapters["global"](self._add_noise(wt_g)) + self.types[1],
            ], dim=1)
            m = self.mut_proj(self.shared_reader(query, w_tok, avail))
        elif self.input_mode == "var_go":
            v_tok = torch.stack([
                self.adapters["mut"](self._add_noise(var_l)) + self.types[0],
                self.adapters["global"](self._add_noise(var_g)) + self.types[1],
            ], dim=1)
            m = self.mut_proj(self.shared_reader(query, v_tok, avail))
        else:
            w_tok = torch.stack([
                self.adapters["mut"](self._add_noise(wt_l)) + self.types[0],
                self.adapters["global"](self._add_noise(wt_g)) + self.types[1],
            ], dim=1)
            v_tok = torch.stack([
                self.adapters["mut"](self._add_noise(var_l)) + self.types[0],
                self.adapters["global"](self._add_noise(var_g)) + self.types[1],
            ], dim=1)
            hw = self.shared_reader(query, w_tok, avail)
            hv = self.shared_reader(query, v_tok, avail)
            pair = self._read_window(wt_w, var_w, wt_m, var_m, query, ps_wt_w, ps_var_w, ps_mask, ps_site_valid)

            if self.arch == "delta_only":
                delta = hv - hw
                m = self.mut_proj(torch.cat([delta, delta.abs(), hw * hv, pair], dim=-1))
            else:
                m = self.mut_proj(torch.cat([hw, hv, hv - hw, pair], dim=-1))

        # Auxiliary conditioning (concat vs Aux-FiLM / Dual-FiLM)
        aux_h = self.aux_encoder(aux) if (self.aux_encoder is not None and aux is not None) else None
        if self.aux_film is not None and aux_h is not None:
            m = self.aux_film(m, aux_h)

        # Main Architecture Fusion with GO
        if self.arch == "go_film" or self.aux_mode == "dual_film":
            m_go = self.go_film(m, query)
            feat = torch.cat([m_go, aux_h], dim=-1) if (self.aux_mode == "concat" and aux_h is not None) else m_go
            h = self.fusion(feat)
        elif self.arch == "bilinear":
            m_ext = torch.cat([m, aux_h], dim=-1) if (self.aux_mode == "concat" and aux_h is not None) else m
            inter = self.bilin_m(m_ext) * self.bilin_g(query)
            h = self.fusion(torch.cat([m_ext, query, inter], dim=-1))
        elif self.arch == "res_mlp":
            parts = [m, query] + ([aux_h] if (self.aux_mode == "concat" and aux_h is not None) else [])
            h = self.fusion_norm(self.res_blocks(self.fusion_in(torch.cat(parts, dim=-1))))
        else:
            parts = [m, query] + ([aux_h] if (self.aux_mode == "concat" and aux_h is not None) else [])
            h = self.fusion(torch.cat(parts, dim=-1))

        # Output Head
        if self.arch == "mag_dir":
            imp_logit = self.impact_head(h).squeeze(-1)   # >0 => Impact
            dir_logit = self.dir_head(h).squeeze(-1)      # >0 => GOF, <0 => LOF
            log_p_imp = F.logsigmoid(imp_logit)
            log_p_neu = F.logsigmoid(-imp_logit)
            log_p_gof = log_p_imp + F.logsigmoid(dir_logit)
            log_p_lof = log_p_imp + F.logsigmoid(-dir_logit)
            # Log-probabilities directly serve as valid logits for CE / KL / softmax!
            logits = torch.stack([log_p_lof, log_p_gof, log_p_neu], dim=-1)
            scores = logits.exp()
            return {"logits": logits, "scores": scores, "prediction": scores.argmax(1)}

        logits = self.class_head(h)
        scores = logits.softmax(-1)
        return {"logits": logits, "scores": scores, "prediction": scores.argmax(1)}


# ---------------------------------------------------------------------------
# Losses: Focal, Logit-Adjusted CE, Reverse-Consistency, Identity-Neutral
# ---------------------------------------------------------------------------
def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 1.5, weight: torch.Tensor | None = None) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    pt = p.gather(1, labels.unsqueeze(1)).squeeze(1)
    log_pt = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
    loss = -((1.0 - pt).clamp_min(1e-6) ** gamma) * log_pt
    if weight is not None:
        loss = loss * weight[labels]
    return loss.mean()


def logit_adjusted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_priors: torch.Tensor,
    tau: float = 0.75,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Menon et al. (ICLR 2021) training-time logit adjustment for class imbalance."""
    adj_logits = logits + tau * log_priors.unsqueeze(0)
    return F.cross_entropy(adj_logits, labels, label_smoothing=label_smoothing)


def reverse_consistency_loss(logits_fwd: torch.Tensor, logits_rev: torch.Tensor) -> torch.Tensor:
    """Symmetric KL enforcing p(VAR, WT) == swap_{LOF<->GOF}(p(WT, VAR))."""
    swap_idx = torch.tensor([1, 0, 2], device=logits_fwd.device)
    p_fwd_swap = F.softmax(logits_fwd[:, swap_idx].detach(), dim=-1)
    log_p_rev = F.log_softmax(logits_rev, dim=-1)
    p_rev_swap = F.softmax(logits_rev[:, swap_idx].detach(), dim=-1)
    log_p_fwd = F.log_softmax(logits_fwd, dim=-1)
    return 0.5 * (F.kl_div(log_p_rev, p_fwd_swap, reduction="batchmean") + F.kl_div(log_p_fwd, p_rev_swap, reduction="batchmean"))


def identity_neutral_loss(logits_id: torch.Tensor) -> torch.Tensor:
    """Enforces p(WT, WT) -> NEUTRAL (class index 2), directly addressing PDF Page 9 limitation."""
    targets = torch.full((len(logits_id),), 2, dtype=torch.long, device=logits_id.device)
    return F.cross_entropy(logits_id, targets)


# ---------------------------------------------------------------------------
# Weight EMA & Post-Hoc Temperature Scaling
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential Moving Average of model weights for smoother hard-split generalization."""

    def __init__(self, model: nn.Module, decay: float = 0.992) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Find optimal scalar temperature T > 0 minimizing validation NLL (zero test leakage)."""
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 4.0, 71):
        nll = float(F.cross_entropy(logits_t / float(t), labels_t).item())
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t
