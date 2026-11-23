"""Shared GO-query readers with joint or separate-score effect classification.

The encoder and Anc2vec inputs are frozen. All parameters defined here are
learned from scratch; no STAR-GO weights or residue windows are loaded.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from .config import CHECKPOINT_FORMAT, LABELS, PARALLEL_THRESHOLD

HeadKind = Literal["parallel", "multiclass"]
PairMode = Literal["joint", "score_difference"]
FeatureMap = Mapping[str, torch.Tensor]


class DecoderOutput(TypedDict):
    """Model outputs; parallel ``scores`` are not a probability simplex.

    Class order is LOF, GOF, NEUTRAL. ``d`` is the signed logit contrast;
    ``direction_index`` is the separate contrast of output scores.
    """

    logits: torch.Tensor
    scores: torch.Tensor
    prediction: torch.Tensor
    d: torch.Tensor
    direction_index: torch.Tensor
    side_scores: dict[str, torch.Tensor] | None
    attention: dict[str, torch.Tensor]


@contextmanager
def deterministic(model: nn.Module) -> Iterator[None]:
    """Disable dropout while preserving autograd and the previous train/eval mode."""
    training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(training)


class GOReader(nn.Module):
    """Contextualize scale tokens, then read them using one GO query.

    One instance is shared between WT and VAR. Each current side has two
    pooled scale tokens, rather than a residue-level sequence memory.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.context = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(
        self, query: torch.Tensor, memory: torch.Tensor, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read ``[B,S,D]`` memory with ``[B,D]`` query; True means available."""
        if not available.any(dim=1).all():
            raise ValueError("Every example needs at least one available scale")
        padding = ~available
        memory = self.context(memory, src_key_padding_mask=padding)
        q = query.unsqueeze(1)
        read, weights = self.attention(q, memory, memory, key_padding_mask=padding)
        h = self.norm1(q + read)
        h = self.norm2(h + self.ffn(h))
        return h.squeeze(1), weights.squeeze(1)


class PairedGODecoder(nn.Module):
    """Joint WT/VAR/GO learning versus shared single-state score subtraction.

    Each side is {scale: [B, input_dim]}. Future pooled local_window/pocket
    scales can have their own dimensions. No delta embedding is added here.
    """

    def __init__(
        self,
        scale_dims: Mapping[str, int] | None = None,
        go_dim: int = 200,
        width: int = 128,
        heads: int = 4,
        head: HeadKind = "multiclass",
        pair_mode: PairMode = "joint",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scale_dims = dict(scale_dims or {"mut": 1280, "global": 1280})
        if (
            head not in ("parallel", "multiclass")
            or pair_mode not in ("joint", "score_difference")
            or width < 1
            or heads < 1
            or width % heads
            or go_dim < 1
            or not 0 <= dropout < 1
            or any(d < 1 for d in scale_dims.values())
        ):
            raise ValueError("Invalid decoder configuration")
        self.spec = dict(
            scale_dims=scale_dims,
            go_dim=go_dim,
            width=width,
            heads=heads,
            head=head,
            pair_mode=pair_mode,
            dropout=dropout,
        )
        self.scales = tuple(scale_dims)
        self.head_kind, self.pair_mode = head, pair_mode
        self.adapters = nn.ModuleDict(
            {
                s: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width), nn.GELU())
                for s, d in scale_dims.items()
            }
        )
        self.types = nn.Parameter(torch.empty(len(scale_dims), width))
        nn.init.normal_(self.types, std=0.02)
        self.go_adapter = nn.Sequential(nn.LayerNorm(go_dim), nn.Linear(go_dim, width), nn.GELU())
        self.shared_reader = GOReader(width, heads)
        if pair_mode == "joint":
            self.fusion = nn.Sequential(
                nn.Linear(3 * width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout)
            )
            if head == "parallel":
                self.lof_head, self.gof_head = nn.Linear(width, 1), nn.Linear(width, 1)
            else:
                self.class_head = nn.Linear(width, 3)
        else:
            # The entire single-state scorer is deterministic, including in train.
            # Independent WT/VAR dropout would manufacture differences.
            self.side_scorer = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, 2 if head == "parallel" else 1),
            )
            if head == "parallel":
                # Shared threshold keeps the signed LOF/GOF logit contrast odd.
                self.raw_radius = nn.Parameter(torch.tensor(0.0))
            else:
                self.neutral_bias = nn.Parameter(torch.tensor(1.0))
                self.raw_neutral_slope = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        wt: FeatureMap,
        var: FeatureMap,
        go: torch.Tensor,
        available: torch.Tensor | None = None,
    ) -> DecoderOutput:
        """Predict one GO-specific effect for paired, pooled protein features.

        Score subtraction uses shared single-state raw scores. Parallel
        decisions keep the historical fixed 0.5 / strongest-positive rule.
        """
        if go.ndim != 2 or go.shape[1] != self.spec["go_dim"] or not torch.isfinite(go).all():
            raise ValueError("Expected finite GO [B, go_dim]")
        batch = len(go)
        if available is None:
            available = torch.ones(batch, len(self.scales), dtype=torch.bool, device=go.device)
        if available.shape != (batch, len(self.scales)) or available.dtype != torch.bool:
            raise ValueError("available must be bool [B, number_of_scales], True=present")
        mask = available.clone()
        w_tokens, v_tokens = [], []
        for i, scale in enumerate(self.scales):
            if (scale in wt) != (scale in var):
                raise ValueError(f"{scale} must be present on both sides or neither")
            if scale not in wt:
                mask[:, i] = False
                w = v = go.new_zeros(batch, self.spec["scale_dims"][scale])
            else:
                w, v = wt[scale], var[scale]
                expected = (batch, self.spec["scale_dims"][scale])
                if w.shape != expected or v.shape != expected:
                    raise ValueError(f"Invalid {scale} shape, expected {expected}")
                w = torch.where(mask[:, i, None], w, torch.zeros_like(w))
                v = torch.where(mask[:, i, None], v, torch.zeros_like(v))
                if not torch.isfinite(w).all() or not torch.isfinite(v).all():
                    raise ValueError(f"Non-finite available {scale} feature")
            w_tokens.append(self.adapters[scale](w) + self.types[i])
            v_tokens.append(self.adapters[scale](v) + self.types[i])
        q = self.go_adapter(go)
        hw, aw = self.shared_reader(q, torch.stack(w_tokens, dim=1), mask)
        hv, av = self.shared_reader(q, torch.stack(v_tokens, dim=1), mask)
        attention = {"wt": aw, "var": av}
        side_scores = None
        if self.pair_mode == "joint":
            h = self.fusion(torch.cat([hw, hv, q], dim=-1))
            logits = (
                torch.cat([self.lof_head(h), self.gof_head(h)], dim=-1)
                if self.head_kind == "parallel"
                else self.class_head(h)
            )
        else:
            sw = self.side_scorer(torch.cat([hw, q], dim=-1))
            sv = self.side_scorer(torch.cat([hv, q], dim=-1))
            delta = sv - sw  # score subtraction, never embedding subtraction
            side_scores = {"wt": sw, "var": sv, "var_minus_wt": delta}
            if self.head_kind == "parallel":
                logits = torch.stack([-delta[:, 0], delta[:, 1]], dim=1) - F.softplus(
                    self.raw_radius
                )
            else:
                direction = delta.squeeze(1)
                neutral = self.neutral_bias - F.softplus(self.raw_neutral_slope) * direction.abs()
                logits = torch.stack([-direction, direction, neutral], dim=1)
        if self.head_kind == "parallel":
            p = logits.sigmoid()
            scores = torch.cat([p, 1 - p.max(dim=1, keepdim=True).values], dim=1)
            # >=.5: strongest positive head wins; neither positive: NEUTRAL.
            prediction = torch.where(p.max(dim=1).values >= PARALLEL_THRESHOLD, p.argmax(dim=1), 2)
        else:
            scores = logits.softmax(dim=-1)
            prediction = scores.argmax(dim=1)
        return dict(
            logits=logits,
            scores=scores,
            prediction=prediction,
            d=logits[:, 1] - logits[:, 0],
            direction_index=scores[:, 1] - scores[:, 0],
            side_scores=side_scores,
            attention=attention,
        )


def classification_loss(
    output: DecoderOutput,
    labels: torch.Tensor,
    head: HeadKind,
    class_weights: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    binary_targets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply raw-logit CE or binary BCE with the native source targets."""
    if head == "multiclass":
        return F.cross_entropy(output["logits"], labels, weight=class_weights)
    targets = (
        torch.stack([labels == 0, labels == 1], dim=1).to(output["logits"].dtype)
        if binary_targets is None
        else binary_targets.to(output["logits"].dtype)
    )
    return F.binary_cross_entropy_with_logits(output["logits"], targets, pos_weight=pos_weight)


def reverse_loss(forward: DecoderOutput, reverse: DecoderOutput) -> torch.Tensor:
    """Mean squared violation of the signed WT/VAR reversal constraint."""
    return (forward["d"] + reverse["d"]).square().mean()


def load_decoder(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[PairedGODecoder, dict[str, Any]]:
    """Load a compatible tensor-only checkpoint, preserving label order."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected a starter decoder checkpoint")
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Checkpoint label order must be LOF/GOF/NEUTRAL")
    model = PairedGODecoder(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
