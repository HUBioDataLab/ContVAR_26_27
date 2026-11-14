"""
Phase 3 loss: Margin Ranking (hinge).

Design: phase_3_variant_decoder/DESIGN.md §4 (single-margin revision)

    d = s_wt - s_var                s_* = sigmoid(logits[go_idx]),  d in [-1, 1]

    LOF      y=+1  ->  w_LOF     * max(0,  m - d)      zero when:  d >= +m
    GOF      y=-1  ->  w_GOF     * max(0,  m + d)      zero when:  d <= -m
    NEUTRAL        ->  w_NEUTRAL * max(0, |d| - m)     zero when: |d| <= m

A single margin `m` is shared by all three classes: LOF/GOF want to clear it,
NEUTRAL wants to stay inside it. The two targets are not contradictory — both
are satisfied simultaneously at exactly d = m — but there is no buffer zone
between them (the earlier two-margin design's `m_n < m` gap is gone by design).
This removed `margin_neutral`/`margin_ratio` as a separate, searched quantity.

BCE was dropped deliberately: the labels are 100% LLM-generated, and because BCE
pins the target to 0/1 it punishes noisy labels hard. The hinge zeroes the gradient
exactly once the margin is met.

NEUTRAL IS TWO-SIDED (|d|). Writing the one-sided max(0, m - d) makes it
identical to the LOF term and trains NEUTRAL rows as if they were LOF.
"""

from __future__ import annotations

import torch

from phase_3_variant_decoder.config import LABEL_TO_IDX

IDX_LOF = LABEL_TO_IDX["LOF"]
IDX_GOF = LABEL_TO_IDX["GOF"]
IDX_NEUTRAL = LABEL_TO_IDX["NEUTRAL"]


def pair_delta(logits_wt: torch.Tensor, logits_var: torch.Tensor,
               go_idx: torch.Tensor) -> torch.Tensor:
    """d = sigmoid(logits_wt[go_idx]) - sigmoid(logits_var[go_idx]).

    Only the row's own go_idx is taken; every other output is treated as masked
    and receives no gradient.
    """
    rows = torch.arange(go_idx.shape[0], device=go_idx.device)
    s_wt = torch.sigmoid(logits_wt[rows, go_idx])
    s_var = torch.sigmoid(logits_var[rows, go_idx])
    return s_wt - s_var


def margin_ranking_loss(
    d: torch.Tensor,
    label_idx: torch.Tensor,
    margin: float,
    w_lof: float = 1.0,
    w_gof: float = 1.0,
    w_neutral: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """The DESIGN §4 hinge. `d` must be in sigmoid-probability space.

    A single `margin` (m) is shared by all three classes — there is no separate
    `margin_neutral` any more (removed at the user's request: the earlier
    two-margin design was judged an unwanted extra degree of freedom).
    """
    zero = torch.zeros((), device=d.device, dtype=d.dtype)

    loss_lof = torch.clamp(margin - d, min=0.0)                 # we want d >= +m
    loss_gof = torch.clamp(margin + d, min=0.0)                 # we want d <= -m
    loss_neu = torch.clamp(d.abs() - margin, min=0.0)           # we want |d| <= m

    per_row = torch.where(label_idx == IDX_LOF, w_lof * loss_lof, zero)
    per_row = per_row + torch.where(label_idx == IDX_GOF, w_gof * loss_gof, zero)
    per_row = per_row + torch.where(label_idx == IDX_NEUTRAL, w_neutral * loss_neu, zero)

    if reduction == "mean":
        return per_row.mean()
    if reduction == "sum":
        return per_row.sum()
    if reduction == "none":
        return per_row
    raise ValueError(f"Unknown reduction: {reduction!r}")
