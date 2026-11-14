"""Two independent GO-conditioned, shared-weight WT/variant score models.

Each stage has its own parameters; within a stage WT and variant share weights.
Delta is always sigmoid(score_WT) - sigmoid(score_variant): positive means LOF.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from phase_3_variant_decoder.config import LABEL_TO_IDX
from phase_3_variant_decoder.data.go_embeddings import decoder_inputs


class PairScorer(nn.Module):
    """Scalar scorer with paired dropout, avoiding artificial WT/variant noise.

The same dropout mask is used on both sides. Identical WT and variant inputs
therefore have exactly zero delta, even in training. LayerNorm allows batch=1.
"""

    def __init__(self, protein_dim, go_dim, hidden_dims, dropout):
        super().__init__()
        if protein_dim <= 0 or go_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid dimensions or dropout")
        dims = [protein_dim + go_dim, *hidden_dims]
        if any(d <= 0 for d in dims):
            raise ValueError("Hidden dimensions must be positive")
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU())
            for a, b in zip(dims, dims[1:])
        ])
        self.output = nn.Linear(dims[-1], 1)
        self.dropout = dropout

    def forward(self, wt, variant, go):
        left, right = decoder_inputs(wt, variant, go, mode="separate")
        for block in self.blocks:
            left, right = block(left), block(right)
            if self.training and self.dropout:
                mask = F.dropout(torch.ones_like(left), self.dropout, training=True)
                left, right = left * mask, right * mask
        s_wt = self.output(left).squeeze(-1).sigmoid()
        s_var = self.output(right).squeeze(-1).sigmoid()
        return {"s_wt": s_wt, "s_var": s_var, "delta": s_wt - s_var}


class BinaryScoreStage(nn.Module):
    def __init__(self, protein_dim, go_dim, hidden_dims, dropout, *, effect):
        super().__init__()
        self.scorer = PairScorer(protein_dim, go_dim, hidden_dims, dropout)
        self.effect = effect
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(10.0))))
        if effect:
            self.raw_radius = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

    def forward(self, wt, variant, go):
        out = self.scorer(wt, variant, go)
        scale = F.softplus(self.raw_scale) + 1e-4
        signal = out["delta"]
        if self.effect:
            signal = signal.abs() - self.raw_radius.sigmoid()
        out["logit"] = scale * signal
        # Compatibility field name. With hinge objectives this is a monotone
        # sigmoid decision score, not an estimated/calibrated probability.
        out["probability"] = out["logit"].sigmoid()
        return out


class CascadeDecoder(nn.Module):
    """Stage 1 P(effect); stage 2 P(LOF | effect), with separate scorers."""

    def __init__(self, protein_dim, go_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        self.spec = {"protein_dim": protein_dim, "go_dim": go_dim,
                     "hidden_dims": list(hidden_dims if hidden_dims is not None else [512, 1024, 512]),
                     "dropout": dropout}
        self.effect_stage = BinaryScoreStage(**self.spec, effect=True)
        self.direction_stage = BinaryScoreStage(**self.spec, effect=False)

    def forward(self, wt, variant, go):
        """Both stages for diagnostics/probabilities; use predict for routing."""
        return {"effect": self.effect_stage(wt, variant, go),
                "direction": self.direction_stage(wt, variant, go)}

    @torch.no_grad()
    def predict(self, wt, variant, go, effect_threshold=0.5):
        """Hard cascade: neutral rows never enter the direction model.

        Direction scores are NaN for rows not routed to stage 2. These are not
        zero-filled predictions. Probability >= 0.5 selects LOF, otherwise GOF.
        Call in eval mode so results do not depend on dropout.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before predict()")
        if not 0 <= effect_threshold <= 1:
            raise ValueError("effect_threshold must be in [0, 1]")
        effect = self.effect_stage(wt, variant, go)
        routed = effect["probability"] >= effect_threshold
        pred = torch.full((len(wt),), LABEL_TO_IDX["NEUTRAL"], dtype=torch.long, device=wt.device)
        direction = {k: torch.full_like(effect[k], float("nan")) for k in effect}
        if routed.any():
            selected = self.direction_stage(wt[routed], variant[routed], go[routed])
            for key, value in selected.items():
                direction[key][routed] = value
            pred[routed] = torch.where(selected["probability"] >= 0.5,
                                       LABEL_TO_IDX["LOF"], LABEL_TO_IDX["GOF"])
        return {"prediction": pred, "routed": routed, "effect": effect, "direction": direction}


def stage_targets(labels, stage):
    """Ground-truth routing for training; neutral never supervises direction."""
    if labels.ndim != 1 or not torch.isin(labels, labels.new_tensor([0, 1, 2])).all():
        raise ValueError("Expected 1-D LOF/GOF/NEUTRAL label indices")
    if stage == "effect":
        mask = torch.ones_like(labels, dtype=torch.bool)
        target = (labels != LABEL_TO_IDX["NEUTRAL"]).float()
    elif stage == "direction":
        mask = labels != LABEL_TO_IDX["NEUTRAL"]
        target = (labels[mask] == LABEL_TO_IDX["LOF"]).float()
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return mask, target
