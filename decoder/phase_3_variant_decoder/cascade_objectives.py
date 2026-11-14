"""Per-row losses on the cascade's scaled, signed decision logits.

Binary targets remain 0/1. Hinge uses t=2*y-1, not probabilities or hard
predictions. Effect logit = a*(abs(delta)-radius); Direction logit = a*delta.
"""
import math

import torch
from torch.nn import functional as F

LOSS_LABELS = {"bce": "BCE", "hinge": "Hinge", "squared_hinge": "Squared hinge"}


def cascade_loss_per_row(logits, targets, *, kind="hinge", margin=1.0):
    if kind not in LOSS_LABELS:
        raise ValueError(f"Unknown cascade loss: {kind}")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Cascade margin must be finite and positive")
    if logits.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("Expected aligned 1-D logits and targets")
    if not torch.isin(targets, targets.new_tensor([0, 1])).all():
        raise ValueError("Cascade targets must be binary 0/1")
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    signed_target = targets.to(logits.dtype).mul(2).sub(1)
    violation = F.relu(margin - signed_target * logits)
    return violation.square() if kind == "squared_hinge" else violation
