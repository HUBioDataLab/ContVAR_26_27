"""Train-only GO lookup helper for the retained input-control comparison."""
from __future__ import annotations

import numpy as np

def fit_lookup(go_ids, labels):
    """Build the rule on train only; ties reproduce the earlier audit exactly."""
    go_ids, labels = np.asarray(go_ids), np.asarray(labels)
    if go_ids.ndim != 1 or labels.ndim != 1 or len(go_ids) != len(labels) or not len(labels):
        raise ValueError("Expected aligned nonempty 1-D GO IDs and labels")
    if not np.isin(labels, [0, 1, 2]).all() or any(not isinstance(g, str) or not g for g in go_ids):
        raise ValueError("Expected nonempty GO strings and LOF/GOF/NEUTRAL indices")
    table, first_seen = {}, {}
    for index, (go, label) in enumerate(zip(go_ids, labels)):
        table.setdefault(go, [0, 0, 0])[int(label)] += 1
        first_seen.setdefault(go, {}).setdefault(int(label), index)
    rules = {go: min(first_seen[go], key=lambda label: (-counts[label], first_seen[go][label]))
             for go, counts in table.items()}
    fallback = int(np.bincount(labels.astype(int), minlength=3).argmax())
    return rules, fallback, table

