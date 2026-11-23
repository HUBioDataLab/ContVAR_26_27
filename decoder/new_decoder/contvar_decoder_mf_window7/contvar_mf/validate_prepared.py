"""End-to-end preflight on real prepared data, without fitting extra models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset, model_inputs
from .model import PairedGODecoder, classification_loss
from .suite import verify_preparation
from .util import LABELS, WORKSPACE, run_conditions, write_json


def main():
    prepared = WORKSPACE / "prepared"
    manifest = verify_preparation(prepared)
    store = FeatureStore(prepared)
    checks = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    for condition in run_conditions()[:4] + run_conditions()[8:16]:
        path = prepared / condition["cohort"] / f"{condition['family']}_train.tsv"
        dataset = PairDataset(path, store, condition["window_mode"])
        batch = next(iter(DataLoader(dataset, batch_size=12)))
        model = PairedGODecoder(go_dim=store.go.dimension, head=condition["head"],
            pair_mode=condition["pair_mode"], window_mode=condition["window_mode"]).to(device)
        output = model(**model_inputs(batch, device))
        loss = classification_loss(output, batch["label"].to(device), condition["head"])
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(condition)
        checks.append(dict(window_mode=condition["window_mode"], head=condition["head"],
                           pair_mode=condition["pair_mode"], batch_loss=float(loss.detach())))
    for cohort in ["old", "full"]:
        for family in ["easy", "hard"]:
            paths = [prepared / cohort / f"{family}_{s}.tsv" for s in ["train", "val", "test"]]
            frames = [pd.read_csv(p, sep="\t", keep_default_na=False) for p in paths]
            if not all(f.namespace.eq("molecular_function").all() for f in frames):
                raise ValueError("Non-MF record")
            if not all(not f.is_combinatorial.astype(str).str.lower().eq("true").any() for f in frames):
                raise ValueError("Combinatorial record")
            if any(f.augmented.astype(str).str.lower().eq("true").any() for f in frames[1:]):
                raise ValueError("Augmented evaluation record")
    result = dict(passed=True, conditions_checked=checks, extra_training_runs=0,
                  common_test=manifest["common_test"], device=device)
    write_json(WORKSPACE / "verification/prepared_and_gradient_checks.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
