"""Frozen indexed feature cache and matched datasets for every window condition."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .go_embeddings import GOEmbeddingTable
from .util import DATA_ROOT, LABELS


class FeatureStore:
    def __init__(self, prepared, data=DATA_ROOT):
        self.go = GOEmbeddingTable(data / "embeddings/go_terms_embeddings_anc2vec.npz")
        with h5py.File(prepared / "features_window7.h5", "r") as f:
            if not f.attrs.get("complete") or int(f.attrs["window_size"]) != 7:
                raise ValueError("Incomplete/wrong feature cache")
            self.maps = {s: {k: i for i, k in enumerate(f[s]["keys"].asstr()[:])} for s in ("global", "mut", "window")}
            self.global_vectors = torch.from_numpy(f["global/vectors"][:])
            self.mut_vectors = torch.from_numpy(f["mut/vectors"][:])
            self.window_tokens = torch.from_numpy(f["window/tokens"][:])
            self.window_masks = torch.from_numpy(f["window/mask"][:])
            self.window_means = torch.from_numpy(f["window/mean"][:])
        print(f"FEATURES_LOADED global={len(self.maps['global'])} mutation/window={len(self.maps['mut'])}", flush=True)


class PairDataset(Dataset):
    def __init__(self, path, store, window_mode="none", frame=None):
        self.path, self.store, self.window_mode = path, store, window_mode
        self.frame = pd.read_csv(path, sep="\t", keep_default_na=False) if frame is None else frame.copy()
        self.frame = self.frame.reset_index(drop=True)
        if self.frame.empty or not self.frame.namespace.eq("molecular_function").all():
            raise ValueError("Expected a nonempty MF-only dataset")
        if (not getattr(store, "allow_combinatorial", False)
                and self.frame.is_combinatorial.astype(str).str.lower().eq("true").any()):
            raise ValueError("Combinatorial records are not allowed")
        if self.frame.split.iloc[0] != "train" and self.frame.augmented.astype(str).str.lower().eq("true").any():
            raise ValueError("Augmented evaluation records are not allowed")
        expected = self.frame.label.map({label: i for i, label in enumerate(LABELS)})
        if not np.array_equal(expected.to_numpy(), self.frame.label_idx.to_numpy()):
            raise ValueError("Label indices disagree with source labels")
        self.labels = expected.to_numpy(dtype=np.int64)
        self.go = store.go.lookup(self.frame.matched_go_id.tolist())
        self.indices = {}
        for scale in ("global", "mut", "window"):
            for side in ("wt", "var"):
                self.indices[f"{side}_{scale}"] = np.array(
                    [store.maps[scale][x] for x in self.frame[f"{side}_{scale}_key"]], dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        result = dict(go=self.go[index], label=int(self.labels[index]), row=index)
        for side in ("wt", "var"):
            result[side] = dict(global_=self.store.global_vectors[self.indices[f"{side}_global"][index]],
                                mut=self.store.mut_vectors[self.indices[f"{side}_mut"][index]])
            result[side]["global"] = result[side].pop("global_")
            wi = self.indices[f"{side}_window"][index]
            if self.window_mode == "mean":
                result[side]["window"] = self.store.window_means[wi]
            elif self.window_mode in ("pair_mean", "esim"):
                result[f"{side}_window"] = self.store.window_tokens[wi]
                result[f"{side}_mask"] = self.store.window_masks[wi]
        return result


def model_inputs(batch, device):
    result = dict(wt={s: x.to(device, non_blocking=True) for s, x in batch["wt"].items()},
                  var={s: x.to(device, non_blocking=True) for s, x in batch["var"].items()},
                  go=batch["go"].to(device, non_blocking=True))
    for side in ("wt", "var"):
        if f"{side}_window" in batch:
            result[f"{side}_window"] = batch[f"{side}_window"].to(device, dtype=torch.float32, non_blocking=True)
            result[f"{side}_mask"] = batch[f"{side}_mask"].to(device, non_blocking=True)
    return result
