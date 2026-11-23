"""Read frozen repository features lazily; descriptions never become inputs."""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_DATA_ROOT, FEATURE_DIM, DataPaths
from .go_embeddings import GOEmbeddingTable


def go_table(data_root: str | Path = DEFAULT_DATA_ROOT) -> GOEmbeddingTable:
    """Load GO by ID using the existing audited project reader."""
    return GOEmbeddingTable(DataPaths(Path(data_root)).go_embeddings)


class FeatureStore:
    """Own read-only H5 handles and an optional unique-vector CPU cache."""

    def __init__(self, data_root: str | Path = DEFAULT_DATA_ROOT) -> None:
        self.paths = DataPaths(Path(data_root))
        self.go = go_table(data_root)
        self.files: dict[str, h5py.File] = {}
        self.memory: dict[tuple[str, str], torch.Tensor] = {}

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @lru_cache(maxsize=8192)
    def vector(self, scale: str, key: str) -> torch.Tensor:
        """Return one finite, 1280-D vector; missing features are errors."""
        if scale not in ("mut", "global"):
            raise ValueError(f"Unsupported pooled scale: {scale}")
        if (scale, key) in self.memory:
            return self.memory[(scale, key)]
        if scale not in self.files:
            path = self.paths.mutation_h5 if scale == "mut" else self.paths.global_h5
            self.files[scale] = h5py.File(path, "r")
        vector = np.asarray(self.files[scale][key][:], dtype=np.float32)
        if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid {scale} vector {key}")
        return torch.from_numpy(vector)

    def preload(self, datasets: Iterable[PairDataset]) -> None:
        """Keep unique frozen features in CPU memory; avoid repeated H5 reads."""
        datasets = tuple(datasets)
        for scale in ("global", "mut"):
            keys = set()
            for dataset in datasets:
                for side in ("wt", "var"):
                    keys.update(dataset.frame[f"{side}_{scale}_key"])
            for key in sorted(keys):
                self.memory[(scale, key)] = self.vector(scale, key)
            print(f"Loaded {len(keys):,} unique {scale} vectors into CPU memory", flush=True)
        self.vector.cache_clear()

    def close(self) -> None:
        """Release file handles and cached tensors, including on failed runs."""
        for handle in self.files.values():
            handle.close()
        self.files.clear()
        self.vector.cache_clear()
        self.memory.clear()


class PairDataset(Dataset):
    """Validated native split rows joined to frozen features by explicit keys."""

    def __init__(
        self,
        path: str | Path,
        store: FeatureStore,
        *,
        limit: int | None = None,
        seed: int = 42,
        exclude_augmented: bool = False,
        head: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.frame = pd.read_csv(path, sep="\t")
        # Older prepared files omitted this field and retained combination evidence.
        # Refuse those files instead of silently reintroducing the scope error.
        from .prepare import combinatorial_mask

        if combinatorial_mask(self.frame).any():
            raise ValueError(
                "Combined-mutation evidence is not allowed; run decoder.new_decoder.prepare"
            )
        if not self.frame["snp"].astype(str).str.fullmatch(r"[A-Z][1-9]\d*[A-Z]").all():
            raise ValueError("Only a single X123Y substitution per example is allowed")
        if exclude_augmented and "augmented" in self.frame:
            augmented = self.frame["augmented"].astype(str).str.lower().eq("true")
            self.frame = self.frame.loc[~augmented]
        if limit is not None and len(self.frame) > limit:
            self.frame = self.frame.sample(limit, random_state=seed)
        self.frame = self.frame.reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Empty dataset: {path}")
        self.store = store
        expected = self.frame["label"].map({"LOF": 0, "GOF": 1, "NEUTRAL": 2})
        if expected.isna().any() or not np.array_equal(expected, self.frame["label_idx"]):
            raise ValueError("Expected label order LOF=0, GOF=1, NEUTRAL=2")
        if head == "parallel":
            for label, column in (("LOF", "target_lof"), ("GOF", "target_gof")):
                if column not in self.frame or not np.array_equal(
                    self.frame[column], self.frame.label.eq(label).astype(int)
                ):
                    raise ValueError(f"Invalid source binary target: {column}")
        if head == "multiclass" and (
            "target" not in self.frame or not self.frame.target.equals(self.frame.label)
        ):
            raise ValueError("Invalid source multiclass target")
        self.go = store.go.lookup(self.frame["matched_go_id"].tolist())
        self.keys = {
            name: self.frame[name].to_numpy()
            for name in ("wt_mut_key", "var_mut_key", "wt_global_key", "var_global_key")
        }
        self.labels = self.frame.label_idx.to_numpy(dtype=np.int64)
        self.binary = np.stack([self.labels == 0, self.labels == 1], axis=1).astype(np.float32)
        if head == "parallel":
            self.binary = self.frame[["target_lof", "target_gof"]].to_numpy(dtype=np.float32).copy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(
            wt={
                s: self.store.vector(s, self.keys[f"wt_{s}_key"][index]) for s in ("mut", "global")
            },
            var={
                s: self.store.vector(s, self.keys[f"var_{s}_key"][index]) for s in ("mut", "global")
            },
            go=self.go[index],
            label=int(self.labels[index]),
            binary_target=torch.from_numpy(self.binary[index]),
            row=index,
        )


def move_batch(
    batch: dict[str, Any], device: str | torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Move model inputs only; labels and metadata are handled by the caller."""
    return (
        {s: x.to(device) for s, x in batch["wt"].items()},
        {s: x.to(device) for s, x in batch["var"].items()},
        batch["go"].to(device),
    )
