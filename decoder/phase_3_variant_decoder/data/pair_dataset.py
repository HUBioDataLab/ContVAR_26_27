"""
WT/variant pair tensors.

Design: phase_3_variant_decoder/DESIGN.md §3.1, §6.3

Input layout:
    raw_esm / seq_phase0 / seq_phase1  ->  [global | local]                    2560
    concat_phase0 / concat_phase1      ->  [pri_global | raw_global |
                                            pri_local  | raw_local ]           5120

phase_2_decoder's GOAnnotationDataset (one sample per protein) does not fit here:
every row is a PAIR and the same WT is shared across many rows.

Memory: unique vectors are stored once and rows point at them by index.
    WT   37,788 x 2560 x 4 B = 0.39 GB
    var  42,659 x 2560 x 4 B = 0.44 GB
Everything fits on the GPU -> no DataLoader, just permutation batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.config import ESM_DIM, SECONDARY_REQUIRED, VariantDecoderConfig


def _read_keys(path: str, keys: list[str], what: str) -> np.ndarray:
    """Read the given keys from an H5 in order and return an [n, D] matrix."""
    import h5py
    missing = []
    with h5py.File(path, "r") as h5:
        # Catch an H5 built with the wrong ESM model early and legibly:
        # esm2_t33_650M -> 1280, esm2_t6_8M -> 320.
        probe = next(iter(h5.keys()), None)
        if probe is not None:
            dim = int(h5[probe].shape[0])
            if dim != ESM_DIM:
                raise ValueError(
                    f"{what}: {path} has embedding dim {dim}, expected {ESM_DIM}. "
                    "Probably built with the wrong --esm_model "
                    "(esm2_t33_650M_UR50D -> 1280, esm2_t6_8M_UR50D -> 320)."
                )
        out = np.empty((len(keys), ESM_DIM), dtype=np.float32)
        for i, k in enumerate(keys):
            if k not in h5:
                missing.append(k)
                if len(missing) > 5:
                    break
                continue
            out[i] = h5[k][:]
    if missing:
        raise KeyError(
            f"{what}: {path} is missing keys (first {len(missing)}): {missing[:5]}. "
            "Embedding generation may be incomplete, or the wrong H5 was given."
        )
    return out


class PairTensors:
    """Unique WT/variant matrices plus per-row indices."""

    def __init__(self, wt_feat: torch.Tensor, var_feat: torch.Tensor, df: pd.DataFrame):
        self.wt_feat = wt_feat
        self.var_feat = var_feat
        self.df = df
        self.input_dim = wt_feat.shape[1]
        self.go_features = None
        self.go_artifact_sha256 = None
        self.go_metadata = None

    def to(self, device) -> "PairTensors":
        self.wt_feat = self.wt_feat.to(device)
        self.var_feat = self.var_feat.to(device)
        if self.go_features is not None:
            self.go_features = self.go_features.to(device)
        return self

    def split_view(self, split: str) -> dict:
        """Index tensors and diagnostic masks for one split."""
        m = self.df["split"].to_numpy() == split
        sub = self.df[m]
        dev = self.wt_feat.device
        return {
            "wt_row": torch.as_tensor(sub["wt_row"].to_numpy(), device=dev),
            "var_row": torch.as_tensor(sub["var_row"].to_numpy(), device=dev),
            "go_idx": torch.as_tensor(sub["go_idx"].to_numpy(), device=dev),
            "label_idx": torch.as_tensor(sub["label_idx"].to_numpy(), device=dev),
            "seen_in_train": sub["seen_in_train"].to_numpy(),
            "is_alanine": sub["is_alanine"].to_numpy(),
            "n": int(m.sum()),
        }

    def gather(self, wt_row: torch.Tensor, var_row: torch.Tensor):
        return self.wt_feat[wt_row], self.var_feat[var_row]

    def attach_go_embeddings(self, path, *, allow_proxies: bool = False):
        """Opt-in GO inputs; validate the actual filtered frame's GO index mapping.

        Existing gather() and legacy training behavior are unchanged. Missing
        terms raise before training; source rows are never silently dropped.
        """
        from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable

        pairs = self.df[["matched_go_id", "go_idx"]].drop_duplicates()
        if pairs["matched_go_id"].duplicated().any() or pairs["go_idx"].duplicated().any():
            raise ValueError("Each GO term must map to exactly one distinct go_idx")
        vocab = {str(g): int(i) for g, i in pairs.itertuples(index=False, name=None)}
        table = GOEmbeddingTable(path, allow_proxies=allow_proxies)
        self.go_features = table.aligned_to_vocab(vocab).to(self.wt_feat.device)
        self.go_artifact_sha256 = table.artifact_sha256
        self.go_metadata = table.metadata
        return self

    def gather_with_go(self, wt_row, var_row, go_idx):
        """Return WT, variant and GO tensors for the same batch rows."""
        if self.go_features is None:
            raise RuntimeError("Call attach_go_embeddings() before gather_with_go()")
        if go_idx.ndim != 1 or go_idx.dtype != torch.long or len(go_idx) != len(wt_row) or len(go_idx) != len(var_row):
            raise ValueError("GO indices must be a 1-D torch.long tensor matching the pair batch")
        if go_idx.numel() and (int(go_idx.min()) < 0 or int(go_idx.max()) >= len(self.go_features)):
            raise IndexError("GO index out of range")
        wt, variant = self.gather(wt_row, var_row)
        return wt, variant, self.go_features[go_idx]


def _cache_signature(cfg: VariantDecoderConfig, wt_keys: list[str],
                     var_keys: list[str]) -> str:
    """Fold everything that should invalidate the cache into one hash.

    The key list is the product of the D6 filters — if a filter setting changes the
    hash changes too, so a stale cache is never read silently. The H5 files' size
    and mtime are included as well, so regenerating embeddings refreshes the cache
    automatically.
    """
    h = hashlib.sha256()
    # input_mode is REQUIRED here: otherwise the `both` cache would be read during
    # a `global` run and the ablation would silently become meaningless.
    h.update(f"{cfg.family}|{cfg.embedding_type}|{cfg.input_mode}|{cfg.input_dim}".encode())
    for path in (cfg.global_h5, cfg.local_h5,
                 cfg.global_h5_secondary, cfg.local_h5_secondary):
        if path:
            st = os.stat(path)
            h.update(f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"{len(wt_keys)}:{len(var_keys)}".encode())
    for k in wt_keys:
        h.update(k.encode())
    for k in var_keys:
        h.update(k.encode())
    return h.hexdigest()[:12]


def build_pair_tensors(df: pd.DataFrame, cfg: VariantDecoderConfig,
                       cache_dir: str | None = None,
                       verbose: bool = False) -> PairTensors:
    """DataFrame + H5 paths -> PairTensors.

    The WT local vector is read under the `{protein}_{pos}` key; it is NOT constant
    per protein but depends on the row's mutation position (DESIGN §3.1).

    If cache_dir is given, the assembled matrices are stored as `.npy`. Reading 80k
    individual datasets from H5 takes ~45 s, and since every sweep trial is a
    separate process that cost repeats per trial. The cache brings it down to
    seconds.
    """
    df = df.copy()
    use_secondary = cfg.embedding_type in SECONDARY_REQUIRED

    wt_keys_df = df[["protein_id", "wt_local_key"]].drop_duplicates("wt_local_key")
    wt_proteins = list(wt_keys_df["protein_id"])
    wt_key_list = list(wt_keys_df["wt_local_key"])
    var_keys = sorted(df["variant_key"].unique())

    wt_index = {k: i for i, k in enumerate(wt_key_list)}
    var_index = {k: i for i, k in enumerate(var_keys)}

    cached = None
    if cache_dir:
        sig = _cache_signature(cfg, wt_key_list, var_keys)
        stem = Path(cache_dir) / f"{cfg.family}_{cfg.embedding_type}_{sig}"
        wt_path, var_path = Path(f"{stem}_wt.npy"), Path(f"{stem}_var.npy")
        if wt_path.exists() and var_path.exists():
            if verbose:
                print(f"[CACHE] reading: {stem}_*.npy")
            cached = (np.load(wt_path), np.load(var_path))

    if cached is not None:
        wt_feat, var_feat = cached
    else:
        wt_feat, var_feat = _assemble(cfg, wt_proteins, wt_key_list, var_keys, use_secondary)
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            np.save(wt_path, wt_feat)
            np.save(var_path, var_feat)
            Path(f"{stem}_meta.json").write_text(json.dumps({
                "family": cfg.family, "embedding_type": cfg.embedding_type,
                "input_dim": int(wt_feat.shape[1]),
                "n_wt": len(wt_key_list), "n_var": len(var_keys),
                "global_h5": cfg.global_h5, "local_h5": cfg.local_h5,
            }, indent=2), encoding="utf-8")
            if verbose:
                print(f"[CACHE] wrote: {stem}_*.npy "
                      f"({(wt_feat.nbytes + var_feat.nbytes) / 1e9:.2f} GB)")

    if wt_feat.shape[1] != cfg.input_dim:
        raise ValueError(
            f"Input dim mismatch: {wt_feat.shape[1]} != cfg.input_dim={cfg.input_dim} "
            f"(embedding_type={cfg.embedding_type})"
        )

    df["wt_row"] = df["wt_local_key"].map(wt_index).astype(np.int64)
    df["var_row"] = df["variant_key"].map(var_index).astype(np.int64)

    return PairTensors(
        torch.from_numpy(np.ascontiguousarray(wt_feat)),
        torch.from_numpy(np.ascontiguousarray(var_feat)),
        df.reset_index(drop=True),
    )


def _assemble(cfg: VariantDecoderConfig, wt_proteins: list[str], wt_key_list: list[str],
              var_keys: list[str], use_secondary: bool) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the input matrices from the H5 files (called when no cache hit).

    Blocks are selected by input_mode. In `global` mode the local H5 is **never
    touched** — for the ablation to be honest the data must not merely go unused,
    it must not be read at all.

    Block order (for concat types: primary first, then raw):
        both    [pri_global | raw_global | pri_local | raw_local]
        global  [pri_global | raw_global]
        local   [pri_local  | raw_local ]
    """
    want_global = cfg.input_mode in ("both", "global")
    want_local = cfg.input_mode in ("both", "local")

    wt_blocks, var_blocks = [], []
    if want_global:
        wt_blocks.append(_read_keys(cfg.global_h5, wt_proteins, "WT global"))
        var_blocks.append(_read_keys(cfg.global_h5, var_keys, "varyant global"))
        if use_secondary:
            wt_blocks.append(_read_keys(cfg.global_h5_secondary, wt_proteins, "WT global (2)"))
            var_blocks.append(_read_keys(cfg.global_h5_secondary, var_keys, "varyant global (2)"))
    if want_local:
        wt_blocks.append(_read_keys(cfg.local_h5, wt_key_list, "WT lokal"))
        var_blocks.append(_read_keys(cfg.local_h5, var_keys, "varyant lokal"))
        if use_secondary:
            wt_blocks.append(_read_keys(cfg.local_h5_secondary, wt_key_list, "WT lokal (2)"))
            var_blocks.append(_read_keys(cfg.local_h5_secondary, var_keys, "varyant lokal (2)"))

    return (np.concatenate(wt_blocks, axis=1), np.concatenate(var_blocks, axis=1))


def iterate_batches(n: int, batch_size: int, shuffle: bool, generator=None):
    """Permutation batches — without DataLoader/worker overhead."""
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield order[start: start + batch_size]


def main():
    """CLI for pre-building the cache.

    Run once per (family, embedding type) before starting sweeps; after that every
    trial loads in seconds.
    """
    from phase_3_variant_decoder.config import EMBEDDING_TYPES, FAMILIES, INPUT_MODES
    from phase_3_variant_decoder.data.variant_dataset import prepare

    p = argparse.ArgumentParser(description="Build the Phase 3 pair-tensor cache")
    p.add_argument("--build_cache", action="store_true", required=True)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--embedding", choices=EMBEDDING_TYPES, required=True)
    p.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    p.add_argument("--global_h5", required=True)
    p.add_argument("--local_h5", required=True)
    p.add_argument("--global_h5_secondary", default=None)
    p.add_argument("--local_h5_secondary", default=None)
    p.add_argument("--cache_dir", required=True)
    args = p.parse_args()

    cfg = VariantDecoderConfig(family=args.family, embedding_type=args.embedding,
                               input_mode=args.input_mode)
    cfg.global_h5 = args.global_h5
    cfg.local_h5 = args.local_h5
    cfg.global_h5_secondary = args.global_h5_secondary
    cfg.local_h5_secondary = args.local_h5_secondary
    cfg.validate()

    df, go_vocab, report = prepare(cfg)
    print(report.render())
    tensors = build_pair_tensors(df, cfg, cache_dir=args.cache_dir, verbose=True)
    print(f"[DONE] wt={tuple(tensors.wt_feat.shape)} var={tuple(tensors.var_feat.shape)} "
          f"vocab={len(go_vocab):,}")


if __name__ == "__main__":
    main()
