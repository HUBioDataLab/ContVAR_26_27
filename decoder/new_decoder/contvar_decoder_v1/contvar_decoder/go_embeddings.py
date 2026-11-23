"""Anc2vec lookup and GO-conditioned decoder input assembly.

Original matched_go_id values are lookup keys, including explicitly recorded
aliases/proxies. Never assume the artifact order equals a run's go_idx order.

The project-native format is pickle-free. The selected external artifact stores
only its GO ID array as a legacy NumPy object pickle. It is accepted through a
restricted unpickler that can only reconstruct NumPy ndarray/dtype objects; the
general-purpose ``allow_pickle=True`` path is deliberately never used.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


_GO_ID = re.compile(r"GO:\d{7}")
_MAX_IDS_NPY_BYTES = 16 * 1024 * 1024
_MAX_ARRAY_NPY_BYTES = 512 * 1024 * 1024
_MAX_GO_TERMS = 200_000
_ALLOWED_MEMBERS = {"go_ids.npy", "embeddings.npy", "status.npy", "metadata_json.npy"}
_TRUSTED_LEGACY_SHA256 = {
    # Selected go_terms_embeddings_anc2vec.npz, audited 2026-09-30.
    "4572ad49d87865dccb80328f6fcec95ed06b103801e9fd212b940ff1dd380fe7",
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Decode old NumPy object arrays without permitting arbitrary imports."""

    def find_class(self, module, name):
        try:
            from numpy._core.multiarray import _reconstruct
        except ImportError:  # NumPy 1.x
            from numpy.core.multiarray import _reconstruct

        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Forbidden global in legacy GO ID array: {module}.{name}"
            ) from exc

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent IDs are forbidden")


def _npy_header(stream: io.BytesIO):
    version = np.lib.format.read_magic(stream)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
        (3, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ValueError(f"Unsupported NPY version: {version}")
    return readers[version](stream)


def _load_go_ids(raw: bytes, *, allow_legacy_object: bool) -> np.ndarray:
    if len(raw) > _MAX_IDS_NPY_BYTES:
        raise ValueError("GO ID array exceeds the safety size limit")
    stream = io.BytesIO(raw)
    shape, fortran, dtype = _npy_header(stream)
    if fortran or len(shape) != 1 or shape[0] > _MAX_GO_TERMS:
        raise ValueError(f"Invalid GO ID array shape/layout: {shape}")
    if dtype.hasobject:
        if not allow_legacy_object:
            raise ValueError(
                "Legacy object-array GO IDs are accepted only for an audited SHA-256; "
                "use the project-native pickle-free format for other files"
            )
        try:
            ids = _RestrictedNumpyUnpickler(stream).load()
        except (pickle.UnpicklingError, AttributeError, EOFError, ValueError, TypeError) as exc:
            raise ValueError(f"Unsafe or invalid legacy GO ID array: {exc}") from exc
    else:
        ids = np.load(io.BytesIO(raw), allow_pickle=False)
    if not isinstance(ids, np.ndarray) or ids.shape != shape or ids.ndim != 1:
        raise ValueError("GO ID payload does not match its NPY header")
    return ids


def _load_numeric_npy(raw: bytes, name: str) -> np.ndarray:
    if len(raw) > _MAX_ARRAY_NPY_BYTES:
        raise ValueError(f"{name} exceeds the safety size limit")
    try:
        return np.load(io.BytesIO(raw), allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"{name} must be a pickle-free NumPy array") from exc


def _load_npz(path: Path, artifact_sha256: str):
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate members in GO embedding NPZ")
            if not {"go_ids.npy", "embeddings.npy"}.issubset(names):
                raise ValueError("GO embedding NPZ requires go_ids and embeddings")
            unexpected = set(names) - _ALLOWED_MEMBERS
            if unexpected:
                raise ValueError(f"Unexpected GO embedding NPZ members: {sorted(unexpected)}")
            by_name = {i.filename: i for i in infos}
            if by_name["go_ids.npy"].file_size > _MAX_IDS_NPY_BYTES:
                raise ValueError("GO ID array exceeds the safety size limit")
            if by_name["embeddings.npy"].file_size > _MAX_ARRAY_NPY_BYTES:
                raise ValueError("Embedding array exceeds the safety size limit")
            ids = _load_go_ids(
                archive.read("go_ids.npy"),
                allow_legacy_object=artifact_sha256 in _TRUSTED_LEGACY_SHA256,
            )
            vectors = _load_numeric_npy(archive.read("embeddings.npy"), "embeddings")
            if "status.npy" in by_name:
                status = _load_numeric_npy(archive.read("status.npy"), "status").tolist()
            else:
                status = ["direct"] * len(ids)
            if "metadata_json.npy" in by_name:
                metadata_array = _load_numeric_npy(
                    archive.read("metadata_json.npy"), "metadata_json"
                )
                metadata = json.loads(str(metadata_array.item()))
            else:
                metadata = {
                    "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else None,
                    "source_format": "legacy_go_ids_embeddings_npz",
                    "source_file": path.name,
                }
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid GO embedding NPZ: {path}") from exc
    return ids.tolist(), vectors, status, metadata


class GOEmbeddingTable(nn.Module):
    def __init__(self, path: str | Path, *, allow_proxies: bool = False):
        super().__init__()
        path = Path(path)
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        ids, vectors, self.status, self.metadata = _load_npz(path, self.artifact_sha256)
        vectors = np.array(vectors, dtype=np.float32, copy=True)
        if (len(ids) != len(set(ids)) or not all(
                isinstance(g, str) and _GO_ID.fullmatch(g) for g in ids)):
            raise ValueError("GO embedding IDs must be unique GO:NNNNNNN strings")
        if (vectors.ndim != 2 or vectors.shape[0] != len(ids)
                or vectors.shape[1] != self.metadata["dimension"]
                or len(self.status) != len(ids) or not np.isfinite(vectors).all()):
            raise ValueError("Invalid GO embedding matrix or metadata")
        if not allow_proxies and any(s == "ancestor_mean_proxy" for s in self.status):
            raise ValueError("Artifact contains ancestor proxies. Pass allow_proxies=True explicitly, or use exact vectors.")
        self.go_ids = tuple(ids)
        self.go_to_row = {go: i for i, go in enumerate(ids)}
        self.dimension = vectors.shape[1]
        self.register_buffer("vectors", torch.from_numpy(vectors))

    def supported_mask(self, go_ids) -> np.ndarray:
        """Coverage only. Caller must explicitly decide whether to filter rows."""
        return np.asarray([go in self.go_to_row for go in go_ids], dtype=bool)

    def encode_ids(self, go_ids) -> torch.Tensor:
        ids = list(go_ids)
        missing = sorted(set(ids) - self.go_to_row.keys())
        if missing:
            raise KeyError(f"No GO embedding for {len(missing)} term(s): {missing[:10]}. See unresolved_terms.tsv; no fallback was applied.")
        return torch.tensor([self.go_to_row[g] for g in ids], dtype=torch.long, device=self.vectors.device)

    def lookup(self, go_ids) -> torch.Tensor:
        return self(self.encode_ids(go_ids))

    def aligned_to_vocab(self, go_vocab: dict[str, int]) -> torch.Tensor:
        """Return [N_GO, D_GO] indexed by this run's go_idx, not artifact order."""
        if (any(type(i) is not int for i in go_vocab.values())
                or set(go_vocab.values()) != set(range(len(go_vocab)))):
            raise ValueError("go_vocab indices must be distinct contiguous integers starting at 0")
        return self.lookup(sorted(go_vocab, key=go_vocab.get))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("GO indices must be a 1-D torch.long tensor")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= len(self.go_ids)):
            raise IndexError("GO embedding index out of range")
        return self.vectors[indices]


def decoder_inputs(wt: torch.Tensor, variant: torch.Tensor, go: torch.Tensor,
                   *, mode: str = "joint", include_delta: bool = True):
    """Prepare either proposed decoder architecture without changing the old one.

    joint:    [WT | variant | variant-WT (optional) | GO] -> one tensor
    separate: [WT | GO], [variant | GO] -> two tensors (shared scalar decoder)

    WT/variant may already contain global/local and primary/raw blocks from
    PairTensors.gather(). No dimension is hardcoded to a particular backbone.
    """
    if wt.ndim != 2 or variant.shape != wt.shape or go.ndim != 2 or go.shape[0] != wt.shape[0]:
        raise ValueError("Expected matching [B,D] WT/variant tensors and [B,G] GO tensor")
    if wt.device != variant.device or wt.device != go.device:
        raise ValueError("WT, variant and GO must be on the same device")
    go = go.to(dtype=wt.dtype)
    if mode == "separate":
        return torch.cat((wt, go), dim=-1), torch.cat((variant, go), dim=-1)
    if mode != "joint":
        raise ValueError(f"Unknown input mode: {mode}")
    blocks = [wt, variant]
    if include_delta:
        blocks.append(variant - wt)
    return torch.cat((*blocks, go), dim=-1)
