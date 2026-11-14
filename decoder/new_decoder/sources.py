"""Read the existing native split directories or their original ZIP artifacts."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pandas as pd

from .config import FAMILIES, SPLITS


def sha256_file(path: str | Path) -> str:
    """Hash an artifact in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(source: str | Path, layout: str) -> Path:
    """Prefer an explicitly supplied layout, then one unambiguous ZIP.

    A repository ``data/splits`` directory contains ``multiclass/``,
    ``parallel/`` and ``cascade/``. Individual layout directories and the
    original layout ZIP files are also accepted. Ambiguous archives are rejected.
    """
    source = Path(source)
    if source.is_file() and source.suffix == ".zip":
        return source
    if (source / layout).is_dir():
        return source / layout
    first = "stage1_easy_train.tsv" if layout == "cascade" else "easy_train.tsv"
    if source.is_dir() and (source / first).is_file():
        return source
    matches = sorted(source.glob(f"{layout}-*.zip"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected {source / layout} or one {layout}-*.zip; found {len(matches)} archives"
        )
    return matches[0]


def read_source_table(
    source: str | Path,
    layout: str,
    family: str,
    split: str,
    *,
    stage: int | None = None,
    strings: bool = False,
) -> pd.DataFrame:
    """Read one native table without changing row order or source targets."""
    if family not in FAMILIES or split not in SPLITS:
        raise ValueError("Unknown split family or partition")
    if layout not in ("parallel", "multiclass", "cascade"):
        raise ValueError(f"Unknown source layout: {layout}")
    if (layout == "cascade" and stage not in (1, 2)) or (layout != "cascade" and stage is not None):
        raise ValueError("Only cascade tables take a stage, which must be 1 or 2")
    resolved = resolve_source(source, layout)
    name = f"{family}_{split}.tsv"
    if stage is not None:
        name = f"stage{stage}_{name}"
    options = dict(sep="\t", keep_default_na=False)
    if strings:
        options["dtype"] = str
    if resolved.is_dir():
        return pd.read_csv(resolved / name, **options)
    with zipfile.ZipFile(resolved) as archive:
        return pd.read_csv(io.BytesIO(archive.read(f"{layout}/{name}")), **options)


def source_fingerprint(source: str | Path, layout: str) -> str:
    """Fingerprint a ZIP or the complete, ordered native table collection."""
    resolved = resolve_source(source, layout)
    if resolved.is_file():
        return sha256_file(resolved)
    prefixes = ("stage1_", "stage2_") if layout == "cascade" else ("",)
    entries = {
        f"{prefix}{family}_{split}.tsv": sha256_file(resolved / f"{prefix}{family}_{split}.tsv")
        for prefix in prefixes
        for family in FAMILIES
        for split in SPLITS
    }
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
