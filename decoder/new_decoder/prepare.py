"""Prepare native source splits against the existing frozen feature cache.

Keeps source partitions; writes every excluded row and reports augmentation,
label conflicts and split overlap. Does not generate new neutral labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_DIR,
    DEFAULT_SOURCE_DIR,
    FEATURE_DIM,
    MAX_SEQUENCE_LENGTH,
    DataPaths,
)
from .data import go_table
from .sources import read_source_table, resolve_source, sha256_file, source_fingerprint

DEFAULT_OUT = DEFAULT_PREPARED_DIR
KEY = ["protein_id", "snp", "matched_go_id"]


def combinatorial_mask(frame: pd.DataFrame) -> pd.Series:
    """A single SNP string alone does not establish single-mutant evidence."""
    if "is_combinatorial" not in frame:
        raise ValueError("Missing is_combinatorial; single-mutation evidence cannot be verified")
    flags = frame["is_combinatorial"].astype(str).str.strip().str.lower()
    if not flags.isin(("true", "false")).all():
        raise ValueError("is_combinatorial must explicitly contain True or False")
    return flags.eq("true")


def overlap(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Report protein, mutation and full-input overlap between partitions."""
    report = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        entry = {}
        for columns in (["protein_id"], ["protein_id", "snp"], KEY):
            left = set(frames[a][columns].itertuples(index=False, name=None))
            right = set(frames[b][columns].itertuples(index=False, name=None))
            entry["+".join(columns)] = len(left & right)
        report[f"{a}/{b}"] = entry
    return report


def prepare(
    source: str | Path,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    out: str | Path = DEFAULT_OUT,
    *,
    layout: str = "multiclass",
) -> dict[str, Any]:
    """Filter a source layout without relabeling or repartitioning examples."""
    if layout not in ("parallel", "multiclass"):
        raise ValueError(f"Unsupported training layout: {layout}")
    source, out = resolve_source(source, layout), Path(out)
    paths = DataPaths(Path(data_root))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already contains files; choose a new --out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    raw_sequences = json.loads(paths.sequences_json.read_text())
    sequences = (
        raw_sequences
        if isinstance(raw_sequences, dict)
        else {r["Entry"]: r["Sequence"] for r in raw_sequences if r.get("Sequence")}
    )
    table = go_table(data_root)
    report = dict(
        source=str(source.resolve()),
        source_sha256=source_fingerprint(source, layout),
        go_sha256=table.artifact_sha256,
        feature_sha256={
            "global": sha256_file(paths.global_h5),
            "local": sha256_file(paths.mutation_h5),
        },
        sequences_sha256=sha256_file(paths.sequences_json),
        labels=["LOF", "GOF", "NEUTRAL"],
        source_layout=layout,
        policy="Single mutation; source partitions and labels preserved; augmented rows excluded from val/test per head_split_report.txt; available-cache subset.",
        single_mutation_only=True,
        families={},
    )
    with h5py.File(paths.global_h5, "r") as glob, h5py.File(paths.mutation_h5, "r") as mut:
        caches = {"global": {}, "mut": {}}

        def valid(scale: str, key: str) -> bool:
            cache = caches[scale]
            if key not in cache:
                handle = mut if scale == "mut" else glob
                if key not in handle:
                    cache[key] = False
                else:
                    v = handle[key][:]
                    cache[key] = bool(
                        v.shape == (FEATURE_DIM,)
                        and np.issubdtype(v.dtype, np.floating)
                        and np.isfinite(v).all()
                    )
            return cache[key]

        for family in ("easy", "hard"):
            frames, ready, excluded, split_report = {}, {}, [], {}
            train_terms = set()
            for split in ("train", "val", "test"):
                member = f"{layout}/{family}_{split}.tsv"
                frame = read_source_table(source, layout, family, split)
                frame["augmented"] = frame["augmented"].astype(str).str.lower().eq("true")
                frame["is_combinatorial"] = combinatorial_mask(frame)
                if layout == "multiclass" and not frame["target"].equals(frame["label"]):
                    raise ValueError(f"Invalid multiclass targets in {member}")
                if layout == "parallel":
                    for label, column in (("LOF", "target_lof"), ("GOF", "target_gof")):
                        if not np.array_equal(frame[column], frame.label.eq(label).astype(int)):
                            raise ValueError(f"Invalid binary targets in {member}: {column}")
                frames[split] = frame
                rows, reasons = [], Counter()
                for index, row in frame.iterrows():
                    reason = None
                    match = re.fullmatch(r"([A-Z])([1-9]\d*)([A-Z])", row.snp)
                    sequence = sequences.get(row.protein_id)
                    if row.label not in report["labels"]:
                        raise ValueError(f"Unknown label in {member}:{index}")
                    if split != "train" and row.augmented:
                        reason = "evaluation_augmentation"
                    elif row.is_combinatorial:
                        reason = "combinatorial_evidence"
                    elif match is None:
                        reason = "invalid_snp"
                    elif sequence is None:
                        reason = "missing_sequence"
                    else:
                        aa, pos, mutant = match[1], int(match[2]), match[3]
                        if pos > MAX_SEQUENCE_LENGTH:
                            reason = "position_beyond_cache_1022"
                        elif pos > len(sequence):
                            reason = "position_beyond_sequence"
                        elif sequence[pos - 1] != aa:
                            reason = "wt_residue_mismatch"
                        elif aa == mutant:
                            reason = "not_a_substitution"
                        else:
                            variant = f"{row.protein_id}_{row.snp}"
                            keys = dict(
                                wt_global_key=row.protein_id,
                                var_global_key=variant,
                                wt_mut_key=f"{row.protein_id}_{pos}",
                                var_mut_key=variant,
                            )
                            bad = [
                                name
                                for name, key in keys.items()
                                if not valid("mut" if "_mut_" in name else "global", key)
                            ]
                            if bad:
                                reason = "missing_or_invalid_embedding:" + ",".join(bad)
                            elif row.matched_go_id not in table.go_to_row:
                                reason = "missing_go_embedding"
                    if reason:
                        reasons[reason] += 1
                        excluded.append(
                            dict(
                                sample_id=f"{family}:{split}:{index}",
                                split=split,
                                protein_id=row.protein_id,
                                snp=row.snp,
                                matched_go_id=row.matched_go_id,
                                label=row.label,
                                augmented=bool(row.augmented),
                                is_combinatorial=bool(row.is_combinatorial),
                                reason=reason,
                            )
                        )
                    else:
                        rows.append(
                            dict(
                                sample_id=f"{family}:{split}:{index}",
                                split=split,
                                protein_id=row.protein_id,
                                snp=row.snp,
                                matched_go_id=row.matched_go_id,
                                label=row.label,
                                label_idx=report["labels"].index(row.label),
                                augmented=bool(row.augmented),
                                match_source=row.match_source,
                                is_combinatorial=False,
                                **(
                                    {"target": row.target}
                                    if layout == "multiclass"
                                    else {
                                        "target_lof": int(row.target_lof),
                                        "target_gof": int(row.target_gof),
                                    }
                                ),
                                context_truncated=len(sequence) > MAX_SEQUENCE_LENGTH,
                                **keys,
                            )
                        )
                if not rows:
                    raise ValueError(f"No usable rows in {member}")
                prepared = pd.DataFrame(rows)
                if split == "train":
                    train_terms = set(prepared.matched_go_id)
                prepared["seen_in_train"] = prepared.matched_go_id.isin(train_terms)
                ready[split] = prepared
                prepared.to_csv(out / f"{family}_{split}.tsv", sep="\t", index=False)
                split_report[split] = dict(
                    raw=len(frame),
                    ready=len(prepared),
                    excluded=dict(reasons),
                    combinatorial_raw=int(frame.is_combinatorial.sum()),
                    combinatorial_ready=int(prepared.is_combinatorial.sum()),
                    augmented_raw=int(frame.augmented.sum()),
                    augmented_ready=int(prepared.augmented.sum()),
                    labels_ready=prepared.label.value_counts().to_dict(),
                )
                print(
                    f"{family}/{split}: raw={len(frame)}, ready={len(prepared)}, "
                    f"excluded={sum(reasons.values())}, augmented={int(prepared.augmented.sum())}",
                    flush=True,
                )
            all_ready = pd.concat(ready.values(), ignore_index=True)
            conflicts = all_ready.groupby(KEY).label.nunique()
            conflicts = conflicts[conflicts > 1]
            conflict_keys = set(conflicts.index)
            for split, frame in ready.items():
                frame["multiple_labels_for_input"] = [
                    key in conflict_keys for key in frame[KEY].itertuples(index=False, name=None)
                ]
                frame.to_csv(out / f"{family}_{split}.tsv", sep="\t", index=False)
                split_report[split]["multiple_label_rows"] = int(
                    frame.multiple_labels_for_input.sum()
                )
            conflict_rows = (
                all_ready.set_index(KEY).loc[conflicts.index].reset_index()
                if len(conflicts)
                else all_ready.iloc[:0]
            )
            conflict_rows.to_csv(out / f"{family}_conflicts.tsv", sep="\t", index=False)
            pd.DataFrame(excluded).to_csv(out / f"{family}_excluded.tsv", sep="\t", index=False)
            report["families"][family] = dict(
                splits=split_report,
                raw_overlap=overlap(frames),
                ready_overlap=overlap(ready),
                conflicting_ready_keys=len(conflicts),
                conflicting_ready_rows=len(conflict_rows),
                conflicts_policy="Single-mutation records with multiple labels retained for evidence review; counts do not establish annotation errors",
            )
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", choices=("multiclass", "parallel"), default="multiclass")
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Native split root, layout directory, or layout ZIP",
    )
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    prepare(args.source, args.data_root, args.out or DEFAULT_OUT / args.layout, layout=args.layout)


if __name__ == "__main__":
    main()
