"""Verify native head targets and prepare matching single-mutation cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_DIR, DEFAULT_SOURCE_DIR, FAMILIES, SPLITS
from .prepare import KEY, prepare
from .sources import read_source_table, sha256_file, source_fingerprint


def validate_source_layouts(source: str | Path) -> dict[str, dict[str, bool]]:
    """Verify equivalent source rows and native multiclass/parallel targets.

    Cascade is a validated reference layout, never extra training examples.
    """
    checks = {}
    for family in FAMILIES:
        for split in SPLITS:
            m = read_source_table(source, "multiclass", family, split, strings=True)
            p = read_source_table(source, "parallel", family, split, strings=True)
            s1 = read_source_table(source, "cascade", family, split, stage=1, strings=True)
            s2 = read_source_table(source, "cascade", family, split, stage=2, strings=True)
            common = [column for column in m.columns if column != "target"]
            row = dict(
                parallel_same_source_rows=m[common].equals(p[common]),
                multiclass_targets=m.target.equals(m.label),
                parallel_lof_targets=p.target_lof.equals(m.label.eq("LOF").astype(int).astype(str)),
                parallel_gof_targets=p.target_gof.equals(m.label.eq("GOF").astype(int).astype(str)),
                cascade_stage1_same_rows=m[common].equals(s1[common]),
                cascade_stage1_targets=s1.target.equals(
                    m.label.map({"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"})
                ),
                cascade_stage2_exact_effect_subset=m.loc[m.label.ne("NEUTRAL"), common]
                .reset_index(drop=True)
                .equals(s2[common]),
            )
            if not all(row.values()):
                raise ValueError(f"Source layout inconsistency: {family}/{split}: {row}")
            checks[f"{family}/{split}"] = row
    return checks


def prepare_all(
    source: str | Path = DEFAULT_SOURCE_DIR,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    out: str | Path = DEFAULT_PREPARED_DIR,
) -> dict[str, Any]:
    """Create both native layouts and pin input and output fingerprints."""
    source, out = Path(source), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose a new prepared directory: {out}")
    checks = validate_source_layouts(source)
    reports = {
        layout: prepare(source, data_root, out / layout, layout=layout)
        for layout in ("parallel", "multiclass")
    }
    tables = {}
    for family in FAMILIES:
        for split in SPLITS:
            paths = {head: out / head / f"{family}_{split}.tsv" for head in reports}
            frames = {head: pd.read_csv(path, sep="\t") for head, path in paths.items()}
            m, p = frames["multiclass"], frames["parallel"]
            columns = [
                "sample_id",
                *KEY,
                "label",
                "wt_global_key",
                "var_global_key",
                "wt_mut_key",
                "var_mut_key",
            ]
            if not m[columns].equals(p[columns]):
                raise ValueError(f"Prepared models have different cohorts: {family}/{split}")
            if m.is_combinatorial.any() or (split != "train" and m.augmented.any()):
                raise ValueError(f"Forbidden prepared rows: {family}/{split}")
            tables[f"{family}/{split}"] = dict(
                rows=len(m),
                labels=m.label.value_counts().to_dict(),
                augmented=int(m.augmented.sum()),
                input_label_review_rows=int(m.multiple_labels_for_input.sum()),
                paths={head: path.relative_to(out).as_posix() for head, path in paths.items()},
                sha256={head: sha256_file(path) for head, path in paths.items()},
            )
        for pair, counts in reports["multiclass"]["families"][family]["ready_overlap"].items():
            if counts["protein_id+snp"] or (family == "hard" and counts["protein_id"]):
                raise ValueError(f"Unexpected split overlap: {family}/{pair}")
    manifest = dict(
        source_sha256={
            layout: source_fingerprint(source, layout)
            for layout in ("parallel", "multiclass", "cascade")
        },
        source_layout_checks=checks,
        prepared=tables,
        policy=dict(
            single_mutation_only=True,
            augmented_eval=False,
            multiple_labels="retain source labels and report groups",
            source_partition_assignments="preserved",
            missing_inputs="explicitly excluded with reasons",
            parallel_source="native parallel layout",
            multiclass_source="native multiclass layout",
            cascade="validated reference layout; not used for the four requested models",
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        "Verified native targets, matching cohorts, no variant leakage, and no augmented evaluation rows.",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREPARED_DIR)
    args = parser.parse_args()
    prepare_all(args.source, args.data_root, args.out)


if __name__ == "__main__":
    main()
