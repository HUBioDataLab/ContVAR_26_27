"""Read-only, exhaustive coverage audit of the current sequence decoder inputs.

Reads source TSVs and the actual H5/GO artifacts, bypassing assembled NPY caches.
Raw rows excluded by the normal loader remain visible in the row manifest.
This checks availability and numeric integrity, not checkpoint/index provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from phase_3_variant_decoder.config import ESM_DIM, FAMILIES, MAX_SEQ_LEN
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable
from phase_3_variant_decoder.data.variant_dataset import (
    _load_source_frames, load_family, load_sequences, parse_snp,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_h5(path, required_keys, *, hash_file=False):
    """Read every requested unique vector, with exact (1280,) shape checks."""
    path = Path(path)
    keys = sorted(set(k for k in required_keys if isinstance(k, str) and k))
    states = {}
    stat = path.stat()
    with h5py.File(path, "r") as h5:
        for i, key in enumerate(keys, 1):
            if key not in h5:
                states[key] = "missing_key"
                continue
            try:
                item = h5[key]
                if not isinstance(item, h5py.Dataset):
                    states[key] = "not_a_dataset"
                elif item.shape != (ESM_DIM,):
                    states[key] = "wrong_shape"
                elif item.dtype.kind != "f":
                    states[key] = "not_floating_point"
                else:
                    vector = item[:]
                    if not np.isfinite(vector).all():
                        states[key] = "nonfinite"
                    elif not np.any(vector != 0):
                        states[key] = "zero_vector"
                    else:
                        states[key] = "valid"
            except (OSError, RuntimeError, ValueError, TypeError):
                states[key] = "unreadable_vector"
            if i % 10000 == 0:
                print(f"[H5] {path.name}: checked {i:,}/{len(keys):,}", flush=True)
        metadata = {"path": str(path.resolve()), "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "total_root_keys": len(h5),
                    "required_unique_keys": len(keys),
                    "expected_vector_shape": [ESM_DIM],
                    "root_attributes": {str(k): str(v) for k, v in h5.attrs.items()},
                    "status_counts": dict(Counter(states.values()))}
    if hash_file:
        metadata["sha256"] = sha256(path)
    return states, metadata


def load_rows(args):
    """Keep raw rows; identify eligibility using the real training loader."""
    sequences = load_sequences(args.sequences_json)
    groups, filters, sources = [], {}, {}
    for family in FAMILIES if args.family == "both" else (args.family,):
        frames, layout = _load_source_frames(family, args.dataset_dir)
        raw = pd.concat(frames, ignore_index=True)
        filtered, report = load_family(
            family, args.dataset_dir, sequences,
            max_seq_len=args.max_seq_len, drop_contradictory=args.drop_contradictory,
        )
        identity = ["protein_id", "snp", "matched_go_id"]
        raw["pipeline_eligible"] = pd.MultiIndex.from_frame(raw[identity]).isin(
            pd.MultiIndex.from_frame(filtered[identity]).unique())
        # The filters depend on these identities; duplicates must remain rows.
        pd.testing.assert_frame_equal(
            raw.loc[raw.pipeline_eligible, identity + ["label", "split"]].reset_index(drop=True),
            filtered[identity + ["label", "split"]].reset_index(drop=True),
        )
        raw["family"] = family
        raw["source_row"] = raw.groupby("split", sort=False).cumcount()
        raw["wt_global_key"] = raw.protein_id.astype(str)
        raw["variant_key"] = raw.protein_id.astype(str) + "_" + raw.snp.astype(str)
        parsed = raw.snp.map(parse_snp)
        raw["wt_local_key"] = [f"{pid}_{value[1]}" if value else ""
                               for pid, value in zip(raw.protein_id, parsed)]
        raw["sequence_length"] = raw.protein_id.map(
            lambda p: len(sequences[p]) if p in sequences else None)
        raw["prefix_truncated"] = raw.sequence_length.gt(args.max_seq_len)
        groups.append(raw)
        filters[family] = {"raw_rows": report.raw_rows, "kept_rows": report.kept_rows,
                           "steps": report.steps, "per_split": report.per_split}
        prefix = "stage1_" if layout == "cascade" else ""
        for split in ("train", "val", "test"):
            path = Path(args.dataset_dir) / f"{prefix}{family}_{split}.tsv"
            sources[f"{family}/{split}"] = {"path": str(path.resolve()), "sha256": sha256(path)}
    return pd.concat(groups, ignore_index=True), filters, sources


def audit(args):
    rows, filters, sources = load_rows(args)
    table = GOEmbeddingTable(args.go_embeddings, allow_proxies=args.allow_go_proxies)
    go_zero = set(table.go_ids[i] for i in np.flatnonzero(
        np.all(table.vectors.numpy() == 0, axis=1)))
    rows["go_status"] = ["missing_key" if g not in table.go_to_row else
                         "zero_vector" if g in go_zero else "valid"
                         for g in rows.matched_go_id]

    global_states, global_meta = audit_h5(
        args.global_h5, list(rows.wt_global_key) + list(rows.variant_key), hash_file=args.hash_h5)
    local_states, local_meta = audit_h5(
        args.local_h5, list(rows.wt_local_key) + list(rows.variant_key), hash_file=args.hash_h5)
    for name, keys, states in (
        ("wt_global", "wt_global_key", global_states),
        ("var_global", "variant_key", global_states),
        ("wt_local", "wt_local_key", local_states),
        ("var_local", "variant_key", local_states),
    ):
        rows[f"{name}_status"] = rows[keys].map(states).fillna("invalid_key")
    statuses = [f"{name}_status" for name in ("wt_global", "var_global", "wt_local", "var_local", "go")]
    rows["all_embeddings_valid"] = rows[statuses].eq("valid").all(axis=1)
    rows["decoder_ready"] = rows.pipeline_eligible & rows.all_embeddings_valid

    summaries = []
    for (family, split), group in rows.groupby(["family", "split"], sort=False):
        kept = group.loc[group.pipeline_eligible]
        summary = {"family": family, "split": split, "raw_rows": len(group),
                   "pipeline_excluded_rows": int((~group.pipeline_eligible).sum()),
                   "pipeline_rows": len(kept), "decoder_ready_rows": int(kept.decoder_ready.sum()),
                   "pipeline_rows_with_bad_or_missing_input": int((~kept.all_embeddings_valid).sum()),
                   "raw_rows_with_all_embeddings": int(group.all_embeddings_valid.sum()),
                   "pipeline_rows_with_truncated_context": int(kept.prefix_truncated.sum())}
        summary.update({f"{name}_bad_or_missing": int(kept[name].ne("valid").sum()) for name in statuses})
        summaries.append(summary)
        print(f"[{family.upper()}/{split}] raw={len(group):,}; excluded={summary['pipeline_excluded_rows']:,}; "
              f"pipeline={len(kept):,}; ready={summary['decoder_ready_rows']:,}; "
              f"bad/missing={summary['pipeline_rows_with_bad_or_missing_input']:,}", flush=True)

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    columns = ["family", "split", "source_row", "protein_id", "snp", "matched_go_id", "label",
               "pipeline_eligible", "sequence_length", "prefix_truncated", "wt_global_key",
               "wt_local_key", "variant_key", *statuses, "all_embeddings_valid", "decoder_ready"]
    rows[columns].to_csv(root / "sample_coverage.csv", index=False)
    rows.loc[rows.pipeline_eligible & ~rows.all_embeddings_valid, columns].to_csv(
        root / "missing_or_invalid_inputs.csv", index=False)
    pd.DataFrame(summaries).to_csv(root / "coverage_summary.csv", index=False)
    pd.DataFrame([{"artifact": name, "key": key, "status": status}
                  for name, states in (("global", global_states), ("local", local_states))
                  for key, status in states.items()]).to_csv(root / "vector_status.csv", index=False)
    kept = rows.loc[rows.pipeline_eligible]
    complete = bool(len(kept)) and bool(kept.all_embeddings_valid.all())
    report = {"complete_for_pipeline_rows": complete, "rows": summaries, "filters": filters,
              "source_files": sources, "sequences_json_sha256": sha256(args.sequences_json),
              "go": {"path": str(Path(args.go_embeddings).resolve()), "sha256": table.artifact_sha256,
                     "terms": len(table.go_ids), "dimension": table.dimension,
                     "metadata": table.metadata}, "global_h5": global_meta, "local_h5": local_meta,
              "scope": "Current primary [global | mutation-position] sequence inputs, not proposal window/pocket tokens.",
              "limitations": ["Key, shape and numeric integrity do not prove the correct backbone was used.",
                              "No live recomputation or assembled NPY-cache verification is performed.",
                              "source_row is a zero-based data row inside its split, not a physical TSV line.",
                              "Training-filtered rows are reported separately; no source data is changed."]}
    (root / "embedding_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[RESULT] {'PASS' if complete else 'FAIL'}: eligible rows have valid WT/VAR global/local and GO inputs.")
    print(f"[SAVED] {root.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=[*FAMILIES, "both"], default="both")
    for name in ("dataset_dir", "sequences_json", "global_h5", "local_h5", "go_embeddings", "out_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--drop_contradictory", action="store_true")
    parser.add_argument("--allow_go_proxies", action="store_true")
    parser.add_argument("--hash_h5", action="store_true", help="Also hash full H5 files; adds sequential disk I/O")
    args = parser.parse_args()
    for name in ("sequences_json", "global_h5", "local_h5", "go_embeddings"):
        if not getattr(args, name) or not Path(getattr(args, name)).is_file():
            parser.error(f"--{name} must name an existing file: {getattr(args, name)!r}")
    if not args.dataset_dir or not Path(args.dataset_dir).is_dir():
        parser.error("--dataset_dir must name the directory containing the split TSVs")
    report = audit(args)
    raise SystemExit(0 if report["complete_for_pipeline_rows"] else 1)


if __name__ == "__main__":
    main()
