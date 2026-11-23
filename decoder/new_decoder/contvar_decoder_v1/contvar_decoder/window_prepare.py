"""Prepare matched none/mean cohorts from native splits and a frozen window cache."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pandas as pd

from .data import WORKSPACE, data_root, go_table
from .prepare import KEY, combinatorial_mask, overlap, sha256_file


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def masked_window_mean(window, mask):
    """Float32 accumulation, excluding padding (even nonfinite padded values)."""
    window, mask = np.asarray(window), np.asarray(mask)
    if window.ndim != 2 or mask.shape != (window.shape[0],) or window.shape[1] < 1:
        raise ValueError("Window must be [positions, features] with a matching mask")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Window mask must be binary")
    valid = mask.astype(bool)
    if not valid.any():
        raise ValueError("Window has no valid positions")
    selected = np.asarray(window[valid], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("Non-finite valid window feature")
    result = selected.mean(axis=0, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite window mean")
    return result


def find_window(root):
    paths = sorted((root / "embeddings").glob("phase3_seq_phase1_window*.h5"))
    if len(paths) != 1:
        raise ValueError(f"Expected one complete window file, found: {paths}")
    return paths[0]


def build_mean_cache(root, destination, input_hashes=None):
    root, destination = Path(root), Path(destination)
    source = find_window(root)
    hashes = input_hashes or {"window_raw": sha256_file(source),
                              "local": sha256_file(root / "embeddings/phase3_seq_phase1_local.h5")}
    if destination.exists():
        with h5py.File(destination, "r") as h:
            if (not h.attrs.get("complete", False) or h.attrs.get("pooling") != "masked_mean_float32"
                    or h.attrs.get("source_window_sha256") != hashes["window_raw"]
                    or h.attrs.get("source_mut_sha256") != hashes["local"]):
                raise ValueError("Existing mean cache belongs to different inputs; choose a new cache path")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Incomplete previous cache exists: {temporary}")
    started = time.monotonic()
    with h5py.File(source, "r") as windows, h5py.File(root / "embeddings/phase3_seq_phase1_local.h5", "r") as mut, \
            h5py.File(temporary, "w") as output:
        keys = sorted(k for k in windows if k != "mask")
        if set(keys) != set(windows["mask"]) or set(keys) != set(mut):
            raise ValueError("Window, mask and mutation-site keys differ")
        padded = 0
        output.attrs["pooling"] = "masked_mean_float32"
        output.attrs["source_window_sha256"] = hashes["window_raw"]
        output.attrs["source_mut_sha256"] = hashes["local"]
        output.attrs["source_window_name"] = source.name
        for index, key in enumerate(keys, 1):
            w, mask = windows[key][:], windows["mask"][key][:]
            if w.shape != (33, 1280) or mask.shape != (33,) or mask[16] != 1:
                raise ValueError(f"Invalid centered window: {key}")
            valid = mask.astype(bool)
            if np.any(w[~valid] != 0):
                raise ValueError(f"Nonzero window padding: {key}")
            center = mut[key][:]
            if center.shape != (1280,) or not np.array_equal(w[16], center.astype(w.dtype)):
                raise ValueError(f"Window center differs from mutation-site cache: {key}")
            output.create_dataset(key, data=masked_window_mean(w, mask), dtype="f4")
            padded += int(not valid.all())
            if index % 5000 == 0 or index == len(keys):
                print(f"WINDOW_MEAN {index}/{len(keys)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        output.attrs["rows"] = len(keys)
        output.attrs["padded_rows"] = padded
        output.attrs["complete"] = True
    temporary.replace(destination)
    return destination


def read_table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def validate_sources(root):
    checks, hashes = {}, {}
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            suffix = f"{family}_{split}.tsv"
            paths = {"multiclass": root / "splits/multiclass" / suffix,
                     "parallel": root / "splits/parallel" / suffix,
                     "stage1": root / "splits/cascade" / f"stage1_{suffix}",
                     "stage2": root / "splits/cascade" / f"stage2_{suffix}"}
            m, p, s1, s2 = [read_table(paths[k]) for k in ("multiclass", "parallel", "stage1", "stage2")]
            common = [c for c in m.columns if c != "target"]
            flags = dict(same_head_rows=m[common].equals(p[common]), multiclass_targets=m.target.equals(m.label),
                         target_lof=p.target_lof.equals(m.label.eq("LOF").astype(int).astype(str)),
                         target_gof=p.target_gof.equals(m.label.eq("GOF").astype(int).astype(str)),
                         cascade_rows=m[common].equals(s1[common]),
                         cascade_targets=s1.target.equals(m.label.map({"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"})),
                         cascade_effect_subset=m.loc[m.label.ne("NEUTRAL"), common].reset_index(drop=True).equals(s2[common]))
            if not all(flags.values()):
                raise ValueError(f"Native split mismatch: {family}/{split}: {flags}")
            checks[f"{family}/{split}"] = flags
            hashes.update({str(path.relative_to(root)): sha256_file(path) for path in paths.values()})
    return checks, hashes


def annotate(frames):
    train_terms = set(frames["train"].matched_go_id)
    combined = pd.concat(frames.values(), ignore_index=True)
    counts = combined.groupby(KEY).label.nunique()
    conflict_keys = set(counts[counts > 1].index)
    for frame in frames.values():
        frame["seen_in_train"] = frame.matched_go_id.isin(train_terms)
        frame["multiple_labels_for_input"] = [k in conflict_keys for k in frame[KEY].itertuples(index=False, name=None)]
    conflicts = combined.loc[[k in conflict_keys for k in combined[KEY].itertuples(index=False, name=None)]]
    return conflicts


def prepare_window(data, old_cohort, cache, out):
    root, old_cohort, cache, out = data_root(data), Path(old_cohort), Path(cache), Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new prepared output: {out}")
    source_window = find_window(root)
    input_paths = {"global": root / "embeddings/phase3_seq_phase1_global.h5",
                   "local": root / "embeddings/phase3_seq_phase1_local.h5",
                   "window_raw": source_window}
    print("Hashing frozen inputs...", flush=True)
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    build_mean_cache(root, cache, hashes)
    hashes["window_mean"] = sha256_file(cache)
    checks, source_hashes = validate_sources(root)
    raw_sequences = json.loads((root / "sequences/uniprot_dict.json").read_text(encoding="utf-8"))
    sequences = raw_sequences if isinstance(raw_sequences, dict) else {r["Entry"]: r["Sequence"] for r in raw_sequences if r.get("Sequence")}
    go = go_table(data)
    policy = dict(single_mutation_only=True, augmented_eval=False, train_augmentation=True,
                  multiple_labels="retain source labels and report groups", source_partition_assignments="preserved",
                  paired_window_comparison=False, pooling="masked_mean_float32",
                  matched_window_off_on_cohort=True,
                  old_cohort="canonical handover prepared_splits; not assumed identical to Gayem's unprovided run")
    collections = {cohort: {head: {} for head in ("multiclass", "parallel")} for cohort in ("old", "full")}
    exclusions, old_hashes = {}, {}
    with h5py.File(input_paths["global"], "r") as glob, h5py.File(input_paths["local"], "r") as mut, h5py.File(cache, "r") as window:
        handles = {"global": glob, "mut": mut, "window": window}
        validity = {}
        centered = bool(glob.attrs.get("center_long_sequences", False)) and bool(mut.attrs.get("center_long_sequences", False))

        def valid(scale, key):
            pair = (scale, key)
            if pair not in validity:
                handle = handles[scale]
                if key not in handle:
                    validity[pair] = False
                else:
                    value = handle[key][:]
                    validity[pair] = value.shape == (1280,) and np.issubdtype(value.dtype, np.floating) and bool(np.isfinite(value).all())
            return validity[pair]

        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                raw = {h: read_table(root / "splits" / h / f"{family}_{split}.tsv") for h in ("multiclass", "parallel")}
                mc = raw["multiclass"]
                combinations = combinatorial_mask(mc)
                augmented = mc.augmented.str.strip().str.lower()
                if not augmented.isin(("true", "false")).all():
                    raise ValueError("Missing/invalid augmented source flag")
                ready = {h: [] for h in raw}
                excluded = []
                for index, row in mc.iterrows():
                    reason, keys = None, None
                    match = re.fullmatch(r"([A-Z])([1-9]\d*)([A-Z])", row.snp)
                    sequence = sequences.get(row.protein_id)
                    if row.label not in ("LOF", "GOF", "NEUTRAL"):
                        raise ValueError(f"Unknown label: {row.label}")
                    if split != "train" and augmented.iloc[index] == "true":
                        reason = "evaluation_augmentation"
                    elif combinations.iloc[index]:
                        reason = "combinatorial_evidence"
                    elif match is None:
                        reason = "invalid_snp"
                    elif sequence is None:
                        reason = "missing_sequence"
                    else:
                        aa, pos, mutant = match[1], int(match[2]), match[3]
                        if pos > len(sequence):
                            reason = "position_beyond_sequence"
                        elif pos > 1022 and not centered:
                            reason = "position_beyond_uncentered_cache"
                        elif sequence[pos - 1] != aa:
                            reason = "wt_residue_mismatch"
                        elif aa == mutant:
                            reason = "not_a_substitution"
                        else:
                            variant = f"{row.protein_id}_{row.snp}"
                            keys = dict(wt_global_key=row.protein_id, var_global_key=variant,
                                        wt_mut_key=f"{row.protein_id}_{pos}", var_mut_key=variant,
                                        wt_window_key=f"{row.protein_id}_{pos}", var_window_key=variant)
                            missing = [name for name, key in keys.items() if not valid(name.split("_")[1], key)]
                            if missing:
                                reason = "missing_or_invalid_embedding:" + ",".join(missing)
                            elif row.matched_go_id not in go.go_to_row:
                                reason = "missing_go_embedding"
                    sample = f"{family}:{split}:{index}"
                    if reason:
                        excluded.append(dict(sample_id=sample, protein_id=row.protein_id, snp=row.snp,
                                             matched_go_id=row.matched_go_id, label=row.label, split=split,
                                             augmented=augmented.iloc[index] == "true", is_combinatorial=bool(combinations.iloc[index]), reason=reason))
                        continue
                    common = dict(sample_id=sample, split=split, protein_id=row.protein_id, snp=row.snp,
                                  matched_go_id=row.matched_go_id, label=row.label,
                                  label_idx=("LOF", "GOF", "NEUTRAL").index(row.label),
                                  augmented=augmented.iloc[index] == "true", is_combinatorial=False,
                                  match_source=row.match_source, context_truncated=len(sequence) > 1022,
                                  pos=pos, **keys)
                    for head in ready:
                        target = ({"target": raw[head].iloc[index].target} if head == "multiclass" else
                                  {"target_lof": int(raw[head].iloc[index].target_lof), "target_gof": int(raw[head].iloc[index].target_gof)})
                        ready[head].append(dict(common, **target))
                exclusions[f"{family}/{split}"] = pd.DataFrame(excluded)
                for head in ready:
                    frame = pd.DataFrame(ready[head])
                    old_path = old_cohort / head / f"{family}_{split}.tsv"
                    old_hashes[str(old_path.relative_to(old_cohort))] = sha256_file(old_path)
                    original = read_table(old_path)
                    old = frame.loc[frame.sample_id.isin(original.sample_id)].copy()
                    columns = ["sample_id", *KEY, "label"]
                    left = old[columns].astype(str).sort_values("sample_id").reset_index(drop=True)
                    right = original[columns].astype(str).sort_values("sample_id").reset_index(drop=True)
                    if not left.equals(right):
                        raise ValueError(f"Old canonical cohort cannot be reproduced: {family}/{split}/{head}")
                    collections["full"][head][f"{family}/{split}"] = frame
                    collections["old"][head][f"{family}/{split}"] = old
                print(f"PREPARE {family}/{split} full={len(frame)} old={len(old)} excluded={len(excluded)}", flush=True)
    out.mkdir(parents=True)
    for cohort, layouts in collections.items():
        manifest = dict(policy=policy, cohort=cohort, feature_sha256=hashes, go_sha256=go.artifact_sha256,
                        sequences_sha256=sha256_file(root / "sequences/uniprot_dict.json"),
                        source_sha256=source_hashes, source_layout_checks=checks, old_cohort_sha256=old_hashes,
                        source_data=str(root.resolve()), window_cache=str(cache.resolve()), prepared={})
        for head, tables in layouts.items():
            folder = out / cohort / head
            folder.mkdir(parents=True)
            audit = dict(single_mutation_only=True, policy=policy, feature_sha256=hashes,
                         go_sha256=go.artifact_sha256, source_layout=head, cohort=cohort, families={})
            for family in ("easy", "hard"):
                frames = {s: tables[f"{family}/{s}"] for s in ("train", "val", "test")}
                conflicts = annotate(frames)
                overlaps = overlap(frames)
                for pair, counts in overlaps.items():
                    if counts["protein_id+snp"] or (family == "hard" and counts["protein_id"]):
                        raise ValueError(f"Split leakage: {cohort}/{family}/{pair}")
                audit["families"][family] = dict(ready_overlap=overlaps, conflicting_ready_rows=len(conflicts), splits={})
                conflicts.to_csv(folder / f"{family}_conflicts.tsv", sep="\t", index=False)
                pd.concat([exclusions[f"{family}/{s}"] for s in frames], ignore_index=True).to_csv(folder / f"{family}_excluded.tsv", sep="\t", index=False)
                for split, frame in frames.items():
                    path = folder / f"{family}_{split}.tsv"
                    frame.to_csv(path, sep="\t", index=False)
                    entry = manifest["prepared"].setdefault(f"{family}/{split}", dict(rows=len(frame), labels=frame.label.value_counts().to_dict(),
                            augmented=int(frame.augmented.sum()), input_label_review_rows=int(frame.multiple_labels_for_input.sum()), paths={}, sha256={}))
                    entry["paths"][head] = str(path.relative_to(out / cohort))
                    entry["sha256"][head] = sha256_file(path)
                    audit["families"][family]["splits"][split] = dict(ready=len(frame), augmented_ready=int(frame.augmented.sum()),
                        labels_ready=frame.label.value_counts().to_dict(), excluded_full_source=dict(Counter(exclusions[f"{family}/{split}"].get("reason", []))))
            write_json(folder / "audit.json", audit)
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                m = layouts["multiclass"][f"{family}/{split}"]
                p = layouts["parallel"][f"{family}/{split}"]
                if not m[["sample_id", *KEY, "label"]].equals(p[["sample_id", *KEY, "label"]]):
                    raise ValueError("Different head cohorts")
        write_json(out / cohort / "manifest.json", manifest)
    write_json(out / "preparation.json", dict(policy=policy, feature_sha256=hashes, source_layout_checks=checks,
                old_cohort_sha256=old_hashes, cohorts={c: str((out / c).resolve()) for c in collections}))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=WORKSPACE.parent / "New_Decoder_Data_with_local_Context")
    p.add_argument("--old-cohort", type=Path, default=WORKSPACE / "prepared_splits")
    p.add_argument("--window-cache", type=Path, default=WORKSPACE / "cache/window_mean.h5")
    p.add_argument("--out", type=Path, default=WORKSPACE / "prepared_window")
    a = p.parse_args()
    prepare_window(a.data_root, a.old_cohort, a.window_cache, a.out)


if __name__ == "__main__":
    main()
