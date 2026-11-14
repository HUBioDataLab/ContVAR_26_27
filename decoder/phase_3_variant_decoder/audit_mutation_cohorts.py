"""Reconcile raw supervisor cohorts with model input filters, without inference.

No source TSV is changed. Stable source_row is a zero-based data-row number
within a split (not a physical line number when TSV fields contain newlines).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from phase_3_variant_decoder.data.variant_dataset import (
    _load_source_frames, load_family, load_sequences, parse_snp,
)
from phase_3_variant_decoder.mutation_sensitivity import (
    GROUP, VARIANT, ORDER, lookup_annotations, pair_counts, clean_variants,
)


def trace_filters(raw, sequences, max_seq_len=1022, drop_contradictory=False):
    """Mirror loader order; the caller verifies surviving rows against the loader."""
    out = raw.copy().reset_index(drop=True)
    out["source_row"] = out.groupby("split", sort=False).cumcount()
    out["exclusion_reason"] = ""
    parsed = out.snp.map(parse_snp)
    out["pos"] = [v[1] if v else None for v in parsed]
    steps = []

    def remove(reason, invalid):
        newly = out.exclusion_reason.eq("") & invalid
        out.loc[newly, "exclusion_reason"] = reason
        kept_test = out.loc[out.split.eq("test") & out.exclusion_reason.eq("")]
        steps.append({"step": reason, "removed_test_rows": int((newly & out.split.eq("test")).sum()),
                      "test_rows": len(kept_test), **pair_counts(kept_test)})

    remove("malformed SNP notation", parsed.isna())
    remove("protein has no sequence", ~out.protein_id.isin(sequences))
    remove(f"pos > {max_seq_len}", out.pos.gt(max_seq_len))
    lengths = out.protein_id.map(lambda p: len(sequences.get(p, "")))
    remove("position beyond sequence", out.pos.gt(lengths))
    valid = out.exclusion_reason.eq("")
    mismatch = pd.Series(False, index=out.index)
    mismatch.loc[valid] = [sequences[p][v[1] - 1] != v[0]
                          for p, v in zip(out.loc[valid, "protein_id"], parsed[valid])]
    remove("WT residue mismatch", mismatch)
    if drop_contradictory:
        valid = out.exclusion_reason.eq("")
        conflict = pd.Series(False, index=out.index)
        conflict.loc[valid] = out.loc[valid].groupby(VARIANT).label.transform("nunique").gt(1)
        remove("contradictory label (optional)", conflict)
    return out, steps


def easy_membership(traced):
    raw_train = traced.loc[traced.split.eq("train")]
    raw_test = traced.loc[traced.split.eq("test")]
    train = raw_train.loc[raw_train.exclusion_reason.eq("")]
    test = raw_test.loc[raw_test.exclusion_reason.eq("")]
    manifest = lookup_annotations(raw_train, raw_test).rename(columns={
        "anti_majority": "raw_anti_majority", "train_pair_majority": "raw_train_majority",
        "lookup_prediction": "raw_lookup_prediction"})
    filtered = lookup_annotations(train, test)
    rules = filtered.set_index("source_row")
    manifest["filtered_anti_majority"] = manifest.source_row.map(rules.anti_majority).eq(True)
    manifest["filtered_train_majority"] = manifest.source_row.map(rules.train_pair_majority).fillna("")
    manifest["model_covered"] = manifest.exclusion_reason.eq("")
    frozen = manifest.raw_anti_majority & manifest.model_covered
    recomputed = manifest.filtered_anti_majority

    def labels(mask):
        return {k: int(v) for k, v in manifest.loc[mask, "label"].value_counts().items()}

    label_sets = raw_train.groupby(GROUP).label.agg(set).to_dict()
    any_opposite, absent, tied, alphabetical = [], [], [], []
    counts = {key: Counter(g.label) for key, g in raw_train.groupby(GROUP, sort=False)}
    for row in manifest.itertuples(index=False):
        key = (row.protein_id, row.matched_go_id)
        ls, cs = label_sets.get(key, set()), counts.get(key, {})
        winners = sorted(k for k, v in cs.items() if v == max(cs.values())) if cs else []
        any_opposite.append(bool(ls - {row.label}))
        absent.append(bool(ls) and row.label not in ls)
        tied.append(len(winners) > 1)
        alphabetical.append(bool(winners) and row.label != winners[0])
    manifest["train_majority_tied"] = tied
    manifest["any_opposite_train_label"] = any_opposite
    manifest["label_absent_in_train_group"] = absent
    return {
        "raw_anti_majority": int(manifest.raw_anti_majority.sum()),
        "raw_labels": labels(manifest.raw_anti_majority),
        "raw_cohort_covered": int(frozen.sum()), "covered_labels": labels(frozen),
        "raw_cohort_excluded_by_reason": manifest.loc[manifest.raw_anti_majority & ~manifest.model_covered,
                                                    "exclusion_reason"].value_counts().to_dict(),
        "filtered_recomputed_cohort": int(recomputed.sum()), "recomputed_labels": labels(recomputed),
        "cohort_added_by_train_filtering": int((recomputed & ~frozen).sum()),
        "cohort_removed_by_train_filtering": int((frozen & ~recomputed).sum()),
        "changed_membership_rows": manifest.loc[frozen ^ recomputed,
            ["source_row", *VARIANT, "label", "raw_train_majority", "filtered_train_majority"]].to_dict("records"),
        "test_rows_in_tied_train_groups": sum(tied),
        "raw_anti_majority_in_tied_train_groups": int((manifest.raw_anti_majority & manifest.train_majority_tied).sum()),
        "any_opposite_train_label_rows": sum(any_opposite),
        "test_label_absent_in_train_group_rows": sum(absent),
        "alphabetical_tie_break_anti_majority": sum(alphabetical),
        "anti_majority_exact_variant_GO_seen_in_train": int((manifest.raw_anti_majority & manifest.exact_variant_GO_seen_in_train).sum()),
    }, manifest


def pair_manifest(test):
    """Enumerate ORIGINAL cross-label row pairs, preserving duplicates/conflicts."""
    rows = []
    covered = test.loc[test.exclusion_reason.eq("")]
    clean, _ = clean_variants(covered)
    clean_ids = set(clean.source_row)
    for (protein, go), group in test.groupby(GROUP, sort=False):
        if group.label.nunique() < 2:
            continue
        by_label = {label: g.to_dict("records") for label, g in group.groupby("label")}
        for lower, higher in combinations(ORDER, 2):
            for a in by_label.get(lower, []):
                for b in by_label.get(higher, []):
                    rows.append({"protein_id": protein, "matched_go_id": go,
                        "lower_source_row": a["source_row"], "higher_source_row": b["source_row"],
                        "lower_snp": a["snp"], "higher_snp": b["snp"],
                        "lower_label": lower, "higher_label": higher,
                        "lower_exclusion": a["exclusion_reason"], "higher_exclusion": b["exclusion_reason"],
                        "model_covered": not a["exclusion_reason"] and not b["exclusion_reason"],
                        "same_variant": a["snp"] == b["snp"],
                        "in_clean_subset": a["source_row"] in clean_ids and b["source_row"] in clean_ids})
    columns = GROUP + ["lower_source_row", "higher_source_row", "lower_snp", "higher_snp",
        "lower_label", "higher_label", "lower_exclusion", "higher_exclusion", "model_covered",
        "same_variant", "in_clean_subset"]
    return pd.DataFrame(rows, columns=columns)


def audit_family(family, dataset_dir, sequences, max_seq_len=1022, drop_contradictory=False):
    frames, layout = _load_source_frames(family, dataset_dir)
    raw = pd.concat(frames, ignore_index=True)
    traced, steps = trace_filters(raw, sequences, max_seq_len, drop_contradictory)
    actual, _ = load_family(family, dataset_dir, sequences, max_seq_len, drop_contradictory)
    expected = traced.loc[traced.exclusion_reason.eq(""), raw.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual.loc[:, raw.columns], check_dtype=False)
    test = traced.loc[traced.split.eq("test")].copy()
    covered = test.loc[test.exclusion_reason.eq("")]
    unique = covered.drop_duplicates(VARIANT + ["label"])
    clean, cleaning = clean_variants(covered)
    pairs = pair_manifest(test)
    assert len(pairs) == pair_counts(test)["different_label_row_pairs"]
    assert int(pairs.model_covered.sum()) == pair_counts(covered)["different_label_row_pairs"]
    assert int(pairs.in_clean_subset.sum()) == pair_counts(clean)["different_label_row_pairs"]
    audit = {"family": family, "source_layout": layout,
        "raw_split_rows": raw.split.value_counts().to_dict(),
        "filtered_split_rows": actual.split.value_counts().to_dict(),
        "waterfall": [{"step": "Raw TSV", "removed_test_rows": 0, "test_rows": len(test), **pair_counts(test)},
                      *steps,
                      {"step": "Deduplicate variant-GO-label", "removed_test_rows": len(covered)-len(unique),
                       "test_rows": len(unique), **pair_counts(unique)},
                      {"step": "Remove conflicting variant-GO", "removed_test_rows": len(unique)-len(clean),
                       "test_rows": len(clean), **pair_counts(clean)}],
        "cleaning": cleaning, "raw_cleaning": clean_variants(test)[1],
        "raw_same_variant_cross_label_pairs": int(pairs.same_variant.sum()),
        "covered_same_variant_cross_label_pairs": int((pairs.same_variant & pairs.model_covered).sum()),
        "raw_pair_types": pairs.groupby(["lower_label", "higher_label"]).size().rename_axis(["lower", "higher"]).reset_index(name="n").to_dict("records"),
        "pair_exclusion_combinations": pairs.loc[~pairs.model_covered].groupby(["lower_exclusion", "higher_exclusion"], dropna=False).size().reset_index(name="pairs").to_dict("records"),
        "protein_overlap_train_test": len(set(raw.loc[raw.split.eq("train"), "protein_id"]) & set(test.protein_id)),
    }
    if family == "easy":
        audit["easy"], test = easy_membership(traced)
    return audit, test, pairs, traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--sequences_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--legacy_dataset_dir")
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--drop_contradictory", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(args.sequences_json)
    result = {"format": "contvar_mutation_cohort_audit_v1", "config": vars(args), "families": {},
              "selection_rule": "EASY: raw train protein-GO majority (first-seen tie break); HARD: all raw cross-label row pairs"}
    sources = [Path(args.sequences_json), *sorted(Path(args.dataset_dir).glob("*.tsv"))]
    result["source_sha256"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for family in ("easy", "hard"):
        data, test, pairs, traced = audit_family(family, args.dataset_dir, sequences, args.max_seq_len, args.drop_contradictory)
        if args.legacy_dataset_dir:
            old, _ = _load_source_frames(family, args.legacy_dataset_dir)
            new, _ = _load_source_frames(family, args.dataset_dir)
            data["legacy_matches_cascade_ordered_rows"] = [
                a[a.columns.intersection(b.columns)].fillna("").equals(b[a.columns.intersection(b.columns)].fillna(""))
                for a, b in zip(old, new)]
        test.to_csv(root / f"{family}_test_row_audit.csv", index=False)
        pairs.to_csv(root / f"{family}_raw_pair_manifest.csv", index=False)
        result["families"][family] = data
        print(json.dumps(data, ensure_ascii=False, indent=2))
    (root / "cohort_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
