"""Post-hoc mutation diagnostics. No fitting or threshold selection on test.

EASY: challenge a train-only protein/GO majority rule.
HARD: rank distinct variants within the same test protein/GO group.
All functions here operate on frames, independently of GPU/checkpoint loading.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from phase_3_variant_decoder.config import LABELS

GROUP = ["protein_id", "matched_go_id"]
VARIANT = GROUP + ["snp"]
ORDER = {"LOF": 0, "NEUTRAL": 1, "GOF": 2}
PAIR_COLUMNS = [*GROUP, "lower_snp", "higher_snp", "lower_label", "higher_label",
                "comparison", "same_position", "score_gap", "credit", "tie",
                "stage_test", "stage_gap", "stage_credit", "stage_tie",
                "lower_source_row", "higher_source_row", "same_variant"]


def validate_frame(frame: pd.DataFrame) -> None:
    required = VARIANT + ["label"]
    if set(required) - set(frame):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(frame))}")
    if frame[required].isna().any().any() or not frame.label.isin(LABELS).all():
        raise ValueError("Missing identifiers or invalid LOF/GOF/NEUTRAL labels")


def majority_rules(train: pd.DataFrame, keys: list[str]) -> dict:
    """Equal counts: first label encountered in original train row order wins."""
    rules = {}
    for key, group in train.groupby(keys, sort=False):
        # Always tuple keys, including the single-GO case (pandas-version safe).
        if not isinstance(key, tuple):
            key = (key,)
        rules[key] = Counter(group.label).most_common(1)[0][0]
    return rules


def lookup_annotations(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    validate_frame(train)
    validate_frame(test)
    if train.empty:
        raise ValueError("Train partition is empty; cannot build train-only lookup")
    pair_rules = majority_rules(train, GROUP)
    go_rules = majority_rules(train, ["matched_go_id"])
    counts = train.label.value_counts()
    # Match the retained GO lookup helper's global tie rule (LABELS order).
    fallback = max(LABELS, key=lambda label: counts.get(label, 0))
    out = test.copy().reset_index(drop=True)
    keys = list(out[GROUP].itertuples(index=False, name=None))
    out["train_pair_seen"] = [key in pair_rules for key in keys]
    out["train_pair_majority"] = [pair_rules.get(key, "") for key in keys]
    out["lookup_prediction"] = [pair_rules.get(key, go_rules.get((key[1],), fallback))
                                for key in keys]
    out["lookup_source"] = ["protein_GO" if key in pair_rules else
                            "GO" if (key[1],) in go_rules else "train_global"
                            for key in keys]
    out["anti_majority"] = out.train_pair_seen & (out.label != out.train_pair_majority)
    train_variants = set(train[VARIANT].itertuples(index=False, name=None))
    out["exact_variant_GO_seen_in_train"] = [
        key in train_variants for key in out[VARIANT].itertuples(index=False, name=None)]
    out["conflicting_test_variant_GO"] = out.groupby(VARIANT).label.transform("nunique") > 1
    return out


def class_metrics(frame: pd.DataFrame, column: str) -> dict:
    n = len(frame)
    result = {"n": n, "correct_n": int((frame.label == frame[column]).sum()),
              "accuracy": float((frame.label == frame[column]).mean()) if n else None,
              "macro_f1": None}
    if n:
        precision, recall, f1, _ = precision_recall_fscore_support(
            frame.label, frame[column], labels=LABELS, zero_division=0)
        result["macro_f1"] = float(f1.mean())
    for i, label in enumerate(LABELS):
        result[f"{label.lower()}_n"] = int((frame.label == label).sum())
        for name, values in (("precision", precision if n else []),
                             ("recall", recall if n else []), ("f1", f1 if n else [])):
            result[f"{label.lower()}_{name}"] = float(values[i]) if n else None
    return result


def easy_diagnostic(train: pd.DataFrame, test: pd.DataFrame, *, reference_train=None,
                    model_name="Cascade") -> tuple[dict, pd.DataFrame]:
    # Freeze the challenge against RAW train so model coverage changes cannot
    # redefine the supervisor's cohort. Filtered-train rule remains a secondary
    # diagnostic: it is the rule the model could learn from its actual training.
    frame = lookup_annotations(train if reference_train is None else reference_train, test)
    filtered = lookup_annotations(train, test)
    frame["filtered_train_anti_majority"] = filtered.anti_majority
    frame["filtered_train_lookup_prediction"] = filtered.lookup_prediction
    frame["filtered_train_pair_majority"] = filtered.train_pair_majority
    # A supplementary stricter cohort: no exact training variant, no conflicting
    # test label, each variant/GO appears once. Primary remains the audit's rule.
    frame["anti_majority_clean"] = (frame.anti_majority &
        ~frame.exact_variant_GO_seen_in_train & ~frame.conflicting_test_variant_GO &
        ~frame.duplicated(VARIANT))
    rows = []
    for cohort, mask in (("Full test", np.ones(len(frame), dtype=bool)),
                         ("Anti-majority", frame.anti_majority),
                         ("Anti-majority / clean novel variants", frame.anti_majority_clean)):
        lookup_name = "Protein + GO lookup" if reference_train is None else "Raw-train protein + GO lookup"
        for model, column in ((model_name, "prediction"), (lookup_name, "lookup_prediction")):
            rows.append({"cohort": cohort, "model": model, **class_metrics(frame.loc[mask], column)})
    if reference_train is not None:
        for model, column in ((model_name, "prediction"), ("Filtered-train protein + GO lookup", "filtered_train_lookup_prediction")):
            rows.append({"cohort": "Anti-majority / filtered-train rule", "model": model,
                         **class_metrics(frame.loc[frame.filtered_train_anti_majority], column)})
    audit = {
        "test_rows": len(frame), "train_pair_seen_rows": int(frame.train_pair_seen.sum()),
        "anti_majority_rows": int(frame.anti_majority.sum()),
        "anti_majority_class_counts": {label: int((frame.anti_majority & (frame.label == label)).sum())
                                       for label in LABELS},
        "anti_majority_exact_variant_seen_rows": int((frame.anti_majority & frame.exact_variant_GO_seen_in_train).sum()),
        "anti_majority_conflicting_test_rows": int((frame.anti_majority & frame.conflicting_test_variant_GO).sum()),
        "clean_novel_anti_majority_variants": int(frame.anti_majority_clean.sum()),
        "reference_train": "raw" if reference_train is not None else "supplied train",
        "filtered_train_anti_majority_rows": int(frame.filtered_train_anti_majority.sum()),
    }
    return {"audit": audit, "classification": rows}, frame


def pair_counts(frame: pd.DataFrame) -> dict:
    """Count row pairs before cleaning; duplicates can inflate this quantity."""
    counts = frame.groupby(GROUP + ["label"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    values = counts.to_numpy(dtype=np.int64)
    pairs = (values.sum(axis=1) ** 2 - (values ** 2).sum(axis=1)) // 2
    return {"different_label_groups": int((pairs > 0).sum()),
            "different_label_row_pairs": int(pairs.sum())}


def clean_variants(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_frame(frame)
    unique = frame.drop_duplicates(VARIANT + ["label"]).copy()
    conflict = unique.groupby(VARIANT).label.transform("nunique") > 1
    audit = {"input_rows": len(frame), **pair_counts(frame),
             "duplicate_variant_GO_label_rows_removed": len(frame) - len(unique),
             "conflicting_variant_GO_keys_removed": len(unique.loc[conflict, VARIANT].drop_duplicates()),
             "conflicting_unique_label_rows_removed": int(conflict.sum())}
    clean = unique.loc[~conflict].reset_index(drop=True)
    audit.update({"clean_variants": len(clean),
                  **{f"clean_{k}": v for k, v in pair_counts(clean).items()}})
    return clean, audit


def signed_index(effect, direction):
    """Soft GOF mass minus LOF mass; decision index, NOT measured activity."""
    effect, direction = np.asarray(effect, dtype=float), np.asarray(direction, dtype=float)
    if effect.shape != direction.shape or effect.ndim != 1:
        raise ValueError("Stage scores must be aligned one-dimensional arrays")
    if not all(np.isfinite(x).all() and ((0 <= x) & (x <= 1)).all() for x in (effect, direction)):
        raise ValueError("Stage sigmoid scores must be finite and in [0,1]")
    return effect * (1 - 2 * direction)


def _credit(gap: float, tolerance: float) -> tuple[float, bool]:
    tie = abs(gap) <= tolerance
    return (0.5 if tie else float(gap > 0)), tie


def ranking_summary(pairs: pd.DataFrame, credit="credit", tie="tie", gap="score_gap") -> dict:
    if pairs.empty:
        return {"n_pairs": 0, "n_groups": 0, "pair_accuracy": None, "group_macro_accuracy": None,
                "tie_rate": None, "correct_pairs": 0, "tied_pairs": 0, "incorrect_pairs": 0,
                "mean_score_gap": None, "constant_score_reference": 0.5}
    return {"n_pairs": len(pairs), "n_groups": pairs.groupby(GROUP).ngroups,
            "pair_accuracy": float(pairs[credit].mean()),
            "group_macro_accuracy": float(pairs.groupby(GROUP)[credit].mean().mean()),
            "tie_rate": float(pairs[tie].mean()),
            "correct_pairs": int((pairs[credit] == 1).sum()),
            "tied_pairs": int(pairs[tie].sum()), "incorrect_pairs": int((pairs[credit] == 0).sum()),
            "mean_score_gap": float(pairs[gap].mean()), "constant_score_reference": 0.5}


def pair_diagnostic(test: pd.DataFrame, tolerance: float = 1e-8, *, clean=True,
                    score_column=None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    unique, audit = clean_variants(test)
    frame = unique if clean else test.copy()
    audit["evaluation_cohort"] = "Clean distinct variants" if clean else "Covered original row pairs"
    if score_column is None:
        frame["signed_index"] = signed_index(frame.effect_score, frame.direction_score)
    else:
        scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Pair ranking requires finite decision indices")
        frame["signed_index"] = scores
    rows = []
    for (protein, go), group in frame.groupby(GROUP, sort=False):
        if group.label.nunique() < 2:
            continue
        # Enumerate only cross-label pairs, not all O(n^2) same-label pairs.
        by_label = {label: sub.to_dict("records") for label, sub in group.groupby("label")}
        for lower, higher in combinations(ORDER, 2):
            for a in by_label.get(lower, []):
                for b in by_label.get(higher, []):
                    gap = b["signed_index"] - a["signed_index"]
                    credit, tie = _credit(gap, tolerance)
                    if score_column is not None:
                        # Generic three-class scores do not have cascade heads.
                        stage, stage_gap, stage_credit, stage_tie = None, None, None, None
                    elif lower == "LOF" and higher == "GOF":
                        stage, stage_gap = "Direction: LOF > GOF", a["direction_score"] - b["direction_score"]
                    elif lower == "LOF":
                        stage, stage_gap = "Effect: LOF > NEUTRAL", a["effect_score"] - b["effect_score"]
                    else:
                        stage, stage_gap = "Effect: GOF > NEUTRAL", b["effect_score"] - a["effect_score"]
                    if score_column is None:
                        stage_credit, stage_tie = _credit(stage_gap, tolerance)
                    rows.append({"protein_id": protein, "matched_go_id": go,
                        "lower_snp": a["snp"], "higher_snp": b["snp"],
                        "lower_label": lower, "higher_label": higher, "comparison": f"{lower} < {higher}",
                        "same_position": a["snp"] != b["snp"] and a.get("pos") is not None and a.get("pos") == b.get("pos"),
                        "score_gap": gap, "credit": credit, "tie": tie,
                        "stage_test": stage, "stage_gap": stage_gap, "stage_credit": stage_credit, "stage_tie": stage_tie,
                        "lower_source_row": a.get("source_row"), "higher_source_row": b.get("source_row"),
                        "same_variant": a["snp"] == b["snp"]})
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    ranking = [{"comparison": "All different-label pairs", **ranking_summary(pairs)}]
    for lower, higher in combinations(ORDER, 2):
        name = f"{lower} < {higher}"
        ranking.append({"comparison": name, **ranking_summary(pairs.loc[pairs.comparison == name])})
    ranking.append({"comparison": "Same-position pairs only", **ranking_summary(pairs.loc[pairs.same_position.astype(bool)])})
    stages = [{"comparison": name, **ranking_summary(sub, "stage_credit", "stage_tie", "stage_gap")}
              for name, sub in pairs.groupby("stage_test", sort=False)]
    group_rows = [{"protein_id": protein, "matched_go_id": go, **ranking_summary(sub)}
                  for (protein, go), sub in pairs.groupby(GROUP, sort=False)]
    groups = pd.DataFrame(group_rows, columns=GROUP + list(ranking_summary(pairs)))
    audit["top5_groups_fraction_of_pairs"] = (float(groups.nlargest(5, "n_pairs").n_pairs.sum() / len(pairs))
                                               if len(pairs) else None)
    return {"audit": audit, "ranking": ranking, "stage_ranking": stages}, pairs, groups


def dataset_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Raw or filtered counts; no model scores required."""
    frame = lookup_annotations(train, test)
    _, cleaning = clean_variants(test)
    return {"train_rows": len(train), "test_rows": len(test),
            "test_proteins_seen_in_train": int(test.protein_id.isin(set(train.protein_id)).sum()),
            "anti_majority_rows": int(frame.anti_majority.sum()),
            "anti_majority_class_counts": {label: int((frame.anti_majority & (frame.label == label)).sum()) for label in LABELS},
            "pair_audit": cleaning}
