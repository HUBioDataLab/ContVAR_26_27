"""Validate seed aggregation with complete synthetic condition matrices."""
from collections import Counter
from statistics import stdev

import pytest

from contvar_mf import three_seed_report as report
from contvar_mf.util import LABELS


def make_scope(f1):
    classes = {label.lower(): dict(precision=f1, recall=f1, f1=f1, support=2) for label in LABELS}
    return dict(n=6, macro_f1=f1, mcc=f1 / 2, accuracy=f1, balanced_accuracy=f1, **classes)


def test_fixed_and_selected_models_and_paired_seed_differences():
    runs = {}
    for seed in report.SEEDS:
        models, selected = {}, {}
        for c in report.V5_CONDITIONS:
            score = c["number"] / 100 + (seed - 42) * (0.01 if c["cohort"] == "distance" else 0.0)
            metrics = {**make_scope(score), "single_substitution": make_scope(score / 2),
                       "combinatorial": make_scope(score / 3)}
            models[c["name"]] = dict(validation=metrics, test=metrics, best_epoch=seed - 41, epochs_run=5)
            key = f"{c['cohort']}/{c['family']}"
            selected.setdefault(key, c["name"])
            if seed == 43 and c["window_mode"] == "esim":
                selected[key] = c["name"]
        runs[seed] = dict(models=models, best_by_validation=selected)
    tables = report.aggregate(runs)
    assert len(tables["conditions_by_seed"]) == 144
    assert len(tables["condition_summary"]) == 48
    assert len(tables["selection_by_seed"]) == 12
    assert len(tables["selection_summary"]) == 4
    assert len(tables["paired_differences_by_seed"]) == 264
    assert Counter(row["comparison"] for row in tables["paired_differences_summary"]) == {
        "encoder": 24, "pair_mode": 16, "window_context": 16, "window_comparison": 8, "head": 24,
    }
    first = tables["condition_summary"][0]
    assert first["test_macro_f1_seed42"] == first["test_macro_f1_seed44"] == 0.01
    assert first["test_macro_f1_sd"] == 0
    assert first["test_combinatorial_lof_precision_mean"] == pytest.approx(0.01 / 3)
    assert tables["selection_summary"][0]["condition_seed43"] != first["condition_seed43"]
    assert tables["selection_summary"][0]["test_macro_f1_mean"] > first["test_macro_f1_mean"]
    pair = tables["paired_differences_summary"][0]
    assert pair["test_macro_f1_delta_mean"] == pytest.approx(0.25)
    assert pair["test_macro_f1_delta_sd"] == pytest.approx(stdev([0.24, 0.25, 0.26]))
    assert "Sabit decoder koşulları" in report.report_text(tables)
