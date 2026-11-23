"""Training-only baselines and refusal to report incomplete experiments."""
import numpy as np
import pandas as pd
import pytest

from contvar_mf.suite import baseline_predictions
from contvar_mf.train import metrics


def test_baselines_never_use_evaluation_labels_and_fallback_to_training_priors():
    train = pd.DataFrame(dict(protein_id=["P", "P", "Q", "R"],
                              matched_go_id=["GO:1", "GO:1", "GO:2", "GO:2"],
                              label=["LOF", "LOF", "GOF", "NEUTRAL"]))
    test = pd.DataFrame(dict(protein_id=["P", "X", "X"], matched_go_id=["GO:1", "GO:2", "GO:3"],
                             label=["GOF", "GOF", "GOF"]))
    first = baseline_predictions(train, test)
    test["label"] = "NEUTRAL"
    second = baseline_predictions(train, test)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_allclose(first["protein_go_majority"][0], [1, 0, 0])
    np.testing.assert_allclose(first["protein_go_majority"][1], [0, .5, .5])
    np.testing.assert_allclose(first["protein_go_majority"][2], [.5, .25, .25])


def test_metrics_are_finite_when_an_evaluation_class_is_absent():
    result = metrics(np.array([0, 0, 2]), np.array([0, 2, 2]),
                     np.array([[.8, .1, .1], [.3, .2, .5], [.1, .1, .8]]), "multiclass")
    assert result["gof"]["support"] == 0
    assert result["gof"]["auprc"] is None
    assert np.isfinite(result["macro_f1"]) and np.isfinite(result["mcc"])


def test_report_refuses_missing_new_summary(tmp_path, monkeypatch):
    from contvar_mf import report
    monkeypatch.setattr("sys.argv", ["report", "--results", str(tmp_path / "missing"), "--out", str(tmp_path / "report")])
    with pytest.raises(RuntimeError, match="NEW 48-run summary missing"):
        report.main()
    assert not (tmp_path / "report").exists()
