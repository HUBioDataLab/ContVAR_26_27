"""Reporting must preserve the original validation choices and stored numbers."""
import csv
import json

import pytest

from decoder.new_decoder.config import DECODER_ROOT
from decoder.new_decoder.report import generate_report


def test_historical_report_preserves_all_conditions_and_validation_selection(tmp_path):
    source = DECODER_ROOT / "docs/experiments/seed42_summary.json"
    report_dir = tmp_path / "report"
    rows = generate_report(source, report_dir)
    assert len(rows) == 12
    selected = [row for row in rows if row["selected_by_validation"]]
    assert len(selected) == 8
    assert all(row["lambda_reverse"] == 0 for row in selected)
    hard_parallel = next(
        row for row in selected if row["family"] == "hard" and row["model"] == "parallel_joint"
    )
    summary = json.loads(source.read_text())
    assert (
        hard_parallel["test_macro_f1"]
        == summary["models"]["hard"]["parallel_joint_lambda0"]["test"]["macro_f1"]
    )
    with (report_dir / "all_runs.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 12
    with pytest.raises(FileExistsError):
        generate_report(source, report_dir)
