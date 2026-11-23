"""Audit completed window experiments against their pinned inputs and selections."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from contvar_decoder.model import load_decoder
from contvar_decoder.prepare import sha256_file


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(suite):
    root = Path(__file__).resolve().parent
    suite = Path(suite)
    summary = read(suite / "summary.json")
    plan = summary["plan"]
    assert not plan["smoke"], "Smoke output is not a full experiment"
    assert len(summary["models"]) == plan["distinct_training_runs"]
    for name, digest in plan["source_code_sha256"].items():
        assert sha256_file(root / "contvar_decoder" / name) == digest, name
    selection = read(suite / "selection_before_test.json")
    checked = []
    for name, result in summary["models"].items():
        cohort, family, pool, architecture = name.split("/")
        directory = suite / name
        config = read(directory / "config.json")
        history = read(directory / "history.json")
        manifest = read(root / "prepared_window" / cohort / "manifest.json")
        checkpoint = directory / "best.pt"
        assert sha256_file(checkpoint) == selection[name]["checkpoint_sha256"]
        _, saved = load_decoder(checkpoint)
        scales = set(saved["model_spec"]["scale_dims"])
        assert scales == ({"mut", "global", "window"} if pool == "mean" else {"mut", "global"})
        assert config["window_pool"] == pool == result["window_pool"]
        assert config["feature_sha256"] == plan["feature_sha256"]
        assert saved["feature_sha256"] == plan["feature_sha256"]
        assert saved["family"] == family
        assert result["best_epoch"] == max(history, key=lambda row: row["validation"]["macro_f1"])["epoch"]
        assert len(history) == result["epochs_run"] <= plan["max_epochs"]
        for split in ("val", "test"):
            path = directory / f"{split}_predictions.tsv"
            pred = pd.read_csv(path, sep="\t")
            target = pd.read_csv(root / "prepared_window" / cohort / result["head"] / f"{family}_{split}.tsv", sep="\t")
            assert pred.sample_id.equals(target.sample_id), (name, split, "row IDs")
            assert pred.label.equals(target.label)
            assert not pred.augmented.any() and not pred.is_combinatorial.any()
            assert np.isfinite(pred[["score_lof", "score_gof", "score_neutral", "d_forward", "d_reverse"]]).all().all()
            assert len(pred) == manifest["prepared"][f"{family}/{split}"]["rows"] == result[split]["n"]
            if split == "test":
                assert path.stat().st_mtime_ns >= (suite / "selection_before_test.json").stat().st_mtime_ns
        if cohort == "full" and "common_old_test" in result:
            pred = pd.read_csv(directory / "common_old_test_predictions.tsv", sep="\t")
            target = pd.read_csv(root / "prepared_window/old" / result["head"] / f"{family}_test.tsv", sep="\t")
            assert pred.sample_id.equals(target.sample_id)
            assert pred.label.equals(target.label)
            assert len(pred) == result["common_old_test"]["n"]
        checked.append(name)
    gains = pd.read_csv(suite / "window_katkisi.csv")
    assert len(gains) == len(checked) // 2
    # Historical results remain byte-for-byte equal to the original handover.
    original = read(root / "metadata/file_manifest.json")
    historical = 0
    for entry in original["files"]:
        if entry["path"].startswith("results/full_seed42/"):
            assert sha256_file(root / entry["path"]) == entry["sha256"]
            historical += 1
    report = dict(all_checks_passed=True, full_training_runs=len(checked), matched_window_pairs=len(gains),
                  original_result_files_unchanged=historical, tests_loaded_after_selection=True,
                  checks="code/input provenance, window scale, checkpoint selection, native rows/labels, finite predictions, no forbidden evidence, common-test alignment, historical result hashes")
    (suite / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", type=Path)
    verify(p.parse_args().suite)
