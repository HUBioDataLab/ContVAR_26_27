"""Repository path, native layout and public CLI integration coverage."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch

from decoder.new_decoder.config import DEFAULT_DATA_ROOT, DEFAULT_PREPARED_DIR, REPO_ROOT, DataPaths
from decoder.new_decoder.data import FeatureStore
from decoder.new_decoder.model import PairedGODecoder
from decoder.new_decoder.prepare_all import prepare_all, validate_source_layouts
from decoder.new_decoder.sources import resolve_source


def run_command(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )


def test_paths_match_the_repository_without_a_working_directory_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert DEFAULT_DATA_ROOT == REPO_ROOT / "data"
    assert DEFAULT_PREPARED_DIR == REPO_ROOT / "data/prepared/new_decoder"
    paths = DataPaths(tmp_path)
    assert paths.go_embeddings == tmp_path / "embeddings/go_terms_embeddings_anc2vec.npz"
    (tmp_path / "go").mkdir()
    (tmp_path / "go/go_terms_embeddings_anc2vec.npz").touch()
    assert paths.go_embeddings == tmp_path / "go/go_terms_embeddings_anc2vec.npz"


def test_zip_and_extracted_sources_produce_identical_native_cohorts(native_data, tmp_path):
    outputs = [tmp_path / "from_directories", tmp_path / "from_archives"]
    for source, output in zip((native_data["source"], native_data["archives"]), outputs):
        manifest = prepare_all(source, native_data["root"], output)
        assert all(all(checks.values()) for checks in manifest["source_layout_checks"].values())
        assert manifest["prepared"]["hard/train"]["rows"] == 5
        assert manifest["prepared"]["hard/val"]["rows"] == 4
        assert manifest["prepared"]["hard/train"]["augmented"] == 1
        assert manifest["prepared"]["hard/test"]["augmented"] == 0
        assert manifest["prepared"]["hard/train"]["input_label_review_rows"] == 2
        for entry in manifest["prepared"].values():
            assert all(not Path(path).is_absolute() for path in entry["paths"].values())
    for head in ("parallel", "multiclass"):
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                name = f"{head}/{family}_{split}.tsv"
                assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()


def test_wrong_native_targets_and_ambiguous_archives_are_rejected(native_data, tmp_path):
    path = native_data["source"] / "parallel/hard_val.tsv"
    table = pd.read_csv(path, sep="\t")
    table.loc[0, "target_lof"] = 0
    table.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="Source layout inconsistency"):
        validate_source_layouts(native_data["source"])
    for suffix in ("a", "b"):
        (tmp_path / f"parallel-{suffix}.zip").touch()
    with pytest.raises(FileNotFoundError, match="found 2 archives"):
        resolve_source(tmp_path, "parallel")


def test_feature_store_closes_handles_and_validates_scale(native_data):
    with FeatureStore(native_data["root"]) as store:
        assert store.vector("global", "P_easy_train_0").shape == (1280,)
        with pytest.raises(ValueError, match="Unsupported pooled scale"):
            store.vector("window", "P_easy_train_0")
        assert store.files
    assert not store.files and not store.memory


@pytest.mark.parametrize(
    "lof,gof,expected",
    [(0.4, 0.4, 2), (0.6, 0.4, 0), (0.4, 0.6, 1), (0.6, 0.7, 1), (0.7, 0.6, 0), (0.5, 0.5, 0)],
)
def test_parallel_historical_decision_including_overlap_and_tie(lof, gof, expected):
    model = PairedGODecoder(
        scale_dims={"mut": 8}, go_dim=5, width=8, heads=2, head="parallel", pair_mode="joint"
    ).eval()
    with torch.no_grad():
        model.lof_head.weight.zero_()
        model.gof_head.weight.zero_()
        model.lof_head.bias.copy_(torch.logit(torch.tensor([lof])))
        model.gof_head.bias.copy_(torch.logit(torch.tensor([gof])))
    features = {"mut": torch.ones(1, 8)}
    out = model(features, features, torch.ones(1, 5))
    assert out["prediction"].item() == expected
    torch.testing.assert_close(out["scores"][0, 2], torch.tensor(1 - max(lof, gof)))


def test_public_train_predict_and_suite_commands(native_data, tmp_path):
    prepared = tmp_path / "prepared"
    prepare_all(native_data["source"], native_data["root"], prepared)
    trained = tmp_path / "trained"
    run_command(
        "decoder.new_decoder.train",
        "--data-root",
        str(native_data["root"]),
        "--data-dir",
        str(prepared),
        "--family",
        "hard",
        "--head",
        "multiclass",
        "--pair-mode",
        "joint",
        "--smoke",
        "--width",
        "8",
        "--attention-heads",
        "2",
        "--threads",
        "1",
        "--out",
        str(trained),
    )
    assert (trained / "best.pt").is_file()
    assert not (trained / "test_predictions.tsv").exists()
    inferred = tmp_path / "inferred"
    run_command(
        "decoder.new_decoder.predict",
        str(trained / "best.pt"),
        "--data-root",
        str(native_data["root"]),
        "--data-dir",
        str(prepared),
        "--family",
        "hard",
        "--split",
        "val",
        "--out",
        str(inferred),
    )
    trained_result = json.loads((trained / "results.json").read_text())
    assert json.loads((inferred / "metrics.json").read_text()) == trained_result["val"]
    suite = tmp_path / "suite"
    run_command(
        "decoder.new_decoder.suite",
        "--data-root",
        str(native_data["root"]),
        "--data-dir",
        str(prepared),
        "--smoke",
        "--width",
        "8",
        "--attention-heads",
        "2",
        "--threads",
        "1",
        "--out",
        str(suite),
    )
    summary = json.loads((suite / "summary.json").read_text())
    assert summary["distinct_training_runs"] == 12
    assert summary["test_used_for_selection"] is False
    assert all(
        run["smoke"] and run["test"]["n"] == 4
        for runs in summary["models"].values()
        for run in runs.values()
    )
    assert (suite / "selection_before_test.json").stat().st_mtime_ns <= min(
        path.stat().st_mtime_ns for path in suite.glob("*/*/test_predictions.tsv")
    )


@pytest.mark.parametrize(
    "module,options",
    [
        ("train", ("--lambda-reverse", "nan")),
        ("suite", ("--lr", "nan")),
        ("train", ("--width", "7", "--attention-heads", "4")),
    ],
)
def test_invalid_training_options_fail_before_artifact_creation(module, options, tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises(subprocess.CalledProcessError):
        run_command(f"decoder.new_decoder.{module}", *options, "--out", str(destination))
    assert not destination.exists()
