"""Seed plumbing, output isolation and a small real training reproducibility check."""
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from contvar_mf import suite, train
from contvar_mf.util import LABELS
from contvar_mf.v5 import V5_FORMAT


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_v5_cli_pins_cpu_plan_in_distinct_output(tmp_path, monkeypatch, seed):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}")
    manifest = dict(format=V5_FORMAT, source_feature_sha256={}, source_cohorts={}, ontology={})
    monkeypatch.setattr(suite, "verify_preparation", lambda path: manifest)
    monkeypatch.setattr(suite, "V5Bundle", lambda path: SimpleNamespace(hashes=lambda: {}))
    monkeypatch.setattr(sys, "argv", ["suite", "--prepared", str(prepared), "--v5-bundle", str(tmp_path),
                                      "--device", "cpu", "--seed", str(seed)])
    def inspect(args, is_v5, bundle, cohorts, shared_tables, plan, source_files):
        assert is_v5 and cohorts == ("mean", "distance")
        assert args.out == tmp_path / f"prepared_seed{seed}"
        assert plan["device"] == "cpu"
        assert plan["settings"] == {**suite.SETTINGS, "seed": seed}
        assert suite.SETTINGS["seed"] == 42
        assert len(plan["conditions"]) == len({c["name"] for c in plan["conditions"]}) == 48
        assert all(sum(c["cohort"] == e and c["family"] == f for c in plan["conditions"]) == 12
                   for e in cohorts for f in ("easy", "hard"))
    observed = []
    def capture(*args):
        inspect(*args)
        observed.append(True)
    monkeypatch.setattr(suite, "run_suite", capture)
    suite.main()
    assert observed == [True]


def test_pinned_outputs_require_explicit_matching_resume(tmp_path):
    target = tmp_path / "seed42"
    target.mkdir()
    plan = {"settings": {"seed": 42}, "provenance": {"sha256": "same"}}
    saved = target / "plan.json"
    saved.write_text(json.dumps(plan))
    original = saved.read_bytes()
    with pytest.raises(FileExistsError, match="requires --resume"):
        suite.check_output_directory(target, plan, False)
    with pytest.raises(ValueError, match="different pinned plan"):
        suite.check_output_directory(target, {**plan, "settings": {"seed": 43}}, True)
    with pytest.raises(ValueError, match="different pinned plan"):
        suite.check_output_directory(target, {**plan, "provenance": {"sha256": "other"}}, True)
    with pytest.raises(ValueError, match="nested"):
        suite.check_output_directory(target / "other", plan, True)
    suite.check_output_directory(target, plan, True)
    assert saved.read_bytes() == original
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "occupied").touch()
    with pytest.raises(FileExistsError, match="without a pinned plan"):
        suite.check_output_directory(unrelated, plan, True)
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "plan.json.tmp").write_text("{")
    suite.check_output_directory(interrupted, plan, True)


def test_parallel_writers_are_excluded_and_lock_can_be_reused(tmp_path):
    output = tmp_path / "seed43"
    with suite.output_lock(output):
        with pytest.raises(RuntimeError, match="in use"):
            with suite.output_lock(output):
                pass
    with suite.output_lock(output):
        assert not output.exists()


class TinyDataset:
    def __init__(self, count):
        self.labels = np.arange(count) % 3
        self.store = SimpleNamespace(go=SimpleNamespace(dimension=5))
        self.frame = pd.DataFrame(dict(label=[LABELS[i] for i in self.labels], common_test=False,
                                       seen_in_train=True, multiple_labels_for_input=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        wt = dict(mut=torch.zeros(1280), **{"global": torch.zeros(1280)})
        var = dict(mut=torch.zeros(1280), **{"global": torch.zeros(1280)})
        wt["mut"][0], var["mut"][i + 1] = i, 1
        return dict(wt=wt, var=var, go=torch.ones(5), label=int(self.labels[i]), row=i)


def test_seed_changes_actual_training_init_and_loader_and_replays(tmp_path, monkeypatch):
    real_model = train.PairedGODecoder
    traces = []
    def observed_model(**kwargs):
        model = real_model(**kwargs)
        trace = dict(initial=model.go_adapter[1].weight.detach().clone(),
                     random=(random.random(), float(np.random.rand()), float(torch.rand(()))), order=[])
        traces.append(trace)
        def record(module, inputs, kwargs):
            if module.training:
                trace["order"].extend(kwargs["wt"]["mut"][:, 0].tolist())
        model.register_forward_pre_hook(record, with_kwargs=True)
        return model
    monkeypatch.setattr(train, "PairedGODecoder", observed_model)
    settings = {**suite.SETTINGS, "epochs": 2, "batch_size": 3, "width": 8, "attention_heads": 2}
    condition = dict(number=1, name="tiny", head="multiclass", pair_mode="joint", window_mode="none")
    datasets = {"train": TinyDataset(12), "val": TinyDataset(6)}
    def fit(index, seed):
        folder = tmp_path / str(index)
        result = train.fit(condition, datasets, torch.device("cpu"), folder, {**settings, "seed": seed}, {})
        best = torch.load(folder / "best.pt", weights_only=True)
        last = torch.load(folder / "last.pt", weights_only=True)
        assert result["seed"] == best["seed"] == last["seed"] == seed
        assert json.loads((folder / "config.json").read_text())["seed"] == seed
        return folder, best
    first, a = fit(0, 43)
    second, b = fit(1, 43)
    fit(2, 44)
    assert traces[0]["random"] == traces[1]["random"]
    assert all(x != y for x, y in zip(traces[0]["random"], traces[2]["random"]))
    assert torch.equal(traces[0]["initial"], traces[1]["initial"])
    assert not torch.equal(traces[0]["initial"], traces[2]["initial"])
    assert traces[0]["order"] == traces[1]["order"] != traces[2]["order"]
    assert sorted(traces[0]["order"][:12]) == list(range(12))
    assert (first / "history.json").read_bytes() == (second / "history.json").read_bytes()
    assert (first / "val_predictions.tsv").read_bytes() == (second / "val_predictions.tsv").read_bytes()
    assert all(torch.equal(value, b["state_dict"][key]) for key, value in a["state_dict"].items())
