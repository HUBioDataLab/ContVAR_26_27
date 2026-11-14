"""Input isolation, train-only fitting, held-out selection, full data/W&B integration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from phase_3_variant_decoder.compare_input_controls import main, run_experiment, upload_saved
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.data.pair_dataset import PairTensors
from phase_3_variant_decoder.input_controls import (
    InputControlMLP, LEARNED, METHODS, MUTATION_DIM, attach_mutation_features,
    class_weights, classification_metrics, fit_normalization, load_input_control,
    predict, train_control,
)
from phase_3_variant_decoder.input_control_reports import SECTIONS
from phase_3_variant_decoder.train import prepare_run


def fixture_config(root):
    rng, sequences = np.random.default_rng(37), {}
    base = dict(feature_type="MUTAGEN", ftId="", atomic_description="synthetic fixture",
                matched_go_name="fixture", cosine_similarity=1., match_source="fixture", notes="", is_combinatorial=False)
    with h5py.File(root / "global.h5", "w") as glob, h5py.File(root / "local.h5", "w") as local:
        for split in ("train", "val", "test"):
            rows = []
            for i, label in enumerate(("LOF", "GOF", "NEUTRAL") * 3):
                protein = f"{split}_{i}"
                sequences[protein] = "ACDE"
                # One held-out GO tests fallback and artifact alignment, not ontology training.
                go = "GO:0000003" if split != "train" and i == 8 else f"GO:{i % 2 + 1:07d}"
                rows.append({**base, "protein_id": protein, "snp": "A1G", "label": label, "matched_go_id": go})
                for key in (protein, protein + "_A1G"):
                    glob[key] = rng.normal(size=1280).astype("float32")
                for key in (protein + "_1", protein + "_A1G"):
                    local[key] = rng.normal(size=1280).astype("float32")
            stage1 = pd.DataFrame(rows)
            stage1["target"] = stage1.label.map({"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"})
            stage2 = stage1.loc[stage1.label != "NEUTRAL"].copy()
            stage2["target"] = stage2.label
            for family in ("easy", "hard"):
                stage1.to_csv(root / f"stage1_{family}_{split}.tsv", sep="\t", index=False)
                stage2.to_csv(root / f"stage2_{family}_{split}.tsv", sep="\t", index=False)
    (root / "seq.json").write_text(json.dumps(sequences))
    np.savez(root / "go.npz", go_ids=np.array(["GO:0000001", "GO:0000002", "GO:0000003"]),
             embeddings=rng.normal(size=(3, 5)).astype("float32"))
    cfg = VariantDecoderConfig("hard", "seq_phase1", "both")
    cfg.dataset_dir, cfg.sequences_json = str(root), str(root / "seq.json")
    cfg.global_h5, cfg.local_h5 = str(root / "global.h5"), str(root / "local.h5")
    cfg.go_embeddings = str(root / "go.npz")
    cfg.hidden_dims, cfg.dropout, cfg.epochs, cfg.batch_size = [8], .1, 2, 4
    cfg.cuda, cfg.wandb_enabled = "cpu", False
    return cfg


class InputControlTests(unittest.TestCase):
    def test_lookup_frequency_ties_and_unseen_fallback(self):
        from phase_3_variant_decoder.go_lookup_rules import fit_lookup
        rules, fallback, _ = fit_lookup(["a", "a", "b", "b"], [1, 0, 2, 0])
        self.assertEqual(rules, {"a": 1, "b": 2})
        self.assertEqual(fallback, 0)
        self.assertEqual(rules.get("unseen", fallback), 0)
        with self.assertRaises(ValueError):
            fit_lookup(["a"], [0, 1])

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_restricted_arms_do_not_read_excluded_inputs(self):
        torch.manual_seed(4)
        go, wt, var = torch.randn(3, 5), torch.randn(3, 7), torch.randn(3, 7)
        mutation = torch.randn(3, MUTATION_DIM)
        bad = torch.full_like(wt, float("nan"))
        for method in LEARNED:
            model = InputControlMLP(method, 7, 5, [9], 0).eval()
            out = model(go, wt, var, mutation)
            if method == "go_only" or method == "mutation_go":
                torch.testing.assert_close(out, model(go, bad, bad, mutation))
            if method == "wt_go":
                torch.testing.assert_close(out, model(go, wt, bad))
            if method == "pair":
                v = var.clone().requires_grad_()
                model(go, wt, v).sum().backward()
                self.assertGreater(v.grad.abs().sum().item(), 0)

    def test_statistics_ignore_mutants_and_heldout_rows(self):
        tensors = SimpleNamespace(wt_feat=torch.tensor([[1., 2.], [3., 2.], [900., 300.]]),
                                  var_feat=torch.randn(3, 2), go_features=torch.tensor([[1., 2.], [3., 6.], [800., 500.]]))
        train = dict(wt_row=torch.tensor([0, 0, 1]), go_idx=torch.tensor([0, 1, 1]))
        before = fit_normalization(tensors, train)
        torch.testing.assert_close(before["protein_mean"], torch.tensor([2., 2.]))
        torch.testing.assert_close(before["protein_std"], torch.tensor([1., 1.]))
        tensors.wt_feat[2] = float("nan")
        tensors.go_features[2] = float("nan")
        tensors.var_feat[:] = float("nan")
        after = fit_normalization(tensors, train)
        for key in before:
            torch.testing.assert_close(before[key], after[key])
        counts, weights = class_weights(torch.tensor([0, 0, 0, 1, 2, 2]))
        torch.testing.assert_close(counts, torch.tensor([3, 1, 2]))
        torch.testing.assert_close(weights, torch.tensor([2/3, 2., 1.]))

    def test_fixed_class_metrics_and_mutation_encoding(self):
        y, pred = np.array([0, 0, 1, 2]), np.array([0, 0, 0, 0])
        self.assertAlmostEqual(classification_metrics(y, pred)["macro_f1"],
                               f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        frame = pd.DataFrame(dict(split=["train", "test"], wt_aa=["A", "X"], mut_aa=["G", "A"], pos=[1, 1022]))
        tensors = SimpleNamespace(df=frame, wt_feat=torch.zeros(1, 2))
        views = {"train": {}, "test": {}}
        attach_mutation_features(tensors, views, 1022)
        self.assertEqual(views["test"]["mutation_features"].shape, (1, MUTATION_DIM))
        self.assertEqual(views["test"]["mutation_features"][0, -1].item(), 1.)
        self.assertEqual(views["test"]["mutation_features"][0, 20].item(), 1.)  # unknown WT

    def test_checkpoint_selection_is_invariant_to_test_labels(self):
        frame = pd.DataFrame([dict(split=split, wt_row=i, var_row=i, go_idx=i % 2, label_idx=i % 3,
                                    seen_in_train=True, is_alanine=False, protein_id=f"p{i}", snp="A1G",
                                    matched_go_id=f"GO:{i%2}", label=("LOF", "GOF", "NEUTRAL")[i % 3])
                              for split in ("train", "val", "test") for i in range(6)])
        torch.manual_seed(3)
        tensors = PairTensors(torch.randn(6, 4), torch.randn(6, 4), frame)
        tensors.go_features = torch.randn(2, 3)
        views = {s: tensors.split_view(s) for s in ("train", "val", "test")}
        norm = fit_normalization(tensors, views["train"])
        cfg = VariantDecoderConfig()
        cfg.hidden_dims, cfg.epochs, cfg.batch_size = [8], 3, 4
        cfg.input_balance_classes = True
        with tempfile.TemporaryDirectory() as temporary:
            a, hist_a = train_control("pair", cfg, tensors, views, norm, Path(temporary) / "a", quiet=True)
            views["test"]["label_idx"] = (views["test"]["label_idx"] + 1) % 3
            b, hist_b = train_control("pair", cfg, tensors, views, norm, Path(temporary) / "b", quiet=True)
            self.assertEqual(hist_a, hist_b)
            self.assertEqual(a["best_epoch"], b["best_epoch"])
            model_a, _ = load_input_control(a["checkpoint"])
            model_b, _ = load_input_control(b["checkpoint"])
            for key in model_a.state_dict():
                torch.testing.assert_close(model_a.state_dict()[key], model_b.state_dict()[key], rtol=0, atol=0)

    def test_real_cascade_h5_reports_reload_and_wandb_reupload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = fixture_config(root)
            result = run_experiment(cfg, cache_dir=root / "cache", out_dir=root / "runs", quiet=True)
            self.assertEqual(cfg.architecture, "legacy")  # Caller and existing workflows unchanged.
            self.assertEqual(set(result["runs"]), set(METHODS))
            self.assertNotIn("cascade_loss", result["config"])
            source = Path(result["manifest_path"])
            for relative in result["artifacts"].values():
                self.assertTrue((source.parent / relative).is_file())
            self.assertEqual(result["provenance"]["split_rows"], dict(train=9, val=9, test=9))
            cfg.architecture = "cascade"
            tensors, views, _, _ = prepare_run(cfg, cache_dir=root / "cache", quiet=True)
            attach_mutation_features(tensors, views, cfg.max_seq_len)
            for method in LEARNED:
                model, ckpt = load_input_control(source.parent / result["runs"][method]["checkpoint"])
                probs = predict(model, tensors, views["test"], cfg.batch_size)
                saved = pd.read_csv(source.parent / method / "test_predictions.csv")
                np.testing.assert_allclose(probs, saved[[f"uncalibrated_softmax_{c}" for c in ("lof", "gof", "neutral")]], rtol=1e-6)
                history = result["histories"][method]
                self.assertEqual(ckpt["selection"]["epoch"], max(history, key=lambda r: r["val_macro_f1"])["epoch"])
            rules = json.loads((source.parent / "go_lookup" / "rules.json").read_text())
            self.assertNotIn("GO:0000003", rules["rules"])
            run = SimpleNamespace(log=Mock(), summary={}, finish=Mock(), log_artifact=Mock(), url="https://wandb.test/inputs", define_metric=Mock())
            wb = SimpleNamespace(Table=Mock(), Html=Mock(), Image=Mock(), Artifact=Mock(), init=Mock(return_value=run))
            with patch.dict("sys.modules", {"wandb": wb}), \
                 patch("phase_3_variant_decoder.compare_input_controls.prepare_run", side_effect=AssertionError("No data reload")), \
                 patch("phase_3_variant_decoder.compare_input_controls.train_control", side_effect=AssertionError("No retraining")):
                self.assertEqual(upload_saved(source), run.url)
            self.assertEqual(wb.init.call_args.kwargs["group"], "input-control-comparison")
            keys = set().union(*(call.args[0].keys() for call in run.log.call_args_list))
            self.assertIn("Input_Control_Comparison/02_Final_F1", keys)
            for method in METHODS:
                self.assertIn(f"{SECTIONS[method]}/Scores", keys)
                self.assertIn(f"{SECTIONS[method]}/test/macro_f1", run.summary)
            for method in LEARNED:
                self.assertIn(f"{SECTIONS[method]}/Training/val_macro_f1", keys)

    def test_both_cli_uses_separate_family_runs_and_names(self):
        with patch("phase_3_variant_decoder.compare_input_controls.run_experiment") as execute:
            main(["--family", "both", "--global_h5", "global.h5", "--local_h5", "local.h5",
                  "--go_embeddings", "go.npz", "--wandb_run_name", "controls"])
            self.assertEqual([c.args[0].family for c in execute.call_args_list], ["easy", "hard"])
            self.assertEqual([c.args[0].wandb_run_name for c in execute.call_args_list], ["controls-easy", "controls-hard"])
            self.assertTrue(all(c.args[0].hidden_dims == [128, 64] for c in execute.call_args_list))


if __name__ == "__main__":
    unittest.main()
