"""Architecture invariants and a real TSV/H5 -> train -> reload -> eval test."""
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from phase_3_variant_decoder.cascade_model import CascadeDecoder, stage_targets
from phase_3_variant_decoder.cascade import (
    attach_checkpoint_go, evaluate_cascade, evaluate_checkpoint,
    fit_cascade, load_cascade,
)
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.data.variant_dataset import load_family
from phase_3_variant_decoder.train import prepare_run


class CascadeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_swap_symmetry_identical_pairs_and_go_conditioning(self):
        torch.manual_seed(9)
        model = CascadeDecoder(5, 3, [12, 8], dropout=0.4)
        wt, var, go = torch.randn(7, 5), torch.randn(7, 5), torch.randn(7, 3)
        for training in (True, False):
            model.train(training)
            torch.manual_seed(8)
            original = model(wt, var, go)
            torch.manual_seed(8)
            swapped = model(var, wt, go)
            for stage in original:
                torch.testing.assert_close(original[stage]["delta"], -swapped[stage]["delta"])
                self.assertEqual(original[stage]["s_wt"].shape, (7,))
                same = model(wt, wt, go)[stage]["delta"]
                self.assertEqual(same.count_nonzero().item(), 0)
            torch.testing.assert_close(original["effect"]["probability"], swapped["effect"]["probability"])
            torch.testing.assert_close(original["direction"]["probability"], 1 - swapped["direction"]["probability"])
        changed_go = model(wt, var, go + 2)
        self.assertFalse(torch.allclose(original["direction"]["delta"], changed_go["direction"]["delta"]))

    def test_training_targets_and_stage_gradient_isolation(self):
        labels = torch.tensor([2, 0, 2, 1])
        mask, effect = stage_targets(labels, "effect")
        self.assertEqual(effect.tolist(), [0, 1, 0, 1])
        mask, direction = stage_targets(labels, "direction")
        self.assertEqual(mask.tolist(), [False, True, False, True])
        self.assertEqual(direction.tolist(), [1, 0])
        model = CascadeDecoder(5, 3, [8], dropout=0)
        wt = torch.randn(4, 5, requires_grad=True)
        var = torch.randn(4, 5, requires_grad=True)
        go = torch.randn(4, 3)
        out = model.direction_stage(wt[mask], var[mask], go[mask])
        F.binary_cross_entropy_with_logits(out["logit"], direction).backward()
        self.assertEqual(wt.grad[~mask].count_nonzero().item(), 0)
        self.assertGreater(wt.grad[mask].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.effect_stage.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.direction_stage.parameters()))

    def test_hard_routing_and_score_sign(self):
        model = CascadeDecoder(2, 1, [], dropout=0).eval()
        wt, var, go = torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 1)
        def output(p):
            p = torch.tensor(p)
            return {"s_wt": p, "s_var": 1 - p, "delta": 2 * p - 1,
                    "probability": p, "logit": torch.logit(p)}
        with patch.object(model.effect_stage, "forward", return_value=output([0.1, 0.9, 0.8])), \
             patch.object(model.direction_stage, "forward", return_value=output([0.8, 0.2])) as second:
            out = model.predict(wt, var, go)
            self.assertEqual(out["prediction"].tolist(), [2, 0, 1])
            torch.testing.assert_close(second.call_args.args[0], wt[1:])
            self.assertTrue(torch.isnan(out["direction"]["delta"][0]))
        with patch.object(model.effect_stage, "forward", return_value=output([0.1, 0.1, 0.1])), \
             patch.object(model.direction_stage, "forward") as second:
            self.assertEqual(model.predict(wt, var, go)["prediction"].tolist(), [2, 2, 2])
            second.assert_not_called()
        # Signed difference really makes higher WT score select LOF.
        with torch.no_grad():
            model.direction_stage.scorer.output.weight.copy_(torch.tensor([[1., 0., 0.]]))
            model.direction_stage.scorer.output.bias.zero_()
        out = model.direction_stage(torch.tensor([[1., 0.]]), torch.tensor([[-1., 0.]]), torch.zeros(1, 1))
        self.assertGreater(out["probability"].item(), 0.5)

    def test_final_macro_f1_threshold_selection_uses_validation_predictions(self):
        from phase_3_variant_decoder.compare_threshold_selection import (
            build_threshold_curve, select_final_threshold, threshold_metrics,
        )

        # LOF=0, GOF=1, NEUTRAL=2. At the low baseline threshold both neutral
        # examples are routed and must become LOF/GOF. A threshold above 0.4
        # produces perfect three-class predictions with the same model outputs.
        y = np.array([0, 0, 1, 1, 2, 2])
        p_effect = np.array([.90, .80, .90, .80, .40, .30])
        p_lof = np.array([.90, .80, .10, .20, .60, .40])
        baseline = threshold_metrics(y, p_effect, p_lof, .10)
        curve = build_threshold_curve(y, p_effect, p_lof, .10)
        selected = select_final_threshold(curve, .10)
        self.assertGreater(selected["macro_f1"], baseline["macro_f1"])
        self.assertEqual(selected["macro_f1"], 1.0)
        self.assertGreater(selected["threshold"], .40)
        self.assertEqual(selected["neutral_routed_rate"], 0.0)

    def test_train_cli_defaults_to_cascade_and_retains_explicit_legacy(self):
        from phase_3_variant_decoder.train import main
        base = ["train", "--family", "hard", "--embedding", "raw_esm",
                "--global_h5", "global.h5", "--local_h5", "local.h5", "--no_wandb"]
        with patch("sys.argv", base + ["--go_embeddings", "go.npz"]), \
             patch("phase_3_variant_decoder.train.train") as train:
            main()
            cfg = train.call_args.args[0]
            self.assertEqual(cfg.architecture, "cascade")
            self.assertEqual(cfg.go_embeddings, "go.npz")
        with patch("sys.argv", base + ["--architecture", "legacy"]), \
             patch("phase_3_variant_decoder.train.train") as train:
            main()
            self.assertEqual(train.call_args.args[0].architecture, "legacy")

    def test_explicit_cascade_splits_are_validated_and_loaded_from_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {
                "feature_type": "MUTAGEN", "ftId": "", "atomic_description": "fixture",
                "matched_go_name": "fixture", "cosine_similarity": 1.0,
                "match_source": "fixture", "notes": "", "is_combinatorial": False,
            }
            sequences = {}
            for split in ("train", "val", "test"):
                rows = []
                for i, label in enumerate(("LOF", "GOF", "NEUTRAL")):
                    protein = f"{split}_{i}"
                    sequences[protein] = "ACDE"
                    rows.append({**base, "protein_id": protein, "snp": "A1G",
                                 "label": label, "matched_go_id": f"GO:{i + 1:07d}"})
                stage1 = pd.DataFrame(rows)
                stage1["target"] = stage1["label"].map(
                    {"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"}
                )
                stage2 = stage1[stage1.label != "NEUTRAL"].copy()
                stage2["target"] = stage2["label"]
                stage1.to_csv(root / f"stage1_hard_{split}.tsv", sep="\t", index=False)
                stage2.to_csv(root / f"stage2_hard_{split}.tsv", sep="\t", index=False)

            frame, report = load_family("hard", str(root), sequences)
            self.assertEqual(report.source_layout, "cascade")
            self.assertEqual(len(frame), 9)
            self.assertNotIn("target", frame.columns)

            bad = pd.read_csv(root / "stage2_hard_train.tsv", sep="\t")
            bad.loc[0, "target"] = "GOF"
            bad.to_csv(root / "stage2_hard_train.tsv", sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "target must equal"):
                load_family("hard", str(root), sequences)

    def test_weighting_comparison_report_writes_figures_and_tables(self):
        from phase_3_variant_decoder.compare_weighting import (
            generate_comparison_report, log_comparison_to_wandb,
        )

        metric = {
            "macro_f1": 0.6, "accuracy": 0.7,
            "effect": {"macro_f1": 0.65},
            "direction_true_effect": {"macro_f1": 0.55},
            "effect_missed_n": 2, "neutral_routed_n": 3,
            "confusion": [[4, 1, 1], [1, 3, 2], [1, 1, 4]],
        }
        for label, value in (("lof", 0.7), ("gof", 0.4), ("neutral", 0.7)):
            for name in ("precision", "recall", "f1", "auprc"):
                metric[f"{label}_{name}"] = value
        result = {
            "selection": {
                "effect": {"epoch": 2, "val_macro_f1": 0.65,
                           "train_class_counts_0_1": [4, 8],
                           "class_weights_0_1": [1.5, 0.75]},
                "direction": {"epoch": 2, "val_macro_f1": 0.55,
                              "train_class_counts_0_1": [2, 6],
                              "class_weights_0_1": [2.0, 0.667]},
            },
            "metrics": {"val": deepcopy(metric), "test": deepcopy(metric)},
        }
        histories = {
            name: [
                {"stage": stage, "epoch": epoch, "train_loss": 1 / epoch,
                 "val_macro_f1": 0.4 + epoch / 10, "threshold": 0.5}
                for stage in ("effect", "direction") for epoch in (1, 2)
            ] for name in ("weighted", "unweighted")
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"MPLCONFIGDIR": str(Path(tmp) / "mpl")}, clear=False):
            artifacts = generate_comparison_report(
                {"weighted": deepcopy(result), "unweighted": deepcopy(result)}, histories, tmp
            )
            for path in artifacts.values():
                self.assertTrue(Path(path).is_file(), path)
                self.assertGreater(Path(path).stat().st_size, 0)

            class FakeRun:
                def __init__(self):
                    self.logged, self.defined, self.summary = [], [], {}

                def log(self, payload):
                    self.logged.append(payload)

                def define_metric(self, *args, **kwargs):
                    self.defined.append((args, kwargs))

            class FakeWandb:
                @staticmethod
                def Table(**kwargs):
                    return ("table", kwargs)

                @staticmethod
                def Image(path, **kwargs):
                    return ("image", path)

            run = FakeRun()
            results = {"weighted": deepcopy(result), "unweighted": deepcopy(result)}
            log_comparison_to_wandb(run, FakeWandb, results, histories, artifacts)
            logged_keys = set().union(*(payload.keys() for payload in run.logged))
            self.assertIn("comparison/effect/weighted_train_loss", logged_keys)
            self.assertIn("comparison/direction/unweighted_val_macro_f1", logged_keys)
            self.assertIn("00_Overview/Score_Table", logged_keys)
            self.assertIn("02_Final_Comparison/F1", logged_keys)
            self.assertEqual(run.summary["weighted/test/macro_f1"], 0.6)

            from phase_3_variant_decoder.compare_architectures import (
                _log_final_wandb, generate_report as generate_architecture_report,
            )
            architecture_results = {
                name: {"metrics": {"val": deepcopy(metric), "test": deepcopy(metric)}}
                for name in ("legacy", "cascade")
            }
            architecture_artifacts = generate_architecture_report(
                architecture_results, Path(tmp) / "architectures"
            )
            architecture_run = FakeRun()
            _log_final_wandb(
                architecture_run, FakeWandb, architecture_results,
                architecture_artifacts,
            )
            architecture_keys = set().union(
                *(payload.keys() for payload in architecture_run.logged)
            )
            self.assertIn("00_Overview/Score_Table", architecture_keys)
            self.assertEqual(architecture_run.summary["legacy/test/macro_f1"], 0.6)

    def test_real_data_pipeline_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(8)
            sequences = {}
            rows = []
            for split in ("train", "val", "test"):
                for i in range(6):
                    protein = f"{split}_{i}"
                    sequences[protein] = "ACDE"
                    rows.append({"split": split, "protein_id": protein, "snp": "A1G",
                                 "matched_go_id": f"GO:{i % 2 + 1:07d}",
                                 "matched_go_name": "fixture", "label": ["NEUTRAL", "LOF", "GOF"][i % 3]})
            frame = pd.DataFrame(rows)
            for split in ("train", "val", "test"):
                split_frame = frame[frame.split == split].drop(columns="split")
                split_frame.to_csv(root / f"hard_{split}.tsv", sep="\t", index=False)
                split_frame.to_csv(root / f"easy_{split}.tsv", sep="\t", index=False)
            (root / "seq.json").write_text(json.dumps(sequences))
            with h5py.File(root / "global.h5", "w") as glob, h5py.File(root / "local.h5", "w") as loc:
                for protein in sequences:
                    for key in (protein, protein + "_A1G"):
                        glob[key] = rng.normal(size=1280).astype("float32")
                    for key in (protein + "_1", protein + "_A1G"):
                        loc[key] = rng.normal(size=1280).astype("float32")
            # Artifact order intentionally differs from sorted GO vocabulary.
            np.savez(root / "go.npz", go_ids=np.array(["GO:0000002", "GO:0000001"]),
                     embeddings=np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32),
                     status=np.array(["direct", "direct"]), metadata_json=np.asarray(json.dumps({"dimension": 3})))
            cfg = VariantDecoderConfig(input_mode="global")
            cfg.architecture = "cascade"
            cfg.dataset_dir, cfg.sequences_json = str(root), str(root / "seq.json")
            cfg.global_h5, cfg.local_h5 = str(root / "global.h5"), str(root / "local.h5")
            cfg.go_embeddings = str(root / "go.npz")
            cfg.hidden_dims, cfg.dropout, cfg.epochs, cfg.batch_size = [8], 0.1, 2, 5
            cfg.cuda, cfg.wandb_enabled = "cpu", False
            tensors, views, vocab, device = prepare_run(cfg, quiet=True)
            torch.testing.assert_close(tensors.go_features, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
            with patch("phase_3_variant_decoder.cascade.evaluate_cascade", wraps=evaluate_cascade) as evaluate:
                best = fit_cascade(cfg, tensors, views, vocab, device, root / "runs", quiet=True)
                self.assertEqual(evaluate.call_count, 2)  # Only val and test, after training both stages.
            self.assertEqual(best["selection"]["effect"]["train_n"], 6)
            self.assertEqual(best["selection"]["direction"]["train_n"], 4)
            model, ckpt = load_cascade(best["checkpoint"])
            self.assertEqual(ckpt["go_metadata"], tensors.go_metadata)
            go_saved = tensors.go_features.clone()
            attach_checkpoint_go(tensors, {"GO:0000001": 1, "GO:0000002": 0}, ckpt)
            torch.testing.assert_close(tensors.go_features, go_saved.flip(0))
            attach_checkpoint_go(tensors, vocab, ckpt)
            self.assertEqual(tensors.go_metadata, ckpt["go_metadata"])
            with self.assertRaisesRegex(ValueError, "no GO vector"):
                attach_checkpoint_go(tensors, {"GO:9999999": 0}, ckpt)
            metrics, scores = evaluate_cascade(model, tensors, views["test"], best["effect_threshold"], batch_size=5)
            self.assertEqual(metrics, best["metrics"]["test"])
            view = views["test"]
            pred = model.predict(*tensors.gather_with_go(view["wt_row"], view["var_row"], view["go_idx"]),
                                 best["effect_threshold"])
            np.testing.assert_array_equal(pred["prediction"].numpy(), scores["prediction"])
            np.testing.assert_allclose(scores["probabilities"].sum(axis=1), 1, atol=1e-6)
            from phase_3_variant_decoder.compare_weighting import run_comparison
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(root / "mpl")}, clear=False):
                comparison = run_comparison(
                    cfg, cache_dir=None, out_dir=str(root / "comparisons"), quiet=True
                )
            self.assertEqual(set(comparison["runs"]), {"weighted", "unweighted"})
            self.assertTrue(Path(comparison["artifacts"]["test_metrics"]).is_file())
            from phase_3_variant_decoder.compare_architectures import (
                run_comparison as run_architecture_comparison,
            )
            architecture_comparison = run_architecture_comparison(
                cfg, cache_dir=None, out_dir=str(root / "architecture_comparisons"),
                reuse_cascade_run=str(Path(best["checkpoint"]).parent), quiet=True,
            )
            self.assertEqual(
                set(architecture_comparison["runs"]), {"legacy", "cascade"}
            )
            self.assertTrue(
                Path(architecture_comparison["artifacts"]["test_metrics"]).is_file()
            )
            self.assertEqual(
                architecture_comparison["format"],
                "contvar_decoder_architecture_comparison_v1",
            )
            self.assertTrue(architecture_comparison["cascade_reused"])
            from phase_3_variant_decoder.compare_threshold_selection import run_experiment
            threshold_comparison = run_experiment(
                best["checkpoint"], cache_dir=None,
                out_dir=str(root / "threshold_comparisons"), cuda="cpu",
                wandb_enabled=False, quiet=True,
            )
            self.assertEqual(
                threshold_comparison["format"],
                "contvar_cascade_threshold_selection_v1",
            )
            self.assertFalse(threshold_comparison["test_used_for_selection"])
            self.assertTrue(
                Path(threshold_comparison["artifacts"]["threshold_sweep"]).is_file()
            )
            from phase_3_variant_decoder.compare_learned_threshold import (
                run_experiment as run_learned_threshold_experiment,
            )
            learned_threshold_comparison = run_learned_threshold_experiment(
                str(Path(best["checkpoint"]).parent), cache_dir=None,
                out_dir=str(root / "learned_threshold_comparisons"), cuda="cpu",
                wandb_enabled=False, quiet=True,
            )
            self.assertEqual(
                learned_threshold_comparison["format"],
                "contvar_learned_effect_threshold_comparison_v1",
            )
            self.assertFalse(learned_threshold_comparison["test_used_for_selection"])
            learned_run = learned_threshold_comparison["runs"]["learned_radius"]
            self.assertEqual(learned_run["effect_threshold"], 0.5)
            self.assertEqual(
                learned_run["selection"]["effect"]["threshold_strategy"],
                "learned_radius",
            )
            self.assertTrue(
                Path(learned_threshold_comparison["artifacts"]["comparison"]).is_file()
            )
            from phase_3_variant_decoder.diagnostic_suite import run_suite
            diagnostic = run_suite(
                cfg,
                embedding_paths={
                    embedding: (str(root / "global.h5"), str(root / "local.h5"))
                    for embedding in ("raw_esm", "seq_phase0", "seq_phase1")
                },
                seeds=[42],
                cache_root=str(root / "diagnostic_cache"),
                out_dir=str(root / "diagnostics"),
                quiet=True,
            )
            self.assertEqual(len(diagnostic["results"]), 6)
            self.assertEqual(diagnostic["reused_keys"], [])
            self.assertTrue(Path(diagnostic["artifacts"]["overview"]).is_file())
            self.assertTrue(Path(diagnostic["artifacts"]["error_breakdown"]).is_file())
            self.assertTrue(Path(diagnostic["artifacts"]["input_ablation"]).is_file())
            self.assertTrue(Path(diagnostic["artifacts"]["encoder_ablation"]).is_file())
            self.assertTrue(Path(diagnostic["artifacts"]["presentation_csv"]).is_file())
            report_text = Path(
                diagnostic["artifacts"]["presentation_markdown"]
            ).read_text(encoding="utf-8")
            self.assertIn("Input comparison", report_text)
            self.assertIn("Single mutation-site residue embedding", report_text)
            from phase_3_variant_decoder.present_diagnostics import present
            rebuilt = present(
                diagnostic["suite_json"], str(root / "rebuilt_presentation"),
                upload=False, project="unused", entity=None, name="unused",
            )
            self.assertIsNone(rebuilt["wandb_url"])
            self.assertTrue(Path(rebuilt["artifacts"]["seed_stability"]).is_file())
            args = SimpleNamespace(checkpoint=best["checkpoint"], cuda="cpu", global_h5=None, local_h5=None,
                                   global_h5_secondary=None, local_h5_secondary=None, dataset_dir=None,
                                   sequences_json=None, research_thresholds=False,
                                   dump_scores=str(root / "scores.csv"), out_json=str(root / "eval.json"))
            # Reloading does not need the original GO artifact.
            (root / "go.npz").unlink()
            from phase_3_variant_decoder.evaluate import run
            result = run(args)  # Checkpoint format dispatch through the public evaluator.
            self.assertEqual(result["test"]["confusion"], metrics["confusion"])
            self.assertTrue((root / "scores_test.csv").is_file())
            self.assertTrue((root / "eval.json").is_file())


if __name__ == "__main__":
    unittest.main()
