"""Hinge objectives, train-only balancing, old checkpoint compatibility and W&B."""
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
from torch.nn import functional as F

from phase_3_variant_decoder.cascade import fit_stage, load_cascade
from phase_3_variant_decoder.cascade_model import CascadeDecoder
from phase_3_variant_decoder.cascade_objectives import cascade_loss_per_row
from phase_3_variant_decoder.compare_cascade_losses import SECTIONS, log_results, run_experiment, upload_saved
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.data.pair_dataset import PairTensors


class CascadeLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_hinge_values_and_margin_gradients_for_both_binary_labels(self):
        logits = torch.tensor([-2., 0., .5, 1., 2.], requires_grad=True)
        targets = torch.ones(5)
        hinge = cascade_loss_per_row(logits, targets, kind="hinge")
        squared = cascade_loss_per_row(logits, targets, kind="squared_hinge")
        torch.testing.assert_close(hinge, torch.tensor([3., 1., .5, 0., 0.]))
        torch.testing.assert_close(squared, torch.tensor([9., 1., .25, 0., 0.]))
        hinge.sum().backward(retain_graph=True)
        torch.testing.assert_close(logits.grad, torch.tensor([-1., -1., -1., 0., 0.]))
        logits.grad.zero_()
        squared.sum().backward()
        torch.testing.assert_close(logits.grad, torch.tensor([-6., -2., -1., 0., 0.]))
        torch.testing.assert_close(cascade_loss_per_row(-logits.detach(), 1-targets), hinge.detach())
        torch.testing.assert_close(cascade_loss_per_row(logits.detach(), targets, kind="bce"),
                                   F.binary_cross_entropy_with_logits(logits.detach(), targets, reduction="none"))
        with self.assertRaises(ValueError):
            cascade_loss_per_row(logits, targets, margin=0)
        with self.assertRaises(ValueError):
            cascade_loss_per_row(logits, targets, margin=float("nan"))

    def test_effect_and_direction_signed_gradient_semantics(self):
        for kind in ("hinge", "squared_hinge"):
            # Effect: targets 0=NEUTRAL, 1=effect. Direction: 0=GOF, 1=LOF.
            score = torch.tensor([0., 0.], requires_grad=True)
            loss = cascade_loss_per_row(score, torch.tensor([0., 1.]), kind=kind)
            loss.sum().backward()
            self.assertGreater(score.grad[0], 0)  # gradient descent decreases the negative-class logit
            self.assertLess(score.grad[1], 0)

    def test_stage_loss_weighting_is_train_only_and_neutral_does_not_train_direction(self):
        frame = pd.DataFrame([
            {"split": split, "wt_row": i, "var_row": i, "go_idx": 0, "label_idx": label,
             "seen_in_train": True, "is_alanine": False}
            for split in ("train", "val", "test") for i, label in enumerate((0, 0, 0, 1, 2, 2))])
        torch.manual_seed(3)
        tensors = PairTensors(torch.randn(6, 4), torch.randn(6, 4), frame)
        tensors.go_features = torch.randn(1, 2)
        views = {s: tensors.split_view(s) for s in ("train", "val", "test")}
        cfg = VariantDecoderConfig()
        cfg.cascade_loss, cfg.epochs, cfg.batch_size = "squared_hinge", 1, 6
        cfg.effect_threshold_strategy = "learned_radius"
        model = CascadeDecoder(4, 2, [8], dropout=0)
        with patch("phase_3_variant_decoder.cascade.select_effect_threshold", side_effect=AssertionError("No threshold sweep")):
            selection, _ = fit_stage(model.direction_stage, "direction", tensors, views, cfg, quiet=True)
        self.assertEqual(selection["train_n"], 4)
        self.assertEqual(selection["train_class_counts_0_1"], [1, 3])
        np.testing.assert_allclose(selection["class_weights_0_1"], [2., 2/3])
        self.assertEqual(selection["loss"], "squared_hinge")
        self.assertTrue(all(p.grad is None for p in model.effect_stage.parameters()))
        selection, _ = fit_stage(model.effect_stage, "effect", tensors, views, cfg, quiet=True)
        self.assertEqual(selection["train_class_counts_0_1"], [2, 4])
        np.testing.assert_allclose(selection["class_weights_0_1"], [1.5, .75])

    def test_train_cli_defaults_hinge_and_allows_explicit_alternatives(self):
        from phase_3_variant_decoder.train import main
        base = ["train", "--family", "hard", "--embedding", "seq_phase1", "--global_h5", "g.h5", "--local_h5", "l.h5", "--go_embeddings", "go.npz", "--no_wandb"]
        for kind in (None, "squared_hinge", "bce"):
            with patch("sys.argv", base + (["--cascade_loss", kind] if kind else [])), \
                 patch("phase_3_variant_decoder.train.train") as train:
                main()
                self.assertEqual(train.call_args.args[0].cascade_loss, kind or "hinge")

    def test_real_tsv_h5_comparison_checkpoints_reports_and_wandb_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(12)
            sequences = {}
            with h5py.File(root / "global.h5", "w") as glob, h5py.File(root / "local.h5", "w") as local:
                for split in ("train", "val", "test"):
                    rows = []
                    for i, label in enumerate(("LOF", "GOF", "NEUTRAL")*2):
                        protein = f"{split}_{i}"
                        sequences[protein] = "ACDE"
                        rows.append({"protein_id": protein, "snp": "A1G", "matched_go_id": "GO:0000001", "label": label})
                        for key in (protein, protein+"_A1G"):
                            glob[key] = rng.normal(size=1280).astype("float32")
                        for key in (protein+"_1", protein+"_A1G"):
                            local[key] = rng.normal(size=1280).astype("float32")
                    pd.DataFrame(rows).to_csv(root / f"hard_{split}.tsv", sep="\t", index=False)
            (root / "seq.json").write_text(json.dumps(sequences))
            np.savez(root / "go.npz", go_ids=np.array(["GO:0000001"]), embeddings=np.array([[1., 2., 3.]], dtype=np.float32))
            cfg = VariantDecoderConfig("hard", "seq_phase1", "both")
            cfg.dataset_dir, cfg.sequences_json = str(root), str(root / "seq.json")
            cfg.global_h5, cfg.local_h5 = str(root / "global.h5"), str(root / "local.h5")
            cfg.go_embeddings = str(root / "go.npz")
            cfg.hidden_dims, cfg.dropout, cfg.epochs, cfg.batch_size = [8], .1, 2, 4
            cfg.cuda, cfg.wandb_enabled = "cpu", False
            cfg.effect_threshold_strategy = "learned_radius"
            with patch("phase_3_variant_decoder.cascade.select_effect_threshold", side_effect=AssertionError("No external threshold")):
                comparison = run_experiment(cfg, out_dir=root / "experiment", quiet=True)
            self.assertEqual(set(comparison["runs"]), {"hinge", "squared_hinge"})
            self.assertEqual(cfg.cascade_loss, "hinge")
            source = Path(comparison["manifest_path"])
            artifacts = {k: str(source.parent / p) for k, p in comparison["artifacts"].items()}
            details = {kind: {k: str(source.parent / p) for k, p in files.items()}
                       for kind, files in comparison["method_artifacts"].items()}
            for path in artifacts.values():
                self.assertTrue(Path(path).is_file())
            for kind, result in comparison["runs"].items():
                self.assertEqual(result["effect_threshold"], .5)
                model, checkpoint = load_cascade(result["checkpoint"])
                self.assertEqual(checkpoint["config"]["cascade_loss"], kind)
                self.assertEqual(checkpoint["loss_spec"]["margin_space"], "scaled_signed_logit")
                # An actual pre-selector checkpoint must still be interpreted as BCE.
                checkpoint["config"].pop("cascade_loss")
                checkpoint["config"].pop("cascade_margin")
                checkpoint.pop("loss_spec")
                old = root / f"old_{kind}.pt"
                torch.save(checkpoint, old)
                _, restored = load_cascade(old)
                self.assertEqual(restored["config"]["cascade_loss"], "bce")
            run = SimpleNamespace(log=Mock(), summary={}, finish=Mock(), log_artifact=Mock(), url="https://wandb.test/losses", define_metric=Mock())
            wb = SimpleNamespace(Table=Mock(), Html=Mock(), Image=Mock(), Artifact=Mock(), init=Mock(return_value=run))
            log_results(run, wb, comparison["runs"], artifacts, details)
            keys = set().union(*(c.args[0].keys() for c in run.log.call_args_list))
            for kind in ("hinge", "squared_hinge"):
                self.assertIn(f"{SECTIONS[kind]}/Results/Scores", keys)
                self.assertIn(f"{SECTIONS[kind]}/test/macro_f1", run.summary)
            self.assertIn("Cascade_Loss_Comparison/02_Final_F1", keys)
            with patch.dict("sys.modules", {"wandb": wb}), \
                 patch("phase_3_variant_decoder.compare_cascade_losses.fit_cascade", side_effect=AssertionError("Do not retrain")):
                url = upload_saved(source)
            self.assertEqual(url, run.url)


if __name__ == "__main__":
    unittest.main()
