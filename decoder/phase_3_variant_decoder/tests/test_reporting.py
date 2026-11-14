"""Reporting must preserve results and distinguish cumulative encoder histories."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.presentation_terms import (
    ENCODER_LABELS, INPUT_LABELS, configuration_metadata, image_caption,
)


class ReportingTests(unittest.TestCase):
    def test_confusion_metric_matches_sklearn_including_absent_class(self):
        import numpy as np
        from sklearn.metrics import confusion_matrix, f1_score
        from phase_3_variant_decoder.metrics import metrics_from_confusion
        y, pred = np.array([0, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 0])
        actual = metrics_from_confusion(confusion_matrix(y, pred, labels=[0, 1, 2]))
        for name, average in (("macro_f1", "macro"), ("weighted_f1", "weighted")):
            self.assertAlmostEqual(actual[name], f1_score(
                y, pred, labels=[0, 1, 2], average=average, zero_division=0))
        self.assertEqual(actual["neutral_f1"], 0)

    def test_tsne_does_not_display_a_mutant_as_missing_wt_local(self):
        import numpy as np
        import torch
        from phase_1_dms_fine_tuning import visualize_tsne as visual
        lengths = []

        class LayoutProjection:
            def __init__(self, **kwargs):
                pass

            def fit_transform(self, values):
                lengths.append(len(values))
                return values[:, :2]

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "fixture.npz"
            values = np.arange(32).reshape(8, 4).astype(np.float32)
            np.savez(source, X=values, X_local=values, y_bin=np.tile([0, 1], 4),
                     y_dms_id=np.repeat([1, 2], 4),
                     wt_embeddings_global={1: np.ones(4), 2: np.ones(4)})
            with patch.object(visual, "TSNE", LayoutProjection):
                visual.plot_tsne(str(source), torch.nn.Identity(), torch.nn.Identity(),
                                 "cpu", save_dir=tmp, n_proteins=2)
        self.assertEqual(lengths, [10, 10, 8, 8, 8, 8])

    def test_encoder_history_and_input_concatenation_are_different_axes(self):
        cfg = VariantDecoderConfig("easy", "concat_phase1", "both")
        _, metadata = cfg.wandb_metadata()
        self.assertIn("Phase 0", metadata["encoder_label"])
        self.assertIn("Phase 1", metadata["encoder_label"])
        self.assertIn("pretrained ESM-2", metadata["encoder_label"])
        self.assertEqual(metadata["input_label"], "Global + Local (concat)")
        self.assertEqual(cfg.input_dim, 5120)
        self.assertIn("not a DMS-only", configuration_metadata({"embedding": "seq_phase1"})["encoder_training_history"])
        self.assertEqual(INPUT_LABELS["local"], "Local only")
        self.assertIn("does not include a DMS-only", image_caption("encoder_ablation"))

    def test_refresh_rebuilds_artifacts_without_training_or_changing_scores(self):
        import numpy as np
        from phase_3_variant_decoder.metrics import metrics_from_confusion
        from phase_3_variant_decoder.refresh_reports import refresh
        # Counts are a synthetic fixture, not experimental model performance.
        metric = metrics_from_confusion(np.array([[4, 1, 1], [1, 3, 2], [1, 1, 4]]))
        metric["confusion"] = [[4, 1, 1], [1, 3, 2], [1, 1, 4]]
        for label in ("lof", "gof", "neutral"):
            for key in ("precision", "recall", "auprc"):
                metric.setdefault(f"{label}_{key}", .5)
        results = {name: {"metrics": {"val": metric, "test": metric}}
                   for name in ("legacy", "cascade")}
        cfg = VariantDecoderConfig("hard", "seq_phase1", "local")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "architecture_comparison.json"
            source.write_text(json.dumps({"format": "contvar_decoder_architecture_comparison_v1",
                                          "config": vars(cfg), "runs": results}))
            original = source.read_bytes()
            with patch("phase_3_variant_decoder.cascade.fit_cascade", side_effect=AssertionError("No training allowed")), \
                 patch("phase_3_variant_decoder.train.prepare_run", side_effect=AssertionError("No data reload allowed")):
                rebuilt = refresh(source)
            self.assertEqual(original, source.read_bytes())
            self.assertTrue(rebuilt["presentation_only"])
            self.assertIsNone(rebuilt["wandb_url"])
            with Path(rebuilt["artifacts"]["summary_csv"]).open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]["macro_f1"]), metric["macro_f1"])
            self.assertEqual(rows[0]["encoder_label"], ENCODER_LABELS["seq_phase1"])
            self.assertEqual(rows[0]["input_label"], "Local only")
            with self.assertRaisesRegex(ValueError, "separate output"):
                refresh(source, output_dir=tmp)


if __name__ == "__main__":
    unittest.main()
