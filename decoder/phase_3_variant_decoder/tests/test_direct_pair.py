"""Concat-only loading, matched cascade inference, raw/clean cohorts and W&B."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.cascade import CHECKPOINT_FORMAT
from phase_3_variant_decoder.cascade_model import CascadeDecoder
from phase_3_variant_decoder.compare_concat_cascade import main, upload_saved, direct_sources
from phase_3_variant_decoder.concat_cascade_evaluation import validate_go_alignment, validate_cascade_config
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.direct_pair_decoder import DirectPairDecoder, load_decoder, train_decoder
from phase_3_variant_decoder.direct_pair_reports import SECTION, create_workspace
from phase_3_variant_decoder.input_controls import fit_normalization
from phase_3_variant_decoder.mutation_sensitivity import pair_diagnostic
from phase_3_variant_decoder.train import prepare_run


def fixture(root):
    rng, sequences, records = np.random.default_rng(7), {}, {}
    base = dict(feature_type="MUTAGEN", ftId="", atomic_description="synthetic fixture",
                matched_go_name="fixture", cosine_similarity=1., match_source="fixture", notes="", is_combinatorial=False)
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            rows = []
            for suffix in ("a", "b"):
                protein = f"{'test' if family == 'easy' and split == 'train' else split}_{suffix}"
                sequences[protein] = "ACDE"
                changes = [("A1G", "LOF"), ("A1V", "GOF"), ("C2A", "NEUTRAL")]
                if family == "easy" and split == "train":
                    changes = [("A1S", "LOF"), ("A1T", "LOF"), ("C2G", "NEUTRAL" if suffix == "a" else "GOF")]
                for snp, label in changes:
                    rows.append({**base, "protein_id": protein, "snp": snp, "label": label,
                                 "matched_go_id": "GO:0000001" if suffix == "a" else "GO:0000002"})
            if family == "hard" and split == "test":
                rows.append(dict(rows[0]))  # Duplicate adds pair weight in primary cohort.
                rows.append({**rows[4], "label": "NEUTRAL"})  # Conflicting same input, clean excludes it.
            stage1 = pd.DataFrame(rows)
            records[family, split] = stage1
            stage1["target"] = stage1.label.map({"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"})
            stage2 = stage1.loc[stage1.label.ne("NEUTRAL")].copy()
            stage2["target"] = stage2.label
            stage1.to_csv(root / f"stage1_{family}_{split}.tsv", sep="\t", index=False)
            stage2.to_csv(root / f"stage2_{family}_{split}.tsv", sep="\t", index=False)
    (root / "seq.json").write_text(json.dumps(sequences))
    np.savez(root / "go.npz", go_ids=np.array(["GO:0000002", "GO:0000001"]), embeddings=rng.normal(size=(2, 5)).astype("float32"))
    all_rows = pd.concat(records.values())
    with h5py.File(root / "global.h5", "w") as g, h5py.File(root / "local.h5", "w") as l:
        for protein, group in all_rows.groupby("protein_id"):
            g[protein] = rng.normal(size=1280).astype("float32")
            for pos in (1, 2):
                l[f"{protein}_{pos}"] = rng.normal(size=1280).astype("float32")
            for snp in group.snp.unique():
                for h5 in (g, l):
                    h5[f"{protein}_{snp}"] = rng.normal(size=1280).astype("float32")
    manifest = dict(format="contvar_direct_pair_comparison_v1", families={})
    suite = dict(format="contvar_cascade_diagnostic_suite_v1", results={})
    for family in ("easy", "hard"):
        cfg = VariantDecoderConfig(family, "seq_phase1", "both")
        cfg.dataset_dir, cfg.sequences_json, cfg.go_embeddings = str(root), str(root / "seq.json"), str(root / "go.npz")
        cfg.global_h5, cfg.local_h5 = str(root / "global.h5"), str(root / "local.h5")
        cfg.projection_dim, cfg.hidden_dims, cfg.balance_classes = 4, [8], True
        cfg.epochs, cfg.batch_size, cfg.dropout, cfg.cuda = 2, 4, .1, "cpu"
        cfg.architecture = "cascade"
        tensors, views, vocab, _ = prepare_run(cfg, quiet=True)
        arm, history = train_decoder("concat", cfg, tensors, views, fit_normalization(tensors, views["train"]),
                                     root / family / "concat", quiet=True)
        arm["checkpoint"] = str(Path(arm["checkpoint"]).relative_to(root))
        columns = ["split", "protein_id", "snp", "matched_go_id", "label"]
        manifest["families"][family] = dict(config=vars(cfg), arms={"concat": arm}, histories={"concat": history},
            provenance=dict(filtered_rows_sha256=hashlib.sha256(tensors.df[columns].to_csv(index=False).encode()).hexdigest()))
        model = CascadeDecoder(2560, 5, [8], dropout=.1)
        reversed_vocab = {g: len(vocab)-i-1 for g, i in vocab.items()}
        path = root / family / "cascade.pt"
        torch.save(dict(format=CHECKPOINT_FORMAT, config=vars(cfg), model_spec=model.spec,
            model_state_dict=model.state_dict(), effect_threshold=.31, direction_threshold=.5,
            selection={"effect": {"epoch": 1}, "direction": {"epoch": 2}}, go_vocab=reversed_vocab,
            go_vectors=tensors.go_features.flip(0).cpu(), go_artifact_sha256=tensors.go_artifact_sha256), path)
        suite["results"][f"{family}_seq_phase1_both_seed42"] = {"checkpoint": str(path)}
    (root / "direct_pair_comparison.json").write_text(json.dumps(manifest))
    (root / "diagnostic_suite.json").write_text(json.dumps(suite))
    return ["--direct_manifest", str(root / "direct_pair_comparison.json"), "--suite_json", str(root / "diagnostic_suite.json"),
            "--no_wandb", "--out_dir", str(root / "reports"), "--cuda", "cpu"]


class DirectPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_only_concat_remains_and_preserves_both_states_and_go(self):
        with self.assertRaises(ValueError):
            DirectPairDecoder("compare", 7, 5)
        model = DirectPairDecoder("concat", 7, 5, 4, [8], 0).eval()
        wt, var, go = torch.randn(3, 7), torch.randn(3, 7), torch.randn(3, 5)
        features = model.features(wt, var, go)
        torch.testing.assert_close(features[:, :4], model.projection(wt))
        torch.testing.assert_close(features[:, 4:8], model.projection(var))
        torch.testing.assert_close(features[:, 8:], go)
        torch.testing.assert_close(model(wt, var, go).softmax(-1).sum(1), torch.ones(3))

    def test_generic_pair_index_uses_softmax_scores_without_inventing_cascade_heads(self):
        frame = pd.DataFrame(dict(protein_id=["P"]*3, matched_go_id=["GO:0000001"]*3,
                                 snp=["A1G", "A1V", "C2A"], label=["LOF", "GOF", "NEUTRAL"],
                                 pos=[1, 1, 2], decision_index=[-.8, .8, 0.]))
        result, pairs, _ = pair_diagnostic(frame, clean=False, score_column="decision_index")
        self.assertEqual(result["ranking"][0]["pair_accuracy"], 1)
        self.assertEqual(result["ranking"][0]["n_pairs"], 3)
        self.assertEqual(result["stage_ranking"], [])
        frame.decision_index = 0.
        result, _, _ = pair_diagnostic(frame, clean=False, score_column="decision_index")
        self.assertEqual(result["ranking"][0]["pair_accuracy"], .5)

    def test_go_alignment_matches_ids_and_rejects_different_vectors(self):
        a = dict(go_vocab={"a": 0, "b": 1}, go_vectors=torch.tensor([[1., 2.], [3., 4.]]))
        b = dict(go_vocab={"a": 1, "b": 0}, go_vectors=a["go_vectors"].flip(0))
        validate_go_alignment(a, b, {"a": 0, "b": 1})
        b["go_vectors"] = b["go_vectors"] + 1
        with self.assertRaisesRegex(ValueError, "different GO vectors"):
            validate_go_alignment(a, b, {"a": 0, "b": 1})

    def test_existing_concat_is_reused_and_cohorts_are_identical_to_cascade(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                args = fixture(root)
                with patch("phase_3_variant_decoder.compare_concat_cascade.train_decoder", side_effect=AssertionError("No retraining")):
                    result = main(args)
            self.assertFalse(result["trained_in_this_experiment"])
            self.assertEqual(set(result["families"]), {"easy", "hard"})
            for family, record in result["families"].items():
                self.assertEqual(set(record["arms"]), {"concat", "cascade"})
                self.assertEqual(record["arms"]["cascade"]["effect_threshold"], .31)
                self.assertIsNone(record["arms"]["concat"]["effect_threshold"])
                for method, arm in record["arms"].items():
                    pred = pd.read_csv(Path(result["manifest_path"]).parent / family / method / "test_predictions.csv")
                    np.testing.assert_allclose(pred.decision_index, pred.class_score_gof-pred.class_score_lof, atol=1e-12)
                    if family == "hard":
                        self.assertEqual(arm["mutation"]["pairs"]["ranking"][0]["n_pairs"], 10)
                        self.assertEqual(arm["mutation"]["clean_pairs"]["ranking"][0]["n_pairs"], 4)
                    else:
                        challenge = next(r for r in arm["mutation"]["easy"]["classification"] if r["cohort"] == "Anti-majority")
                        self.assertEqual(challenge["n"], 4)
            source = Path(result["manifest_path"])
            self.assertEqual(set(direct_sources(source, ("easy", "hard"))), {"easy", "hard"})
            for relative in result["artifacts"].values():
                self.assertTrue((source.parent/relative).is_file())
            run = SimpleNamespace(log=Mock(), summary={}, finish=Mock(), log_artifact=Mock(), config={},
                                  id="fixture", url="https://wandb.test/concat", define_metric=Mock())
            wandb = SimpleNamespace(Table=Mock(), Html=Mock(), Image=Mock(), Artifact=Mock(), init=Mock(return_value=run))
            with patch.dict("sys.modules", {"wandb": wandb}), \
                 patch("phase_3_variant_decoder.compare_concat_cascade.prepare_run", side_effect=AssertionError("No inference")):
                self.assertEqual(upload_saved(source), run.url)
            keys = set().union(*(call.args[0].keys() for call in run.log.call_args_list))
            self.assertIn(f"{SECTION}/02_HARD_Mutation_Sensitivity", keys)
            self.assertIn(f"{SECTION}/02_EASY_Challenge", keys)
            self.assertIn(f"{SECTION}/hard_concat_pair_accuracy", run.summary)

    def test_filter_mismatch_is_rejected(self):
        cfg = VariantDecoderConfig("hard", "seq_phase1", "both")
        saved = {**vars(cfg), "max_seq_len": 512}
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            validate_cascade_config(cfg, saved)

    def test_changed_dataset_identity_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                args = fixture(root)
                path = root / "direct_pair_comparison.json"
                manifest = json.loads(path.read_text())
                manifest["families"]["easy"]["provenance"]["filtered_rows_sha256"] = "different-data"
                path.write_text(json.dumps(manifest))
                with patch("phase_3_variant_decoder.compare_concat_cascade.evaluate_models", side_effect=AssertionError("No inference")):
                    with self.assertRaisesRegex(ValueError, "rows/order/labels"):
                        main(args)

    def test_saved_workspace_has_explicit_first_section_and_run_filter(self):
        import sys
        ws, wr = SimpleNamespace(Workspace=Mock(), Section=Mock(), RunsetSettings=Mock()), SimpleNamespace(MediaBrowser=Mock())
        parent, reports = ModuleType("wandb_workspaces"), ModuleType("wandb_workspaces.reports")
        parent.workspaces, parent.reports, reports.v2 = ws, reports, wr
        with patch.dict(sys.modules, {"wandb_workspaces": parent, "wandb_workspaces.workspaces": ws,
            "wandb_workspaces.reports": reports, "wandb_workspaces.reports.v2": wr}):
            create_workspace(SimpleNamespace(entity="entity", project="project", id="id123"), ["00_Comparison_Table"])
        self.assertEqual(ws.Section.call_args.kwargs["name"], SECTION)
        self.assertIn("id123", ws.RunsetSettings.call_args.kwargs["filters"])


if __name__ == "__main__":
    unittest.main()
