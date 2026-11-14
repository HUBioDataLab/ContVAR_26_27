"""Meaningful diagnostic invariants plus real TSV/H5/checkpoint inference."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.mutation_sensitivity import (
    class_metrics, dataset_audit, easy_diagnostic, lookup_annotations,
    pair_counts, pair_diagnostic, signed_index,
)
from phase_3_variant_decoder.evaluate_mutation_sensitivity import (
    checkpoints_from_suite, evaluate_one, sha256,
)
from phase_3_variant_decoder.mutation_sensitivity_reports import SECTION, log_wandb
from phase_3_variant_decoder.audit_mutation_cohorts import trace_filters, easy_membership, pair_manifest


def frame(rows):
    return pd.DataFrame(rows, columns=["protein_id", "matched_go_id", "snp", "label"])


def scored(rows, effect, direction):
    out = frame(rows)
    out["pos"] = out.snp.str.extract(r"(\d+)", expand=False).astype(int)
    out["effect_score"], out["direction_score"] = effect, direction
    return out


def fixture(root):
    """Small real files with different variants per group and reverse GO order."""
    from phase_3_variant_decoder.cascade import CHECKPOINT_FORMAT
    from phase_3_variant_decoder.cascade_model import CascadeDecoder
    from phase_3_variant_decoder.config import VariantDecoderConfig
    records = []
    for split in ("train", "val", "test"):
        for protein in (f"{split}_a", f"{split}_b"):
            for snp, label in (("A1G", "LOF"), ("A1V", "NEUTRAL"), ("C2A", "GOF")):
                records.append({"split": split, "protein_id": protein, "snp": snp,
                                "label": label, "matched_go_id": "GO:0000001" if protein.endswith("a") else "GO:0000002",
                                "matched_go_name": "fixture"})
    df = pd.DataFrame(records)
    easy_train = df[df.split == "train"].copy()
    easy_train.protein_id = easy_train.protein_id.str.replace("train_", "test_")
    easy_train.label = "LOF"
    for split in ("train", "val", "test"):
        df[df.split == split].drop(columns="split").to_csv(root / f"hard_{split}.tsv", sep="\t", index=False)
        (easy_train if split == "train" else df[df.split == split]).drop(columns="split").to_csv(root / f"easy_{split}.tsv", sep="\t", index=False)
    (root / "sequences.json").write_text(json.dumps({p: "ACDE" for p in df.protein_id.unique()}))
    rng = np.random.default_rng(7)
    with h5py.File(root / "global.h5", "w") as g, h5py.File(root / "local.h5", "w") as l:
        for protein, sub in df.groupby("protein_id"):
            g[protein] = rng.normal(size=1280).astype("float32")
            for pos in (1, 2):
                l[f"{protein}_{pos}"] = rng.normal(size=1280).astype("float32")
            for snp in sub.snp:
                g[f"{protein}_{snp}"] = rng.normal(size=1280).astype("float32")
                l[f"{protein}_{snp}"] = rng.normal(size=1280).astype("float32")
    cfg = VariantDecoderConfig("hard", "seq_phase1", "both")
    cfg.dataset_dir, cfg.sequences_json = str(root), str(root / "sequences.json")
    cfg.global_h5, cfg.local_h5 = str(root / "global.h5"), str(root / "local.h5")
    cfg.cuda, cfg.batch_size = "cpu", 2
    cfg.go_embeddings = "/does/not/exist.npz"  # Must use checkpoint's GO table.
    model = CascadeDecoder(2560, 3, [8], dropout=.2)
    checkpoint = {"format": CHECKPOINT_FORMAT, "config": vars(cfg), "model_spec": model.spec,
                  "model_state_dict": model.state_dict(), "effect_threshold": .31,
                  "go_vocab": {"GO:0000002": 0, "GO:0000001": 1},
                  "go_vectors": torch.tensor([[4., 5., 6.], [1., 2., 3.]]),
                  "go_artifact_sha256": "fixture"}
    paths = {}
    for family in ("hard", "easy"):
        checkpoint["config"]["family"] = family
        paths[family] = root / f"{family}.pt"
        torch.save(checkpoint, paths[family])
    return paths


class MutationSensitivityTests(unittest.TestCase):
    def test_frozen_raw_challenge_survives_train_majority_change(self):
        # Removing out-of-window LOF training variants flips the learned prior.
        # The target GOF test row must stay in the ORIGINAL raw-data challenge.
        train = frame([("P", "G", "A1023G", "LOF"), ("P", "G", "A1024G", "LOF"),
                       ("P", "G", "A2G", "GOF")]).assign(split="train")
        test = frame([("P", "G", "A1G", "GOF")]).assign(split="test", prediction="GOF")
        raw = pd.concat([train, test], ignore_index=True)
        traced, _ = trace_filters(raw, {"P": "A" * 1100})
        a, manifest = easy_membership(traced)
        self.assertEqual(a["raw_cohort_covered"], 1)
        self.assertEqual(a["filtered_recomputed_cohort"], 0)
        self.assertEqual(a["cohort_removed_by_train_filtering"], 1)
        filtered_train = traced.loc[traced.split.eq("train") & traced.exclusion_reason.eq("")]
        r, annotated = easy_diagnostic(filtered_train, test, reference_train=train)
        self.assertTrue(annotated.anti_majority.iloc[0])
        self.assertFalse(annotated.filtered_train_anti_majority.iloc[0])
        self.assertEqual(r["audit"]["anti_majority_rows"], 1)

    def test_pair_manifest_preserves_raw_pairs_and_marks_unavailable_endpoints(self):
        raw = frame([("P", "G", "A1G", "LOF"), ("P", "G", "A1V", "GOF"),
                     ("P", "G", "A1023G", "NEUTRAL"),
                     ("Q", "H", "A1G", "LOF"), ("Q", "H", "A1V", "GOF")]).assign(split="test")
        traced, steps = trace_filters(raw, {"P": "A" * 1100})
        manifest = pair_manifest(traced)
        self.assertEqual(len(manifest), 4)
        self.assertEqual(int(manifest.model_covered.sum()), 1)
        self.assertEqual(traced.exclusion_reason.value_counts().to_dict(),
                         {"": 2, "protein has no sequence": 2, "pos > 1022": 1})
        self.assertEqual(set(manifest.lower_source_row) | set(manifest.higher_source_row), set(range(5)))

    def test_original_row_ranking_is_separate_from_clean_variant_ranking(self):
        data = scored([("P", "G", "A1G", "LOF"), ("P", "G", "A1G", "LOF"),
                       ("P", "G", "A2G", "GOF"), ("P", "G", "A3G", "GOF"),
                       ("P", "G", "A3G", "NEUTRAL")], [1.] * 5, [.9, .9, .1, .1, .1])
        data["source_row"] = range(5)
        raw, pairs, _ = pair_diagnostic(data, clean=False)
        clean, _, _ = pair_diagnostic(data)
        self.assertEqual(raw["ranking"][0]["n_pairs"], 8)
        self.assertEqual(clean["ranking"][0]["n_pairs"], 1)
        self.assertEqual(int(pairs.same_variant.sum()), 1)
        self.assertTrue((pairs.loc[pairs.same_variant, "credit"] == .5).all())
        self.assertFalse(pairs.loc[pairs.same_variant, "same_position"].any())

    def test_easy_is_anti_train_majority_not_any_opposite_label(self):
        train = frame([("P", "G", "A1G", "LOF"), ("P", "G", "A2G", "LOF"),
                       ("P", "G", "A3G", "GOF"), ("Q", "G", "A1G", "NEUTRAL")])
        test = frame([("P", "G", "A4G", "LOF"), ("P", "G", "A5G", "GOF"),
                      ("Q", "G", "A6G", "LOF"), ("NEW", "G", "A1G", "GOF")])
        test["prediction"] = ["LOF", "GOF", "NEUTRAL", "LOF"]
        result, annotated = easy_diagnostic(train, test)
        self.assertEqual(annotated.anti_majority.tolist(), [False, True, True, False])
        self.assertEqual(result["audit"]["anti_majority_rows"], 2)
        lookup = next(r for r in result["classification"] if r["cohort"] == "Anti-majority" and r["model"] != "Cascade")
        self.assertEqual(lookup["accuracy"], 0)
        self.assertEqual(annotated.lookup_source.tolist(), ["protein_GO", "protein_GO", "protein_GO", "GO"])

    def test_first_seen_tie_and_train_only_fallback(self):
        train = frame([("P", "G", "A1G", "NEUTRAL"), ("P", "G", "A2G", "LOF")])
        test = frame([("P", "G", "A3G", "GOF"), ("Q", "G", "A1G", "GOF"), ("R", "H", "A1G", "GOF")])
        a = lookup_annotations(train, test)
        test.label = "LOF"
        b = lookup_annotations(train, test)
        self.assertEqual(a.lookup_prediction.tolist(), ["NEUTRAL", "NEUTRAL", "LOF"])
        self.assertEqual(a.lookup_prediction.tolist(), b.lookup_prediction.tolist())

    def test_clean_easy_removes_training_variant_and_test_conflict(self):
        train = frame([("P", "G", "A1G", "LOF")])
        test = frame([("P", "G", "A1G", "GOF"), ("P", "G", "A2G", "GOF"),
                      ("P", "G", "A2G", "NEUTRAL"), ("P", "G", "A3G", "GOF"),
                      ("P", "G", "A3G", "GOF")])
        test["prediction"] = "GOF"
        result, annotated = easy_diagnostic(train, test)
        self.assertEqual(result["audit"]["anti_majority_rows"], 5)
        self.assertEqual(int(annotated.anti_majority_clean.sum()), 1)

    def test_signed_score_orientation_and_validation(self):
        np.testing.assert_allclose(signed_index([1, 0, 1], [1, .5, 0]), [-1, 0, 1])
        with self.assertRaises(ValueError):
            signed_index([np.nan], [.5])
        with self.assertRaises(ValueError):
            signed_index([.5], [1.1])

    def test_perfect_reverse_and_constant_ranking(self):
        data = scored([("P", "G", "A1G", "LOF"), ("P", "G", "A1V", "NEUTRAL"),
                       ("P", "G", "C2A", "GOF")], [1., 0., 1.], [1., .5, 0.])
        result, pairs, groups = pair_diagnostic(data)
        self.assertEqual(result["ranking"][0]["pair_accuracy"], 1)
        self.assertEqual(result["ranking"][-1]["n_pairs"], 1)
        self.assertTrue((pairs.stage_credit == 1).all())
        self.assertEqual(len(groups), 1)
        data.direction_score = 1 - data.direction_score
        self.assertEqual(pair_diagnostic(data)[0]["ranking"][0]["pair_accuracy"], 0)
        data.effect_score, data.direction_score = .7, .3
        constant = pair_diagnostic(data)[0]["ranking"][0]
        self.assertEqual(constant["pair_accuracy"], .5)
        self.assertEqual(constant["tie_rate"], 1)

    def test_duplicate_and_conflicting_variant_exclusion(self):
        data = scored([("P", "G", "A1G", "LOF"), ("P", "G", "A1G", "LOF"),
                       ("P", "G", "A2G", "GOF"), ("P", "G", "A3G", "GOF"),
                       ("P", "G", "A3G", "NEUTRAL")], [1.] * 5, [.9, .9, .1, .1, .1])
        result, pairs, _ = pair_diagnostic(data)
        self.assertEqual(result["audit"]["duplicate_variant_GO_label_rows_removed"], 1)
        self.assertEqual(result["audit"]["conflicting_variant_GO_keys_removed"], 1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pair_counts(data)["different_label_row_pairs"], 8)

    def test_group_macro_does_not_weight_big_groups_more(self):
        data = scored([("P", "G", "A1G", "LOF"), ("P", "G", "A2G", "NEUTRAL"),
                       ("P", "G", "A3G", "GOF"), ("Q", "G", "A1G", "LOF"),
                       ("Q", "G", "A2G", "GOF")], [1, 0, 1, 1, 1], [1, .5, 0, 0, 1])
        ranking = pair_diagnostic(data)[0]["ranking"][0]
        self.assertEqual(ranking["pair_accuracy"], .75)
        self.assertEqual(ranking["group_macro_accuracy"], .5)

    def test_empty_cohorts_are_missing_not_zero(self):
        data = scored([("P", "G", "A1G", "LOF")], [.7], [.8])
        result, pairs, _ = pair_diagnostic(data)
        self.assertTrue(pairs.empty)
        self.assertIsNone(result["ranking"][0]["pair_accuracy"])
        empty = data.iloc[:0].assign(prediction=pd.Series(dtype=str))
        self.assertIsNone(class_metrics(empty, "prediction")["macro_f1"])
        with self.assertRaises(ValueError):
            pair_diagnostic(data, -1)

    def test_macro_f1_fixed_class_set_and_tie_tolerance(self):
        data = frame([("P", "G", "A1G", "LOF"), ("P", "G", "A2G", "GOF")])
        data["prediction"] = data.label
        self.assertAlmostEqual(class_metrics(data, "prediction")["macro_f1"], 2/3)
        data["effect_score"], data["direction_score"] = 1., [.5, .5 - 1e-10]
        self.assertEqual(pair_diagnostic(data)[0]["ranking"][0]["pair_accuracy"], .5)

    def test_suite_selects_configuration_not_test_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "suite.json"
            results = {f"{family}_seq_phase1_both_seed42": {"checkpoint": f"/{family}.pt"} for family in ("easy", "hard")}
            results["hard_seq_phase1_both_seed43"] = {"checkpoint": "/not_selected.pt", "test_macro_f1": 1.0}
            path.write_text(json.dumps({"format": "contvar_cascade_diagnostic_suite_v1", "results": results}))
            self.assertEqual(checkpoints_from_suite(path), ["/easy.pt", "/hard.pt"])
            with self.assertRaises(ValueError):
                checkpoints_from_suite(path, seed=44)

    def test_real_inference_matches_existing_cascade_without_changing_checkpoint(self):
        from phase_3_variant_decoder.cascade import attach_checkpoint_go, evaluate_cascade, load_cascade
        from phase_3_variant_decoder.config import VariantDecoderConfig
        from phase_3_variant_decoder.data.pair_dataset import build_pair_tensors
        from phase_3_variant_decoder.data.variant_dataset import prepare
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = fixture(root)
                before = sha256(paths["hard"])
                result = evaluate_one(paths["hard"], root / "output", quiet=True)
                model, ckpt = load_cascade(paths["hard"])
                cfg = VariantDecoderConfig()
                for key, value in ckpt["config"].items():
                    setattr(cfg, key, value)
                df, vocab, _ = prepare(cfg)
                tensors = build_pair_tensors(df[df.split == "test"], cfg)
                attach_checkpoint_go(tensors, vocab, ckpt)
                torch.testing.assert_close(tensors.go_features, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
                metrics, _ = evaluate_cascade(model, tensors, tensors.split_view("test"), ckpt["effect_threshold"])
                self.assertEqual(result["full_test"]["macro_f1"], metrics["macro_f1"])
                self.assertEqual(result["full_test"]["accuracy"], metrics["accuracy"])
                self.assertEqual(sha256(paths["hard"]), before)
                self.assertEqual(result["pairs"]["ranking"][0]["n_pairs"], 6)
                self.assertEqual(result["cohort_audit"]["waterfall"][0]["different_label_row_pairs"], 6)
                self.assertTrue((root / "output/raw_pair_manifest.csv").is_file())
                self.assertEqual(result["clean_pairs"]["ranking"][0]["n_pairs"], 6)
                predictions = pd.read_csv(root / "output/test_predictions.csv")
                self.assertTrue(np.isfinite(predictions.direction_score).all())
                for path in result["figures"].values():
                    self.assertTrue(Path(path).is_file())
                easy = evaluate_one(paths["easy"], root / "easy_output", quiet=True)
                self.assertEqual(easy["easy"]["audit"]["anti_majority_rows"], 4)
                # A mislabeled hard split must fail rather than produce a deceptive report.
                bad = pd.read_csv(root / "hard_train.tsv", sep="\t")
                bad.loc[0, "protein_id"] = "test_a"
                bad.to_csv(root / "hard_train.tsv", sep="\t", index=False)
                with self.assertRaisesRegex(ValueError, "not protein-disjoint"):
                    evaluate_one(paths["hard"], root / "bad_output", quiet=True)
        finally:
            torch.set_num_threads(previous)

    def test_wandb_all_panels_in_one_section(self):
        logged = []
        artifact = SimpleNamespace(add_dir=lambda path: None)
        wandb = SimpleNamespace(Table=lambda **kw: kw, Image=lambda *a, **kw: (a, kw), Artifact=lambda *a, **kw: artifact)
        run = SimpleNamespace(id="test", summary={}, log=lambda p: logged.append(p), log_artifact=lambda a: None)
        data = scored([("P", "G", "A1G", "LOF"), ("P", "G", "A1V", "GOF")], [1., 1.], [1., 0.])
        pairs = pair_diagnostic(data)[0]
        result = {"config": {"family": "hard", "embedding_type": "seq_phase1", "input_mode": "both"},
                  "figures": {"02_HARD_Pair_Ranking": "test.png"}, "pairs": pairs,
                  "full_test": {"macro_f1": .5}}
        log_wandb(run, wandb, {"hard": result}, Path("/tmp"))
        for key in [*logged[0], *run.summary]:
            self.assertTrue(key.startswith(SECTION + "/"))
            self.assertEqual(key.count("/"), 1)
        self.assertEqual(run.summary[f"{SECTION}/hard_ranking_pair_accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
