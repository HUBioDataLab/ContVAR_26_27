"""Coverage invariants using actual TSV, NPZ and H5 fixtures, without ESM/GPU."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

from phase_3_variant_decoder.audit_input_embeddings import audit, audit_h5


def fixture(root):
    for split in ("train", "val", "test"):
        pd.DataFrame([
            ("P", "A1G", "GO:0000001", "LOF"),
            ("P", "C2A", "GO:0000002", "NEUTRAL"),
            ("P", "A1023G", "GO:0000001", "LOF"),
            ("Q", "A1G", "GO:0000001", "GOF"),
        ], columns=["protein_id", "snp", "matched_go_id", "label"]).to_csv(
            root / f"hard_{split}.tsv", sep="\t", index=False)
    (root / "seq.json").write_text(json.dumps({"P": "ACDE"}))
    np.savez(root / "go.npz", go_ids=np.array(["GO:0000002", "GO:0000001"]),
             embeddings=np.array([[4., 5., 6.], [1., 2., 3.]], dtype="float32"))
    with h5py.File(root / "global.h5", "w") as g, h5py.File(root / "local.h5", "w") as l:
        for key in ("P", "P_A1G", "P_C2A"):
            g[key] = np.ones(1280, dtype="float32")
        for key in ("P_1", "P_2", "P_A1G", "P_C2A"):
            l[key] = np.ones(1280, dtype="float32")
    return SimpleNamespace(family="hard", dataset_dir=str(root), sequences_json=str(root / "seq.json"),
                           global_h5=str(root / "global.h5"), local_h5=str(root / "local.h5"),
                           go_embeddings=str(root / "go.npz"), out_dir=str(root / "report"),
                           max_seq_len=1022, drop_contradictory=False, allow_go_proxies=False,
                           hash_h5=False)


class InputEmbeddingAuditTests(unittest.TestCase):
    def run_audit(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return audit(args)

    def test_filtered_out_rows_remain_visible_without_causing_false_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            report = self.run_audit(args)
            self.assertTrue(report["complete_for_pipeline_rows"])
            rows = pd.read_csv(Path(args.out_dir) / "sample_coverage.csv")
            self.assertEqual(len(rows), 12)
            self.assertEqual(int(rows.pipeline_eligible.sum()), 6)
            self.assertEqual(int(rows.decoder_ready.sum()), 6)
            for group in report["rows"]:
                self.assertEqual(group["raw_rows"], 4)
                self.assertEqual(group["pipeline_excluded_rows"], 2)
                self.assertEqual(group["pipeline_rows_with_bad_or_missing_input"], 0)

    def test_missing_wt_position_is_found_even_when_variant_vector_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with h5py.File(args.local_h5, "a") as h5:
                del h5["P_2"]
            report = self.run_audit(args)
            self.assertFalse(report["complete_for_pipeline_rows"])
            bad = pd.read_csv(Path(args.out_dir) / "missing_or_invalid_inputs.csv")
            self.assertEqual(len(bad), 3)
            self.assertEqual(set(bad.snp), {"C2A"})
            self.assertEqual(set(bad.wt_local_status), {"missing_key"})
            self.assertEqual(set(bad.var_local_status), {"valid"})

    def test_missing_go_is_reported_per_sample_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            np.savez(args.go_embeddings, go_ids=np.array(["GO:0000001"]),
                     embeddings=np.ones((1, 3), dtype="float32"))
            report = self.run_audit(args)
            self.assertFalse(report["complete_for_pipeline_rows"])
            bad = pd.read_csv(Path(args.out_dir) / "missing_or_invalid_inputs.csv")
            self.assertEqual(len(bad), 3)
            self.assertEqual(set(bad.matched_go_id), {"GO:0000002"})
            self.assertEqual(set(bad.go_status), {"missing_key"})

    def test_wrong_shape_nonfinite_zero_and_nonfloat_are_not_counted_as_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vectors.h5"
            with h5py.File(path, "w") as h5:
                h5["valid"] = np.ones(1280, dtype="float32")
                h5["broadcastable_but_wrong"] = np.ones((1, 1280), dtype="float32")
                h5["nonfinite"] = np.full(1280, np.nan, dtype="float32")
                h5["zero"] = np.zeros(1280, dtype="float32")
                h5["integer"] = np.ones(1280, dtype="int32")
                h5.create_group("group")
            states, metadata = audit_h5(path, ["valid", "broadcastable_but_wrong", "nonfinite",
                                                "zero", "integer", "group", "missing"], hash_file=True)
            self.assertEqual(states, {"valid": "valid", "broadcastable_but_wrong": "wrong_shape",
                                      "nonfinite": "nonfinite", "zero": "zero_vector",
                                      "integer": "not_floating_point", "group": "not_a_dataset",
                                      "missing": "missing_key"})
            self.assertEqual(metadata["status_counts"]["valid"], 1)
            self.assertEqual(len(metadata["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
