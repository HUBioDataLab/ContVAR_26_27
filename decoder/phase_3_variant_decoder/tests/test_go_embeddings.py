"""Offline correctness tests; python -m unittest discover -s phase_3_variant_decoder/tests."""
import json
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from phase_3_variant_decoder.prepare_go_embeddings import (
    Term, acquire_sources, build_records, load_pretrained, nearest_ancestors,
    read_obo, resolve_identity, sequence_eligible,
)
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable, decoder_inputs
from phase_2_decoder.models.ffn_decoder import FFNDecoder


def save_fixture(path, proxy=False):
    np.savez(path, go_ids=np.array(["GO:0000001", "GO:0000002"]),
             embeddings=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
             status=np.array(["direct", "ancestor_mean_proxy" if proxy else "direct"]),
             metadata_json=np.asarray(json.dumps({"dimension": 3})))


class OntologyTests(unittest.TestCase):
    def test_parser_preserves_obsolete_and_filters_unsafe_edges(self):
        text = """format-version: 1.2
data-version: test
[Term]
id: GO:0000001
name: child
namespace: molecular_function
alt_id: GO:0000011
is_a: GO:0000002 ! parent
relationship: part_of GO:0000003 ! whole
relationship: regulates GO:0000004
[Term]
id: GO:0000005
is_obsolete: true
replaced_by: GO:0000002
consider: GO:0000003
[Typedef]
id: regulates
"""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "go.obo"
            path.write_text(text)
            terms, aliases, version = read_obo(path)
        self.assertEqual(version, "test")
        self.assertEqual(terms["GO:0000001"].parents, ["GO:0000002", "GO:0000003"])
        self.assertEqual(aliases["GO:0000011"], "GO:0000001")
        self.assertTrue(terms["GO:0000005"].obsolete)
        self.assertNotIn("regulates", terms)

    def test_identity_chains_do_not_use_consider_or_ambiguous_replacements(self):
        vectors = {"target": np.ones(3), "historical": np.ones(3)}
        terms = {"old": Term(obsolete=True, replaced_by=["target"]),
                 "maybe": Term(obsolete=True, consider=["target"]),
                 "ambiguous": Term(obsolete=True, replaced_by=["target", "other"]),
                 "loop1": Term(obsolete=True, replaced_by=["loop2"]),
                 "loop2": Term(obsolete=True, replaced_by=["loop1"])}
        self.assertEqual(resolve_identity("alias", vectors, terms, {"alias": "old"}), ("target", "alt_id+replaced_by"))
        for g in ("maybe", "ambiguous", "loop1"):
            self.assertEqual(resolve_identity(g, vectors, terms, {})[1], "missing")
        self.assertEqual(resolve_identity("historical", vectors, terms, {}), ("historical", "direct"))

    def test_nearest_ancestor_level_same_aspect_and_nonroot(self):
        terms = {"child": Term(namespace="F", parents=["p1", "p2", "cross"]),
                 "p1": Term(namespace="F", parents=["grand"]),
                 "p2": Term(namespace="F"), "grand": Term(namespace="F"),
                 "cross": Term(namespace="P"),
                 "rootchild": Term(namespace="F", parents=["GO:0003674"]),
                 "GO:0003674": Term(namespace="F")}
        vectors = {g: np.ones(3) for g in ["p2", "grand", "cross", "GO:0003674"]}
        self.assertEqual(nearest_ancestors("child", vectors, terms, {}), (["p2"], 1))
        self.assertEqual(nearest_ancestors("rootchild", vectors, terms, {}), ([], 0))
        records = build_records(["child"], vectors, terms, {})
        self.assertEqual(records["child"]["identity_status"], "missing")
        self.assertEqual(records["child"]["proxy_candidates"], ["p2"])

    def test_unverified_pickle_is_rejected_before_numpy_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.npz"
            path.write_bytes(b"untrusted")
            with patch("numpy.load") as load:
                with self.assertRaisesRegex(ValueError, "unverified"):
                    load_pretrained(path)
                load.assert_not_called()

    def test_offline_never_downloads(self):
        with tempfile.TemporaryDirectory() as d, patch("urllib.request.urlopen") as request:
            with self.assertRaises(FileNotFoundError):
                acquire_sources(Path(d), offline=True)
            request.assert_not_called()

    def test_download_verification_atomic_cache_and_failed_cleanup(self):
        payload = b"valid test asset"
        sources = {"asset": ("https://example.org/asset", hashlib.sha256(payload).hexdigest())}
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            with patch("phase_3_variant_decoder.prepare_go_embeddings.SOURCES", sources), patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                acquire_sources(directory)
                acquire_sources(directory, offline=True)
                self.assertEqual(request.call_count, 1)
                self.assertEqual((directory / "asset").read_bytes(), payload)
            (directory / "asset").unlink()
            with patch("phase_3_variant_decoder.prepare_go_embeddings.SOURCES", sources), patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    acquire_sources(directory)
            self.assertEqual(list(directory.iterdir()), [])

    def test_sequence_filter_order_and_bounds(self):
        cases = [("P", "A1G"), ("missing", "A999G"), ("P", "A999G"),
                 ("P", "A3G"), ("P", "C1G"), ("P", "C0G")]
        rows = [{"family": "hard", "protein_id": p, "snp": snp} for p, snp in cases]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seq.json"
            path.write_text(json.dumps([{"Entry": "P", "Sequence": "AC"}]))
            kept, excluded, report = sequence_eligible(rows, path, max_seq_len=10)
        self.assertEqual([r["snp"] for r in kept], ["A1G"])
        self.assertEqual([r["reason"] for r in excluded], ["missing_sequence", "beyond_truncation", "position_out_of_bounds", "wt_residue_mismatch", "position_out_of_bounds"])
        self.assertEqual(report["counts"]["hard"]["kept"], 1)


class DecoderInputTests(unittest.TestCase):
    def test_selected_legacy_two_array_format_loads_without_general_pickle(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "legacy.npz"
            np.savez(path,
                     go_ids=np.array(["GO:0000002", "GO:0000001"], dtype=object),
                     embeddings=np.array([[4, 5, 6], [1, 2, 3]], dtype=np.float32))
            trusted = {hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch("phase_3_variant_decoder.data.go_embeddings._TRUSTED_LEGACY_SHA256", trusted), \
                 patch("numpy.load", wraps=np.load) as load:
                table = GOEmbeddingTable(path)
            # Every NumPy load performed by the adapter must explicitly forbid pickle.
            self.assertTrue(load.call_args_list)
            self.assertTrue(all(call.kwargs.get("allow_pickle") is False for call in load.call_args_list))
            self.assertEqual(table.metadata["source_format"], "legacy_go_ids_embeddings_npz")
            self.assertEqual(table.status, ["direct", "direct"])
            torch.testing.assert_close(
                table.aligned_to_vocab({"GO:0000001": 0, "GO:0000002": 1}),
                torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            )

    def test_legacy_object_array_cannot_import_arbitrary_globals(self):
        class Evil:
            def __reduce__(self):
                return os.system, ("touch should_never_exist",)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "malicious.npz"
            marker = Path(d) / "should_never_exist"
            np.savez(path, go_ids=np.array([Evil()], dtype=object),
                     embeddings=np.ones((1, 3), dtype=np.float32))
            trusted = {hashlib.sha256(path.read_bytes()).hexdigest()}
            old_cwd = os.getcwd()
            try:
                os.chdir(d)
                with patch("phase_3_variant_decoder.data.go_embeddings._TRUSTED_LEGACY_SHA256", trusted):
                    with self.assertRaisesRegex(ValueError, "Forbidden global"):
                        GOEmbeddingTable(path)
            finally:
                os.chdir(old_cwd)
            self.assertFalse(marker.exists())

    def test_untrusted_legacy_object_array_is_rejected_by_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "unknown.npz"
            np.savez(path, go_ids=np.array(["GO:0000001"], dtype=object),
                     embeddings=np.ones((1, 3), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "audited SHA-256"):
                GOEmbeddingTable(path)

    def test_lookup_aligns_vocab_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "go.npz"
            save_fixture(path)
            table = GOEmbeddingTable(path)
        aligned = table.aligned_to_vocab({"GO:0000002": 0, "GO:0000001": 1})
        torch.testing.assert_close(aligned, torch.tensor([[4., 5., 6.], [1., 2., 3.]]))
        self.assertEqual(table.supported_mask(["GO:0000001", "GO:9999999"]).tolist(), [True, False])
        with self.assertRaises(KeyError):
            table.lookup(["GO:9999999"])
        with self.assertRaises(ValueError):
            table.aligned_to_vocab({"GO:0000001": 1})
        with self.assertRaises(IndexError):
            table(torch.tensor([-1]))

    def test_proxy_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "go.npz"
            save_fixture(path, proxy=True)
            with self.assertRaisesRegex(ValueError, "proxies"):
                GOEmbeddingTable(path)
            self.assertEqual(GOEmbeddingTable(path, allow_proxies=True).dimension, 3)

    def test_both_layouts_feed_scalar_decoder_and_backpropagate(self):
        torch.manual_seed(42)
        wt = torch.randn(4, 5, requires_grad=True)
        var = torch.randn(4, 5, requires_grad=True)
        go = torch.randn(4, 3)
        joint = decoder_inputs(wt, var, go)
        self.assertEqual(joint.shape, (4, 18))
        torch.testing.assert_close(joint[:, 10:15], var - wt)
        torch.testing.assert_close(joint[:, -3:], go)
        model = FFNDecoder(18, 1, hidden_dims=[8], dropout=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(joint).squeeze(-1), torch.tensor([0., 1., 0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(wt.grad).all())
        self.assertTrue(torch.isfinite(var.grad).all())
        self.assertGreater(float(wt.grad.abs().sum()), 0)
        left, right = decoder_inputs(wt, var, go, mode="separate")
        self.assertEqual(left.shape, (4, 8))
        torch.testing.assert_close(left[:, -3:], right[:, -3:])
        shared = FFNDecoder(8, 1, hidden_dims=[8], dropout=0)
        self.assertEqual((shared(left) - shared(right)).shape, (4, 1))

    def test_pair_tensors_attach_after_filtering(self):
        import pandas as pd
        from phase_3_variant_decoder.data.pair_dataset import PairTensors
        frame = pd.DataFrame({"matched_go_id": ["GO:0000002", "GO:0000001"], "go_idx": [0, 1]})
        tensors = PairTensors(torch.randn(2, 5), torch.randn(2, 5), frame)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "go.npz"
            save_fixture(path)
            tensors.attach_go_embeddings(path).to("cpu")
        idx = torch.tensor([1, 0])
        wt, var, go = tensors.gather_with_go(idx, idx, idx)
        torch.testing.assert_close(go, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        self.assertEqual(decoder_inputs(wt, var, go).shape, (2, 18))


if __name__ == "__main__":
    unittest.main()
