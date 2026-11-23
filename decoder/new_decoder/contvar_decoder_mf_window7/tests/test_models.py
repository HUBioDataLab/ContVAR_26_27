"""Behavioral tests for preserved scorers and the new aligned comparison paths."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from contvar_mf.model import PairedGODecoder, classification_loss, masked_mean
from contvar_mf.prepare import boolean_column, ontology_namespaces, row_statistics
from contvar_mf.util import run_conditions

torch.set_num_threads(2)


def inputs():
    torch.manual_seed(123)
    return dict(wt={"mut": torch.randn(4, 8), "global": torch.randn(4, 8)},
                var={"mut": torch.randn(4, 8), "global": torch.randn(4, 8)},
                go=torch.randn(4, 5))


def add_windows(batch):
    batch.update(wt_window=torch.randn(4, 7, 8), var_window=torch.randn(4, 7, 8),
                 wt_mask=torch.ones(4, 7, dtype=torch.bool), var_mask=torch.ones(4, 7, dtype=torch.bool))
    return batch


@pytest.mark.parametrize("head", ["parallel", "multiclass"])
@pytest.mark.parametrize("pair_mode", ["joint", "score_difference"])
@pytest.mark.parametrize("mode", ["none", "mean"])
def test_preserved_models_match_handover_logits(head, pair_mode, mode):
    root = Path(__file__).resolve().parents[2]
    source = root / "contvar_decoder_v1/contvar_decoder/model.py"
    spec = importlib.util.spec_from_file_location("historical_decoder_for_parity", source)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    scales = {"mut": 8, "global": 8, **({"window": 8} if mode == "mean" else {})}
    old = legacy.PairedGODecoder(scale_dims=scales, go_dim=5, width=8, heads=2,
                                 head=head, pair_mode=pair_mode).eval()
    new = PairedGODecoder(input_dim=8, go_dim=5, width=8, heads=2,
                          head=head, pair_mode=pair_mode, window_mode=mode).eval()
    new.load_state_dict(old.state_dict())
    batch = inputs()
    if mode == "mean":
        batch["wt"]["window"] = torch.randn(4, 8)
        batch["var"]["window"] = torch.randn(4, 8)
    with torch.no_grad():
        a, b = old(**batch), new(**batch)
    torch.testing.assert_close(a["logits"], b["logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a["scores"], b["scores"], atol=1e-6, rtol=1e-6)
    assert torch.equal(a["prediction"], b["prediction"])


@pytest.mark.parametrize("mode,tokens", [("pair_mean", 1), ("esim", 7)])
@pytest.mark.parametrize("head", ["parallel", "multiclass"])
def test_pair_branch_receives_gradients_and_reads_expected_tokens(mode, tokens, head):
    model = PairedGODecoder(input_dim=8, go_dim=5, width=8, heads=2, head=head, window_mode=mode)
    batch = add_windows(inputs())
    output = model(**batch)
    assert output["attention"]["wt"].shape == (4, 2)
    assert output["attention"]["pair"].shape == (4, tokens)
    loss = classification_loss(output, torch.tensor([0, 1, 2, 0]), head)
    loss.backward()
    for module in [model.window_adapter, model.comparison, model.local_reader]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in module.parameters())


@pytest.mark.parametrize("mode", ["pair_mean", "esim"])
def test_padding_values_cannot_change_predictions(mode):
    model = PairedGODecoder(input_dim=8, go_dim=5, width=8, heads=2, window_mode=mode).eval()
    batch = add_windows(inputs())
    batch["wt_mask"][:, :2] = False
    batch["var_mask"][:, :2] = False
    with torch.no_grad():
        before = model(**batch)
        batch["wt_window"][:, :2] = float("nan")
        batch["var_window"][:, :2] = float("inf")
        after = model(**batch)
    torch.testing.assert_close(before["logits"], after["logits"])
    if mode == "esim":
        assert torch.equal(after["attention"]["pair"][:, :2], torch.zeros(4, 2))


def test_esim_attention_is_go_conditioned():
    model = PairedGODecoder(input_dim=8, go_dim=5, width=8, heads=2, window_mode="esim").eval()
    batch = add_windows(inputs())
    with torch.no_grad():
        a = model(**batch)["attention"]["pair"]
        batch["go"] = torch.randn(4, 5) * 3
        b = model(**batch)["attention"]["pair"]
    assert not torch.allclose(a, b)


def test_pair_mean_and_esim_have_equal_parameter_counts():
    models = [PairedGODecoder(input_dim=8, go_dim=5, width=8, heads=2, window_mode=m)
              for m in ["pair_mean", "esim"]]
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())


def test_crop_and_masked_mean_include_exactly_center_seven():
    raw = torch.arange(33.).view(1, 33, 1)
    crop, mask = raw[:, 13:20], torch.ones(1, 7, dtype=torch.bool)
    assert crop[:, 3].item() == 16
    assert masked_mean(crop, mask).item() == 16
    mask[:, :3] = False
    crop = crop.clone()
    crop[:, :3] = float("nan")
    assert masked_mean(crop, mask).item() == 17.5


def test_no_joint_pair_branch_in_pure_score_difference():
    with pytest.raises(ValueError):
        PairedGODecoder(pair_mode="score_difference", window_mode="esim")


def test_only_approved_48_conditions():
    plan = run_conditions()
    assert len(plan) == 48 and len({c["name"] for c in plan}) == 48
    assert [c["number"] for c in plan] == list(range(1, 49))
    assert sum(c["window_mode"] == "none" for c in plan) == 16
    assert sum(c["window_mode"] == "mean" for c in plan) == 16
    assert sum(c["window_mode"] == "pair_mean" for c in plan) == 8
    assert sum(c["window_mode"] == "esim" for c in plan) == 8
    assert all(c["pair_mode"] == "joint" for c in plan if c["window_mode"] in ["pair_mean", "esim"])


def test_duplicate_and_conflicting_labels_are_reported_not_removed():
    import pandas as pd
    frame = pd.DataFrame(dict(protein_id=["P", "P", "P"], snp=["A1V"] * 3,
                              matched_go_id=["GO:0005509"] * 3,
                              label=["LOF", "LOF", "NEUTRAL"], augmented=[False] * 3))
    stats = row_statistics(frame)
    assert len(frame) == stats["rows"] == stats["conflicting_rows"] == 3
    assert stats["same_label_duplicate_extra"] == 1


def test_namespace_uses_official_metadata_and_preserves_original_alias(tmp_path):
    import json
    path = tmp_path / "go.json"
    path.write_text(json.dumps({"graphs": [{"id": "go", "nodes": [{
        "id": "http://purl.obolibrary.org/obo/GO_0005509", "lbl": "misleading name",
        "meta": {"basicPropertyValues": [
            {"pred": "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace", "val": "molecular_function"},
            {"pred": "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId", "val": "GO:9999999"}]}}]}]}))
    table, _ = ontology_namespaces(path)
    assert table["GO:0005509"]["namespace"] == "molecular_function"
    assert table["GO:9999999"]["go_id"] == "GO:9999999"
    assert table["GO:9999999"]["canonical_go_id"] == "GO:0005509"
