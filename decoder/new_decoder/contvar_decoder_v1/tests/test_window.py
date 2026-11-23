import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from contvar_decoder.data import FeatureStore, PairDataset
from contvar_decoder.model import LABELS, PairedGODecoder, classification_loss, load_decoder
from contvar_decoder.window_prepare import build_mean_cache, masked_window_mean


def test_padding_is_excluded_and_accumulation_is_float32():
    values = np.array([[1, 4], [3, 8], [np.nan, np.inf]], dtype=np.float32)
    result = masked_window_mean(values, np.array([1, 1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(result, [2, 6])
    assert result.dtype == np.float32
    many = np.full((33, 2), 40000, dtype=np.float16)
    assert np.isfinite(masked_window_mean(many, np.ones(33))).all()
    with pytest.raises(ValueError, match="no valid"):
        masked_window_mean(values, np.zeros(3))
    with pytest.raises(ValueError, match="binary"):
        masked_window_mean(values, [1, 2, 0])
    with pytest.raises(ValueError, match="Non-finite valid"):
        masked_window_mean(values, [1, 1, 1])


@pytest.mark.parametrize("head", ["parallel", "multiclass"])
@pytest.mark.parametrize("mode", ["joint", "score_difference"])
def test_window_is_used_and_learns_in_all_four_models(head, mode, tmp_path):
    torch.manual_seed(12)
    net = PairedGODecoder(scale_dims={"mut": 8, "global": 8, "window": 8},
                         go_dim=5, width=8, heads=2, head=head, pair_mode=mode, dropout=0)
    wt = {s: torch.randn(6, 8) for s in net.scales}
    var = {s: torch.randn(6, 8) for s in net.scales}
    go = torch.randn(6, 5)
    out = net(wt, var, go)
    changed = dict(var, window=var["window"] + torch.randn(6, 8) * 3)
    assert not torch.allclose(out["logits"], net(wt, changed, go)["logits"])
    loss = classification_loss(out, torch.tensor([0, 1, 2, 0, 1, 2]), head)
    loss.backward()
    assert net.adapters["window"][1].weight.grad.abs().sum() > 0
    if mode == "score_difference":
        reverse = net(var, wt, go)
        torch.testing.assert_close(out["d"], -reverse["d"], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(out["side_scores"]["wt"], net(wt, changed, go)["side_scores"]["wt"])
    path = tmp_path / "checkpoint.pt"
    net.eval()
    torch.save(dict(format="contvar_go_decoder_starter_v1", model_spec=net.spec,
                    state_dict=net.state_dict(), labels=list(LABELS)), path)
    loaded, _ = load_decoder(path)
    torch.testing.assert_close(net(wt, var, go)["logits"], loaded(wt, var, go)["logits"], atol=0, rtol=0)


def test_real_hdf_layout_mask_pool_and_wt_position_lookup(tmp_path):
    root = tmp_path / "inputs"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    np.savez(emb / "go_terms_embeddings_anc2vec.npz", go_ids=np.array(["GO:0000001"]),
             embeddings=np.ones((1, 200), dtype=np.float32))
    keys = ("P1_1", "P1_A1G")
    with h5py.File(emb / "phase3_seq_phase1_local.h5", "w") as local, \
            h5py.File(emb / "phase3_seq_phase1_window-001.h5", "w") as windows:
        masks = windows.create_group("mask")
        for i, key in enumerate(keys):
            value = np.zeros((33, 1280), dtype=np.float16)
            value[16:18] = i + np.array([2, 6])[:, None]
            mask = np.zeros(33, dtype=np.uint8)
            mask[16:18] = 1
            windows[key] = value
            masks[key] = mask
            local[key] = value[16].astype(np.float32)
    with h5py.File(emb / "phase3_seq_phase1_global.h5", "w") as h:
        h["P1"] = np.ones(1280, dtype=np.float32)
        h["P1_A1G"] = np.ones(1280, dtype=np.float32) * 2
    cache = tmp_path / "mean.h5"
    build_mean_cache(root, cache)
    build_mean_cache(root, cache)  # verified reuse, never a duplicate append
    frame = pd.DataFrame([dict(snp="A1G", is_combinatorial=False, augmented=False, label="LOF", label_idx=0,
        matched_go_id="GO:0000001", target="LOF", wt_mut_key="P1_1", var_mut_key="P1_A1G",
        wt_global_key="P1", var_global_key="P1_A1G")])
    path = tmp_path / "split.tsv"
    frame.to_csv(path, sep="\t", index=False)
    store = FeatureStore(root, window_pool="mean", window_cache=cache)
    try:
        ds = PairDataset(path, store, head="multiclass")
        batch = ds[0]
        torch.testing.assert_close(batch["wt"]["window"], torch.full((1280,), 4.))
        torch.testing.assert_close(batch["var"]["window"], torch.full((1280,), 5.))
        store.preload([ds])
        torch.testing.assert_close(ds[0]["wt"]["window"], batch["wt"]["window"])
    finally:
        store.close()
    with h5py.File(cache, "r+") as h:
        h.attrs["complete"] = False
    with pytest.raises(ValueError, match="Existing mean cache"):
        build_mean_cache(root, cache)
