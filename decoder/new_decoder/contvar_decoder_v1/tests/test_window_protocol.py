import json

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from contvar_decoder.model import LABELS, PairedGODecoder
from contvar_decoder.window_prepare import prepare_window
from contvar_decoder.window_suite import baseline_predictions, common_evaluation


def test_reference_rules_never_use_evaluation_labels():
    train = pd.DataFrame(dict(protein_id=["P1", "P1", "P2"], matched_go_id=["G1", "G1", "G2"], label=["GOF", "GOF", "LOF"]))
    test = pd.DataFrame(dict(protein_id=["P1", "P3", "P4"], matched_go_id=["G1", "G2", "G9"], label=["NEUTRAL"]*3))
    original = baseline_predictions(train, test, "protein_go")
    np.testing.assert_array_equal(original, [1, 0, 1])
    test["label"] = "GOF"
    np.testing.assert_array_equal(original, baseline_predictions(train, test, "protein_go"))


def test_common_test_subset_keeps_original_prediction_row_alignment():
    class Pairs(Dataset):
        def __init__(self):
            self.frame = pd.DataFrame(dict(sample_id=["s0", "s1", "s2", "s3"], label_idx=[0, 1, 2, 0], label=[LABELS[i] for i in [0, 1, 2, 0]]))
            torch.manual_seed(7)
            self.wt = {s: torch.randn(4, 8) for s in ("mut", "global")}
            self.var = {s: torch.randn(4, 8) for s in ("mut", "global")}
            self.go = torch.randn(4, 5)

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return dict(wt={s: t[i] for s, t in self.wt.items()}, var={s: t[i] for s, t in self.var.items()},
                        go=self.go[i], label=int(self.frame.iloc[i].label_idx), row=i)

    ds = Pairs()
    model = PairedGODecoder(scale_dims={"mut": 8, "global": 8}, go_dim=5, width=8, heads=2, dropout=0).eval()
    result, predictions = common_evaluation(ds.frame.iloc[[1, 3]], model, ds, "cpu", 2)
    assert result["n"] == 2
    assert predictions.sample_id.tolist() == ["s1", "s3"]
    with torch.no_grad():
        expected = model({s: t[[1, 3]] for s, t in ds.wt.items()}, {s: t[[1, 3]] for s, t in ds.var.items()}, ds.go[[1, 3]])["scores"]
    np.testing.assert_allclose(predictions[["score_lof", "score_gof", "score_neutral"]], expected.numpy(), atol=1e-6)
    assert len(ds.frame) == 4


def test_preparation_preserves_old_cohort_and_accepts_centered_long_positions(tmp_path):
    data = tmp_path / "data"
    emb = data / "embeddings"
    emb.mkdir(parents=True)
    (data / "sequences").mkdir()
    proteins = {f"P_{split}": "A"*1100 for split in ("train", "val", "test")}
    (data / "sequences/uniprot_dict.json").write_text(json.dumps(proteins), encoding="utf-8")
    np.savez(emb / "go_terms_embeddings_anc2vec.npz", go_ids=np.array(["GO:0000001"]), embeddings=np.ones((1, 200), dtype=np.float32))
    with h5py.File(emb / "phase3_seq_phase1_global.h5", "w") as g, h5py.File(emb / "phase3_seq_phase1_local.h5", "w") as m, h5py.File(emb / "phase3_seq_phase1_window-001.h5", "w") as w:
        g.attrs["center_long_sequences"] = True
        m.attrs["center_long_sequences"] = True
        masks = w.create_group("mask")
        for protein in proteins:
            g[protein] = np.ones(1280, dtype=np.float32)
            for pos in (1, 1050, 2, 3):
                variant = f"{protein}_A{pos}G"
                g[variant] = np.ones(1280, dtype=np.float32)
                for key in (f"{protein}_{pos}", variant):
                    m[key] = np.ones(1280, dtype=np.float32)
                    value = np.ones((33, 1280), dtype=np.float16)
                    mask = np.array([1 <= pos+off <= 1100 for off in range(-16, 17)], dtype=np.uint8)
                    value[mask == 0] = 0
                    w[key] = value
                    masks[key] = mask
    old_root = tmp_path / "old"
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            rows = []
            for pos, label, combo, aug in [(1, "LOF", False, False), (1050, "NEUTRAL", False, False), (2, "GOF", True, False), (3, "NEUTRAL", False, True)]:
                rows.append(dict(protein_id=f"P_{split}", snp=f"A{pos}G", matched_go_id="GO:0000001", label=label,
                                 match_source="test", is_combinatorial=combo, augmented=aug))
            common = pd.DataFrame(rows)
            mc = common.assign(target=common.label)
            parallel = common.assign(target_lof=common.label.eq("LOF").astype(int), target_gof=common.label.eq("GOF").astype(int))
            stage1 = common.assign(target=common.label.map({"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"}))
            stage2 = common.loc[common.label.ne("NEUTRAL")].assign(target=common.loc[common.label.ne("NEUTRAL")].label)
            for head, frame in [("multiclass", mc), ("parallel", parallel), ("cascade", stage1), ("cascade", stage2)]:
                folder = data / "splits" / head
                folder.mkdir(parents=True, exist_ok=True)
                prefix = "stage1_" if head == "cascade" and frame is stage1 else "stage2_" if head == "cascade" else ""
                frame.to_csv(folder / f"{prefix}{family}_{split}.tsv", sep="\t", index=False)
            for head in ("multiclass", "parallel"):
                folder = old_root / head
                folder.mkdir(parents=True, exist_ok=True)
                mc.iloc[:1].assign(sample_id=f"{family}:{split}:0").to_csv(folder / f"{family}_{split}.tsv", sep="\t", index=False)
    output = tmp_path / "prepared"
    prepare_window(data, old_root, tmp_path / "mean.h5", output)
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            full = pd.read_csv(output / "full/multiclass" / f"{family}_{split}.tsv", sep="\t")
            old = pd.read_csv(output / "old/multiclass" / f"{family}_{split}.tsv", sep="\t")
            assert len(old) == 1
            assert "A1050G" in set(full.snp)
            assert not full.is_combinatorial.any()
            assert int(full.augmented.sum()) == (1 if split == "train" else 0)
            assert len(full) == (3 if split == "train" else 2)
