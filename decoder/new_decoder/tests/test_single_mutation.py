import json
import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest

from decoder.new_decoder.data import FeatureStore, PairDataset
from decoder.new_decoder.prepare import combinatorial_mask, prepare


def test_single_mutation_requires_an_explicit_evidence_flag():
    assert combinatorial_mask(
        pd.DataFrame({"is_combinatorial": [False, "False", True, " true "]})
    ).tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match="Missing is_combinatorial"):
        combinatorial_mask(pd.DataFrame({"snp": ["A1G"]}))
    with pytest.raises(ValueError, match="explicitly"):
        combinatorial_mask(pd.DataFrame({"is_combinatorial": [""]}))


def test_preparation_excludes_combination_evidence_and_multiple_substitutions(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    for name in ("embeddings", "sequences", "go"):
        (package / "data" / name).mkdir(parents=True)
    (package / "data/sequences/uniprot_dict.json").write_text(json.dumps({"P1": "AC"}))
    for scale, keys in (("global", ("P1", "P1_A1G")), ("local", ("P1_1", "P1_A1G"))):
        with h5py.File(package / f"data/embeddings/phase3_seq_phase1_{scale}.h5", "w") as handle:
            for key in keys:
                handle[key] = np.ones(1280, dtype=np.float32)
    np.savez(
        package / "data/go/go_terms_embeddings_anc2vec.npz",
        go_ids=np.array(["GO:0000001"]),
        embeddings=np.ones((1, 200), dtype=np.float32),
    )
    base = dict(
        protein_id="P1",
        snp="A1G",
        matched_go_id="GO:0000001",
        augmented=False,
        match_source="manual",
    )
    frame = pd.DataFrame(
        [
            dict(base, label="LOF", target="LOF", is_combinatorial=True),
            dict(base, label="NEUTRAL", target="NEUTRAL", is_combinatorial=False),
            dict(base, snp="A1G:C2D", label="GOF", target="GOF", is_combinatorial=False),
        ]
    )
    archive = tmp_path / "multiclass.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for family in ("easy", "hard"):
            for split in ("train", "val", "test"):
                z.writestr(f"multiclass/{family}_{split}.tsv", frame.to_csv(sep="\t", index=False))
    destination = tmp_path / "prepared"
    report = prepare(archive, package / "data", destination)
    assert report["single_mutation_only"] is True
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            row_report = report["families"][family]["splits"][split]
            assert row_report["excluded"] == {"combinatorial_evidence": 1, "invalid_snp": 1}
            ready = pd.read_csv(destination / f"{family}_{split}.tsv", sep="\t")
            assert ready.label.tolist() == ["NEUTRAL"]
            assert not ready.is_combinatorial.any()
    store = FeatureStore(package / "data")
    try:
        assert len(PairDataset(destination / "hard_train.tsv", store)) == 1
    finally:
        store.close()


@pytest.mark.parametrize("flags", [None, [True]])
def test_loader_rejects_old_or_combination_data_before_loading_embeddings(tmp_path, flags):
    data = {"snp": ["A1G"]}
    if flags is not None:
        data["is_combinatorial"] = flags
    path = tmp_path / "unsafe.tsv"
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="combinatorial|Combined-mutation"):
        PairDataset(path, store=None)
