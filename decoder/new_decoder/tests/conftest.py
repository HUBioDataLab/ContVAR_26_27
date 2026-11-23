"""Small native artifacts for end-to-end preparation and command-line tests."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def native_data(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "data"
    for name in ("embeddings", "sequences", "splits"):
        (root / name).mkdir(parents=True)
    rng = np.random.default_rng(17)
    sequences = {}
    vectors = {"global": {}, "local": {}}
    for family in ("easy", "hard"):
        for split in ("train", "val", "test"):
            rows = []
            for i, label in enumerate(("LOF", "GOF", "NEUTRAL")):
                protein = f"P_{family}_{split}_{i}"
                sequences[protein] = "AC"
                rows.append(
                    dict(
                        protein_id=protein,
                        snp="A1G",
                        matched_go_id=f"GO:{i+1:07d}",
                        label=label,
                        augmented=False,
                        is_combinatorial=False,
                        match_source="manual",
                    )
                )
                vectors["global"][protein] = rng.normal(size=1280).astype(np.float32)
                vectors["local"][f"{protein}_1"] = rng.normal(size=1280).astype(np.float32)
                for kind in vectors:
                    vectors[kind][f"{protein}_A1G"] = rng.normal(size=1280).astype(np.float32)
            # Keep a source-label conflict; augmentation has its own train/eval policy.
            rows.append(dict(rows[0], label="NEUTRAL"))
            rows.append(dict(rows[2], augmented=True))
            rows.append(dict(rows[0], is_combinatorial=True))
            rows.append(dict(rows[0], snp="A1G:C2D"))
            common = pd.DataFrame(rows)
            layouts = {
                "multiclass": {f"{family}_{split}.tsv": common.assign(target=common.label)},
                "parallel": {
                    f"{family}_{split}.tsv": common.assign(
                        target_lof=common.label.eq("LOF").astype(int),
                        target_gof=common.label.eq("GOF").astype(int),
                    )
                },
                "cascade": {
                    f"stage1_{family}_{split}.tsv": common.assign(
                        target=common.label.map(
                            {"LOF": "FUNCTIONAL", "GOF": "FUNCTIONAL", "NEUTRAL": "NEUTRAL"}
                        )
                    ),
                    f"stage2_{family}_{split}.tsv": common.loc[common.label.ne("NEUTRAL")].assign(
                        target=common.loc[common.label.ne("NEUTRAL"), "label"]
                    ),
                },
            }
            for layout, tables in layouts.items():
                (root / "splits" / layout).mkdir(exist_ok=True)
                for name, table in tables.items():
                    table.to_csv(root / "splits" / layout / name, sep="\t", index=False)
    (root / "sequences/uniprot_dict.json").write_text(json.dumps(sequences))
    for kind, table in vectors.items():
        with h5py.File(root / f"embeddings/phase3_seq_phase1_{kind}.h5", "w") as handle:
            for key, vector in table.items():
                handle[key] = vector
    np.savez(
        root / "embeddings/go_terms_embeddings_anc2vec.npz",
        go_ids=np.array([f"GO:{i+1:07d}" for i in range(3)]),
        embeddings=rng.normal(size=(3, 200)).astype(np.float32),
    )
    archives = tmp_path / "archives"
    archives.mkdir()
    for layout in ("parallel", "multiclass", "cascade"):
        with zipfile.ZipFile(archives / f"{layout}-fixture.zip", "w") as archive:
            for path in sorted((root / "splits" / layout).glob("*.tsv")):
                archive.write(path, f"{layout}/{path.name}")
    return dict(root=root, source=root / "splits", archives=archives)
