from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from contvar_mf.data import PairDataset
from contvar_mf.v5 import FEATURE_KEYS, select_rows


def test_v5_preserves_combination_and_all_explicit_feature_keys():
    keys = {column: f"repaired_{column}" for column in FEATURE_KEYS}
    combo = dict(sample_id="hard:test:0", family="hard", split="test", protein_id="P1", snp="A1G",
                 mutations="A1G;C2D", n_mutations="2", variant_key="P1_A1G;C2D", matched_go_id="GO:0000001",
                 label="LOF", label_idx="0", is_combinatorial="True", augmented="False",
                 match_source="goa_candidate", wt_status="ok", **keys)
    rows = [
        combo,
        dict(combo, sample_id="hard:test:1", label="GOF", label_idx="1"),
        dict(combo, sample_id="hard:test:2", augmented="True", match_source="broad_neutral_broadcast"),
    ]
    frame, report = select_rows(pd.DataFrame(rows), "hard", "test",
                                {"GO:0000001": {"namespace": "molecular_function"}})
    assert frame.label.tolist() == ["LOF", "GOF"]
    assert frame.mutations.tolist() == ["A1G;C2D"] * 2
    assert frame.is_combinatorial.tolist() == [True, True]
    assert frame[FEATURE_KEYS].iloc[0].to_dict() == keys
    assert report["evaluation_augmentation"] == 1
    store = SimpleNamespace(
        allow_combinatorial=True,
        go=SimpleNamespace(lookup=lambda ids: torch.ones(len(ids), 200)),
        maps={scale: {keys[f"wt_{scale}_key"]: 0, keys[f"var_{scale}_key"]: 1}
              for scale in ("global", "mut", "window")},
        global_vectors=torch.stack([torch.zeros(1280), torch.ones(1280)]),
        mut_vectors=torch.stack([torch.zeros(1280), torch.ones(1280)]),
        window_tokens=torch.stack([torch.zeros(7, 1280), torch.ones(7, 1280)]),
        window_masks=torch.ones(2, 7, dtype=torch.bool),
    )
    sample = PairDataset(None, store, "esim", frame)[0]
    assert sample["wt_window"].eq(0).all()
    assert sample["var_window"].eq(1).all()
    store.allow_combinatorial = False
    with pytest.raises(ValueError, match="Combinatorial records"):
        PairDataset(None, store, "esim", frame)
