import numpy as np
import pandas as pd
import pytest

from decoder.new_decoder.data import PairDataset


class DummyGO:
    def lookup(self, identifiers):
        import torch

        return torch.ones(len(identifiers), 200)


class DummyStore:
    go = DummyGO()


def test_parallel_loader_uses_source_binary_targets_and_rejects_wrong_mapping(tmp_path):
    frame = pd.DataFrame(
        dict(
            snp=["A1G", "A1G"],
            is_combinatorial=[False, False],
            label=["LOF", "NEUTRAL"],
            label_idx=[0, 2],
            matched_go_id=["GO:0000001"] * 2,
            target_lof=[1, 0],
            target_gof=[0, 0],
            augmented=[False, False],
            wt_mut_key=["P1_1"] * 2,
            var_mut_key=["P1_A1G"] * 2,
            wt_global_key=["P1"] * 2,
            var_global_key=["P1_A1G"] * 2,
        )
    )
    path = tmp_path / "parallel.tsv"
    frame.to_csv(path, sep="\t", index=False)
    dataset = PairDataset(path, DummyStore(), head="parallel")
    np.testing.assert_array_equal(dataset.binary, [[1, 0], [0, 0]])
    frame.loc[0, "target_lof"] = 0
    frame.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="binary target"):
        PairDataset(path, DummyStore(), head="parallel")


def test_evaluation_excludes_augmented_rows_without_changing_training_labels(tmp_path):
    frame = pd.DataFrame(
        dict(
            snp=["A1G"] * 2,
            is_combinatorial=[False] * 2,
            label=["LOF", "NEUTRAL"],
            label_idx=[0, 2],
            matched_go_id=["GO:0000001"] * 2,
            target=["LOF", "NEUTRAL"],
            augmented=[False, True],
            wt_mut_key=["P1_1"] * 2,
            var_mut_key=["P1_A1G"] * 2,
            wt_global_key=["P1"] * 2,
            var_global_key=["P1_A1G"] * 2,
        )
    )
    path = tmp_path / "multiclass.tsv"
    frame.to_csv(path, sep="\t", index=False)
    train = PairDataset(path, DummyStore(), head="multiclass")
    evaluation = PairDataset(path, DummyStore(), head="multiclass", exclude_augmented=True)
    assert train.frame.label.tolist() == ["LOF", "NEUTRAL"]
    assert evaluation.frame.label.tolist() == ["LOF"]
