"""
Three-stage decoder grid search for the sequence decoder.

This mirrors the ContVar-main grid-search/reporting flow, while keeping the
sequence embedding choices from updated_version.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .config import DecoderConfig
    from .train import train
    from .benchmark_eval import BM_METRIC_KEYS
    from .evaluate import FINAL_METRIC_KEYS
except ImportError:
    from config import DecoderConfig
    from train import train
    from benchmark_eval import BM_METRIC_KEYS
    from evaluate import FINAL_METRIC_KEYS


LR_GRID = [2.5e-4, 5e-4, 1e-3]
DROPOUT_GRID = [0.2, 0.3, 0.4]

HIDDEN_DIMS_GRID = [
    [512, 512],
    [1024, 512],
    [512, 1024, 512],
]
WEIGHT_DECAY_GRID = [1e-5, 1e-4, 5e-4]
POS_WEIGHT_CLAMP_GRID = [10.0, 20.0, 40.0]

BASE_LR = 5e-4
BASE_DROPOUT = 0.3
BASE_HIDDEN_DIMS = [512, 1024, 512]
BASE_WEIGHT_DECAY = 1e-4
BASE_POS_WEIGHT_CLAMP = 20.0

CSV_FIELDS = [
    "tag",
    "stage",
    "aspect",
    "embedding_type",
    "lr",
    "dropout",
    "hidden_dims",
    "weight_decay",
    "pos_weight_clamp",
    "epochs",
    "patience",
    "best_epoch",
    "best_val_map",
    "val_mAP",
    "val_F1_macro",
    "val_MCC",
    "val_threshold",
    "test_mAP",
    "test_F1_macro",
    "test_MCC",
    "test_threshold",
    "checkpoint",
    *[f"bm_val_{key}" for key in BM_METRIC_KEYS],
    *[f"bm_test_{key}" for key in BM_METRIC_KEYS],
    *[f"final_val_{key}" for key in FINAL_METRIC_KEYS],
    *[f"final_test_{key}" for key in FINAL_METRIC_KEYS],
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT_H5 = {
    "raw_esm": REPO_ROOT / "data" / "processed" / "variant_embeddings_raw.h5",
    "seq_phase0": REPO_ROOT / "data" / "processed" / "variant_embeddings_phase_0.h5",
    "seq_phase1": REPO_ROOT / "data" / "processed" / "variant_embeddings_phase_1.h5",
}


@dataclass(frozen=True)
class RunParams:
    lr: float = BASE_LR
    dropout: float = BASE_DROPOUT
    hidden_dims: tuple[int, ...] = tuple(BASE_HIDDEN_DIMS)
    weight_decay: float = BASE_WEIGHT_DECAY
    pos_weight_clamp: float = BASE_POS_WEIGHT_CLAMP


def _choices(value: str, all_values: list[str]) -> list[str]:
    return all_values if value == "all" else [value]


def _tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _tag_hidden(hidden_dims: Iterable[int]) -> str:
    return "-".join(str(x) for x in hidden_dims)


def _run_id(stage: str, params: RunParams) -> str:
    return (
        f"{stage}"
        f"_lr{_tag_float(params.lr)}"
        f"_do{_tag_float(params.dropout)}"
        f"_hd{_tag_hidden(params.hidden_dims)}"
        f"_wd{_tag_float(params.weight_decay)}"
        f"_pwc{_tag_float(params.pos_weight_clamp)}"
    )


def _variant_h5_for(embedding_type: str, args: argparse.Namespace) -> tuple[str | None, str | None]:
    """(primary, secondary) variant embedding H5 paths for a given embedding_type,
    mirroring DecoderConfig.resolve_embeddings_h5()'s concat_phase0/concat_phase1 pairing."""
    raw = args.raw_esm_variant_h5
    phase0 = args.seq_phase0_variant_h5
    phase1 = args.seq_phase1_variant_h5
    source_by_type = {
        "raw_esm": (raw, None),
        "seq_phase0": (phase0, None),
        "seq_phase1": (phase1, None),
        "concat_phase0": (phase0, raw),
        "concat_phase1": (phase1, raw),
    }
    return source_by_type[embedding_type]


def _make_cfg(
    aspect: str,
    embedding_type: str,
    params: RunParams,
    stage: str,
    args: argparse.Namespace,
) -> DecoderConfig:
    cfg = DecoderConfig(aspect=aspect, embedding_type=embedding_type)
    cfg.lr = params.lr
    cfg.dropout = params.dropout
    cfg.hidden_dims = list(params.hidden_dims)
    cfg.weight_decay = params.weight_decay
    cfg.pos_weight_clamp = params.pos_weight_clamp
    cfg.epochs = args.epochs if stage != "final" else args.final_epochs
    cfg.early_stop_patience = args.patience if stage != "final" else args.final_patience
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg._return_metrics = True
    cfg.use_go_propagation = args.propagation
    cfg.use_qualifier_filter = args.use_qualifier_filter
    cfg.cuda = args.cuda
    cfg.wandb_enabled = not args.no_wandb
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project

    if args.goa_tsv:
        cfg.goa_tsv = args.goa_tsv
    if args.split_json:
        cfg.split_json = args.split_json
    if args.uniref_tsv:
        cfg.uniref_tsv = args.uniref_tsv
    if args.raw_esm_h5:
        cfg.raw_esm_h5 = args.raw_esm_h5
    if args.seq_phase0_h5:
        cfg.seq_phase0_h5 = args.seq_phase0_h5
    if args.seq_phase1_h5:
        cfg.seq_phase1_h5 = args.seq_phase1_h5
    cfg.resolve_embeddings_h5()

    if args.benchmark_tsv:
        cfg.benchmark_tsv = args.benchmark_tsv
    if stage == "final" and not args.no_final_benchmark:
        variant_primary, variant_secondary = _variant_h5_for(embedding_type, args)
        if variant_primary:
            cfg.variant_h5 = variant_primary
        if variant_secondary:
            cfg.variant_h5_secondary = variant_secondary

    run_id = _run_id(stage, params)
    ckpt_dir = args.checkpoint_dir / args.tag / aspect.lower() / embedding_type
    cfg.decoder_checkpoint = str(ckpt_dir / f"{run_id}.pt")
    cfg.go_vocab_json = str(ckpt_dir / f"{run_id}_go_vocab.json")
    cfg.wandb_run_name = f"decoder-grid-{args.tag}-{aspect.lower()}-{embedding_type}-{run_id}"
    cfg.wandb_tags = [
        f"aspect_{aspect.lower()}",
        f"emb_{embedding_type}",
        f"stage_{stage}",
        f"tag_{args.tag}",
        f"hd_{_tag_hidden(params.hidden_dims)}",
    ]
    if stage == "final":
        cfg.wandb_tags.append("final_run")
    cfg.wandb_extra_config = {
        "stage": stage,
        "grid_tag": args.tag,
        "hidden_dims_str": _tag_hidden(params.hidden_dims),
        "is_final": stage == "final",
    }
    return cfg


def _to_row(tag: str, stage: str, result: dict) -> dict:
    hp = result["hyperparams"]
    val = result["val"]
    test = result["test"]
    bm = result.get("benchmark", {})
    bm_val = bm.get("val", {})
    bm_test = bm.get("test", {})
    final_val = result.get("final_val", {})
    final_test = result.get("final_test", {})
    return {
        "tag": tag,
        "stage": stage,
        "aspect": result["aspect"],
        "embedding_type": result["embedding_type"],
        "lr": hp["lr"],
        "dropout": hp["dropout"],
        "hidden_dims": "-".join(str(x) for x in hp["hidden_dims"]),
        "weight_decay": hp["weight_decay"],
        "pos_weight_clamp": hp["pos_weight_clamp"],
        "epochs": hp["epochs"],
        "patience": hp["early_stop_patience"],
        "best_epoch": result["best_epoch"],
        "best_val_map": result["best_val_map"],
        "val_mAP": val["mAP"],
        "val_F1_macro": val["F1_macro"],
        "val_MCC": val["MCC"],
        "val_threshold": val["threshold"],
        "test_mAP": test["mAP"],
        "test_F1_macro": test["F1_macro"],
        "test_MCC": test["MCC"],
        "test_threshold": test["threshold"],
        "checkpoint": result["checkpoint"],
        **{f"bm_val_{key}": bm_val.get(key, "") for key in BM_METRIC_KEYS},
        **{f"bm_test_{key}": bm_test.get(key, "") for key in BM_METRIC_KEYS},
        **{f"final_val_{key}": final_val.get(key, "") for key in FINAL_METRIC_KEYS},
        **{f"final_test_{key}": final_test.get(key, "") for key in FINAL_METRIC_KEYS},
    }


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def _print_leaderboard(rows: list[dict], title: str, limit: int = 10):
    rows = [r for r in rows if r.get("val_mAP", "") != ""]
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    ranked = sorted(rows, key=lambda r: float(r["val_mAP"]), reverse=True)
    for i, row in enumerate(ranked[:limit], 1):
        print(
            f"{i:2d}. val_mAP={float(row['val_mAP']):.4f} "
            f"test_mAP={float(row['test_mAP']):.4f} "
            f"test_MCC={float(row['test_MCC']):.4f} "
            f"aspect={row['aspect']} emb={row['embedding_type']} "
            f"lr={row['lr']} dropout={row['dropout']} "
            f"hidden={row['hidden_dims']} wd={row['weight_decay']} "
            f"pwc={row['pos_weight_clamp']}"
        )


def _run_one(
    aspect: str,
    embedding_type: str,
    params: RunParams,
    stage: str,
    args: argparse.Namespace,
) -> dict:
    run_id = _run_id(stage, params)
    print(f"\n[{aspect}/{embedding_type}] {run_id}")
    if args.dry_run:
        return {
            "tag": args.tag,
            "stage": stage,
            "aspect": aspect,
            "embedding_type": embedding_type,
            "lr": params.lr,
            "dropout": params.dropout,
            "hidden_dims": _tag_hidden(params.hidden_dims),
            "weight_decay": params.weight_decay,
            "pos_weight_clamp": params.pos_weight_clamp,
            "epochs": args.epochs if stage != "final" else args.final_epochs,
            "patience": args.patience if stage != "final" else args.final_patience,
            "best_epoch": "",
            "best_val_map": "",
            "val_mAP": "",
            "val_F1_macro": "",
            "val_MCC": "",
            "val_threshold": "",
            "test_mAP": "",
            "test_F1_macro": "",
            "test_MCC": "",
            "test_threshold": "",
            "checkpoint": "",
        }

    cfg = _make_cfg(aspect, embedding_type, params, stage, args)
    Path(cfg.decoder_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    result = train(cfg)
    row = _to_row(args.tag, stage, result)
    if args.output:
        _append_csv(args.output, row)
    return row


def run_grid_for_pair(aspect: str, embedding_type: str, args: argparse.Namespace) -> list[dict]:
    rows = []

    stage1_rows = []
    for lr in LR_GRID:
        for dropout in DROPOUT_GRID:
            params = RunParams(lr=lr, dropout=dropout)
            stage1_rows.append(_run_one(aspect, embedding_type, params, "stage1", args))
    rows.extend(stage1_rows)
    if args.dry_run:
        return rows

    best1 = max(stage1_rows, key=lambda r: float(r["val_mAP"]))
    best1_params = RunParams(
        lr=float(best1["lr"]),
        dropout=float(best1["dropout"]),
    )
    print(
        f"\nBest stage1 for {aspect}/{embedding_type}: "
        f"lr={best1_params.lr}, dropout={best1_params.dropout}, "
        f"val_mAP={float(best1['val_mAP']):.4f}"
    )

    stage2_rows = []
    for hidden_dims in HIDDEN_DIMS_GRID:
        for weight_decay in WEIGHT_DECAY_GRID:
            params = RunParams(
                lr=best1_params.lr,
                dropout=best1_params.dropout,
                hidden_dims=tuple(hidden_dims),
                weight_decay=weight_decay,
            )
            stage2_rows.append(_run_one(aspect, embedding_type, params, "stage2", args))
    rows.extend(stage2_rows)

    best2 = max(stage2_rows, key=lambda r: float(r["val_mAP"]))
    best2_params = RunParams(
        lr=float(best2["lr"]),
        dropout=float(best2["dropout"]),
        hidden_dims=tuple(int(x) for x in best2["hidden_dims"].split("-")),
        weight_decay=float(best2["weight_decay"]),
    )
    print(
        f"\nBest stage2 for {aspect}/{embedding_type}: "
        f"hidden={best2_params.hidden_dims}, wd={best2_params.weight_decay}, "
        f"val_mAP={float(best2['val_mAP']):.4f}"
    )

    stage3_rows = []
    for pos_weight_clamp in POS_WEIGHT_CLAMP_GRID:
        params = RunParams(
            lr=best2_params.lr,
            dropout=best2_params.dropout,
            hidden_dims=best2_params.hidden_dims,
            weight_decay=best2_params.weight_decay,
            pos_weight_clamp=pos_weight_clamp,
        )
        stage3_rows.append(_run_one(aspect, embedding_type, params, "stage3", args))
    rows.extend(stage3_rows)

    if args.run_final:
        best3 = max(stage3_rows, key=lambda r: float(r["val_mAP"]))
        final_params = RunParams(
            lr=float(best3["lr"]),
            dropout=float(best3["dropout"]),
            hidden_dims=tuple(int(x) for x in best3["hidden_dims"].split("-")),
            weight_decay=float(best3["weight_decay"]),
            pos_weight_clamp=float(best3["pos_weight_clamp"]),
        )
        rows.append(_run_one(aspect, embedding_type, final_params, "final", args))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aspect", default="all", choices=["F", "P", "C", "all"])
    parser.add_argument(
        "--embedding",
        default="all",
        choices=["raw_esm", "seq_phase0", "seq_phase1", "concat_phase0", "concat_phase1", "all"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--final_epochs", type=int, default=120)
    parser.add_argument("--final_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override DecoderConfig batch_size (default: DecoderConfig default, 32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="sequence_decoder")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("updated_version/phase_2_decoder/grid_runs"),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_final", action="store_true")
    parser.add_argument("--propagation", action="store_true")
    parser.add_argument("--use_qualifier_filter", action="store_true")
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None, help="Override DecoderConfig wandb_project")
    parser.add_argument("--goa_tsv", default=None)
    parser.add_argument("--split_json", default=None)
    parser.add_argument("--uniref_tsv", default=None)
    parser.add_argument("--raw_esm_h5", default=None)
    parser.add_argument("--seq_phase0_h5", default=None)
    parser.add_argument("--seq_phase1_h5", default=None)
    parser.add_argument("--benchmark_tsv", default=None, help="Override variant-specific GO benchmark TSV path")
    parser.add_argument("--raw_esm_variant_h5", default=str(DEFAULT_VARIANT_H5["raw_esm"]),
                         help="Variant embedding H5 for raw_esm (and concat_* secondary)")
    parser.add_argument("--seq_phase0_variant_h5", default=str(DEFAULT_VARIANT_H5["seq_phase0"]),
                         help="Variant embedding H5 for seq_phase0 (and concat_phase0 primary)")
    parser.add_argument("--seq_phase1_variant_h5", default=str(DEFAULT_VARIANT_H5["seq_phase1"]),
                         help="Variant embedding H5 for seq_phase1 (and concat_phase1 primary)")
    parser.add_argument("--no_final_benchmark", action="store_true",
                         help="Disable automatic variant-benchmark eval for final-stage runs")
    args = parser.parse_args()

    aspects = _choices(args.aspect, ["F", "P", "C"])
    embeddings = _choices(
        args.embedding,
        ["raw_esm", "seq_phase0", "seq_phase1", "concat_phase0", "concat_phase1"],
    )

    all_rows = []
    for aspect in aspects:
        for embedding_type in embeddings:
            rows = run_grid_for_pair(aspect, embedding_type, args)
            all_rows.extend(rows)
            _print_leaderboard(rows, f"Leaderboard: {aspect}/{embedding_type}")

    _print_leaderboard(all_rows, "Overall Leaderboard")


if __name__ == "__main__":
    main()
