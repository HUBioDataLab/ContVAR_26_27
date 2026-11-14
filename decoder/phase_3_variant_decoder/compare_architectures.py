"""Compare the original per-GO decoder with the GO-conditioned cascade.

When both models are trained, the dataset is prepared exactly once. When an
existing cascade run is reused, its data paths, vocabulary, split sizes and GO
artifact are checked against the freshly prepared legacy data. Both routes keep
the filtered train/validation/test rows and WT/variant embeddings aligned.
The comparison intentionally changes the full decoder system: GO representation,
objective and decision rule all differ between the legacy and cascade models.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from phase_3_variant_decoder.cascade import fit_cascade
from phase_3_variant_decoder.presentation_terms import (
    contextual_caption, save_figure, finalize_report,
    log_reporting_guide,
)
from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, INPUT_MODES, SECONDARY_REQUIRED,
    VariantDecoderConfig, resolve_default_go_embeddings,
)
from phase_3_variant_decoder.train import fit, prepare_run


MODEL_LABELS = {
    "legacy": "Original per-GO decoder",
    "cascade": "GO-conditioned cascade",
}
COLORS = {"legacy": "#6b7280", "cascade": "#2563eb"}
COMMON_METRICS = (
    "macro_f1", "accuracy",
    "lof_precision", "lof_recall", "lof_f1", "lof_auprc",
    "gof_precision", "gof_recall", "gof_f1", "gof_auprc",
    "neutral_precision", "neutral_recall", "neutral_f1", "neutral_auprc",
    "confusion",
)


def _legacy_metrics(best: dict, split: str) -> dict:
    metrics = {}
    for key in COMMON_METRICS:
        source = f"{split}_{key}"
        if source not in best:
            raise KeyError(f"Legacy result is missing {source}")
        metrics[key] = best[source]
    return metrics


def _normalise_results(legacy: dict, cascade: dict) -> dict[str, dict]:
    return {
        "legacy": {
            "selection": {
                "epoch": legacy["epoch"],
                "center": legacy["center"],
                "width": legacy["width"],
            },
            "checkpoint": legacy.get("checkpoint", ""),
            "metrics": {split: _legacy_metrics(legacy, split) for split in ("val", "test")},
        },
        "cascade": {
            "selection": cascade["selection"],
            "effect_threshold": cascade["effect_threshold"],
            "direction_threshold": cascade["direction_threshold"],
            "checkpoint": cascade.get("checkpoint", ""),
            "metrics": cascade["metrics"],
        },
    }


def _summary_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for architecture in MODEL_LABELS:
        for split in ("val", "test"):
            metrics = results[architecture]["metrics"][split]
            row = {
                "architecture": architecture,
                "split": split,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
            for label in ("lof", "gof", "neutral"):
                for metric in ("precision", "recall", "f1", "auprc"):
                    row[f"{label}_{metric}"] = metrics[f"{label}_{metric}"]
            rows.append(row)
    return rows


def _annotate(ax, bars) -> None:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.3f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )


def _plot_metrics(results: dict[str, dict], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(MODEL_LABELS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    width = 0.32
    for offset, metric in enumerate(("macro_f1", "accuracy")):
        bars = axes[0].bar(
            x + (offset - 0.5) * width,
            [results[name]["metrics"]["test"][metric] for name in names],
            width, label=metric.replace("_", " ").title(),
        )
        _annotate(axes[0], bars)
    axes[0].set_title("End-to-end test performance")
    axes[0].set_xticks(x, [MODEL_LABELS[name] for name in names])
    axes[0].legend()

    classes = ("lof", "gof", "neutral")
    x2 = np.arange(len(classes))
    for offset, name in enumerate(names):
        bars = axes[1].bar(
            x2 + (offset - 0.5) * width,
            [results[name]["metrics"]["test"][f"{label}_f1"] for label in classes],
            width, label=MODEL_LABELS[name], color=COLORS[name],
        )
        _annotate(axes[1], bars)
    axes[1].set_title("Test F1 by final class")
    axes[1].set_xticks(x2, [label.upper() for label in classes])
    axes[1].legend()
    for ax in axes:
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Original decoder vs GO-conditioned cascade", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_confusions(results: dict[str, dict], path: Path, config=None) -> None:
    from phase_3_variant_decoder.outcome_tables import plot_prediction_counts
    plot_prediction_counts(
        {MODEL_LABELS[name]: results[name]["metrics"]["test"] for name in MODEL_LABELS},
        path, title="Test prediction counts — correct versus incorrect assignments", config=config,
    )


def generate_report(results: dict[str, dict], output_dir: str | Path, config=None) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "test_metrics": output / "architecture_test_metrics.png",
        "confusion_matrices": output / "architecture_test_prediction_counts.png",
        "summary_csv": output / "architecture_comparison_summary.csv",
        "summary_markdown": output / "architecture_comparison_summary.md",
    }
    _plot_metrics(results, artifacts["test_metrics"], config=config)
    _plot_confusions(results, artifacts["confusion_matrices"], config=config)
    rows = _summary_rows(results)
    with artifacts["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Decoder architecture comparison", "",
        "Both models use the same filtered rows, split and WT/variant protein embeddings. ",
        "This is a full-system comparison: GO representation, objective and decision rule differ.",
        "", "| Architecture | Split | Macro-F1 | Accuracy | LOF F1 | GOF F1 | Neutral F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {MODEL_LABELS[row['architecture']]} | {row['split']} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | {row['lof_f1']:.4f} | "
            f"{row['gof_f1']:.4f} | {row['neutral_f1']:.4f} |"
        )
    artifacts["summary_markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return finalize_report(artifacts, config)


def _start_wandb(cfg: VariantDecoderConfig, *, cascade_reused: bool = False):
    if not cfg.wandb_enabled:
        return None, None
    try:
        import wandb
        tags, meta = cfg.wandb_metadata()
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=[*tags, "architecture-comparison", "legacy-vs-cascade"],
            config={
                **meta, "architecture": "architecture_comparison",
                "comparison": "original per-GO decoder vs GO-conditioned cascade",
                "same_data_configuration": True,
                "same_in_memory_tensors": not cascade_reused,
                "cascade_reused": cascade_reused,
                "cascade_balance_classes": cfg.cascade_balance_classes,
                "epochs": cfg.epochs,
                "early_stop_patience": cfg.early_stop_patience,
                "batch_size": cfg.batch_size,
                "hidden_dims": cfg.hidden_dims,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "legacy_margin": cfg.margin,
                "legacy_class_weights": [cfg.w_lof, cfg.w_gof, cfg.w_neutral],
                "dataset_dir": cfg.dataset_dir,
                "go_artifact": cfg.go_embeddings,
            },
        )
        log_reporting_guide(run, wandb, [vars(cfg)])
        return run, wandb
    except Exception as exc:
        raise RuntimeError(
            "WandB is enabled but the architecture-comparison run could not be initialized. "
            "Run `wandb login`, fix the connection/configuration, or explicitly use "
            "--no_wandb for a local-only run."
        ) from exc


def _log_final_wandb(run, wandb, results: dict[str, dict], artifacts: dict[str, str]) -> None:
    if run is None:
        return
    rows = _summary_rows(results)
    columns = list(rows[0])
    run.log({
        "00_Overview/Score_Table": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in rows]
        ),
        "01_Final_Comparison/F1": wandb.Image(artifacts["test_metrics"], caption=contextual_caption("test_metrics", dict(getattr(run, "config", {})))),
        "02_Prediction_Counts/Test": wandb.Image(artifacts["confusion_matrices"], caption=contextual_caption("confusion_matrices", dict(getattr(run, "config", {})))),
    })
    for name in MODEL_LABELS:
        for split in ("val", "test"):
            metrics = results[name]["metrics"][split]
            for metric in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1"):
                run.summary[f"{name}/{split}/{metric}"] = metrics[metric]


def _load_cascade_run(directory: str | Path, cfg: VariantDecoderConfig,
                      go_artifact_sha256: str | None, go_vocab: dict,
                      split_sizes: dict[str, int], run=None,
                      log_prefix: str = "cascade") -> dict:
    root = Path(directory).resolve()
    checkpoint_path = root / "decoder.pt"
    metrics_path = root / "metrics.json"
    history_path = root / "history.json"
    for path in (checkpoint_path, metrics_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"Reusable cascade artifact not found: {path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint.get("config", {})
    expected = {
        "family": cfg.family,
        "embedding_type": cfg.embedding_type,
        "input_mode": cfg.input_mode,
        "seed": cfg.seed,
        "hidden_dims": cfg.hidden_dims,
        "dropout": cfg.dropout,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "cascade_balance_classes": cfg.cascade_balance_classes,
        "dataset_dir": cfg.dataset_dir,
        "sequences_json": cfg.sequences_json,
        "global_h5": cfg.global_h5,
        "local_h5": cfg.local_h5,
        "global_h5_secondary": cfg.global_h5_secondary,
        "local_h5_secondary": cfg.local_h5_secondary,
        "drop_contradictory": cfg.drop_contradictory,
        "max_seq_len": cfg.max_seq_len,
    }
    mismatches = {
        key: (saved_cfg.get(key), value)
        for key, value in expected.items()
        if saved_cfg.get(key) != value
    }
    if checkpoint.get("go_artifact_sha256") != go_artifact_sha256:
        mismatches["go_artifact_sha256"] = (
            checkpoint.get("go_artifact_sha256"), go_artifact_sha256
        )
    if checkpoint.get("go_vocab") != go_vocab:
        mismatches["go_vocab"] = ("checkpoint vocabulary", "current vocabulary")
    if mismatches:
        detail = "; ".join(
            f"{key}: checkpoint={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"Reusable cascade run does not match this comparison: {detail}")

    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    saved_sizes = {
        "train": result["selection"]["effect"]["train_n"],
        "val": result["metrics"]["val"]["n"],
        "test": result["metrics"]["test"]["n"],
    }
    if saved_sizes != split_sizes:
        raise ValueError(
            "Reusable cascade split sizes do not match the current prepared data: "
            f"checkpoint={saved_sizes}, current={split_sizes}"
        )
    result["checkpoint"] = str(checkpoint_path)
    if run is not None:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        for row in history:
            run.log({
                f"{log_prefix}/{row['stage']}/{key}": value
                for key, value in row.items() if key != "stage"
            })
    return result


def run_comparison(cfg: VariantDecoderConfig, *, cache_dir: str | None,
                   out_dir: str, reuse_cascade_run: str | None = None,
                   quiet: bool = False) -> dict:
    prepare_cfg = copy.deepcopy(cfg)
    prepare_cfg.architecture = "cascade"
    tensors, views, go_vocab, device = prepare_run(
        prepare_cfg, cache_dir=cache_dir, quiet=quiet
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(
        prefix=f"architectures_{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}_",
        dir=Path(out_dir).resolve(),
    ))
    run, wandb = _start_wandb(cfg, cascade_reused=bool(reuse_cascade_run))
    try:
        if reuse_cascade_run:
            if not quiet:
                print(f"\n[COMPARE] Reusing {MODEL_LABELS['cascade']} <- {reuse_cascade_run}")
            cascade = _load_cascade_run(
                reuse_cascade_run, cfg, tensors.go_artifact_sha256, go_vocab,
                {split: views[split]["n"] for split in ("train", "val", "test")},
                run=run,
            )
        else:
            cascade_cfg = copy.deepcopy(cfg)
            cascade_cfg.architecture = "cascade"
            if not quiet:
                print(f"\n[COMPARE] {MODEL_LABELS['cascade']} -> {root / 'cascade'}")
            cascade = fit_cascade(
                cascade_cfg, tensors, views, go_vocab, device,
                out_dir=root / "cascade", run=run, quiet=quiet, log_prefix="cascade",
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        legacy_cfg = copy.deepcopy(cfg)
        legacy_cfg.architecture = "legacy"
        torch.manual_seed(legacy_cfg.seed)
        np.random.seed(legacy_cfg.seed)
        if not quiet:
            print(f"\n[COMPARE] {MODEL_LABELS['legacy']} -> {root / 'legacy'}")
        legacy = fit(
            legacy_cfg, tensors, views, go_vocab, device,
            out_dir=root / "legacy", run=run, quiet=quiet,
            results_csv=None, log_prefix="legacy",
        )
        results = _normalise_results(legacy, cascade)
        artifacts = generate_report(results, root, config=vars(cfg))
        _log_final_wandb(run, wandb, results, artifacts)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    comparison = {
        "format": "contvar_decoder_architecture_comparison_v1",
        "comparison_scope": "full decoder system",
        "controlled": ["filtered rows", "train/val/test split", "protein embeddings", "seed"],
        "different": ["GO representation", "output head", "loss", "decision rule"],
        "cascade_reused": bool(reuse_cascade_run),
        "config": vars(cfg),
        "runs": results,
        "artifacts": artifacts,
        "wandb_url": run.url if run is not None else None,
    }
    comparison_path = root / "architecture_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    comparison["comparison_json"] = str(comparison_path)
    if run is not None:
        run.summary["architecture_comparison/output_dir"] = str(root)
        run.finish()
    if not quiet:
        print(f"\n[DONE] architecture comparison -> {root}")
        for label, artifact in artifacts.items():
            print(f"  {label}: {artifact}")
        if comparison["wandb_url"]:
            print(f"  wandb: {comparison['wandb_url']}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the original and cascade decoders on the same prepared split"
    )
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--embedding", choices=EMBEDDING_TYPES, required=True)
    parser.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    parser.add_argument("--global_h5", required=True)
    parser.add_argument("--local_h5", required=True)
    parser.add_argument("--global_h5_secondary", default=None)
    parser.add_argument("--local_h5_secondary", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--sequences_json", default=None)
    parser.add_argument("--go_embeddings", default=resolve_default_go_embeddings())
    parser.add_argument("--allow_go_proxies", action="store_true")
    parser.add_argument("--drop_contradictory", action="store_true")
    parser.add_argument("--unweighted_cascade", action="store_true")
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--w_lof", type=float, default=None)
    parser.add_argument("--w_gof", type=float, default=None)
    parser.add_argument("--w_neutral", type=float, default=None)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--reuse_cascade_run", default=None,
        help="Existing cascade_* directory containing decoder.pt, metrics.json and history.json",
    )
    parser.add_argument("--out_dir", default="phase_3_variant_decoder/runs")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    if args.embedding in SECONDARY_REQUIRED and not (
        args.global_h5_secondary and args.local_h5_secondary
    ):
        parser.error(
            f"embedding={args.embedding} requires --global_h5_secondary and --local_h5_secondary"
        )
    if not args.go_embeddings:
        parser.error("--go_embeddings is required for the cascade side")

    cfg = VariantDecoderConfig(args.family, args.embedding, args.input_mode)
    cfg.architecture = "cascade"
    cfg.global_h5, cfg.local_h5 = args.global_h5, args.local_h5
    cfg.global_h5_secondary, cfg.local_h5_secondary = (
        args.global_h5_secondary, args.local_h5_secondary
    )
    cfg.go_embeddings = args.go_embeddings
    cfg.allow_go_proxies = args.allow_go_proxies
    cfg.drop_contradictory = args.drop_contradictory
    cfg.cascade_balance_classes = not args.unweighted_cascade
    cfg.cuda = args.cuda
    cfg.wandb_enabled = not args.no_wandb
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity:
        cfg.wandb_entity = args.wandb_entity
    cfg.wandb_run_name = args.wandb_run_name or (
        f"phase3-architectures-{args.family}-{args.embedding}-{args.input_mode}"
    )
    for name in (
        "dataset_dir", "sequences_json", "margin", "w_lof", "w_gof", "w_neutral",
        "hidden_dims", "dropout", "lr", "batch_size", "epochs",
        "early_stop_patience", "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_comparison(
        cfg, cache_dir=args.cache_dir, out_dir=args.out_dir,
        reuse_cascade_run=args.reuse_cascade_run,
    )


if __name__ == "__main__":
    main()
