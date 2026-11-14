"""Train weighted and unweighted cascade controls and plot them side by side.

Both runs reuse the same prepared tensors and start from the same random seed.
The only intended difference is inverse-frequency class weighting in the two
binary BCE objectives.
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
from phase_3_variant_decoder.train import prepare_run


RUN_LABELS = {"weighted": "Weighted BCE", "unweighted": "Unweighted BCE"}
COLORS = {"weighted": "#2563eb", "unweighted": "#dc2626"}


def _history_by_stage(history: list[dict], stage: str) -> list[dict]:
    return [row for row in history if row["stage"] == stage]


def _summary_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for weighting, result in results.items():
        for split in ("val", "test"):
            metrics = result["metrics"][split]
            row = {
                "weighting": weighting,
                "split": split,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "effect_macro_f1": metrics["effect"]["macro_f1"],
                "direction_macro_f1": metrics["direction_true_effect"]["macro_f1"],
                "effect_missed_n": metrics["effect_missed_n"],
                "neutral_routed_n": metrics["neutral_routed_n"],
            }
            for label in ("lof", "gof", "neutral"):
                for metric in ("precision", "recall", "f1", "auprc"):
                    row[f"{label}_{metric}"] = metrics[f"{label}_{metric}"]
            rows.append(row)
    return rows


def _save_summary_csv(results: dict[str, dict], path: Path, config=None) -> None:
    rows = _summary_rows(results)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(f"{value:.3f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)


def _plot_training(histories: dict[str, list[dict]], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for col, stage in enumerate(("effect", "direction")):
        for name, history in histories.items():
            rows = _history_by_stage(history, stage)
            epochs = [row["epoch"] for row in rows]
            axes[0, col].plot(epochs, [row["train_loss"] for row in rows],
                              label=RUN_LABELS[name], color=COLORS[name], linewidth=2)
            axes[1, col].plot(epochs, [row["val_macro_f1"] for row in rows],
                              label=RUN_LABELS[name], color=COLORS[name], linewidth=2)
        title = "Stage 1: EFFECT versus NEUTRAL (all rows)" if stage == "effect" else "Stage 2: LOF versus GOF (true-effect rows)"
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Training loss")
        axes[1, col].set_ylabel("Validation macro-F1")
        for row in range(2):
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].grid(alpha=0.25)
            axes[row, col].legend()
    fig.suptitle("Cascade training: inverse-frequency weighting ablation", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_test_metrics(results: dict[str, dict], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(RUN_LABELS)
    x = np.arange(len(names))
    width = 0.22
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for i, metric in enumerate(("macro_f1", "accuracy")):
        bars = axes[0].bar(x + (i - 0.5) * width,
                           [results[name]["metrics"]["test"][metric] for name in names],
                           width, label=metric.replace("_", " ").title())
        _annotate_bars(axes[0], bars)
    axes[0].set_title("End-to-end test performance")
    axes[0].set_xticks(x, [RUN_LABELS[name] for name in names])
    axes[0].legend()

    labels = ("lof", "gof", "neutral")
    x2 = np.arange(len(labels))
    for offset, name in enumerate(names):
        bars = axes[1].bar(x2 + (offset - 0.5) * 0.34,
                           [results[name]["metrics"]["test"][f"{label}_f1"] for label in labels],
                           0.34, label=RUN_LABELS[name], color=COLORS[name])
        _annotate_bars(axes[1], bars)
    axes[1].set_title("Test F1 by final class")
    axes[1].set_xticks(x2, [label.upper() for label in labels])
    axes[1].legend()

    stage_labels = ("Effect detection\nBinary / all rows", "Effect direction\nBinary / true LOF + GOF")
    x3 = np.arange(len(stage_labels))
    for offset, name in enumerate(names):
        test = results[name]["metrics"]["test"]
        values = [test["effect"]["macro_f1"], test["direction_true_effect"]["macro_f1"]]
        bars = axes[2].bar(x3 + (offset - 0.5) * 0.34, values, 0.34,
                           label=RUN_LABELS[name], color=COLORS[name])
        _annotate_bars(axes[2], bars)
    axes[2].set_title("Binary stage macro-F1\nDifferent evaluation populations")
    axes[2].set_xticks(x3, stage_labels)
    axes[2].legend()

    for ax in axes:
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Metric value")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Weighted vs unweighted cascade decoder", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_confusions(results: dict[str, dict], path: Path, config=None) -> None:
    from phase_3_variant_decoder.outcome_tables import plot_prediction_counts
    plot_prediction_counts(
        {RUN_LABELS[name]: results[name]["metrics"]["test"] for name in RUN_LABELS},
        path, title="Test prediction counts — correct versus incorrect assignments", config=config,
    )


def _write_markdown(results: dict[str, dict], path: Path, config=None) -> None:
    lines = [
        "# Cascade weighting comparison",
        "",
        "Both models use the same split, architecture, seed and initialization. "
        "Only inverse-frequency BCE weighting changes.",
        "",
        "| Run | Split | Macro-F1 | Accuracy | LOF F1 | GOF F1 | Neutral F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUN_LABELS:
        for split in ("val", "test"):
            m = results[name]["metrics"][split]
            lines.append(
                f"| {RUN_LABELS[name]} | {split} | {m['macro_f1']:.4f} | "
                f"{m['accuracy']:.4f} | {m['lof_f1']:.4f} | {m['gof_f1']:.4f} | "
                f"{m['neutral_f1']:.4f} |"
            )
    lines.extend(["", "## Train-derived weights", ""])
    for name in RUN_LABELS:
        lines.append(f"### {RUN_LABELS[name]}")
        lines.append("")
        for stage in ("effect", "direction"):
            selected = results[name]["selection"][stage]
            class_order = "[NEUTRAL, FUNCTIONAL]" if stage == "effect" else "[GOF, LOF]"
            lines.append(
                f"- {stage} {class_order}: counts `{selected['train_class_counts_0_1']}`, "
                f"weights `{[round(v, 6) for v in selected['class_weights_0_1']]}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_comparison_report(results: dict[str, dict], histories: dict[str, list[dict]],
                               output_dir: str | Path, config=None) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "training_curves": output / "training_curves.png",
        "test_metrics": output / "test_metrics.png",
        "confusion_matrices": output / "test_prediction_counts.png",
        "summary_csv": output / "comparison_summary.csv",
        "summary_markdown": output / "comparison_summary.md",
    }
    _plot_training(histories, artifacts["training_curves"], config=config)
    _plot_test_metrics(results, artifacts["test_metrics"], config=config)
    _plot_confusions(results, artifacts["confusion_matrices"], config=config)
    _save_summary_csv(results, artifacts["summary_csv"])
    _write_markdown(results, artifacts["summary_markdown"])
    return finalize_report(artifacts, config)


def _start_wandb(cfg: VariantDecoderConfig):
    if not cfg.wandb_enabled:
        return None, None
    try:
        import wandb
        tags, meta = cfg.wandb_metadata()
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=[*tags, "weighting-comparison", "weighted-vs-unweighted"],
            config={
                **meta,
                "comparison": "inverse-frequency BCE vs unweighted BCE",
                "controlled_seed": cfg.seed,
                "same_prepared_tensors": True,
                "epochs": cfg.epochs,
                "early_stop_patience": cfg.early_stop_patience,
                "batch_size": cfg.batch_size,
                "hidden_dims": cfg.hidden_dims,
                "dropout": cfg.dropout,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "dataset_dir": cfg.dataset_dir,
                "go_artifact": cfg.go_embeddings,
            },
        )
        log_reporting_guide(run, wandb, [vars(cfg) | {"cascade_balance_classes": balance}
                                       for balance in (True, False)])
        return run, wandb
    except Exception as exc:
        raise RuntimeError(
            "WandB is enabled but the run could not be initialized. "
            "Run `wandb login`, fix the connection/configuration, or explicitly "
            "use --no_wandb for a local-only run."
        ) from exc


def log_comparison_to_wandb(run, wandb, results: dict[str, dict],
                            histories: dict[str, list[dict]], artifacts: dict[str, str]) -> None:
    """Upload aligned curves, final metrics and presentation-ready figures."""
    if run is None:
        return
    for stage in ("effect", "direction"):
        step_key = f"comparison/{stage}/epoch"
        run.define_metric(step_key)
        run.define_metric(f"comparison/{stage}/*", step_metric=step_key)
        indexed = {
            name: {row["epoch"]: row for row in _history_by_stage(history, stage)}
            for name, history in histories.items()
        }
        epochs = sorted(set().union(*(rows.keys() for rows in indexed.values())))
        for epoch in epochs:
            payload = {step_key: epoch}
            for name, rows in indexed.items():
                if epoch not in rows:
                    continue
                payload[f"comparison/{stage}/{name}_train_loss"] = rows[epoch]["train_loss"]
                payload[f"comparison/{stage}/{name}_val_macro_f1"] = rows[epoch]["val_macro_f1"]
                payload[f"comparison/{stage}/{name}_threshold"] = rows[epoch]["threshold"]
            run.log(payload)

    summary_rows = _summary_rows(results)
    columns = list(summary_rows[0])
    run.log({
        "00_Overview/Score_Table": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in summary_rows]
        ),
        "01_Training/Curves": wandb.Image(artifacts["training_curves"], caption=contextual_caption("training_curves", dict(getattr(run, "config", {})))),
        "02_Final_Comparison/F1": wandb.Image(artifacts["test_metrics"], caption=contextual_caption("test_metrics", dict(getattr(run, "config", {})))),
        "03_Prediction_Counts/Test": wandb.Image(artifacts["confusion_matrices"], caption=contextual_caption("confusion_matrices", dict(getattr(run, "config", {})))),
    })
    for name, result in results.items():
        for split in ("val", "test"):
            metrics = result["metrics"][split]
            for metric in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1",
                           "lof_recall", "gof_recall", "neutral_recall"):
                run.summary[f"{name}/{split}/{metric}"] = metrics[metric]
        for stage in ("effect", "direction"):
            selection = result["selection"][stage]
            run.summary[f"{name}/{stage}/best_epoch"] = selection["epoch"]
            run.summary[f"{name}/{stage}/best_val_macro_f1"] = selection["val_macro_f1"]
            run.summary[f"{name}/{stage}/class_counts_0_1"] = selection["train_class_counts_0_1"]
            run.summary[f"{name}/{stage}/class_weights_0_1"] = selection["class_weights_0_1"]


def run_comparison(cfg: VariantDecoderConfig, *, cache_dir: str | None, out_dir: str,
                   quiet: bool = False) -> dict:
    cfg.architecture = "cascade"
    cfg.cascade_loss = "bce"  # This historical experiment explicitly compares BCE weighting.
    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=cache_dir, quiet=quiet)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(
        prefix=f"weighting_{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}_",
        dir=Path(out_dir).resolve(),
    ))
    run, wandb = _start_wandb(cfg)
    results, histories = {}, {}
    try:
        for name, balanced in (("weighted", True), ("unweighted", False)):
            run_cfg = copy.deepcopy(cfg)
            run_cfg.cascade_balance_classes = balanced
            run_dir = root / name
            if not quiet:
                print(f"\n[COMPARE] {RUN_LABELS[name]} -> {run_dir}")
            result = fit_cascade(run_cfg, tensors, views, go_vocab, device,
                                 out_dir=run_dir, run=run, quiet=quiet,
                                 log_prefix=name)
            results[name] = result
            histories[name] = json.loads(
                (Path(result["checkpoint"]).parent / "history.json").read_text(encoding="utf-8")
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        artifacts = generate_comparison_report(results, histories, root, config=vars(cfg))
        log_comparison_to_wandb(run, wandb, results, histories, artifacts)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    comparison = {
        "format": "contvar_cascade_weighting_comparison_v1",
        "controlled_difference": "inverse-frequency BCE class weighting",
        "config": vars(cfg),
        "runs": results,
        "artifacts": artifacts,
        "wandb_url": run.url if run is not None else None,
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    comparison["comparison_json"] = str(comparison_path)
    if run is not None:
        run.summary["comparison/output_dir"] = str(root)
        run.finish()
    if not quiet:
        print(f"\n[DONE] weighting comparison -> {root}")
        for label, artifact in artifacts.items():
            print(f"  {label}: {artifact}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train weighted and unweighted cascade decoders and generate comparison figures"
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
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--cache_dir", default=None)
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
        parser.error("--go_embeddings is required")

    cfg = VariantDecoderConfig(args.family, args.embedding, args.input_mode)
    cfg.architecture = "cascade"
    cfg.global_h5, cfg.local_h5 = args.global_h5, args.local_h5
    cfg.global_h5_secondary, cfg.local_h5_secondary = (
        args.global_h5_secondary, args.local_h5_secondary
    )
    cfg.go_embeddings = args.go_embeddings
    cfg.allow_go_proxies = args.allow_go_proxies
    cfg.drop_contradictory = args.drop_contradictory
    cfg.cuda = args.cuda
    cfg.wandb_enabled = not args.no_wandb
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity:
        cfg.wandb_entity = args.wandb_entity
    cfg.wandb_run_name = args.wandb_run_name or (
        f"phase3-weighting-{args.family}-{args.embedding}-{args.input_mode}"
    )
    for name in ("dataset_dir", "sequences_json", "hidden_dims", "dropout", "lr",
                 "batch_size", "epochs", "early_stop_patience", "seed"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    run_comparison(cfg, cache_dir=args.cache_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
