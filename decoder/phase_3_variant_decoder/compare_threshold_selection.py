"""Compare stage-wise and end-to-end validation threshold selection.

This is a post-hoc experiment: it loads one trained cascade checkpoint, keeps
all weights fixed and changes only the Effect routing threshold. The baseline
threshold was selected for binary Effect macro-F1. The alternative threshold is
selected for final three-class macro-F1 on validation. Test is never consulted
during selection and is evaluated once for each frozen threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from phase_3_variant_decoder.cascade import (
    attach_checkpoint_go, load_cascade, stage_outputs,
)
from phase_3_variant_decoder.presentation_terms import (
    contextual_caption, save_figure, finalize_report,
    configuration_metadata, log_reporting_guide,
)
from phase_3_variant_decoder.config import LABELS, LABEL_TO_IDX, VariantDecoderConfig
from phase_3_variant_decoder.train import prepare_run


METHOD_LABELS = {
    "stage_f1": "Effect F1 threshold",
    "final_f1": "Final 3-class F1 threshold",
}
COLORS = {"stage_f1": "#64748b", "final_f1": "#2563eb"}


def threshold_metrics(y, p_effect, p_lof, threshold: float) -> dict:
    """Evaluate a hard cascade from cached probabilities."""
    y = np.asarray(y, dtype=np.int64)
    p_effect = np.asarray(p_effect)
    p_lof = np.asarray(p_lof)
    actual_effect = y != LABEL_TO_IDX["NEUTRAL"]
    routed = p_effect >= threshold
    pred = np.full(len(y), LABEL_TO_IDX["NEUTRAL"], dtype=np.int64)
    pred[routed] = np.where(
        p_lof[routed] >= 0.5, LABEL_TO_IDX["LOF"], LABEL_TO_IDX["GOF"]
    )
    _, _, class_f1, _ = precision_recall_fscore_support(
        y, pred, labels=[0, 1, 2], zero_division=0
    )
    effect_pred = routed.astype(np.int64)
    effect_macro_f1 = f1_score(
        actual_effect, effect_pred, labels=[0, 1], average="macro", zero_division=0
    )
    effect_n = int(actual_effect.sum())
    neutral_n = int((~actual_effect).sum())
    missed_n = int((actual_effect & ~routed).sum())
    neutral_routed_n = int((~actual_effect & routed).sum())
    result = {
        "threshold": float(threshold),
        "macro_f1": float(class_f1.mean()),
        "accuracy": float(np.mean(pred == y)),
        "effect_macro_f1": float(effect_macro_f1),
        "effect_missed_rate": missed_n / effect_n if effect_n else None,
        "neutral_routed_rate": neutral_routed_n / neutral_n if neutral_n else None,
        "effect_missed_n": missed_n,
        "neutral_routed_n": neutral_routed_n,
        "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }
    for label, score in zip(LABELS, class_f1):
        result[f"{label.lower()}_f1"] = float(score)
    return result


def build_threshold_curve(y, p_effect, p_lof, baseline_threshold: float) -> list[dict]:
    grid = np.unique(np.concatenate((
        np.linspace(0.0, 1.0, 101), np.asarray([baseline_threshold], dtype=float)
    )))
    return [threshold_metrics(y, p_effect, p_lof, float(t)) for t in grid]


def select_final_threshold(curve: list[dict], baseline_threshold: float) -> dict:
    """Maximize validation final macro-F1; ties stay nearest the old threshold."""
    if not curve:
        raise ValueError("Threshold curve is empty")
    return max(
        curve,
        key=lambda row: (row["macro_f1"], -abs(row["threshold"] - baseline_threshold)),
    )


def _outputs(model, tensors, view, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_effect = stage_outputs(
        model.effect_stage, tensors, view, batch_size=batch_size
    )["probability"]
    p_lof = stage_outputs(
        model.direction_stage, tensors, view, batch_size=batch_size
    )["probability"]
    return view["label_idx"].cpu().numpy(), p_effect, p_lof


def _annotate(ax, bars, percent=False) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            (f"{value:.1%}" if percent else f"{value:.3f}"),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
        )


def _plot_sweep(curve: list[dict], thresholds: dict[str, float], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["threshold"] for row in curve]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(x, [row["macro_f1"] for row in curve], label="Final three-class macro-F1")
    axes[0].plot(x, [row["effect_macro_f1"] for row in curve], label="Binary effect-detection macro-F1")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title("Validation objective versus effect-gate threshold")
    axes[0].legend()

    axes[1].plot(x, [row["neutral_routed_rate"] for row in curve],
                 label="NEUTRAL false-positive rate")
    axes[1].plot(x, [row["effect_missed_rate"] for row in curve],
                 label="Effect false-negative rate")
    axes[1].set_ylabel("Validation conditional error rate (%)")
    from matplotlib.ticker import PercentFormatter
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Effect-gate routing trade-off")
    axes[1].legend()
    for ax in axes:
        for method, threshold in thresholds.items():
            ax.axvline(
                threshold, color=COLORS[method], linestyle="--", alpha=.9,
                label=METHOD_LABELS[method] if ax is axes[1] else None,
            )
        ax.set_xlabel("Effect threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.25)
    # Add the threshold legend after vertical lines have been drawn.
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=8)
    fig.suptitle("Validation threshold selection: model weights held fixed", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_comparison(results: dict[str, dict], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(METHOD_LABELS)
    x = np.arange(len(methods))
    width = .34
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for offset, split in enumerate(("val", "test")):
        bars = axes[0].bar(
            x + (offset - .5) * width,
            [results[name][split]["macro_f1"] for name in methods],
            width, label=split.upper(),
        )
        _annotate(axes[0], bars)
    axes[0].set_xticks(x, [METHOD_LABELS[name] for name in methods])
    axes[0].set_title("Final three-class macro-F1")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend()

    classes = ("lof", "gof", "neutral")
    x2 = np.arange(len(classes))
    for offset, method in enumerate(methods):
        bars = axes[1].bar(
            x2 + (offset - .5) * width,
            [results[method]["test"][f"{label}_f1"] for label in classes],
            width, label=METHOD_LABELS[method], color=COLORS[method],
        )
        _annotate(axes[1], bars)
    axes[1].set_xticks(x2, [label.upper() for label in classes])
    axes[1].set_title("Per-class F1 on the test set")
    axes[1].set_ylabel("Class F1")
    axes[1].legend(fontsize=8)

    error_names = ("effect_missed_rate", "neutral_routed_rate")
    x3 = np.arange(len(error_names))
    for offset, method in enumerate(methods):
        bars = axes[2].bar(
            x3 + (offset - .5) * width,
            [results[method]["test"][name] for name in error_names],
            width, label=METHOD_LABELS[method], color=COLORS[method],
        )
        _annotate(axes[2], bars, percent=True)
    axes[2].set_xticks(x3, ["Effect false-negative rate", "NEUTRAL false-positive rate"])
    axes[2].set_title("Effect-gate conditional error rates on the test set")
    axes[2].set_ylabel("Conditional error rate (%)")
    from matplotlib.ticker import PercentFormatter
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Effect-gate threshold-selection comparison", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_confusions(results: dict[str, dict], path: Path, config=None) -> None:
    from phase_3_variant_decoder.outcome_tables import plot_prediction_counts
    plot_prediction_counts(
        {METHOD_LABELS[name]: results[name]["test"] for name in METHOD_LABELS},
        path, title="Test prediction counts — correct versus incorrect assignments", config=config,
    )


def generate_report(curve: list[dict], results: dict[str, dict], output: Path, config=None) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "threshold_sweep": output / "validation_threshold_sweep.png",
        "comparison": output / "threshold_comparison.png",
        "confusions": output / "test_prediction_counts.png",
        "curve_csv": output / "validation_threshold_curve.csv",
        "summary_csv": output / "threshold_summary.csv",
        "summary_markdown": output / "threshold_summary.md",
    }
    thresholds = {name: result["threshold"] for name, result in results.items()}
    _plot_sweep(curve, thresholds, artifacts["threshold_sweep"], config=config)
    _plot_comparison(results, artifacts["comparison"], config=config)
    _plot_confusions(results, artifacts["confusions"], config=config)

    curve_columns = [key for key in curve[0] if key != "confusion"]
    with artifacts["curve_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in curve_columns} for row in curve)

    rows = []
    for method, result in results.items():
        for split in ("val", "test"):
            metrics = result[split]
            rows.append({
                "method": method, "description": METHOD_LABELS[method], "split": split,
                "threshold": result["threshold"],
                **{key: metrics[key] for key in (
                    "macro_f1", "accuracy", "effect_macro_f1",
                    "lof_f1", "gof_f1", "neutral_f1",
                    "effect_missed_rate", "neutral_routed_rate",
                )},
            })
    with artifacts["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    old, new = results["stage_f1"], results["final_f1"]
    lines = [
        "# Effect-gate threshold-selection comparison", "",
        "Model weights are identical in both methods; only the effect-gate threshold changes. "
        "Both thresholds are selected on validation data; test labels are not used for selection.", "",
        "| Method | Threshold | Split | Final macro-F1 | Binary effect macro-F1 | LOF F1 | GOF F1 | NEUTRAL F1 | Effect false-negative rate | NEUTRAL false-positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['description']} | {row['threshold']:.2f} | {row['split']} | "
            f"{row['macro_f1']:.4f} | {row['effect_macro_f1']:.4f} | "
            f"{row['lof_f1']:.4f} | {row['gof_f1']:.4f} | {row['neutral_f1']:.4f} | "
            f"{row['effect_missed_rate']:.1%} | {row['neutral_routed_rate']:.1%} |"
        )
    lines.extend([
        "", "## Descriptive comparison", "",
        f"Optimizing validation final macro-F1 changes the threshold from `{old['threshold']:.2f}` to "
        f"`{new['threshold']:.2f}`.",
        f"Change in test final macro-F1: `{new['test']['macro_f1'] - old['test']['macro_f1']:+.4f}`.",
        f"Change in test NEUTRAL false-positive rate: "
        f"`{new['test']['neutral_routed_rate'] - old['test']['neutral_routed_rate']:+.1%}`.",
        f"Change in test effect false-negative rate: "
        f"`{new['test']['effect_missed_rate'] - old['test']['effect_missed_rate']:+.1%}`.", "",
    ])
    artifacts["summary_markdown"].write_text("\n".join(lines), encoding="utf-8")
    return finalize_report(artifacts, config)


def _start_wandb(cfg: VariantDecoderConfig, checkpoint: str, name: str | None,
                 project: str | None, entity: str | None, enabled: bool):
    if not enabled:
        return None, None
    try:
        import wandb
        run = wandb.init(
            project=project or cfg.wandb_project,
            entity=entity or cfg.wandb_entity,
            name=name or "phase3-effect-threshold-selection",
            tags=["phase3", "cascade", "threshold-selection", "posthoc"],
            config={
                **configuration_metadata(cfg),
                "experiment": "effect_threshold_selection",
                "checkpoint": str(Path(checkpoint).resolve()),
                "selection_data": "validation_only",
                "test_used_for_selection": False,
                "family": cfg.family,
                "embedding_type": cfg.embedding_type,
                "input_mode": cfg.input_mode,
                "seed": cfg.seed,
            },
        )
        log_reporting_guide(run, wandb, [vars(cfg)])
        return run, wandb
    except Exception as exc:
        raise RuntimeError("Could not initialize W&B; run `wandb login` or use --no_wandb.") from exc


def _log_wandb(run, wandb, curve: list[dict], results: dict[str, dict],
               artifacts: dict[str, str]) -> None:
    if run is None:
        return
    curve_columns = [key for key in curve[0] if key != "confusion"]
    summary_rows = []
    for method, result in results.items():
        for split in ("val", "test"):
            m = result[split]
            summary_rows.append([
                METHOD_LABELS[method], split, result["threshold"], m["macro_f1"],
                m["effect_macro_f1"], m["lof_f1"], m["gof_f1"], m["neutral_f1"],
                m["effect_missed_rate"], m["neutral_routed_rate"],
            ])
    run.log({
        "00_Overview/Threshold_Comparison": wandb.Table(
            columns=["method", "split", "threshold", "final_macro_f1", "effect_macro_f1",
                     "lof_f1", "gof_f1", "neutral_f1", "effect_missed_rate",
                     "neutral_routed_rate"],
            data=summary_rows,
        ),
        "01_Validation_Sweep/Curves": wandb.Image(artifacts["threshold_sweep"], caption=contextual_caption("threshold_sweep", dict(getattr(run, "config", {})))),
        "01_Validation_Sweep/All_Thresholds": wandb.Table(
            columns=curve_columns,
            data=[[row[key] for key in curve_columns] for row in curve],
        ),
        "02_Final_Comparison/F1_and_Routing": wandb.Image(artifacts["comparison"], caption=contextual_caption("comparison", dict(getattr(run, "config", {})))),
        "03_Prediction_Counts/Test": wandb.Image(artifacts["confusions"], caption=contextual_caption("confusions", dict(getattr(run, "config", {})))),
    })
    for method, result in results.items():
        run.summary[f"{method}/threshold"] = result["threshold"]
        for split in ("val", "test"):
            for metric in ("macro_f1", "effect_missed_rate", "neutral_routed_rate",
                           "lof_f1", "gof_f1", "neutral_f1"):
                run.summary[f"{method}/{split}/{metric}"] = result[split][metric]


def run_experiment(checkpoint: str, *, cache_dir: str | None, out_dir: str,
                   cuda: str | None = None, dataset_dir: str | None = None,
                   sequences_json: str | None = None, global_h5: str | None = None,
                   local_h5: str | None = None, wandb_enabled: bool = True,
                   wandb_project: str | None = None, wandb_entity: str | None = None,
                   wandb_run_name: str | None = None, quiet: bool = False) -> dict:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if raw.get("format") != "contvar_go_scalar_cascade_v1":
        raise ValueError("Checkpoint is not a GO-conditioned cascade")
    saved = raw["config"]
    cfg = VariantDecoderConfig(
        saved["family"], saved["embedding_type"], saved["input_mode"]
    )
    for key, value in saved.items():
        setattr(cfg, key, value)
    overrides = {
        "dataset_dir": dataset_dir, "sequences_json": sequences_json,
        "global_h5": global_h5, "local_h5": local_h5, "cuda": cuda,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    # GO vectors are loaded from the checkpoint; the original NPZ may have moved.
    cfg.architecture = "legacy"
    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=cache_dir, quiet=quiet)
    model, ckpt = load_cascade(checkpoint, device)
    attach_checkpoint_go(tensors, go_vocab, ckpt)
    if tensors.input_dim != model.spec["protein_dim"]:
        raise ValueError("Protein input dimension differs from checkpoint")

    probabilities = {
        split: _outputs(model, tensors, views[split], cfg.batch_size)
        for split in ("val", "test")
    }
    baseline_threshold = float(ckpt["effect_threshold"])
    curve = build_threshold_curve(*probabilities["val"], baseline_threshold)
    selected = select_final_threshold(curve, baseline_threshold)
    thresholds = {"stage_f1": baseline_threshold, "final_f1": selected["threshold"]}
    results = {}
    for method, threshold in thresholds.items():
        results[method] = {"threshold": threshold}
        for split in ("val", "test"):
            results[method][split] = threshold_metrics(*probabilities[split], threshold)

    output_parent = Path(out_dir).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="threshold_selection_", dir=output_parent))
    artifacts = generate_report(curve, results, root, config=vars(cfg))
    run, wandb = _start_wandb(
        cfg, checkpoint, wandb_run_name, wandb_project, wandb_entity, wandb_enabled
    )
    try:
        _log_wandb(run, wandb, curve, results, artifacts)
        if run is not None:
            run.summary["output_dir"] = str(root)
            wandb_url = run.url
            run.finish()
        else:
            wandb_url = None
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    experiment = {
        "format": "contvar_cascade_threshold_selection_v1",
        "config": vars(cfg),
        "checkpoint": str(Path(checkpoint).resolve()),
        "selection": "validation_final_macro_f1",
        "test_used_for_selection": False,
        "curve": curve,
        "results": results,
        "artifacts": artifacts,
        "wandb_url": wandb_url,
    }
    output_json = root / "threshold_selection.json"
    output_json.write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    experiment["output_json"] = str(output_json)
    if not quiet:
        print(f"[DONE] threshold selection -> {root}")
        print(f"  old threshold: {baseline_threshold:.2f}")
        print(f"  new threshold: {selected['threshold']:.2f}")
        for method in METHOD_LABELS:
            test = results[method]["test"]
            print(f"  {method}: test macro-F1={test['macro_f1']:.4f} | "
                  f"missed={test['effect_missed_rate']:.1%} | "
                  f"neutral routed={test['neutral_routed_rate']:.1%}")
        if wandb_url:
            print(f"  wandb: {wandb_url}")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare binary Effect-F1 and final three-class-F1 threshold selection"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--sequences_json", default=None)
    parser.add_argument("--global_h5", default=None)
    parser.add_argument("--local_h5", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="phase3-effect-threshold-selection")
    args = parser.parse_args()
    run_experiment(
        args.checkpoint, cache_dir=args.cache_dir, out_dir=args.out_dir,
        cuda=args.cuda, dataset_dir=args.dataset_dir,
        sequences_json=args.sequences_json, global_h5=args.global_h5,
        local_h5=args.local_h5, wandb_enabled=not args.no_wandb,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
