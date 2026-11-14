"""Compare validation-tuned routing with a learned Effect decision boundary.

The baseline checkpoint learned its Effect radius with BCE, then selected an
additional probability threshold on validation macro-F1.  The alternative is
trained from scratch with the same data and hyperparameters, but fixes the
probability threshold at 0.5 during model selection and inference.  Therefore
the trainable radius is the complete routing boundary.  Test labels never
select an epoch, a threshold, or a model.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from phase_3_variant_decoder.cascade import (
    CHECKPOINT_FORMAT, attach_checkpoint_go, evaluate_cascade, fit_cascade,
    load_cascade,
)
from phase_3_variant_decoder.presentation_terms import (
    contextual_caption, save_figure, finalize_report,
    configuration_metadata, log_reporting_guide,
)
from phase_3_variant_decoder.config import LABELS, VariantDecoderConfig
from phase_3_variant_decoder.train import prepare_run


METHOD_LABELS = {
    "validation_threshold": "Validation threshold",
    "learned_radius": "Learned radius (p=0.50)",
}
COLORS = {"validation_threshold": "#64748b", "learned_radius": "#2563eb"}


def _checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    checkpoint = candidate / "decoder.pt" if candidate.is_dir() else candidate
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cascade checkpoint not found: {checkpoint}")
    return checkpoint


def _load_saved_config(checkpoint: dict, overrides: dict) -> VariantDecoderConfig:
    saved = checkpoint.get("config", {})
    cfg = VariantDecoderConfig(
        saved["family"], saved["embedding_type"], saved["input_mode"]
    )
    for key, value in saved.items():
        setattr(cfg, key, value)
    cfg.cascade_loss = saved.get("cascade_loss", "bce")
    cfg.cascade_margin = saved.get("cascade_margin", 1.0)
    # Old checkpoints predate the explicit strategy field.
    cfg.effect_threshold_strategy = saved.get(
        "effect_threshold_strategy", "validation"
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _routing_rates(metrics: dict) -> dict:
    effect_n = int(metrics["direction_true_effect"]["n"])
    neutral_n = int(metrics["n"]) - effect_n
    return {
        "effect_missed_rate": (
            metrics["effect_missed_n"] / effect_n if effect_n else None
        ),
        "neutral_routed_rate": (
            metrics["neutral_routed_n"] / neutral_n if neutral_n else None
        ),
    }


def _boundary(checkpoint: dict) -> dict:
    state = checkpoint["model_state_dict"]
    radius = float(torch.sigmoid(state["effect_stage.raw_radius"]).item())
    scale = float((F.softplus(state["effect_stage.raw_scale"]) + 1e-4).item())
    threshold = float(checkpoint["effect_threshold"])
    logit_threshold = math.log(threshold / (1.0 - threshold))
    return {
        "radius": radius,
        "scale": scale,
        "probability_threshold": threshold,
        # p(effect)>=t iff |delta| >= this value.
        "effective_abs_delta_cutoff": radius + logit_threshold / scale,
    }


def _result(model, checkpoint: dict, tensors, views, batch_size: int) -> dict:
    threshold = float(checkpoint["effect_threshold"])
    metrics = {}
    for split in ("val", "test"):
        split_metrics, _ = evaluate_cascade(
            model, tensors, views[split], threshold, batch_size
        )
        split_metrics.update(_routing_rates(split_metrics))
        metrics[split] = split_metrics
    return {
        "checkpoint": "",
        "effect_threshold": threshold,
        "boundary": _boundary(checkpoint),
        "selection": checkpoint["selection"],
        "metrics": metrics,
    }


def _summary_rows(results: dict[str, dict]) -> list[dict]:
    rows = []
    for method in METHOD_LABELS:
        result = results[method]
        for split in ("val", "test"):
            metrics = result["metrics"][split]
            rows.append({
                "method": method,
                "description": METHOD_LABELS[method],
                "split": split,
                "threshold": result["effect_threshold"],
                "learned_radius": result["boundary"]["radius"],
                "learned_scale": result["boundary"]["scale"],
                "effective_abs_delta_cutoff": result["boundary"]["effective_abs_delta_cutoff"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "effect_macro_f1": metrics["effect"]["macro_f1"],
                "direction_macro_f1": metrics["direction_true_effect"]["macro_f1"],
                "lof_f1": metrics["lof_f1"],
                "gof_f1": metrics["gof_f1"],
                "neutral_f1": metrics["neutral_f1"],
                "effect_missed_rate": metrics["effect_missed_rate"],
                "neutral_routed_rate": metrics["neutral_routed_rate"],
            })
    return rows


def _annotate(ax, bars, percent=False) -> None:
    for bar in bars:
        ax.annotate(
            (f"{bar.get_height():.1%}" if percent else f"{bar.get_height():.3f}"),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8,
        )


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
            [results[m]["metrics"][split]["macro_f1"] for m in methods],
            width, label=split.upper(),
        )
        _annotate(axes[0], bars)
    axes[0].set_xticks(x, [METHOD_LABELS[m] for m in methods])
    axes[0].set_title("Final three-class macro-F1")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend()

    classes = ("lof", "gof", "neutral")
    x2 = np.arange(3)
    for offset, method in enumerate(methods):
        bars = axes[1].bar(
            x2 + (offset - .5) * width,
            [results[method]["metrics"]["test"][f"{label}_f1"] for label in classes],
            width, label=METHOD_LABELS[method], color=COLORS[method],
        )
        _annotate(axes[1], bars)
    axes[1].set_xticks(x2, [label.upper() for label in classes])
    axes[1].set_title("Per-class F1 on the test set")
    axes[1].set_ylabel("Class F1")
    axes[1].legend(fontsize=8)

    errors = ("effect_missed_rate", "neutral_routed_rate")
    x3 = np.arange(2)
    for offset, method in enumerate(methods):
        bars = axes[2].bar(
            x3 + (offset - .5) * width,
            [results[method]["metrics"]["test"][name] for name in errors],
            width, label=METHOD_LABELS[method], color=COLORS[method],
        )
        _annotate(axes[2], bars, percent=True)
    axes[2].set_xticks(x3, ["Effect false-negative rate", "NEUTRAL false-positive rate"])
    axes[2].set_title("Effect-gate conditional error rates")
    axes[2].set_ylabel("Conditional error rate (%)")
    from matplotlib.ticker import PercentFormatter
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=8)
    fig.suptitle("External validation threshold vs learned Effect boundary", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_histories(histories: dict[str, list[dict]], path: Path, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, stage in enumerate(("effect", "direction")):
        for method in METHOD_LABELS:
            rows = [r for r in histories[method] if r["stage"] == stage]
            axes[0, column].plot(
                [r["epoch"] for r in rows], [r["train_loss"] for r in rows],
                label=METHOD_LABELS[method], color=COLORS[method],
            )
            axes[1, column].plot(
                [r["epoch"] for r in rows], [r["val_macro_f1"] for r in rows],
                label=METHOD_LABELS[method], color=COLORS[method],
            )
        axes[0, column].set_title(f"{stage.title()} train loss")
        axes[1, column].set_title(f"{stage.title()} validation macro-F1")
        for row in range(2):
            axes[row, column].set_xlabel("Epoch")
            axes[row, column].grid(alpha=.25)
            axes[row, column].legend(fontsize=8)
    fig.suptitle("Training and model-selection curves", fontsize=15)
    save_figure(fig, path, config)
    plt.close(fig)


def _plot_confusions(results: dict[str, dict], path: Path, config=None) -> None:
    from phase_3_variant_decoder.outcome_tables import plot_prediction_counts
    plot_prediction_counts(
        {METHOD_LABELS[name]: results[name]["metrics"]["test"] for name in METHOD_LABELS},
        path, title="Test prediction counts — correct versus incorrect assignments", config=config,
    )


def generate_report(results: dict[str, dict], histories: dict[str, list[dict]],
                    output: str | Path, config=None) -> dict[str, str]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "comparison": root / "learned_threshold_comparison.png",
        "training_curves": root / "learned_threshold_training_curves.png",
        "confusions": root / "learned_threshold_prediction_counts.png",
        "summary_csv": root / "learned_threshold_summary.csv",
        "summary_markdown": root / "learned_threshold_summary.md",
    }
    _plot_comparison(results, artifacts["comparison"], config=config)
    _plot_histories(histories, artifacts["training_curves"], config=config)
    _plot_confusions(results, artifacts["confusions"], config=config)
    rows = _summary_rows(results)
    with artifacts["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Learned Effect boundary comparison", "",
        "Both conditions use the same filtered data, split, cached protein inputs, GO vectors, "
        "architecture, seed, loss weighting and training hyperparameters. The baseline chooses an "
        "external probability threshold on validation Effect macro-F1. The learned-radius condition "
        "keeps the probability threshold at 0.50; BCE therefore learns the routing boundary through "
        "the model's radius parameter. Test is evaluation-only.", "",
        "| Method | Split | Threshold | Radius | Effective abs(delta) cutoff | Final macro-F1 | Effect F1 | Direction F1 | LOF F1 | GOF F1 | Neutral F1 | Effect missed | Neutral routed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['description']} | {row['split']} | {row['threshold']:.2f} | "
            f"{row['learned_radius']:.4f} | {row['effective_abs_delta_cutoff']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['effect_macro_f1']:.4f} | "
            f"{row['direction_macro_f1']:.4f} | {row['lof_f1']:.4f} | "
            f"{row['gof_f1']:.4f} | {row['neutral_f1']:.4f} | "
            f"{row['effect_missed_rate']:.1%} | {row['neutral_routed_rate']:.1%} |"
        )
    artifacts["summary_markdown"].write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return finalize_report(artifacts, config)


def _start_wandb(cfg: VariantDecoderConfig, baseline_checkpoint: Path,
                 name: str | None, project: str | None, entity: str | None,
                 enabled: bool):
    if not enabled:
        return None, None
    try:
        import wandb
        run = wandb.init(
            project=project or cfg.wandb_project,
            entity=entity or cfg.wandb_entity,
            name=name or "phase3-learned-effect-threshold",
            tags=["phase3", "cascade", "learned-threshold", "controlled-comparison"],
            config={
                **configuration_metadata(cfg),
                "experiment": "validation_threshold_vs_learned_radius",
                "baseline_checkpoint": str(baseline_checkpoint),
                "controlled_variables": [
                    "data", "split", "protein_embeddings", "GO_embeddings",
                    "architecture", "seed", "loss", "class_weighting",
                    "training_hyperparameters",
                ],
                "changed_variable": "effect_threshold_strategy",
                "test_used_for_selection": False,
                "family": cfg.family,
                "embedding_type": cfg.embedding_type,
                "input_mode": cfg.input_mode,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "early_stop_patience": cfg.early_stop_patience,
                "batch_size": cfg.batch_size,
                "cascade_balance_classes": cfg.cascade_balance_classes,
            },
        )
        log_reporting_guide(run, wandb, [vars(cfg)])
        return run, wandb
    except Exception as exc:
        raise RuntimeError(
            "W&B could not start. Run `wandb login`, fix the connection, or use --no_wandb."
        ) from exc


def _log_wandb(run, wandb, results: dict[str, dict],
               histories: dict[str, list[dict]], artifacts: dict[str, str]) -> None:
    if run is None:
        return
    # The learned-radius history was logged live by fit_cascade. Re-log the
    # frozen baseline history in its own namespace for an aligned comparison.
    for row in histories["validation_threshold"]:
        run.log({
            f"validation_threshold/{row['stage']}/{key}": value
            for key, value in row.items() if key != "stage"
        })
    rows = _summary_rows(results)
    columns = list(rows[0])
    boundary_rows = [[
        METHOD_LABELS[method], results[method]["effect_threshold"],
        results[method]["boundary"]["radius"], results[method]["boundary"]["scale"],
        results[method]["boundary"]["effective_abs_delta_cutoff"],
    ] for method in METHOD_LABELS]
    run.log({
        "00_Overview/Final_Metrics": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in rows]
        ),
        "01_Training/Curves": wandb.Image(artifacts["training_curves"], caption=contextual_caption("learned_training_curves", dict(getattr(run, "config", {})))),
        "02_Final_Comparison/F1_and_Routing": wandb.Image(artifacts["comparison"], caption=contextual_caption("comparison", dict(getattr(run, "config", {})))),
        "03_Prediction_Counts/Test": wandb.Image(artifacts["confusions"], caption=contextual_caption("confusions", dict(getattr(run, "config", {})))),
        "04_Learned_Boundary/Parameters": wandb.Table(
            columns=["method", "probability_threshold", "learned_radius", "learned_scale",
                     "effective_abs_delta_cutoff"],
            data=boundary_rows,
        ),
    })
    for method in METHOD_LABELS:
        boundary = results[method]["boundary"]
        for key, value in boundary.items():
            run.summary[f"{method}/boundary/{key}"] = value
        for split in ("val", "test"):
            metrics = results[method]["metrics"][split]
            for key in (
                "macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1",
                "effect_missed_rate", "neutral_routed_rate",
            ):
                run.summary[f"{method}/{split}/{key}"] = metrics[key]


def run_experiment(baseline_run: str, *, cache_dir: str | None, out_dir: str,
                   cuda: str | None = None, dataset_dir: str | None = None,
                   sequences_json: str | None = None, global_h5: str | None = None,
                   local_h5: str | None = None,
                   wandb_enabled: bool = True, wandb_project: str | None = None,
                   wandb_entity: str | None = None,
                   wandb_run_name: str | None = None,
                   quiet: bool = False) -> dict:
    baseline_checkpoint = _checkpoint_path(baseline_run)
    raw = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
    if raw.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Baseline is not a GO-conditioned cascade checkpoint")
    cfg = _load_saved_config(raw, {
        "cuda": cuda,
        "dataset_dir": dataset_dir,
        "sequences_json": sequences_json,
        "global_h5": global_h5,
        "local_h5": local_h5,
    })
    if cfg.effect_threshold_strategy != "validation":
        raise ValueError(
            "Baseline must use validation threshold selection; got "
            f"{cfg.effect_threshold_strategy!r}"
        )

    # The checkpoint carries its GO table, so the original NPZ is unnecessary.
    prepare_cfg = copy.deepcopy(cfg)
    prepare_cfg.architecture = "legacy"
    tensors, views, go_vocab, device = prepare_run(
        prepare_cfg, cache_dir=cache_dir, quiet=quiet
    )
    baseline_model, baseline_ckpt = load_cascade(baseline_checkpoint, device)
    if baseline_ckpt["go_vocab"] != go_vocab:
        raise ValueError("Current filtered GO vocabulary differs from the baseline checkpoint")
    attach_checkpoint_go(tensors, go_vocab, baseline_ckpt)
    if tensors.input_dim != baseline_model.spec["protein_dim"]:
        raise ValueError("Protein input dimension differs from the baseline checkpoint")

    baseline_result = _result(
        baseline_model, baseline_ckpt, tensors, views, cfg.batch_size
    )
    baseline_result["checkpoint"] = str(baseline_checkpoint)
    baseline_history_path = baseline_checkpoint.parent / "history.json"
    if not baseline_history_path.is_file():
        raise FileNotFoundError(f"Baseline history not found: {baseline_history_path}")
    baseline_history = json.loads(baseline_history_path.read_text(encoding="utf-8"))

    output_parent = Path(out_dir).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="learned_threshold_", dir=output_parent))
    run, wandb = _start_wandb(
        cfg, baseline_checkpoint, wandb_run_name, wandb_project,
        wandb_entity, wandb_enabled,
    )
    try:
        learned_cfg = copy.deepcopy(cfg)
        learned_cfg.architecture = "cascade"
        learned_cfg.effect_threshold_strategy = "learned_radius"
        if not quiet:
            print("\n[TRAIN] Learned Effect boundary: threshold fixed at 0.50")
        learned_fit = fit_cascade(
            learned_cfg, tensors, views, go_vocab, device,
            out_dir=root / "learned_radius", run=run, quiet=quiet,
            log_prefix="learned_radius",
        )
        learned_checkpoint = Path(learned_fit["checkpoint"])
        learned_model, learned_ckpt = load_cascade(learned_checkpoint, device)
        learned_result = _result(
            learned_model, learned_ckpt, tensors, views, learned_cfg.batch_size
        )
        learned_result["checkpoint"] = str(learned_checkpoint)
        learned_history = json.loads(
            (learned_checkpoint.parent / "history.json").read_text(encoding="utf-8")
        )
        results = {
            "validation_threshold": baseline_result,
            "learned_radius": learned_result,
        }
        histories = {
            "validation_threshold": baseline_history,
            "learned_radius": learned_history,
        }
        artifacts = generate_report(results, histories, root, config=vars(cfg))
        _log_wandb(run, wandb, results, histories, artifacts)
        wandb_url = run.url if run is not None else None
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    experiment = {
        "format": "contvar_learned_effect_threshold_comparison_v1",
        "controlled_comparison": True,
        "test_used_for_selection": False,
        "config": vars(cfg),
        "runs": results,
        "artifacts": artifacts,
        "wandb_url": wandb_url,
    }
    output_json = root / "learned_threshold_comparison.json"
    output_json.write_text(
        json.dumps(experiment, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    experiment["output_json"] = str(output_json)
    if run is not None:
        run.summary["output_dir"] = str(root)
        run.finish()
    if not quiet:
        print(f"\n[DONE] learned threshold comparison -> {root}")
        for method in METHOD_LABELS:
            test = results[method]["metrics"]["test"]
            boundary = results[method]["boundary"]
            print(
                f"  {method}: threshold={results[method]['effect_threshold']:.2f} "
                f"radius={boundary['radius']:.4f} "
                f"effective_cutoff={boundary['effective_abs_delta_cutoff']:.4f} | "
                f"test macro-F1={test['macro_f1']:.4f} | "
                f"missed={test['effect_missed_rate']:.1%} | "
                f"neutral routed={test['neutral_routed_rate']:.1%}"
            )
        for name, artifact in artifacts.items():
            print(f"  {name}: {artifact}")
        if wandb_url:
            print(f"  wandb: {wandb_url}")
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain Effect with a learned radius boundary and compare it to a validation threshold"
    )
    parser.add_argument(
        "--baseline_run", required=True,
        help="Existing validation-threshold cascade directory or decoder.pt",
    )
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
    parser.add_argument("--wandb_run_name", default="phase3-learned-effect-threshold")
    args = parser.parse_args()
    run_experiment(
        args.baseline_run, cache_dir=args.cache_dir, out_dir=args.out_dir,
        cuda=args.cuda, dataset_dir=args.dataset_dir,
        sequences_json=args.sequences_json, global_h5=args.global_h5,
        local_h5=args.local_h5,
        wandb_enabled=not args.no_wandb, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
