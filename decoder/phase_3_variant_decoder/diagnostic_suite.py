"""Run the Phase-3 encoder, input, split and seed diagnostic matrix.

The suite keeps the weighted GO-conditioned cascade fixed while changing one
diagnostic axis at a time:

* hard / seq_phase1 / {global, local, both}: local-variant signal
* hard / {raw_esm, seq_phase0, seq_phase1} / both: encoder representation
* {easy, hard} / seq_phase1 / both: protein-disjoint generalisation
* extra seeds for hard / seq_phase1 / both: run-to-run stability

All runs are logged into one WandB run and summarized in CSV, Markdown, JSON and
presentation-ready PNGs. A previously completed weighted baseline can be
reused after strict artifact/configuration checks.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from phase_3_variant_decoder.cascade import fit_cascade
from phase_3_variant_decoder.compare_architectures import _load_cascade_run
from phase_3_variant_decoder.presentation_terms import (
    METRIC_DEFINITIONS, REPORTING_VERSION, INPUT_LABELS, encoder_label,
    configuration_metadata, image_caption, log_reporting_guide, save_figure,
)
from phase_3_variant_decoder.config import VariantDecoderConfig, resolve_default_go_embeddings
from phase_3_variant_decoder.train import prepare_run


EMBEDDINGS = ("raw_esm", "seq_phase0", "seq_phase1")


@dataclass(frozen=True)
class DiagnosticSpec:
    family: str
    embedding: str
    input_mode: str
    seed: int
    axis: str

    @property
    def key(self) -> str:
        return f"{self.family}_{self.embedding}_{self.input_mode}_seed{self.seed}"

    @property
    def label(self) -> str:
        return f"{self.family} / {encoder_label(self.embedding)} / {INPUT_LABELS[self.input_mode]} / seed {self.seed}"

    @property
    def display_metadata(self) -> dict:
        return configuration_metadata(asdict(self))


def build_specs(seeds: list[int]) -> list[DiagnosticSpec]:
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    primary = seeds[0]
    specs = [
        DiagnosticSpec("hard", "seq_phase1", "both", primary, "baseline"),
        DiagnosticSpec("hard", "seq_phase1", "global", primary, "input"),
        DiagnosticSpec("hard", "seq_phase1", "local", primary, "input"),
        DiagnosticSpec("hard", "raw_esm", "both", primary, "encoder"),
        DiagnosticSpec("hard", "seq_phase0", "both", primary, "encoder"),
        DiagnosticSpec("easy", "seq_phase1", "both", primary, "split"),
    ]
    specs.extend(
        DiagnosticSpec("hard", "seq_phase1", "both", seed, "seed")
        for seed in seeds[1:]
    )
    return specs


def _cfg_for_spec(base: VariantDecoderConfig, spec: DiagnosticSpec,
                  embedding_paths: dict[str, tuple[str, str]]) -> VariantDecoderConfig:
    cfg = copy.deepcopy(base)
    cfg.family = spec.family
    cfg.embedding_type = spec.embedding
    cfg.input_mode = spec.input_mode
    cfg.seed = spec.seed
    cfg.architecture = "cascade"
    cfg.global_h5, cfg.local_h5 = embedding_paths[spec.embedding]
    cfg.global_h5_secondary = None
    cfg.local_h5_secondary = None
    cfg.wandb_run_name = f"phase3-diagnostic-{spec.key}"
    return cfg


def _validate_paths(base: VariantDecoderConfig,
                    embedding_paths: dict[str, tuple[str, str]]) -> None:
    missing = []
    for embedding in EMBEDDINGS:
        if embedding not in embedding_paths:
            missing.append(f"{embedding}: no path pair")
            continue
        for kind, path in zip(("global", "local"), embedding_paths[embedding]):
            if not Path(path).is_file():
                missing.append(f"{embedding}/{kind}: {path}")
    for name, path in (("dataset_dir", base.dataset_dir),
                       ("sequences_json", base.sequences_json),
                       ("go_embeddings", base.go_embeddings)):
        if path is None or not Path(path).exists():
            missing.append(f"{name}: {path}")
    if missing:
        raise FileNotFoundError("Diagnostic preflight failed:\n  - " + "\n  - ".join(missing))


def _start_wandb(base: VariantDecoderConfig, specs: list[DiagnosticSpec],
                 reuse_baseline_run: str | None):
    if not base.wandb_enabled:
        return None, None
    try:
        import wandb
        tags, meta = base.wandb_metadata()
        run = wandb.init(
            project=base.wandb_project,
            entity=base.wandb_entity,
            name=base.wandb_run_name,
            tags=[*tags, "diagnostic-suite", "encoder-ablation", "input-ablation"],
            config={
                **meta,
                "architecture": "cascade_diagnostic_suite",
                "matrix": [asdict(spec) | spec.display_metadata for spec in specs],
                "weighted_bce": base.cascade_balance_classes and base.cascade_loss == "bce",
                "cascade_loss": base.cascade_loss,
                "reuse_baseline_run": reuse_baseline_run,
                "epochs": base.epochs,
                "early_stop_patience": base.early_stop_patience,
                "batch_size": base.batch_size,
                "hidden_dims": base.hidden_dims,
                "dropout": base.dropout,
                "lr": base.lr,
                "weight_decay": base.weight_decay,
                "dataset_dir": base.dataset_dir,
                "go_artifact": base.go_embeddings,
                "configuration_scope": "Matrix of configurations; top-level fields describe the reference only.",
            },
        )
        return run, wandb
    except Exception as exc:
        raise RuntimeError(
            "WandB is enabled but the diagnostic suite could not initialize its run. "
            "Run `wandb login`, fix the connection/configuration, or explicitly use "
            "--no_wandb for a local-only run."
        ) from exc


def _route_rates(metrics: dict) -> tuple[float | None, float | None]:
    confusion = metrics["effect"]["confusion"]
    neutral_n = sum(confusion[0])
    effect_n = sum(confusion[1])
    missed = metrics["effect_missed_n"] / effect_n if effect_n else None
    routed = metrics["neutral_routed_n"] / neutral_n if neutral_n else None
    return missed, routed


def _row(spec: DiagnosticSpec, result: dict, split: str, reused: bool) -> dict:
    metrics = result["metrics"][split]
    effect_missed_rate, neutral_routed_rate = _route_rates(metrics)
    row = {
        "key": spec.key,
        "axis": spec.axis,
        "family": spec.family,
        "embedding": spec.embedding,
        "input_mode": spec.input_mode,
        "seed": spec.seed,
        "reused": reused,
        "split": split,
        "encoder_label": encoder_label(spec.embedding),
        "encoder_training_history": spec.display_metadata["encoder_training_history"],
        "input_label": INPUT_LABELS[spec.input_mode],
        "n_rows": metrics["n"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "effect_macro_f1": metrics["effect"]["macro_f1"],
        "direction_macro_f1": metrics["direction_true_effect"]["macro_f1"],
        "effect_missed_rate": effect_missed_rate,
        "neutral_routed_rate": neutral_routed_rate,
        "seen_go_macro_f1": metrics.get("seen_in_train", {}).get("macro_f1"),
        "unseen_go_macro_f1": metrics.get("not_seen_in_train", {}).get("macro_f1"),
        "seen_go_n": metrics.get("seen_in_train", {}).get("n"),
        "unseen_go_n": metrics.get("not_seen_in_train", {}).get("n"),
    }
    for label in ("lof", "gof", "neutral"):
        for metric in ("precision", "recall", "f1", "auprc"):
            row[f"{label}_{metric}"] = metrics[f"{label}_{metric}"]
    return row


def _rows(specs: list[DiagnosticSpec], results: dict[str, dict],
          reused_keys: set[str]) -> list[dict]:
    return [
        _row(spec, results[spec.key], split, spec.key in reused_keys)
        for spec in specs for split in ("val", "test")
    ]


def _primary_result(specs: list[DiagnosticSpec], results: dict[str, dict],
                    *, family: str, embedding: str, input_mode: str) -> dict:
    seed = specs[0].seed
    key = DiagnosticSpec(family, embedding, input_mode, seed, "").key
    return results[key]


def _annotate(ax, bars) -> None:
    for bar in bars:
        value = bar.get_height()
        if np.isfinite(value):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)


def _plot_overview(specs: list[DiagnosticSpec], results: dict[str, dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = specs[0].seed
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    def paired(ax, labels, values, title):
        x = np.arange(len(labels))
        width = 0.34
        for offset, split in enumerate(("val", "test")):
            bars = ax.bar(x + (offset - 0.5) * width, values[split], width,
                          label=split.upper())
            _annotate(ax, bars)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    modes = ("global", "local", "both")
    paired(
        axes[0, 0], [INPUT_LABELS[mode].replace(" (concat)", "") for mode in modes],
        {split: [results[DiagnosticSpec("hard", "seq_phase1", mode, primary, "").key]
                 ["metrics"][split]["macro_f1"] for mode in modes]
         for split in ("val", "test")},
        "Input ablation: hard / Phase 0 -> Phase 1",
    )

    embeddings = EMBEDDINGS
    paired(
        axes[0, 1], [encoder_label(name, multiline=True) for name in embeddings],
        {split: [results[DiagnosticSpec("hard", emb, "both", primary, "").key]
                 ["metrics"][split]["macro_f1"] for emb in embeddings]
         for split in ("val", "test")},
        "Encoder training history: hard / Global + Local",
    )

    families = ("easy", "hard")
    paired(
        axes[1, 0], [family.upper() for family in families],
        {split: [results[DiagnosticSpec(family, "seq_phase1", "both", primary, "").key]
                 ["metrics"][split]["macro_f1"] for family in families]
         for split in ("val", "test")},
        "Split comparison: Phase 0 -> Phase 1 / Global + Local",
    )

    seed_specs = [spec for spec in specs
                  if spec.family == "hard" and spec.embedding == "seq_phase1"
                  and spec.input_mode == "both"]
    seed_values = [results[spec.key]["metrics"]["test"]["macro_f1"] for spec in seed_specs]
    seed_x = np.arange(len(seed_specs))
    axes[1, 1].scatter(seed_x, seed_values, zorder=3)
    axes[1, 1].set_xticks(seed_x, [str(spec.seed) for spec in seed_specs])
    mean, std = float(np.mean(seed_values)), float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
    axes[1, 1].axhline(mean, color="#dc2626", linestyle="--", label=f"mean={mean:.3f}")
    if len(seed_values) > 1:
        axes[1, 1].fill_between(
            [-.4, len(seed_specs) - .6],
            mean - std, mean + std, color="#dc2626", alpha=0.15, label=f"std={std:.3f}",
        )
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xlabel("Seed")
    axes[1, 1].set_ylabel("Test macro-F1")
    axes[1, 1].set_title("Seed stability: hard / Phase 0 -> Phase 1 / Global + Local")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()

    fig.suptitle("Phase-3 cascade diagnostic suite", fontsize=16)
    save_figure(fig, path)
    plt.close(fig)


def _plot_error_breakdown(specs: list[DiagnosticSpec], results: dict[str, dict],
                          path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_specs = [spec for spec in specs if spec.seed == specs[0].seed]
    labels = [f"{spec.family}\n{encoder_label(spec.embedding, multiline=True)}\n{INPUT_LABELS[spec.input_mode]}"
              for spec in primary_specs]
    x = np.arange(len(primary_specs))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    stage_metrics = ("effect", "direction", "final")
    width = 0.25
    for offset, name in enumerate(stage_metrics):
        values = []
        for spec in primary_specs:
            metrics = results[spec.key]["metrics"]["test"]
            values.append({
                "effect": metrics["effect"]["macro_f1"],
                "direction": metrics["direction_true_effect"]["macro_f1"],
                "final": metrics["macro_f1"],
            }[name])
        axes[0, 0].bar(x + (offset - 1) * width, values, width, label={"effect": "Effect (binary, all rows)", "direction": "Direction (binary, true effects)", "final": "Final (three-class, all rows)"}[name])
    axes[0, 0].set_title("Binary stage metrics versus end-to-end classification")
    axes[0, 0].set_ylabel("Test macro-F1")
    axes[0, 0].legend()

    for offset, name in enumerate(("Effect false-negative rate", "NEUTRAL false-positive rate")):
        values = []
        for spec in primary_specs:
            rates = _route_rates(results[spec.key]["metrics"]["test"])
            values.append(rates[offset])
        axes[0, 1].bar(x + (offset - 0.5) * 0.34, values, 0.34, label=name)
    axes[0, 1].set_title("Cascade routing errors")
    axes[0, 1].set_ylabel("Conditional error rate (%)")
    from matplotlib.ticker import PercentFormatter
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].legend()

    for offset, name in enumerate(("GO seen in\ndecoder training", "GO unseen in\ndecoder training")):
        values = []
        for spec in primary_specs:
            metrics = results[spec.key]["metrics"]["test"]
            key = "seen_in_train" if offset == 0 else "not_seen_in_train"
            value = metrics.get(key, {}).get("macro_f1")
            values.append(np.nan if value is None else value)
        axes[1, 0].bar(x + (offset - 0.5) * 0.34, values, 0.34, label=name)
    axes[1, 0].set_title("Target GO-ID coverage groups")
    axes[1, 0].set_ylabel("Test macro-F1")
    axes[1, 0].legend()

    class_width = 0.25
    for offset, label in enumerate(("lof", "gof", "neutral")):
        values = [results[spec.key]["metrics"]["test"][f"{label}_f1"]
                  for spec in primary_specs]
        axes[1, 1].bar(x + (offset - 1) * class_width, values, class_width,
                       label=label.upper())
    axes[1, 1].set_title("Class-specific test F1")
    axes[1, 1].set_ylabel("F1")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cascade error and generalisation diagnostics", fontsize=16)
    save_figure(fig, path)
    plt.close(fig)


def _comparison_result(specs: list[DiagnosticSpec], results: dict[str, dict],
                       family: str, embedding: str, input_mode: str) -> dict:
    key = DiagnosticSpec(
        family, embedding, input_mode, specs[0].seed, "presentation"
    ).key
    return results[key]


def _plot_f1_comparison(items: list[tuple[str, dict]], path: Path,
                        title: str, note: str) -> None:
    """Create one slide-friendly F1 comparison with a fixed visual grammar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [label for label, _ in items]
    x = np.arange(len(items))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    width = 0.34
    for offset, split in enumerate(("val", "test")):
        values = [result["metrics"][split]["macro_f1"] for _, result in items]
        bars = axes[0].bar(x + (offset - .5) * width, values, width,
                           label=split.upper())
        _annotate(axes[0], bars)
    axes[0].set_title("End-to-end three-class performance")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend()

    class_width = .25
    colors = ("#2563eb", "#f97316", "#16a34a")
    for offset, (label, color) in enumerate(zip(("lof", "gof", "neutral"), colors)):
        values = [result["metrics"]["test"][f"{label}_f1"] for _, result in items]
        bars = axes[1].bar(x + (offset - 1) * class_width, values, class_width,
                           label=label.upper(), color=color)
        _annotate(axes[1], bars)
    axes[1].set_title("Per-class F1 on the test set")
    axes[1].set_ylabel("F1")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle(title, fontsize=15)
    save_figure(fig, path, context=note)
    plt.close(fig)


def _plot_seed_stability(specs: list[DiagnosticSpec], results: dict[str, dict],
                         path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [spec for spec in specs
                if spec.family == "hard" and spec.embedding == "seq_phase1"
                and spec.input_mode == "both"]
    seeds = [spec.seed for spec in selected]
    val = [results[spec.key]["metrics"]["val"]["macro_f1"] for spec in selected]
    test = [results[spec.key]["metrics"]["test"]["macro_f1"] for spec in selected]
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    width = .34
    _annotate(ax, ax.bar(x - width / 2, val, width, label="VAL"))
    _annotate(ax, ax.bar(x + width / 2, test, width, label="TEST"))
    mean = float(np.mean(test))
    std_note = f"sample SD={np.std(test, ddof=1):.3f}" if len(test) > 1 else "SD not estimable (one seed)"
    ax.axhline(mean, color="#dc2626", linestyle="--",
               label=f"Test mean={mean:.3f}; {std_note}")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Random-seed sensitivity: hard / Phase 0 -> Phase 1 / Global + Local")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    save_figure(fig, path)
    plt.close(fig)


def _plot_cascade_diagnostics(specs: list[DiagnosticSpec], results: dict[str, dict],
                              path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _comparison_result(
        specs, results, "hard", "seq_phase1", "both"
    )["metrics"]["test"]
    missed, routed = _route_rates(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    values = [metrics["effect"]["macro_f1"],
              metrics["direction_true_effect"]["macro_f1"], metrics["macro_f1"]]
    _annotate(axes[0], axes[0].bar(
        ["Effect detection\nBinary: all rows",
         "Effect direction\nBinary: true effects",
         "Final classification\nThree-class: all rows"], values,
        color=["#0ea5e9", "#8b5cf6", "#334155"]
    ))
    axes[0].set_title("Cascade stages\nDifferent tasks and evaluation populations")
    axes[0].set_ylabel("Test macro-F1")

    rates = [np.nan if missed is None else missed,
             np.nan if routed is None else routed]
    effect_n = sum(metrics["effect"]["confusion"][1])
    neutral_n = sum(metrics["effect"]["confusion"][0])
    rate_bars = axes[1].bar(
        [f"Effect false-negative rate\n{metrics['effect_missed_n']:,} / {effect_n:,} true EFFECT",
         f"NEUTRAL false-positive rate\n{metrics['neutral_routed_n']:,} / {neutral_n:,} true NEUTRAL"], rates,
        color=["#dc2626", "#f59e0b"]
    )
    for bar in rate_bars:
        value = bar.get_height()
        if np.isfinite(value):
            axes[1].annotate(f"{value:.1%}", (bar.get_x() + bar.get_width()/2, value),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_title("Effect-gate routing errors")
    axes[1].set_ylabel("Conditional error rate (%)")
    from matplotlib.ticker import PercentFormatter
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))

    seen = metrics.get("seen_in_train", {}).get("macro_f1")
    unseen = metrics.get("not_seen_in_train", {}).get("macro_f1")
    go_values = [np.nan if seen is None else seen, np.nan if unseen is None else unseen]
    _annotate(axes[2], axes[2].bar(
        [f"GO seen in decoder train\nn={metrics.get('seen_in_train', {}).get('n', 'NA')}",
         f"GO unseen in decoder train\nn={metrics.get('not_seen_in_train', {}).get('n', 'NA')}"], go_values,
        color=["#16a34a", "#64748b"]
    ))
    axes[2].set_title("Target GO-ID coverage groups")
    axes[2].set_ylabel("Test macro-F1")

    for ax in axes:
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
    fig.suptitle(
        f"Reference: hard / Phase 0 -> Phase 1 / Global + Local / seed {specs[0].seed}\n"
        "Stage macro-F1 values are not directly comparable to final three-class macro-F1",
        fontsize=14,
    )
    save_figure(fig, path)
    plt.close(fig)


def _fmt(value) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _presentation_rows(specs: list[DiagnosticSpec], results: dict[str, dict]) -> list[dict]:
    primary = specs[0].seed
    definitions = [
        ("Input", "Global", "hard", "seq_phase1", "global", primary,
         "Mean-pooled protein embedding only"),
        ("Input", "Local", "hard", "seq_phase1", "local", primary,
         "Single mutation-site residue embedding only; not a sequence window or 3D pocket"),
        ("Input", "Both", "hard", "seq_phase1", "both", primary,
         "Concatenated global and mutation-site residue embeddings"),
        ("Encoder", "Pretrained ESM-2", "hard", "raw_esm", "both", primary,
         "Pretrained ESM-2 representation before task-specific fine-tuning"),
        ("Encoder", "Phase 0", "hard", "seq_phase0", "both", primary,
         "Sequence encoder after Phase 0 GO semantic pretraining"),
        ("Encoder", encoder_label("seq_phase1"), "hard", "seq_phase1", "both", primary,
         "Phase 0 GO-semantic pretraining followed by Phase 1 DMS fine-tuning; not DMS-only"),
        ("Split", "Easy", "easy", "seq_phase1", "both", primary,
         "Easy split: protein overlap between partitions is permitted"),
        ("Split", "Hard", "hard", "seq_phase1", "both", primary,
         "Hard split: protein-disjoint partitions"),
    ]
    rows = []
    for category, name, family, embedding, mode, seed, explanation in definitions:
        key = DiagnosticSpec(family, embedding, mode, seed, "").key
        val, test = (results[key]["metrics"][split] for split in ("val", "test"))
        rows.append({
            "comparison": category, "configuration": name, "seed": seed,
            "family": family, "embedding_type": embedding, "input_mode": mode,
            "encoder_label": encoder_label(embedding), "input_label": INPUT_LABELS[mode],
            "val_n": val["n"], "test_n": test["n"],
            "val_macro_f1": val["macro_f1"], "test_macro_f1": test["macro_f1"],
            "test_lof_f1": test["lof_f1"], "test_gof_f1": test["gof_f1"],
            "test_neutral_f1": test["neutral_f1"], "explanation": explanation,
        })
    for spec in specs:
        if (spec.family, spec.embedding, spec.input_mode) != ("hard", "seq_phase1", "both"):
            continue
        val, test = (results[spec.key]["metrics"][split] for split in ("val", "test"))
        rows.append({
            "comparison": "Seed", "configuration": f"Seed {spec.seed}",
            "family": spec.family, "embedding_type": spec.embedding, "input_mode": spec.input_mode,
            "encoder_label": encoder_label(spec.embedding), "input_label": INPUT_LABELS[spec.input_mode],
            "val_n": val["n"], "test_n": test["n"],
            "seed": spec.seed, "val_macro_f1": val["macro_f1"],
            "test_macro_f1": test["macro_f1"], "test_lof_f1": test["lof_f1"],
            "test_gof_f1": test["gof_f1"], "test_neutral_f1": test["neutral_f1"],
            "explanation": "Repeat run: hard / cumulative Phase 0 -> Phase 1 / Global + Local",
        })
    return rows


def _write_presentation_markdown(specs: list[DiagnosticSpec], results: dict[str, dict],
                                 rows: list[dict], path: Path) -> None:
    lines = [
        "# Phase-3 decoder: presentation report", "",
        "Model selection uses validation macro-F1. Test metrics describe the selected "
        "configuration and are not used for model selection.", "",
    ]
    lines.extend(["## Metric and representation definitions", "",
                  "| Term | Definition |", "|---|---|"])
    lines.extend(f"| {term} | {definition} |" for term, definition in METRIC_DEFINITIONS.items())
    lines.append("")
    for category, title in (("Input", "1. Global versus mutation-site input"),
                            ("Encoder", "2. Encoder training-stage ablation"),
                            ("Split", "3. Easy versus hard split"),
                            ("Seed", "4. Random-seed sensitivity")):
        lines.extend([
            f"## {title}", "",
            "| Configuration | Encoder history | Split / input / seed | Val n / Test n | Validation macro-F1 | Test macro-F1 | LOF F1 | GOF F1 | NEUTRAL F1 | Definition |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        selected = [row for row in rows if row["comparison"] == category]
        for row in selected:
            lines.append(
                f"| {row['configuration']} | {row['encoder_label']} | {row['family']} / {row['input_label']} / {row['seed']} | "
                f"{row['val_n']} / {row['test_n']} | {_fmt(row['val_macro_f1'])} | "
                f"{_fmt(row['test_macro_f1'])} | {_fmt(row['test_lof_f1'])} | "
                f"{_fmt(row['test_gof_f1'])} | {_fmt(row['test_neutral_f1'])} | "
                f"{row['explanation']} |"
            )
        lines.append("")

    def metric(family, embedding, mode, split="test"):
        return _comparison_result(specs, results, family, embedding, mode)["metrics"][split]["macro_f1"]

    both, local, global_only = (metric("hard", "seq_phase1", mode)
                                for mode in ("both", "local", "global"))
    raw, phase0, phase1 = (metric("hard", emb, "both") for emb in EMBEDDINGS)
    easy, hard = (metric(family, "seq_phase1", "both") for family in ("easy", "hard"))
    seed_specs = [spec for spec in specs
                  if (spec.family, spec.embedding, spec.input_mode)
                  == ("hard", "seq_phase1", "both")]
    seed_test = [results[spec.key]["metrics"]["test"]["macro_f1"] for spec in seed_specs]
    baseline = _comparison_result(specs, results, "hard", "seq_phase1", "both")["metrics"]["test"]
    class_values = {label: baseline[f"{label.lower()}_f1"]
                    for label in ("LOF", "GOF", "NEUTRAL")}
    weakest = min(class_values, key=class_values.get)
    lines.extend([
        "## Descriptive comparisons", "",
        f"- **Input comparison:** Global + mutation-site minus global-only: `{both-global_only:+.4f}`; "
        f"global + mutation-site minus mutation-site-only: `{both-local:+.4f}` test macro-F1. Positive values favor the combined input.",
        f"- **Encoder training history:** Pretrained ESM-2 `{raw:.4f}`, Phase 0 (GO) `{phase0:.4f}`, "
        f"Phase 0 (GO) -> Phase 1 (DMS) `{phase1:.4f}` test macro-F1. "
        "These are descriptive single-seed results, not a ranking for model selection. No DMS-only condition was tested.",
        f"- **Split comparison:** Easy minus hard test macro-F1: `{easy-hard:.4f}`. "
        "The evaluation populations differ, so this gap alone does not establish the cause of the difference.",
        f"- **Random-seed sensitivity:** Hard / Phase 0 -> Phase 1 / Global + Local test macro-F1 mean "
        f"`{np.mean(seed_test):.4f}`, sample standard deviation "
        f"`{_fmt(float(np.std(seed_test, ddof=1))) if len(seed_test) > 1 else 'NA (one seed)'}`.",
        f"- **Lowest per-class F1:** The reference configuration has its lowest class F1 for `{weakest}` "
        f"(`{class_values[weakest]:.4f}`).", "",
        "Effect, Direction and Final macro-F1 values refer to different tasks and "
        "evaluation populations: Effect is binary on all test rows; Direction is binary on true "
        "LOF/GOF rows, irrespective of gate routing; Final is three-class on all test rows. A lower final "
        "score does not establish that both stages failed on the same examples. "
        "", "",
        "Small differences between encoder stages measured with one seed do not establish "
        "superiority; confirm promising configurations with additional seeds.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_presentation(specs: list[DiagnosticSpec], results: dict[str, dict],
                          output_dir: str | Path, config: dict | None = None) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "input_ablation": output / "01_input_ablation_f1.png",
        "encoder_ablation": output / "02_encoder_ablation_f1.png",
        "split_generalization": output / "03_split_generalization_f1.png",
        "seed_stability": output / "04_seed_stability_f1.png",
        "cascade_diagnostics": output / "05_cascade_diagnostics.png",
        "presentation_csv": output / "presentation_scores.csv",
        "presentation_markdown": output / "presentation_report.md",
        "configuration_manifest": output / "configuration_manifest.json",
    }
    primary = specs[0].seed
    get = lambda family, embedding, mode: results[
        DiagnosticSpec(family, embedding, mode, primary, "").key
    ]
    _plot_f1_comparison(
        [(label, get("hard", "seq_phase1", mode))
         for label, mode in (("Global only", "global"), ("Local only", "local"), ("Global + Local", "both"))],
        artifacts["input_ablation"], "Input ablation: Global versus Local representations",
        f"Fixed: hard / Phase 0 -> Phase 1 / seed {primary}. Local = one mutation-site residue.",
    )
    _plot_f1_comparison(
        [(label, get("hard", emb, "both")) for label, emb in
         [(encoder_label(emb, multiline=True), emb) for emb in EMBEDDINGS]],
        artifacts["encoder_ablation"], "Encoder training-stage ablation",
        f"Fixed: hard / Global + Local / seed {primary}. Phase 0 -> Phase 1 is cumulative; no DMS-only encoder.",
    )
    _plot_f1_comparison(
        [(label, get(family, "seq_phase1", "both"))
         for label, family in (("Easy", "easy"), ("Hard", "hard"))],
        artifacts["split_generalization"], "Split comparison: Easy versus Hard",
        f"Fixed: Phase 0 -> Phase 1 / Global + Local / seed {primary}. Evaluation populations differ.",
    )
    _plot_seed_stability(specs, results, artifacts["seed_stability"])
    _plot_cascade_diagnostics(specs, results, artifacts["cascade_diagnostics"])

    rows = _presentation_rows(specs, results)
    with artifacts["presentation_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_presentation_markdown(
        specs, results, rows, artifacts["presentation_markdown"]
    )
    manifest = {
        "reporting_version": REPORTING_VERSION,
        "decoder": "GO-conditioned two-stage scalar cascade; not the proposed attention decoder",
        "class_weighting": (configuration_metadata(config).get("class_weighting", "Not recorded")
                            if config is not None else "Not supplied; inspect source configuration"),
        "configurations": [asdict(spec) | spec.display_metadata for spec in specs],
        "metric_definitions": METRIC_DEFINITIONS,
        "comparison_captions": {key: image_caption(key) for key in (
            "input_ablation", "encoder_ablation", "split_generalization", "seed_stability", "cascade_diagnostics")},
    }
    artifacts["configuration_manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with artifacts["presentation_markdown"].open("a", encoding="utf-8") as handle:
        handle.write(f"\nDecoder: {manifest['decoder']}. Class weighting: {manifest['class_weighting']}.\n")
    return {key: str(value) for key, value in artifacts.items()}


def generate_report(specs: list[DiagnosticSpec], results: dict[str, dict],
                    reused_keys: set[str], output_dir: str | Path, config: dict | None = None) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "overview": output / "diagnostic_overview.png",
        "error_breakdown": output / "diagnostic_error_breakdown.png",
        "summary_csv": output / "diagnostic_summary.csv",
        "summary_markdown": output / "diagnostic_summary.md",
    }
    artifacts.update({key: Path(value) for key, value in
                      generate_presentation(specs, results, output, config=config).items()})
    _plot_overview(specs, results, artifacts["overview"])
    _plot_error_breakdown(specs, results, artifacts["error_breakdown"])
    rows = _rows(specs, results, reused_keys)
    with artifacts["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Phase-3 cascade diagnostic suite", "",
        "All configurations use the GO-conditioned two-stage scalar cascade. Consult configuration_manifest.json for loss weighting. Configuration "
        "selection must use validation metrics; test metrics are final diagnostics.", "",
        "| Configuration | Split | Macro-F1 | Effect F1 | Direction F1 | LOF F1 | GOF F1 | Neutral F1 | Seen GO F1 | Unseen GO F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        value = lambda name: _fmt(row[name])
        lines.append(
            f"| {row['family']} / {row['encoder_label']} / {row['input_label']} / seed {row['seed']} | {row['split']} | {value('macro_f1')} | "
            f"{value('effect_macro_f1')} | {value('direction_macro_f1')} | "
            f"{value('lof_f1')} | {value('gof_f1')} | {value('neutral_f1')} | "
            f"{value('seen_go_macro_f1')} | {value('unseen_go_macro_f1')} |"
        )

    seed_specs = [spec for spec in specs
                  if spec.family == "hard" and spec.embedding == "seq_phase1"
                  and spec.input_mode == "both"]
    seed_test = [results[spec.key]["metrics"]["test"]["macro_f1"] for spec in seed_specs]
    lines.extend(["", "## Seed stability", "",
                  f"Test macro-F1 mean: `{np.mean(seed_test):.4f}`",
                  f"Test macro-F1 sample SD: `{_fmt(float(np.std(seed_test, ddof=1))) if len(seed_test) > 1 else 'NA (one seed)'}`", ""])
    artifacts["summary_markdown"].write_text("\n".join(lines), encoding="utf-8")
    return {key: str(value) for key, value in artifacts.items()}


def log_presentation_to_wandb(run, wandb, specs: list[DiagnosticSpec],
                              results: dict[str, dict], artifacts: dict[str, str]) -> None:
    """Log a small, ordered set of panels intended for presenting the suite."""
    rows = _presentation_rows(specs, results)
    columns = list(rows[0])
    log_reporting_guide(run, wandb, [asdict(spec) for spec in specs])
    run.log({
        "00_Overview/Score_Table": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in rows]
        ),
        "01_Input_Ablation/F1": wandb.Image(artifacts["input_ablation"], caption=image_caption("input_ablation")),
        "02_Encoder_Ablation/F1": wandb.Image(artifacts["encoder_ablation"], caption=image_caption("encoder_ablation")),
        "03_Split_Generalization/F1": wandb.Image(artifacts["split_generalization"], caption=image_caption("split_generalization")),
        "04_Seed_Stability/F1": wandb.Image(artifacts["seed_stability"], caption=image_caption("seed_stability")),
        "05_Cascade_Diagnostics/F1_and_Routing": wandb.Image(artifacts["cascade_diagnostics"], caption=image_caption("cascade_diagnostics")),
    })


def _log_final_wandb(run, wandb, specs: list[DiagnosticSpec], results: dict[str, dict],
                     reused_keys: set[str], artifacts: dict[str, str]) -> None:
    if run is None:
        return
    log_presentation_to_wandb(run, wandb, specs, results, artifacts)
    rows = _rows(specs, results, reused_keys)
    columns = list(rows[0])
    run.log({
        "80_Detailed_Metrics/All_Scores": wandb.Table(
            columns=columns, data=[[row[column] for column in columns] for row in rows]
        ),
        "80_Detailed_Metrics/Combined_Overview": wandb.Image(artifacts["overview"], caption=image_caption("overview")),
        "80_Detailed_Metrics/Combined_Error_Breakdown": wandb.Image(artifacts["error_breakdown"], caption=image_caption("error_breakdown")),
    })
    for spec in specs:
        for split in ("val", "test"):
            metrics = results[spec.key]["metrics"][split]
            for metric in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1"):
                run.summary[f"80_Detailed_Metrics/{spec.key}/{split}/{metric}"] = metrics[metric]


def run_suite(base: VariantDecoderConfig, *, embedding_paths: dict[str, tuple[str, str]],
              seeds: list[int], cache_root: str, out_dir: str,
              reuse_baseline_run: str | None = None, quiet: bool = False) -> dict:
    specs = build_specs(seeds)
    _validate_paths(base, embedding_paths)
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(
        prefix="diagnostic_suite_", dir=Path(out_dir).resolve()
    ))
    run, wandb = _start_wandb(base, specs, reuse_baseline_run)
    results: dict[str, dict] = {}
    reused_keys: set[str] = set()
    try:
        for spec in specs:
            cfg = _cfg_for_spec(base, spec, embedding_paths)
            cache_dir = str(Path(cache_root) / f"{spec.family}_{spec.embedding}_{spec.input_mode}")
            if not quiet:
                print(f"\n[DIAGNOSTIC] {spec.label}")
            tensors, views, go_vocab, device = prepare_run(
                cfg, cache_dir=cache_dir, quiet=quiet
            )
            is_reused_baseline = (
                reuse_baseline_run is not None
                and spec == specs[0]
            )
            # Keep epoch-level curves available without mixing them into the
            # presentation-first WandB sections.
            prefix = f"90_Training_Details/{spec.key}"
            if is_reused_baseline:
                if not quiet:
                    print(f"[DIAGNOSTIC] reusing baseline <- {reuse_baseline_run}")
                result = _load_cascade_run(
                    reuse_baseline_run, cfg, tensors.go_artifact_sha256, go_vocab,
                    {split: views[split]["n"] for split in ("train", "val", "test")},
                    run=run, log_prefix=prefix,
                )
                reused_keys.add(spec.key)
            else:
                result = fit_cascade(
                    cfg, tensors, views, go_vocab, device,
                    out_dir=root / "runs" / spec.key,
                    run=run, quiet=quiet, log_prefix=prefix,
                )
            results[spec.key] = result
            del tensors, views
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        artifacts = generate_report(specs, results, reused_keys, root, config=vars(base))
        _log_final_wandb(run, wandb, specs, results, reused_keys, artifacts)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise

    suite = {
        "format": "contvar_cascade_diagnostic_suite_v1",
        "specs": [asdict(spec) | {"key": spec.key} for spec in specs],
        "config": vars(base),
        "embedding_paths": embedding_paths,
        "reused_keys": sorted(reused_keys),
        "results": results,
        "artifacts": artifacts,
        "wandb_url": run.url if run is not None else None,
    }
    suite_path = root / "diagnostic_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    suite["suite_json"] = str(suite_path)
    if run is not None:
        run.summary["diagnostics/output_dir"] = str(root)
        run.finish()
    if not quiet:
        print(f"\n[DONE] diagnostic suite -> {root}")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
        if suite["wandb_url"]:
            print(f"  wandb: {suite['wandb_url']}")
    return suite


def _resolve_embedding_paths(args) -> dict[str, tuple[str, str]]:
    root = Path(args.embedding_dir) if args.embedding_dir else None
    paths = {}
    for embedding in EMBEDDINGS:
        global_path = getattr(args, f"{embedding}_global_h5")
        local_path = getattr(args, f"{embedding}_local_h5")
        if global_path is None and root is not None:
            global_path = str(root / f"phase3_{embedding}_global.h5")
        if local_path is None and root is not None:
            local_path = str(root / f"phase3_{embedding}_local.h5")
        paths[embedding] = (global_path, local_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run encoder/input/split/seed diagnostics for the weighted cascade"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--sequences_json", required=True)
    parser.add_argument("--go_embeddings", default=resolve_default_go_embeddings())
    parser.add_argument(
        "--embedding_dir", default=None,
        help="Directory containing phase3_{raw_esm,seq_phase0,seq_phase1}_{global,local}.h5",
    )
    for embedding in EMBEDDINGS:
        parser.add_argument(f"--{embedding}_global_h5", default=None)
        parser.add_argument(f"--{embedding}_local_h5", default=None)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reuse_baseline_run", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--allow_go_proxies", action="store_true")
    parser.add_argument("--drop_contradictory", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default="phase3-cascade-diagnostic-suite")
    args = parser.parse_args()

    if not args.go_embeddings:
        parser.error("--go_embeddings is required")
    embedding_paths = _resolve_embedding_paths(args)
    if any(path is None for pair in embedding_paths.values() for path in pair):
        parser.error(
            "Provide --embedding_dir or all six explicit --*_global_h5/--*_local_h5 paths"
        )

    base = VariantDecoderConfig("hard", "seq_phase1", "both")
    base.architecture = "cascade"
    base.dataset_dir = args.dataset_dir
    base.sequences_json = args.sequences_json
    base.go_embeddings = args.go_embeddings
    base.allow_go_proxies = args.allow_go_proxies
    base.drop_contradictory = args.drop_contradictory
    base.cascade_balance_classes = True
    base.cascade_loss = "bce"  # Preserve the documented historical diagnostic baseline.
    base.cuda = args.cuda
    base.wandb_enabled = not args.no_wandb
    base.wandb_run_name = args.wandb_run_name
    if args.wandb_project:
        base.wandb_project = args.wandb_project
    if args.wandb_entity:
        base.wandb_entity = args.wandb_entity
    for name in ("hidden_dims", "dropout", "lr", "batch_size", "epochs",
                 "early_stop_patience"):
        value = getattr(args, name)
        if value is not None:
            setattr(base, name, value)

    run_suite(
        base,
        embedding_paths=embedding_paths,
        seeds=args.seeds,
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        reuse_baseline_run=args.reuse_baseline_run,
    )


if __name__ == "__main__":
    main()
