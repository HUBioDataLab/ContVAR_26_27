"""Train matched hinge and squared-hinge cascades, with separate W&B sections.

Two independent binary stages and hard-cascade inference are retained. Loss is
the controlled variable: inputs, architecture, seed, margins, class weighting,
threshold strategy and training limits are identical. BCE is an optional control.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import html
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from phase_3_variant_decoder.cascade import fit_cascade
from phase_3_variant_decoder.cascade_objectives import LOSS_LABELS
from phase_3_variant_decoder.config import (
    EFFECT_THRESHOLD_STRATEGIES, EMBEDDING_TYPES, FAMILIES, INPUT_MODES,
    SECONDARY_REQUIRED, VariantDecoderConfig,
)
from phase_3_variant_decoder.outcome_tables import plot_prediction_counts, write_outcome_reports
from phase_3_variant_decoder.presentation_terms import configuration_metadata, save_figure
from phase_3_variant_decoder.train import prepare_run

FORMAT = "contvar_cascade_loss_comparison_v1"
SECTION = "Cascade_Loss_Comparison"
SECTIONS = {"hinge": "Hinge_Cascade", "squared_hinge": "Squared_Hinge_Cascade", "bce": "BCE_Cascade_Control"}
COLORS = {"hinge": "#2563eb", "squared_hinge": "#7c3aed", "bce": "#64748b"}
CAPTIONS = {
    "f1": "Aynı filtrelenmiş veri, seed, model, margin, class weighting ve karar stratejisiyle loss karşılaştırmasıdır. Macro-F1 üç sınıfa eşit ağırlık verir; Accuracy değildir. Her loss'un checkpoint'i yalnız validation ile seçilir. Sigmoid çıktılar calibrated probability değildir.",
    "training": "Üst sıra train loss, alt sıra validation binary Macro-F1'dır. Hinge ve squared hinge loss sayıları farklı ölçeklerdedir; küçüklükleri doğrudan başarı kıyası değildir. Effect bütün örneklerde, Direction yalnız gerçek LOF/GOF örneklerinde ölçülür. Test eğrisiyle seçim yapılmaz.",
    "outcomes": "Correct doğru, Incorrect yanlış tahmin sayısıdır. Class recovery her gerçek sınıfta doğru bulunan / mevcut örnek sayısını gösterir. Hinge ve squared hinge için aynı hard-cascade kararı kullanılır; bu deney argmax veya joint training içermez.",
}


def score_rows(results):
    rows = []
    for kind, result in results.items():
        for split in ("val", "test"):
            metric = result["metrics"][split]
            effect_n = metric["direction_true_effect"]["n"]
            neutral_n = metric["n"] - effect_n
            rows.append({"loss": kind, "description": LOSS_LABELS[kind], "split": split,
                         "cascade_margin": result["cascade_margin"],
                         "margin_space": "scaled_signed_logit",
                         "effect_threshold_strategy": result["selection"]["effect"]["threshold_strategy"],
                         "n": metric["n"], "effect_threshold": result["effect_threshold"],
                         "effect_macro_f1": metric["effect"]["macro_f1"],
                         "direction_true_effect_macro_f1": metric["direction_true_effect"]["macro_f1"],
                         "effect_missed_rate": metric["effect_missed_n"] / effect_n if effect_n else None,
                         "neutral_routed_rate": metric["neutral_routed_n"] / neutral_n if neutral_n else None,
                         **{key: metric[key] for key in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1",
                                                        "lof_precision", "gof_precision", "neutral_precision",
                                                        "lof_recall", "gof_recall", "neutral_recall")}})
    return rows


def _outcome_result(results, family):
    records = {}
    for kind, result in results.items():
        for split, metric in result["metrics"].items():
            records[f"{split}/{kind}"] = {**metric, "method": kind, "description": LOSS_LABELS[kind],
                                          "split": split, "mean_confusion": metric["confusion"], "repeats": 1}
    return {"data": {"family": family}, "results": records}


def generate_report(results, histories, directory, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    kinds = list(results)
    artifacts, details = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    x = np.arange(len(kinds))
    for offset, split in enumerate(("val", "test")):
        bars = axes[0].bar(x + (offset-.5)*.34,
                          [results[k]["metrics"][split]["macro_f1"] for k in kinds], .34, label=split.upper())
        axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set(xticks=x, xticklabels=[LOSS_LABELS[k] for k in kinds], title="Final three-class macro-F1", ylabel="Macro-F1 (0–1)")
    width = .8 / len(kinds)
    for offset, kind in enumerate(kinds):
        bars = axes[1].bar(np.arange(3) + (offset-(len(kinds)-1)/2)*width,
                          [results[kind]["metrics"]["test"][f"{label}_f1"] for label in ("lof", "gof", "neutral")],
                          width, color=COLORS[kind], label=LOSS_LABELS[kind])
        axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axes[1].set(xticks=range(3), xticklabels=["LOF", "GOF", "NEUTRAL"], title="Test per-class F1", ylabel="Class F1 (0–1)")
    for ax in axes:
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.suptitle("Cascade objective comparison — hinge and squared hinge", fontsize=16)
    artifacts["f1"] = str(root / "loss_comparison_f1.png")
    save_figure(fig, artifacts["f1"], vars(cfg), description=CAPTIONS["f1"])
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, stage in enumerate(("effect", "direction")):
        for kind in kinds:
            records = [row for row in histories[kind] if row["stage"] == stage]
            epochs = [row["epoch"] for row in records]
            for row, metric in enumerate(("train_loss", "val_macro_f1")):
                axes[row, col].plot(epochs, [r[metric] for r in records], color=COLORS[kind], label=LOSS_LABELS[kind])
        axes[0, col].set(title=f"{stage.title()} training objective", ylabel="Train loss — objective-specific scale")
        axes[1, col].set(title="Effect / all validation rows" if stage == "effect" else "Direction / true-effect validation rows",
                         ylabel="Validation binary macro-F1", ylim=(0, 1.03))
        for ax in axes[:, col]:
            ax.set_xlabel("Epoch within stage")
            ax.legend()
            ax.grid(alpha=.2)
    artifacts["training"] = str(root / "loss_training_curves.png")
    save_figure(fig, artifacts["training"], vars(cfg), description=CAPTIONS["training"])
    plt.close(fig)

    artifacts["prediction_counts"] = str(root / "test_prediction_counts.png")
    plot_prediction_counts({LOSS_LABELS[k]: results[k]["metrics"]["test"] for k in kinds},
                           Path(artifacts["prediction_counts"]), title="Test prediction counts — cascade loss comparison", config=vars(cfg))
    artifacts.update(write_outcome_reports(_outcome_result(results, cfg.family), root))
    for kind, result in results.items():
        details[kind] = write_outcome_reports(_outcome_result({kind: result}, cfg.family), root / kind)
    rows = score_rows(results)
    artifacts["scores"] = str(root / "comparison_summary.csv")
    with Path(artifacts["scores"]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Cascade loss comparison", "", CAPTIONS["f1"], "",
            "Both stages are trained independently. t=2*y-1 converts binary targets to signed labels. "
            "Hinge=max(0,m-t*z); squared hinge=max(0,m-t*z)^2. "
            "z_effect=scale*(abs(delta_effect)-radius); z_direction=scale*delta_direction. "
            "The margin is in scaled decision-logit space, not raw delta or physical-distance space. "
            "Scorer, radius and scale are trainable under either loss. "
            "Stage-specific inverse-frequency class weights use TRAIN counts only, or are both disabled. "
            "NEUTRAL never supervises direction. Checkpoints and thresholds are selected on validation only. "
            "Fixed-0.5 learned_radius inference is available; validation threshold search is a separate explicit strategy. "
            "Sigmoid values are uncalibrated decision scores, not biological effect strengths. "
            "No joint NLL or argmax decision was introduced. Same seed/epoch cap does not guarantee equal early-stopping epochs or compute.", "",
            f"Margin: {cfg.cascade_margin}; weighting: {'train inverse-frequency' if cfg.cascade_balance_classes else 'unweighted'}; "
            f"Effect strategy: {cfg.effect_threshold_strategy}; seed: {cfg.seed}.", "",
            "| Loss | Partition | Macro-F1 | LOF F1 | GOF F1 | NEUTRAL F1 | Accuracy | Effect threshold |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['description']} | {row['split']} | {row['macro_f1']:.4f} | {row['lof_f1']:.4f} | "
                    f"{row['gof_f1']:.4f} | {row['neutral_f1']:.4f} | {row['accuracy']:.4f} | {row['effect_threshold']:.2f} |")
    for key in ("f1", "training", "test_outcomes", "test_class_recall", "prediction_counts"):
        text.extend(["", f"![{key}]({Path(artifacts[key]).name})"])
    artifacts["report"] = str(root / "comparison_report.md")
    Path(artifacts["report"]).write_text("\n".join(text) + "\n", encoding="utf-8")
    return artifacts, details


def start_wandb(cfg, kinds, provenance):
    if not cfg.wandb_enabled:
        return None, None
    import wandb
    tags, meta = cfg.wandb_metadata()
    run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.wandb_run_name,
                     mode=getattr(cfg, "wandb_mode", "online"), group="cascade-loss-comparison",
                     job_type="hinge-vs-squared-hinge", tags=[*tags, *kinds, "loss-comparison"],
                     config={**meta, "experiment": FORMAT, "losses": kinds, "controlled_variable": "loss function",
                             "training_config": vars(cfg), "provenance": provenance,
                             "same_prepared_inputs": True, "test_used_for_selection": False,
                             "decision_rule": "hard_cascade", "margin_space": "scaled_signed_logit"})
    for kind in kinds:
        for stage in ("effect", "direction"):
            prefix = f"{SECTIONS[kind]}/Training/{stage}"
            run.define_metric(f"{prefix}/epoch")
            run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
    return run, wandb


def log_results(run, wandb, results, artifacts, details):
    if run is None:
        return
    rows = score_rows(results)
    columns = list(rows[0])
    run.log({f"{SECTION}/01_Scores": wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in rows]),
             f"{SECTION}/02_Final_F1": wandb.Image(artifacts["f1"], caption=CAPTIONS["f1"]),
             f"{SECTION}/03_Training": wandb.Image(artifacts["training"], caption=CAPTIONS["training"]),
             f"{SECTION}/00_Method": wandb.Html("<pre>" + html.escape(Path(artifacts["report"]).read_text()) + "</pre>", inject=False)})
    for kind, result in results.items():
        selected = [r for r in rows if r["loss"] == kind]
        run.log({f"{SECTIONS[kind]}/Results/Scores": wandb.Table(columns=columns, data=[[r[c] for c in columns] for r in selected]),
                 f"{SECTIONS[kind]}/Results/Test_Outcomes": wandb.Image(details[kind]["test_outcomes"], caption=CAPTIONS["outcomes"]),
                 f"{SECTIONS[kind]}/Results/Test_Class_Recovery": wandb.Image(details[kind]["test_class_recall"], caption=CAPTIONS["outcomes"]),
                 f"{SECTIONS[kind]}/Results/Validation_Outcomes": wandb.Image(details[kind]["val_outcomes"], caption=CAPTIONS["outcomes"])})
        for split, metric in result["metrics"].items():
            for key in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1", "effect_missed_n", "neutral_routed_n"):
                run.summary[f"{SECTIONS[kind]}/{split}/{key}"] = metric[key]
        for stage, selection in result["selection"].items():
            run.summary[f"{SECTIONS[kind]}/{stage}/best_epoch"] = selection["epoch"]
            run.summary[f"{SECTIONS[kind]}/{stage}/class_weights_0_1"] = selection["class_weights_0_1"]
        run.summary[f"{SECTIONS[kind]}/effect_threshold"] = result["effect_threshold"]


def _package(run, wandb, root):
    artifact = wandb.Artifact("phase3-cascade-loss-comparison", type="evaluation")
    for file in root.rglob("*"):
        if file.is_file():
            artifact.add_file(str(file), name=str(file.relative_to(root)))
    run.log_artifact(artifact)


def run_experiment(cfg, *, cache_dir=None, out_dir="runs/phase3_cascade_losses", include_bce=False, quiet=False):
    cfg = copy.deepcopy(cfg)
    cfg.architecture = "cascade"
    kinds = (["bce"] if include_bce else []) + ["hinge", "squared_hinge"]
    tensors, views, go_vocab, device = prepare_run(cfg, cache_dir=cache_dir, quiet=quiet)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"losses_{cfg.family}_{cfg.embedding_type}_{cfg.input_mode}_", dir=Path(out_dir).resolve()))
    identities = ["split", "protein_id", "snp", "matched_go_id", "label"]
    provenance = {"filtered_rows_sha256": hashlib.sha256(tensors.df[identities].to_csv(index=False).encode()).hexdigest(),
                  "split_rows": {s: views[s]["n"] for s in views}, "go_artifact_sha256": tensors.go_artifact_sha256}
    run, wandb = start_wandb(cfg, kinds, provenance)
    results, histories = {}, {}
    try:
        for kind in kinds:
            run_cfg = copy.deepcopy(cfg)
            run_cfg.cascade_loss = kind
            result = fit_cascade(run_cfg, tensors, views, go_vocab, device, root / kind,
                                 run=run, quiet=quiet, log_prefix=f"{SECTIONS[kind]}/Training")
            results[kind] = result
            histories[kind] = json.loads((Path(result["checkpoint"]).parent / "history.json").read_text())
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        artifacts, details = generate_report(results, histories, root, cfg)
        manifest = {"format": FORMAT, "config": vars(cfg), "provenance": provenance, "runs": results,
                    "controlled_variable": "loss function", "test_used_for_selection": False,
                    "artifacts": {k: str(Path(p).relative_to(root)) for k, p in artifacts.items()},
                    "method_artifacts": {kind: {k: str(Path(p).relative_to(root)) for k, p in paths.items()} for kind, paths in details.items()},
                    "wandb_url": run.url if run else None}
        source = root / "loss_comparison.json"
        source.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if run is not None:
            log_results(run, wandb, results, artifacts, details)
            _package(run, wandb, root)
            run.summary[f"{SECTION}/output_dir"] = str(root)
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise
    if run is not None:
        run.finish()
    if not quiet:
        for kind, result in results.items():
            print(f"[TEST] {LOSS_LABELS[kind]} macro-F1={result['metrics']['test']['macro_f1']:.4f}")
        print(f"[DONE] loss comparison -> {source}")
        if run:
            print(f"  wandb: {manifest['wandb_url']}")
    return {**manifest, "manifest_path": str(source)}


def upload_saved(path, *, project=None, entity=None, name=None, mode="online"):
    source = Path(path).expanduser().resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Expected loss_comparison.json from compare_cascade_losses")
    def resolve(paths):
        resolved = {}
        for key, relative in paths.items():
            candidate = (source.parent / relative).resolve()
            if not candidate.is_relative_to(source.parent) or not candidate.is_file():
                raise ValueError(f"Missing or external comparison artifact: {relative}")
            resolved[key] = str(candidate)
        return resolved
    artifacts = resolve(manifest["artifacts"])
    details = {kind: resolve(files) for kind, files in manifest["method_artifacts"].items()}
    cfg = VariantDecoderConfig()
    for key, value in manifest["config"].items():
        setattr(cfg, key, value)
    cfg.wandb_enabled, cfg.wandb_mode = True, mode
    for key, value in (("wandb_project", project), ("wandb_entity", entity), ("wandb_run_name", name)):
        if value is not None:
            setattr(cfg, key, value)
    run, wandb = start_wandb(cfg, list(manifest["runs"]), manifest["provenance"])
    try:
        log_results(run, wandb, manifest["runs"], artifacts, details)
        manifest["wandb_url"] = run.url
        source.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        _package(run, wandb, source.parent)
    except Exception:
        run.finish(exit_code=1)
        raise
    run.finish()
    return manifest["wandb_url"]


def main():
    defaults = VariantDecoderConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--embedding", choices=EMBEDDING_TYPES)
    parser.add_argument("--input_mode", choices=INPUT_MODES, default="both")
    for key in ("global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "cache_dir", "cuda"):
        parser.add_argument(f"--{key}")
    for key in ("dataset_dir", "sequences_json", "go_embeddings"):
        parser.add_argument(f"--{key}", default=getattr(defaults, key))
    for key in ("allow_go_proxies", "drop_contradictory", "unweighted_cascade", "no_wandb", "include_bce"):
        parser.add_argument(f"--{key}", action="store_true")
    parser.add_argument("--hidden_dims", type=int, nargs="+")
    for key, dtype in (("dropout", float), ("lr", float), ("batch_size", int), ("epochs", int), ("early_stop_patience", int), ("seed", int)):
        parser.add_argument(f"--{key}", type=dtype)
    parser.add_argument("--cascade_margin", type=float, default=defaults.cascade_margin)
    parser.add_argument("--effect_threshold_strategy", choices=EFFECT_THRESHOLD_STRATEGIES, default=defaults.effect_threshold_strategy)
    parser.add_argument("--out_dir", default="runs/phase3_cascade_losses")
    for key in ("wandb_project", "wandb_entity", "wandb_run_name"):
        parser.add_argument(f"--{key}")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--upload_existing", metavar="LOSS_COMPARISON_JSON")
    args = parser.parse_args()
    if args.upload_existing:
        if args.no_wandb:
            parser.error("--upload_existing cannot be combined with --no_wandb")
        print("[DONE] saved loss comparison:", upload_saved(args.upload_existing, project=args.wandb_project,
              entity=args.wandb_entity, name=args.wandb_run_name, mode=args.wandb_mode))
        return
    if not all((args.family, args.embedding, args.global_h5, args.local_h5, args.dataset_dir, args.sequences_json, args.go_embeddings)):
        parser.error("Set family, embedding, H5 paths and nonempty dataset/sequence/GO paths; check exports")
    if args.embedding in SECONDARY_REQUIRED and not (args.global_h5_secondary and args.local_h5_secondary):
        parser.error("Concatenated encoders require both secondary H5 paths")
    cfg = VariantDecoderConfig(args.family, args.embedding, args.input_mode)
    for key in ("global_h5", "local_h5", "global_h5_secondary", "local_h5_secondary", "dataset_dir", "sequences_json",
                "go_embeddings", "allow_go_proxies", "drop_contradictory", "hidden_dims", "dropout", "lr", "batch_size",
                "epochs", "early_stop_patience", "seed", "cuda", "cascade_margin", "effect_threshold_strategy", "wandb_project", "wandb_entity"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    cfg.cascade_balance_classes = not args.unweighted_cascade
    cfg.wandb_enabled, cfg.wandb_mode = not args.no_wandb, args.wandb_mode
    cfg.wandb_run_name = args.wandb_run_name or f"phase3-hinge-vs-squared-hinge-{args.family}-{args.embedding}-{args.input_mode}"
    run_experiment(cfg, cache_dir=args.cache_dir, out_dir=args.out_dir, include_bce=args.include_bce)


if __name__ == "__main__":
    main()
