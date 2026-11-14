"""Matched Concat/cascade performance and mutation-sensitivity reports."""
from pathlib import Path
import html

import numpy as np
import pandas as pd

from phase_3_variant_decoder.outcome_tables import _table, plot_prediction_counts
from phase_3_variant_decoder.presentation_terms import save_figure

SECTION = "00_Concat_vs_Cascade"
METHODS = ("concat", "cascade")
NAMES = {"concat": "Concat (3-class)", "cascade": "Cascade (saved)"}
COLORS = {"concat": "#2563eb", "cascade": "#c2410c"}


def fmt(value, percentage=False):
    return "N/A" if value is None else f"{value:.1%}" if percentage else f"{value:.4f}"


def score_rows(results):
    return [dict(family=family.upper(), model=NAMES[method], split=split, parameters=arm["parameter_count"],
                 **{k: v for k, v in metric.items() if k != "confusion"})
            for family, result in results.items() for method, arm in result["arms"].items()
            for split, metric in arm["metrics"].items()]


def mutation_rows(results):
    rows = []
    for family, result in results.items():
        for method, arm in result["arms"].items():
            for cohort, key in (("Covered original pairs (primary)", "pairs"), ("Clean pairs (secondary)", "clean_pairs")):
                rows.extend(dict(family=family.upper(), model=NAMES[method], cohort=cohort, **r)
                            for r in arm["mutation"][key]["ranking"])
    return rows


def generate_reports(results, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    artifacts = {}
    scores, mutations = score_rows(results), mutation_rows(results)
    pd.DataFrame(scores).to_csv(root / "scores.csv", index=False)
    pd.DataFrame(mutations).to_csv(root / "mutation_scores.csv", index=False)
    visible = [r for r in scores if r["split"] != "train"]
    path = root / "comparison_table.png"
    _table(["Family", "Model", "Partition", "Samples", "Macro-F1", "Accuracy", "LOF F1", "GOF F1", "NEUTRAL F1"],
        [[r["family"], r["model"], r["split"].upper(), f"{r['n']:,}", *[fmt(r[k]) for k in
          ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1")]] for r in visible], path,
        "Concat vs saved cascade — same-example performance",
        "Her family içinde aynı örnekler, aynı encoder girdileri ve aynı GO vektörleri kullanılır. Concat argmax, cascade kayıtlı hard-routing kuralıyla değerlendirilir. Testte threshold veya checkpoint seçilmez.",
        widths=[.08, .17, .11, .10, .10, .10, .10, .10, .14])
    artifacts["00_Comparison_Table"] = str(path)
    report = ["# Concat versus saved cascade", "",
        "Concat is the only new decoder architecture. Existing checkpoints are reused unless training was explicitly requested.",
        "Final Concat classification: three-class softmax argmax. Cascade: original saved effect threshold and direction=0.5.",
        "Ranking index for both: GOF class decision score minus LOF class decision score.",
        "Cascade uses e*(1-2*d); Concat uses softmax_GOF-softmax_LOF. Scores are uncalibrated decision indices, not activity/severity.",
        "Correct order=1; tied indices=0.5; reversed order=0. Reference for constant scores is 0.5.",
        "Raw-train EASY challenge is frozen before filtering; original covered HARD row pairs are primary; clean pairs supplementary.",
        "Same-position pairs hold mutation position fixed and require different mutant amino acids.",
        "Checkpoints can differ in capacity, optimization and loss; this is a model comparison, not a loss-only ablation.",
        "Current data identities/filters and aligned GO vectors are checked. Legacy cascade training-data hashes are not available for independent provenance verification.", ""]
    for family, result in results.items():
        cfg, arms = result["config"], result["arms"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for j, method in enumerate(METHODS):
            for ax, values in ((axes[0], [arms[method]["metrics"][s]["macro_f1"] for s in ("train", "val", "test")]),
                               (axes[1], [arms[method]["metrics"]["test"][f"{s}_f1"] for s in ("lof", "gof", "neutral")])):
                bars = ax.bar(np.arange(3)+(j-.5)*.34, values, .34, color=COLORS[method], label=NAMES[method])
                ax.bar_label(bars, fmt="%.3f", padding=3)
        axes[0].set(xticks=range(3), xticklabels=["Train", "Validation", "Test"], title="Three-class macro-F1")
        axes[1].set(xticks=range(3), xticklabels=["LOF", "GOF", "NEUTRAL"], title="Test per-class F1")
        for ax in axes:
            ax.set(ylabel="F1 (0–1)", ylim=(0, 1.13))
            ax.grid(axis="y", alpha=.2)
            ax.legend()
        fig.suptitle(f"{family.upper()} — Concat vs saved cascade", fontsize=16)
        key = f"01_{family.upper()}_F1"
        artifacts[key] = str(root / f"{family}_f1.png")
        save_figure(fig, artifacts[key], cfg, description="İki model kendi önceden seçilmiş checkpoint'iyle ölçülür. Cascade threshold'u değiştirilmez. Train ve validation değerleri mevcut checkpoint'lerin yeniden değerlendirmesidir; yeni checkpoint seçimi yapılmaz.")
        plt.close(fig)
        for split in ("val", "test"):
            key = f"05_{family.upper()}_{split.upper()}_Prediction_Counts"
            artifacts[key] = str(root / f"{family}_{split}_prediction_counts.png")
            plot_prediction_counts({NAMES[m]: arms[m]["metrics"][split] for m in METHODS}, Path(artifacts[key]),
                title=f"{family.upper()} / {split.upper()} — prediction counts", config=cfg)
        audit = result["cohort_audit"]
        if family == "easy":
            challenge = [{**r, "model": NAMES[m]} for m in METHODS for r in arms[m]["mutation"]["easy"]["classification"]
                         if r["cohort"] in ("Anti-majority", "Anti-majority / clean novel variants")]
            pd.DataFrame(challenge).to_csv(root / "easy_challenge.csv", index=False)
            key = "02_EASY_Challenge"
            artifacts[key] = str(root / "easy_challenge.png")
            _table(["Cohort", "Model", "Samples", "Correct", "Accuracy", "Macro-F1"],
                [[r["cohort"].replace(" / ", "\n"), r["model"], r["n"], r["correct_n"], fmt(r["accuracy"], True), fmt(r["macro_f1"])] for r in challenge],
                Path(artifacts[key]), "EASY — variants against the memorized protein–GO majority",
                f"Ham train çoğunluğuna ters etiketli sabit challenge: raw={audit['easy']['raw_anti_majority']}, covered={audit['easy']['raw_cohort_covered']}. İki model aynı satırlarda ölçülür. Clean novel varyantlar ek analizdir; ana challenge'ın yerine geçmez.",
                widths=[.31, .22, .11, .11, .13, .12])
        if family == "hard":
            fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
            counts = []
            for j, method in enumerate(METHODS):
                rank = arms[method]["mutation"]["pairs"]["ranking"][0]
                counts.append(f"{NAMES[method]}: {rank['n_pairs']:,} pairs / {rank['n_groups']} groups")
                values = [rank["pair_accuracy"], rank["group_macro_accuracy"]]
                bars = ax.bar(np.arange(2)+(j-.5)*.34, [np.nan if x is None else x for x in values], .34,
                              color=COLORS[method], label=NAMES[method])
                ax.bar_label(bars, fmt="%.3f", padding=3)
            ax.axhline(.5, color="gray", linestyle="--", label="Constant-score reference (ties = 0.5)")
            ax.set(xticks=range(2), xticklabels=["Pairwise ranking accuracy", "Group macro accuracy"],
                   ylabel="Accuracy (0–1)", ylim=(0, 1.13), title="HARD — same protein and GO, different variant effects")
            ax.legend(fontsize=9)
            key = "02_HARD_Mutation_Sensitivity"
            artifacts[key] = str(root / "hard_mutation_sensitivity.png")
            save_figure(fig, artifacts[key], cfg, description="Protein ve GO sabitken LOF < NEUTRAL < GOF sırası karar indeksi üzerinden ölçülür. Doğru sıra=1, tie=0.5, ters sıra=0. Group macro her protein–GO grubuna eşit ağırlık verir. Bunlar macro-F1 değildir. " + "; ".join(counts))
            plt.close(fig)
            primary = [r for r in mutations if r["family"] == "HARD" and r["cohort"].endswith("(primary)")]
            key = "03_HARD_Pair_Details"
            artifacts[key] = str(root / "hard_pair_details.png")
            _table(["Model", "Pair comparison", "Pairs", "Groups", "Pair accuracy", "Group macro", "Ties"],
                [[r["model"], r["comparison"], r["n_pairs"], r["n_groups"], fmt(r["pair_accuracy"], True), fmt(r["group_macro_accuracy"], True), fmt(r["tie_rate"], True)] for r in primary],
                Path(artifacts[key]), "HARD — covered original pair results",
                "Primary kapsam hocanın ham satır çiftlerini korur; girdi filtrelerinden geçen çiftler değerlendirilir. Same-position satırı protein, GO ve konumu sabit tutar. Çiftler bağımsız gözlemler değildir.",
                widths=[.18, .28, .10, .10, .12, .12, .10])
            selected = [r for r in mutations if r["family"] == "HARD" and r["comparison"] == "All different-label pairs"]
            key = "04_HARD_Coverage_and_Clean"
            artifacts[key] = str(root / "hard_coverage_clean.png")
            _table(["Cohort", "Model", "Pairs", "Groups", "Pair accuracy", "Group macro"],
                [[r["cohort"].replace(" (", "\n("), r["model"], r["n_pairs"], r["n_groups"], fmt(r["pair_accuracy"], True), fmt(r["group_macro_accuracy"], True)] for r in selected],
                Path(artifacts[key]), "HARD — original-row protocol and clean sensitivity analysis",
                f"Ham protokol: {audit['waterfall'][0]['different_label_groups']} grup / {audit['waterfall'][0]['different_label_row_pairs']} çift. Covered ana sonuçtur; clean tekrarları/çelişkili girdileri çıkarır. İki model her kapsamda aynı çiftleri kullanır.",
                widths=[.32, .22, .10, .10, .13, .13])
        report.extend([f"## {family.upper()}", "", f"Counts: {result['provenance']['split_rows']}",
                       f"Cohort audit: {audit}", ""])
        for method in METHODS:
            arm = arms[method]
            report.extend([f"{NAMES[method]}: {arm['parameter_count']} parameters; {arm['decision']}; checkpoint={arm['checkpoint']}",
                           f"Full test: {arm['metrics']['test']}", f"Primary ranking: {arm['mutation']['pairs']['ranking'][0]}", ""])
    (root / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return artifacts


def log_results(run, wandb, report, root):
    payload = {f"{SECTION}/00_Scores": wandb.Table(dataframe=pd.DataFrame(score_rows(report["families"]))),
        f"{SECTION}/00_Mutation_Scores": wandb.Table(dataframe=pd.DataFrame(mutation_rows(report["families"]))),
        f"{SECTION}/00_Methods": wandb.Table(columns=["Model / metric", "Definition"], data=[
            ["Concat", "WT/VAR shared projection + GO -> three logits -> softmax argmax"],
            ["Cascade", "Saved effect/direction models and original hard-routing thresholds"],
            ["EASY challenge", "Test labels differing from RAW-train protein–GO majority; frozen before filtering"],
            ["HARD ranking", "Same protein and GO; different-label original row pairs; s=GOF score-LOF score"],
            ["Ranking accuracy", "Correct=1, tie=0.5, incorrect=0. Constant score reference=0.5; not macro-F1"],
            ["Training", str(report["trained_in_this_experiment"])]]),
        f"{SECTION}/06_Protocol": wandb.Html("<pre>" + html.escape((Path(root)/"comparison_report.md").read_text()) + "</pre>", inject=False)}
    for key, relative in report["artifacts"].items():
        payload[f"{SECTION}/{key}"] = wandb.Image(str(Path(root)/relative))
    run.log(payload)
    for family, result in report["families"].items():
        for method, arm in result["arms"].items():
            for split, metric in arm["metrics"].items():
                for key in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1"):
                    run.summary[f"{SECTION}/{family}_{method}_{split}_{key}"] = metric[key]
            rank = arm["mutation"]["pairs"]["ranking"][0]
            for key in ("pair_accuracy", "group_macro_accuracy", "n_pairs", "n_groups"):
                run.summary[f"{SECTION}/{family}_{method}_{key}"] = rank[key]
            if "easy" in arm["mutation"]:
                challenge = next(r for r in arm["mutation"]["easy"]["classification"] if r["cohort"] == "Anti-majority")
                for key in ("accuracy", "macro_f1", "n", "correct_n"):
                    run.summary[f"{SECTION}/{family}_{method}_challenge_{key}"] = challenge[key]


def create_workspace(run, artifact_keys):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    panels = [wr.MediaBrowser(title=key.replace("_", " "), media_keys=[f"{SECTION}/{key}"]) for key in sorted(artifact_keys)]
    return ws.Workspace(entity=run.entity, project=run.project, name=f"Concat vs Cascade - {run.id}",
        sections=[ws.Section(name=SECTION, panels=panels, is_open=True)],
        runset_settings=ws.RunsetSettings(filters=f"Config('concat_cascade_comparison_id') == '{run.id}'"),
        auto_generate_panels=False).save().url
