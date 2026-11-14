"""Readable tables and a dedicated W&B section for mutation diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from phase_3_variant_decoder.presentation_terms import configuration_metadata, save_figure

SECTION = "00_Mutation_Sensitivity"
DEFINITIONS = [
    ("EASY / Anti-majority", "v2: grup ham train çoğunluğuna göre sabitlenir. Test etiketi buna ters olan, modelin işleyebildiği satırlarda ölçüm yapılır. Filtrelenmiş train kuralı ayrıca raporlanır."),
    ("HARD / Pairwise ranking", "v2: ham testte aynı protein–GO içindeki farklı etiketli satır çiftleri korunur. İki girdisi de mevcut çiftlerde LOF < NEUTRAL < GOF sırası kontrol edilir."),
    ("Ranking credit", "Doğru sıra = 1, eşit skor = 0.5, ters sıra = 0. Böylece sabit tahminin ranking accuracy değeri %50 olur; bu macro-F1 değildir."),
    ("Signed decision index", "s = effect_score × (1 − 2 × direction_score). Direction score arttıkça LOF yönüne gider. Bu ölçülmüş aktivite ya da kalibre olasılık değildir."),
    ("Pair accuracy / Group macro", "İlki her çifte, ikincisi her protein–GO grubuna eşit ağırlık verir. Çok variantlı gruplar ilk metriği baskılayabilir."),
    ("Same-position pairs", "Protein, GO ve mutasyon konumu sabit; mutant aminoasit farklı. Position etkisini ayıran daha dar bir kontrol."),
    ("Clean pairs", "Tekrarlar ve çelişkili variant–GO etiketleri çıkarılmış ek analiz. v1'de yalnız bu alt küme ana sonuçtu; v2'de covered original row pairs ana sonuçtur."),
    ("Coverage", "Ham gruptaki her satır/çift manifestte korunur. Eksik sekans veya konum sınırı nedeniyle tahmin üretilemeyenler coverage kaybı olarak sayılır; başarı paydasına eklenmez."),
    ("Interpretation", "Başarı, sabit protein/GO kuralından fazlasını gösterir; tek başına biyolojik mekanizmanın öğrenildiğini kanıtlamaz. Test sonuçları seçim/eğitim için kullanılmaz."),
]


def _fmt(value, percent=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def table_figure(title, columns, rows, path, config, note, widths=None):
    fig, ax = plt.subplots(figsize=(15, max(3.3, 1.3 + len(rows) * .56)))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, loc="left", pad=20, fontsize=17, weight="bold", color="#172e49")
    if not rows:
        ax.text(.5, .5, "No eligible observations", ha="center", va="center", fontsize=14)
    else:
        table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colWidths=widths)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.3)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#dae3ec")
            if r == 0:
                cell.set_facecolor("#1c3f60")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#edf4fa" if r % 2 else "#ffffff")
                cell.set_text_props(color="#172e49")
    save_figure(fig, path, config, description=note)
    plt.close(fig)


def generate_report(result: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    family = result["config"]["family"].upper()
    cfg = result["config"]
    artifacts = {}
    if "cohort_audit" in result:
        audit = result["cohort_audit"]
        path = output / "protocol_coverage.png"
        if family == "EASY":
            e = audit["easy"]
            rows = [["Raw-train challenge", e["raw_anti_majority"]],
                    ["Covered frozen challenge (primary)", e["raw_cohort_covered"]],
                    ["Filtered-train challenge (secondary)", e["filtered_recomputed_cohort"]]]
            table_figure("EASY | Original challenge and prediction coverage", ["Cohort definition", "Test rows"],
                rows, path, cfg,
                "Ham train kuralı sabit tutulur. Eksilen satırlar source_test_row_audit.csv dosyasında nedenleriyle kayıtlıdır. Filtrelenmiş train çoğunluğu farklı bir alt küme oluşturabilir.", [.77, .23])
        else:
            rows = [[r["step"], r["test_rows"], r["different_label_groups"], r["different_label_row_pairs"]]
                    for r in audit["waterfall"] if r["step"] == "Raw TSV" or r["removed_test_rows"]]
            table_figure("HARD | Why do the cohort counts change?",
                ["Filtering / supplementary cleaning", "Test rows", "Mixed-label\ngroups", "Cross-label\nrow pairs"], rows, path, cfg,
                "Ana değerlendirme yalnız model girdi filtrelerine kadardır. Deduplicate ve Remove conflicting satırları ayrı clean analize aittir. Ham çift listesi raw_pair_manifest.csv içinde korunur.", [.49, .17, .17, .17])
        artifacts[f"00_{family}_Protocol_Coverage"] = str(path)
    if "easy" in result:
        records = result["easy"]["classification"]
        path = output / "easy_challenge.png"
        rows = [[r["cohort"].replace(" / ", "\n"), r["model"].replace(" protein + GO lookup", "\nprotein + GO lookup"), r["n"], r["correct_n"],
                 _fmt(r["accuracy"], True), _fmt(r["macro_f1"])] for r in records]
        table_figure("EASY | Can the cascade beat a memorized protein–GO label?",
            ["Evaluation cohort", "Model / rule", "Samples", "Correct", "Accuracy", "Macro-F1"], rows, path, cfg,
            "Anti-majority: v2'de ham train çoğunluğuna ters olan sabit grup. Lookup'ın sıfır başarısı seçimin sonucudur. Filtered-train rule ayrı analizdir. Clean novel tekrar/çelişki ve train'deki aynı variantı çıkarır.",
            [.32, .22, .10, .10, .13, .13])
        artifacts["01_EASY_Challenge"] = str(path)
        pd.DataFrame(records).to_csv(output / "classification.csv", index=False)
    ranking = result["pairs"]["ranking"]
    path = output / "pair_ranking.png"
    rows = [[r["comparison"], r["n_pairs"], r["n_groups"], _fmt(r["pair_accuracy"], True),
             _fmt(r["group_macro_accuracy"], True), _fmt(r["tie_rate"], True)] for r in ranking]
    scope = "Covered original row pairs" if "clean_pairs" in result else "Clean distinct variants"
    table_figure(f"{family} | {scope}: within-protein–GO ranking",
        ["Signed-index comparison", "Pairs", "Groups", "Pair\naccuracy", "Group macro\naccuracy", "Ties"], rows, path, cfg,
        "Sabit skor referansı %50. v2 ana analiz ham satır tekrarlarını ve etiket çelişkilerini korur; clean karşılaştırması ayrıca verilir. Pair accuracy çiftleri, Group macro grupları eşit sayar. Bu metrik macro-F1 değildir.",
        [.34, .10, .10, .16, .17, .13])
    artifacts[f"02_{family}_Pair_Ranking"] = str(path)
    pd.DataFrame(ranking).to_csv(output / "ranking.csv", index=False)
    if "clean_pairs" in result:
        records = [{"cohort": scope, **ranking[0]},
                   {"cohort": "Clean distinct variants (secondary)", **result["clean_pairs"]["ranking"][0]}]
        path = output / "pair_cohort_comparison.png"
        rows = [[r["cohort"].replace(" (secondary)", "\n(secondary)"), r["n_pairs"], r["n_groups"],
                 _fmt(r["pair_accuracy"], True), _fmt(r["group_macro_accuracy"], True)] for r in records]
        table_figure(f"{family} | Original-row protocol versus clean sensitivity analysis",
            ["Evaluation cohort", "Pairs", "Groups", "Pair accuracy", "Group macro"], rows, path, cfg,
            "İki sonuç aynı checkpoint ve tahminleri kullanır. Yalnız örnek seçimi/ağırlığı değişir. Clean alt küme, ham verideki hocanın sayımının yerine geçmez.", [.44, .12, .12, .16, .16])
        artifacts[f"02_{family}_Cohort_Comparison"] = str(path)
    stages = result["pairs"]["stage_ranking"]
    path = output / "stage_ranking.png"
    rows = [[r["comparison"], r["n_pairs"], _fmt(r["pair_accuracy"], True),
             _fmt(r["group_macro_accuracy"], True), _fmt(r["tie_rate"], True)] for r in stages]
    table_figure(f"{family} | Which cascade stage separates the variants?",
        ["Stage-score comparison", "Pairs", "Pair accuracy", "Group macro accuracy", "Ties"], rows, path, cfg,
        "Effect: LOF/GOF skoru NEUTRAL'dan yüksek olmalı. Direction: LOF skoru GOF'tan yüksek olmalı. Direction head NEUTRAL ile eğitilmediği için o sınıfla doğrudan direction kıyası yapılmaz. Referans %50.",
        [.36, .12, .17, .22, .13])
    artifacts[f"03_{family}_Stage_Ranking"] = str(path)
    pd.DataFrame(stages).to_csv(output / "stage_ranking.csv", index=False)
    audit_rows = []
    for scope, values in result["data_audits"].items():
        a = values["pair_audit"]
        audit_rows.append([scope, values["test_rows"], values["anti_majority_rows"],
            a["different_label_groups"], a["different_label_row_pairs"],
            a["clean_different_label_groups"], a["clean_different_label_row_pairs"]])
    path = output / "coverage_audit.png"
    table_figure(f"{family} | Data coverage: original rows versus valid model inputs",
        ["Cohort", "Test rows", "Anti-majority\nrows", "Raw pair\ngroups", "Raw row\npairs", "Clean pair\ngroups", "Clean variant\npairs"],
        audit_rows, path, cfg,
        "Bu yardımcı tabloda Anti-majority her satırın kendi train kapsamıyla yeniden hesaplanır; iki satır sabit bir challenge değildir. v2 ana challenge ham train'de sabitlenir ve Protocol Coverage tablosunda ayrıca gösterilir.",
        [.22, .11, .14, .12, .12, .14, .15])
    if "cohort_audit" not in result:
        artifacts[f"04_{family}_Coverage"] = str(path)
    markdown = [f"# {family} mutation sensitivity", "", "## Configuration", "",
                json.dumps(configuration_metadata(cfg), indent=2, ensure_ascii=False), "",
                "## Interpretation", ""]
    markdown += [f"- **{name}:** {definition}" for name, definition in DEFINITIONS]
    markdown += ["", "Full-test cascade metrics: " + json.dumps(result["full_test"], ensure_ascii=False),
                 "", "Pair diagnostics: " + json.dumps(result["pairs"], ensure_ascii=False, indent=2), ""]
    if "easy" in result:
        markdown.append("EASY diagnostics: " + json.dumps(result["easy"], ensure_ascii=False, indent=2))
    (output / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return artifacts


def log_wandb(run, wandb, results: dict, root: Path):
    """A single slash keeps everything in ONE clearly named W&B section."""
    payload = {f"{SECTION}/00_Read_Me": wandb.Table(columns=["Term", "Aciklama"], data=DEFINITIONS)}
    configs = [configuration_metadata(r["config"]) for r in results.values()]
    payload[f"{SECTION}/00_Configurations"] = wandb.Table(dataframe=pd.DataFrame(configs))
    for family, result in results.items():
        for key, path in result["figures"].items():
            payload[f"{SECTION}/{key}"] = wandb.Image(path, caption="Açıklama, örnek sayıları ve metric tanımı görselin altında verilmiştir.")
        for name, records in (("Pair_Ranking", result["pairs"]["ranking"]),
                              ("Clean_Pair_Ranking", result.get("clean_pairs", {}).get("ranking", [])),
                              ("Stage_Ranking", result["pairs"]["stage_ranking"]),
                              ("Classification", result.get("easy", {}).get("classification", []))):
            if records:
                payload[f"{SECTION}/{family.upper()}_{name}_Table"] = wandb.Table(dataframe=pd.DataFrame(records))
        if "cohort_audit" in result:
            audit = result["cohort_audit"]
            payload[f"{SECTION}/{family.upper()}_Cohort_Audit_Table"] = wandb.Table(dataframe=pd.DataFrame(audit["waterfall"]))
            run.summary[f"{SECTION}/{family}_raw_pair_count"] = audit["waterfall"][0]["different_label_row_pairs"]
            if "easy" in audit:
                for name in ("raw_anti_majority", "raw_cohort_covered", "filtered_recomputed_cohort"):
                    run.summary[f"{SECTION}/easy_{name}"] = audit["easy"][name]
        if "clean_pairs" in result:
            for name, value in result["clean_pairs"]["ranking"][0].items():
                if name != "comparison" and value is not None:
                    run.summary[f"{SECTION}/{family}_clean_ranking_{name}"] = value
        for name, value in result["full_test"].items():
            if value is not None:
                run.summary[f"{SECTION}/{family}_full_test_{name}"] = value
        for name, value in result["pairs"]["ranking"][0].items():
            if name != "comparison" and value is not None:
                run.summary[f"{SECTION}/{family}_ranking_{name}"] = value
        if "easy" in result:
            record = next(r for r in result["easy"]["classification"] if r["cohort"] == "Anti-majority" and r["model"] == "Cascade")
            for name, value in record.items():
                if name not in ("cohort", "model") and value is not None:
                    run.summary[f"{SECTION}/easy_challenge_{name}"] = value
    run.log(payload)
    artifact = wandb.Artifact(f"mutation-sensitivity-{run.id}", type="evaluation")
    artifact.add_dir(str(root))
    run.log_artifact(artifact)


def create_workspace(run, results: dict):
    """New saved view: ordered section at top, without altering old workspaces.

    Requires optional wandb-workspaces. The regular W&B run uses the same keys.
    """
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    panels = [wr.MediaBrowser(title=key.replace("_", " "), media_keys=[f"{SECTION}/{key}"])
              for result in results.values() for key in result["figures"]]
    workspace = ws.Workspace(
        entity=run.entity, project=run.project,
        name=f"Mutation Sensitivity - {run.id}",
        sections=[ws.Section(name=SECTION, panels=panels, is_open=True)],
        runset_settings=ws.RunsetSettings(filters=f"Config('mutation_diagnostic_id') == '{run.id}'"),
        auto_generate_panels=False,
    )
    return workspace.save().url
