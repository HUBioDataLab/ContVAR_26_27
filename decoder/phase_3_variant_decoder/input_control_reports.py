"""Readable local and W&B reports for the isolated input-control comparison."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from phase_3_variant_decoder.input_controls import METHODS, LEARNED, INPUTS, EXPLANATIONS_TR
from phase_3_variant_decoder.outcome_tables import plot_prediction_counts, write_outcome_reports
from phase_3_variant_decoder.presentation_terms import configuration_metadata, configuration_text, save_figure

SECTION = "Input_Control_Comparison"
SECTIONS = {"go_lookup": "Input_Control_GO_Lookup", "go_only": "Input_Control_GO_Only_MLP",
            "wt_go": "Input_Control_WT_GO_MLP", "mutation_go": "Input_Control_Mutation_GO_MLP", "pair": "Input_Control_Pair_MLP"}
COLORS = ["#64748b", "#d97706", "#0891b2", "#059669", "#7c3aed"]
CAPTIONS = {
    "f1": "Aynı örneklerde input karşılaştırması: GO-only yalnız GO'yu; WT + GO, WT bağlamını; Mutation + GO, basit mutasyon bilgilerini; Pair-MLP mutant embeddingini ve farkları da kullanır. Macro-F1, LOF/GOF/NEUTRAL F1 ortalamasıdır; Accuracy değildir. Checkpoint yalnız validation ile seçilir. Tek seed sonucu kesin üstünlük göstermez.",
    "training": "Dört MLP aynı cross-entropy objective ile eğitilir. Train loss eğitim hatası, Validation Macro-F1 üç sınıflı doğrulama skorudur. Dikey çizgiler seçilen checkpoint epoch'larını gösterir. GO lookup eğitilmediğinden eğrisi yoktur.",
    "outcomes": "Correct / Actual, o sınıftaki doğru tahmin / toplam örnek sayısıdır. F1 veya satır yüzdesi değildir. Tüm yöntemler aynı filtrelenmiş örneklerde ölçülür.",
}


def score_rows(results, config):
    rows = []
    context = configuration_metadata(config)
    for method, result in results.items():
        for split, metric in result["metrics"].items():
            rows.append(dict(family=context["family"], encoder=context["encoder_label"],
                             input_mode=context["input_label"], seed=context["seed"],
                             method=METHODS[method], split=split, n=metric["n"],
                             parameter_count=result["parameter_count"], best_epoch=result["best_epoch"],
                             **{k: metric[k] for k in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1")},
                             delta_macro_f1_vs_lookup=metric["macro_f1"] - results["go_lookup"]["metrics"][split]["macro_f1"],
                             delta_macro_f1_vs_wt_go=metric["macro_f1"] - results["wt_go"]["metrics"][split]["macro_f1"]))
    return rows


def generate_reports(results, histories, root, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    paths = {}
    names = list(METHODS)
    labels = ["GO-only\nlookup", "GO-only\nMLP", "WT + GO\nMLP", "Mutation + GO\nMLP", "Pair-MLP"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for j, split in enumerate(("val", "test")):
        bars = axes[0].bar(np.arange(len(names)) + (j-.5)*.36,
                          [results[m]["metrics"][split]["macro_f1"] for m in names], .36, label=split.upper())
        axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set(xticks=range(len(names)), xticklabels=labels, ylabel="Macro-F1 (0–1)", title="Final three-class macro-F1")
    for j, method in enumerate(names):
        bars = axes[1].bar(np.arange(3) + (j-(len(names)-1)/2)*.16,
                          [results[method]["metrics"]["test"][f"{c}_f1"] for c in ("lof", "gof", "neutral")],
                          .16, label=METHODS[method], color=COLORS[j])
        axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axes[1].set(xticks=range(3), xticklabels=["LOF", "GOF", "NEUTRAL"], ylabel="Class F1 (0–1)", title="Test per-class F1")
    for ax in axes:
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Input controls — contribution of protein and mutant information", fontsize=16)
    paths["f1"] = root / "input_controls_f1.png"
    save_figure(fig, paths["f1"], config, description=CAPTIONS["f1"])
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for j, method in enumerate(LEARNED, 1):
        records = histories[method]
        for ax, field in zip(axes, ("train_loss", "val_macro_f1")):
            ax.plot([r["epoch"] for r in records], [r[field] for r in records], label=METHODS[method], color=COLORS[j])
            ax.axvline(results[method]["best_epoch"], color=COLORS[j], linestyle=":", alpha=.6)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=.2)
            ax.legend()
    axes[0].set(title="Training objective", ylabel="Mean weighted CE" if config["input_balance_classes"] else "Mean CE")
    axes[1].set(title="Validation selection metric", ylabel="Three-class macro-F1", ylim=(0, 1.05))
    paths["training"] = root / "input_controls_training.png"
    save_figure(fig, paths["training"], config, description=CAPTIONS["training"])
    plt.close(fig)

    outcome_result = {"data": {"family": config["family"]}, "results": {
        f"{split}/{method}": {**metric, "method": method, "description": METHODS[method],
                              "split": split, "mean_confusion": metric["confusion"], "repeats": 1}
        for method, result in results.items() for split, metric in result["metrics"].items()}}
    outcome_result.update(configuration=config, class_recall_description=CAPTIONS["outcomes"],
                          outcome_description="Correct doğru, Incorrect yanlış tahmin sayısıdır; toplamları Total'a eşittir. Accuracy doğru tahmin oranıdır. Macro-F1 üç sınıfa eşit ağırlık verir; başarı yüzdesi değildir.")
    paths.update(write_outcome_reports(outcome_result, root))
    paths["test_prediction_counts"] = root / "test_prediction_counts.png"
    plot_prediction_counts({METHODS[m]: results[m]["metrics"]["test"] for m in names},
                           paths["test_prediction_counts"], title="Test prediction counts — input controls", config=config)
    rows = score_rows(results, config)
    paths["scores_csv"] = root / "input_control_scores.csv"
    with paths["scores_csv"].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# Input-control comparison", "", configuration_text(config), "",
            "| Model | Input | Açıklama |", "|---|---|---|"]
    text.extend(f"| {METHODS[m]} | {INPUTS[m]} | {EXPLANATIONS_TR[m]} |" for m in names)
    text += ["", "| Model | Split | Macro-F1 | LOF F1 | GOF F1 | NEUTRAL F1 | vs lookup | vs WT + GO |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    text.extend(f"| {r['method']} | {r['split']} | {r['macro_f1']:.4f} | {r['lof_f1']:.4f} | {r['gof_f1']:.4f} | {r['neutral_f1']:.4f} | {r['delta_macro_f1_vs_lookup']:+.4f} | {r['delta_macro_f1_vs_wt_go']:+.4f} |" for r in rows)
    text += ["", "## Nasıl yorumlanır?", "",
             "GO-only MLP, yalnız GO embeddinginin ne kadar bilgi taşıdığını ölçer; GO lookup ile aynı yöntem değildir.",
             "WT + GO'nun GO-only MLP'ye göre kazancı WT bağlamının katkısını; Pair-MLP'nin WT + GO'ya göre kazancı mutant ve fark girdilerinin ek katkısını sınar.",
             "Pair-MLP lookup'ı geçse bile WT + GO'yu geçmiyorsa mutant bilgisinin ek faydası gösterilmiş olmaz.",
             "Tek seed veya küçük bir skor farkı biyolojik genelleme kanıtı değildir. Veri kaynaklı etiket ipuçları bu deneyle ortadan kalkmaz.",
             "", "## Protokol", "",
             "Encoder sabittir; dört MLP sıfırdan, aynı train satırları/seed/batch sırası/hidden widths/optimizer/epoch limiti ile eğitilir.",
             "Mutation + GO, WT/VAR aminoasitleri için ayrı one-hot vektörleri ve position/max_seq_len kullanır (20 aminoasit + unknown kategorisi). Protein embeddingi almaz.",
             "Input genişlikleri farklı olduğundan parameter count eşit değildir; CSV ve W&B tablosunda ayrıca verilir.",
             "Her modelin checkpoint'i validation üç sınıflı macro-F1 ile seçilir; eşitlikte ilk epoch korunur. Test seçimde kullanılmaz.",
             "Protein standardization yalnız unique train WT satırlarından, GO standardization yalnız unique train GO satırlarından hesaplanır. Aynı protein dönüşümü WT ve VAR'a uygulanır.",
             "Loss: mean(weight[label] * CrossEntropy(logits, label)); balanced seçildiyse weight[c] = N_train / (3 * N_train,c), aksi hâlde 1.",
             "Bu deneyde üç sınıflı argmax vardır; cascade veya validation threshold yoktur. Softmax çıktıları ayrıca kalibre edilmiş değildir.",
             "Local, mutasyon pozisyonundaki tek residue embeddingidir; sequence window veya 3D pocket değildir. WT + GO bu pozisyon bilgisini içerir.",
             "Delta = standardized VAR - standardized WT. İşareti tek başına LOF/GOF değildir; sınıf ilişkisini MLP öğrenir.",
             "Phase 0 -> Phase 1 etiketi repository pipeline tanımıdır; dış H5 dosyasının checkpoint geçmişi ayrıca doğrulanmış sayılmaz.", ""]
    text += ["## Train sınıf ağırlıkları", "", "| Class | Train rows | Loss weight |", "|---|---:|---:|"]
    text.extend(f"| {label} | {count} | {weight:.6f} |" for label, count, weight in
                zip(("LOF", "GOF", "NEUTRAL"), results["pair"]["class_counts"], results["pair"]["class_weights"]))
    paths["report"] = root / "input_control_report.md"
    paths["report"].write_text("\n".join(text), encoding="utf-8")
    paths["metadata"] = root / "report_metadata.json"
    paths["metadata"].write_text(json.dumps(configuration_metadata(config), indent=2) + "\n")
    return {key: str(path) for key, path in paths.items()}


def log_results(run, wandb, manifest, paths):
    rows = score_rows(manifest["runs"], manifest["config"])
    columns = list(rows[0])
    run.log({
        f"{SECTION}/00_Input_Definitions": wandb.Table(columns=["Model", "Inputs", "Kısa açıklama"],
            data=[[METHODS[m], INPUTS[m], EXPLANATIONS_TR[m]] for m in METHODS]),
        f"{SECTION}/01_Scores": wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in rows]),
        f"{SECTION}/02_Final_F1": wandb.Image(paths["f1"], caption=CAPTIONS["f1"]),
        f"{SECTION}/03_Training": wandb.Image(paths["training"], caption=CAPTIONS["training"]),
        f"{SECTION}/04_Test_Class_Recovery": wandb.Image(paths["test_class_recall"], caption=CAPTIONS["outcomes"]),
        f"{SECTION}/05_Test_Outcomes": wandb.Image(paths["test_outcomes"], caption=CAPTIONS["outcomes"]),
        f"{SECTION}/06_Test_Prediction_Counts": wandb.Image(paths["test_prediction_counts"], caption=CAPTIONS["outcomes"]),
        f"{SECTION}/07_Protocol": wandb.Html("<pre>" + html.escape(Path(paths["report"]).read_text()) + "</pre>", inject=False),
    })
    for method, result in manifest["runs"].items():
        selected = [row for row in rows if row["method"] == METHODS[method]]
        run.log({f"{SECTIONS[method]}/Scores": wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in selected])})
        for key in ("parameter_count", "best_epoch", "epochs_run"):
            run.summary[f"{SECTIONS[method]}/{key}"] = result[key]
        if method in LEARNED:
            run.summary[f"{SECTIONS[method]}/train_class_counts_LOF_GOF_NEUTRAL"] = result["class_counts"]
            run.summary[f"{SECTIONS[method]}/loss_class_weights_LOF_GOF_NEUTRAL"] = result["class_weights"]
        for row in selected:
            for key in ("macro_f1", "accuracy", "lof_f1", "gof_f1", "neutral_f1", "delta_macro_f1_vs_lookup", "delta_macro_f1_vs_wt_go"):
                run.summary[f"{SECTIONS[method]}/{row['split']}/{key}"] = row[key]
