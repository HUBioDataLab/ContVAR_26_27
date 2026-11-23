"""Tables and figures from completed matched window experiments; no training."""
import argparse
import json
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report_suite(root):
    root = Path(root)
    data = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows, baseline_rows = [], []
    for name, r in data["models"].items():
        cohort, family, pool, architecture = name.split("/")
        t = r["test"]
        rows.append(dict(cohort=cohort, family=family, window=pool, architecture=architecture,
                         train_rows=r["n_train"], val_rows=r["n_val"], test_rows=t["n"],
                         best_epoch=r["best_epoch"], epochs=r["epochs_run"],
                         val_macro_f1=r["val"]["macro_f1"], test_macro_f1=t["macro_f1"],
                         accuracy=t["accuracy"], mcc=t["mcc"], lof_f1=t["lof"]["f1"],
                         gof_f1=t["gof"]["f1"], neutral_f1=t["neutral"]["f1"],
                         macro_auprc=t["macro_auprc"], reverse_mse=t["reverse_mse"],
                         both_heads_positive_rate=t.get("both_heads_positive_rate"),
                         parameters=r["parameter_count"], seconds=r["seconds"]))
    table = pd.DataFrame(rows)
    table.to_csv(root / "tum_sonuclar.csv", index=False, float_format="%.8g")
    for name, methods in data["baselines"].items():
        cohort, family = name.split("/")
        for kind, splits in methods.items():
            baseline_rows.append(dict(cohort=cohort, family=family, baseline=kind,
                                      val_macro_f1=splits["val"]["macro_f1"], test_macro_f1=splits["test"]["macro_f1"],
                                      test_rows=splits["test"]["n"]))
    base = pd.DataFrame(baseline_rows)
    base.to_csv(root / "baseline_sonuclari.csv", index=False)
    gains = []
    for (cohort, family, architecture), group in table.groupby(["cohort", "family", "architecture"], sort=False):
        indexed = group.set_index("window")
        if not {"none", "mean"}.issubset(indexed.index):
            continue
        n, m = indexed.loc["none"], indexed.loc["mean"]
        if (n.train_rows, n.val_rows, n.test_rows) != (m.train_rows, m.val_rows, m.test_rows):
            raise ValueError("Window off/on cohort sizes differ")
        paths = [root / cohort / family / pool / architecture / "test_predictions.tsv" for pool in ("none", "mean")]
        p0, p1 = [pd.read_csv(path, sep="\t") for path in paths]
        if not p0[["sample_id", "label"]].equals(p1[["sample_id", "label"]]):
            raise ValueError("Window off/on predictions use different rows")
        a, b = p0.prediction.eq(p0.label), p1.prediction.eq(p1.label)
        gains.append(dict(cohort=cohort, family=family, architecture=architecture,
                          none_macro_f1=n.test_macro_f1, mean_macro_f1=m.test_macro_f1,
                          delta_macro_f1=m.test_macro_f1-n.test_macro_f1,
                          none_val_macro_f1=n.val_macro_f1, mean_val_macro_f1=m.val_macro_f1,
                          mean_only_correct=int((~a & b).sum()), none_only_correct=int((a & ~b).sum())))
    gains = pd.DataFrame(gains)
    gains.to_csv(root / "window_katkisi.csv", index=False)
    common_rows = []
    for name, r in data["models"].items():
        if "common_old_test" not in r:
            continue
        _, family, pool, architecture = name.split("/")
        old = data["models"][f"old/{family}/{pool}/{architecture}"]["test"]
        new = r["common_old_test"]
        common_rows.append(dict(family=family, window=pool, architecture=architecture, test_rows=old["n"],
                                old_train_macro_f1=old["macro_f1"], full_train_macro_f1=new["macro_f1"],
                                delta_macro_f1=new["macro_f1"]-old["macro_f1"]))
    common = pd.DataFrame(common_rows)
    common.to_csv(root / "ortak_test_veri_katkisi.csv", index=False)
    architecture_order = ["parallel_score_difference", "parallel_joint", "multiclass_score_difference", "multiclass_joint"]
    labels = ["Parallel\nscore diff", "Parallel\njoint", "Multiclass\nscore diff", "Multiclass\njoint"]
    plots = root / "grafikler"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    for (cohort, family), group in table.groupby(["cohort", "family"], sort=False):
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        pools = [p for p in ("none", "mean") if p in set(group.window)]
        x = np.arange(4)
        for j, pool in enumerate(pools):
            values = group.loc[group.window.eq(pool)].set_index("architecture").reindex(architecture_order).test_macro_f1.to_numpy()
            bars = ax.bar(x + (j-(len(pools)-1)/2)*.34, values, width=.32,
                          label="Window yok" if pool == "none" else "Masked mean window", color="#4765A8" if pool == "none" else "#D28A3A")
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=3)
        for kind, color, style in [("majority", "#777777", ":"), ("go", "#2F8F6B", "--"), ("protein_go", "#8A4E9C", "-.")]:
            value = data["baselines"][f"{cohort}/{family}"][kind]["test"]["macro_f1"]
            ax.axhline(value, color=color, linestyle=style, linewidth=1.3, label=f"{kind} baseline: {value:.3f}")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test Macro-F1")
        ax.set_title(f"{cohort} cohort / {family} - seed {data['plan']['seed']}")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", ncols=2, fontsize=9)
        fig.savefig(plots / f"{cohort}_{family}_karsilastirma.png", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for ax, architecture in zip(axes.flat, architecture_order):
            for pool in pools:
                history = json.loads((root / cohort / family / pool / architecture / "history.json").read_text(encoding="utf-8"))
                ax.plot([r["epoch"] for r in history], [r["validation"]["macro_f1"] for r in history],
                        label=pool, color="#4765A8" if pool == "none" else "#D28A3A")
            ax.set_title(architecture, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Macro-F1")
            ax.grid(alpha=.15)
            ax.legend()
        fig.suptitle(f"{cohort}/{family} - validation curves")
        fig.savefig(plots / f"{cohort}_{family}_egitim.png", dpi=150)
        plt.close(fig)
    sections = ["# Mean window decoder deney sonuçları", "",
                "Bu rapor kayıtlı eğitim ve tahmin dosyalarından otomatik üretildi.", "",
                "## Kapsam ve seçim", "",
                f"- Eğitim sayısı: {len(table)}; seed: {data['plan']['seed']}; smoke: {data['plan']['smoke']}.",
                "- İki çıktı tasarımı (parallel/multiclass), iki çift işleme yolu (joint/score difference).",
                "- Window, her protein tarafında padding hariç float32 ortalama ile üçüncü ölçek olarak eklenir. WT/VAR adapter ve reader ağırlıkları paylaşılır. Comparison reader/ESIM dalı eklenmemiştir.",
                "- Bütün koşullarda yeni global/mutation-site embeddingleri aynıdır; pencere kapalı/açık koşulları aynı örnekleri kullanır.",
                "- Old: devirde doğrulanmış canonical prepared_splits kapsamı. Gayem'in tam koşu kaydıyla birebir aynı olduğu iddia edilmez. Paketlerdeki eski prepared tablolar doğrudan kullanılmadı.",
                "- Full: kaynak splitlerden geniş embedding kapsamıyla yeniden hazırlanmış uygun kayıtlar. Birleşik kanıtlar ve evaluation augmentation dışarıda; kaynak etiketleri ve partition atamaları korunur.",
                "- En iyi epoch yalnız validation Macro-F1 ile seçilir. Bütün eğitimler tamamlandıktan sonra selection_before_test.json yazılır, ardından test değerlendirilir. Reverse lambda=0 sabittir; reverse metrikleri raporlanır.",
                "- Eski test partition'ları geçmişte incelenmiştir. Bu çalışma, aynı partition'larda kontrollü takip deneyidir; hiç görülmemiş yeni test olarak sunulmamalıdır.",
                "- Tek seed, istatistiksel üstünlük veya belirsizlik aralığı kanıtı sağlamaz.", "",
                "## Model sonuçları", "",
                markdown_table(table[["cohort", "family", "window", "architecture", "val_macro_f1", "test_macro_f1", "mcc", "gof_f1"]]), "",
                "## Aynı örneklerde window katkısı", ""]
    if len(gains):
        positive = int(gains.delta_macro_f1.gt(0).sum())
        sections += [f"Mean window, {len(gains)} eşleştirilmiş karşılaştırmanın {positive} tanesinde test Macro-F1'i artırdı. Bu bir seçim kuralı veya istatistiksel anlamlılık sonucu değildir.", "",
                     markdown_table(gains[["cohort", "family", "architecture", "none_macro_f1", "mean_macro_f1", "delta_macro_f1"]]), ""]
    sections += ["## Train-only referanslar", "", markdown_table(base), "",
                 "Çoğunluk, GO çoğunluğu ve protein+GO çoğunluğu kuralları yalnız ilgili train verisinden kuruldu. Bilinmeyen protein+GO için GO, bilinmeyen GO için global çoğunluk fallback'i kullanılır. Bunlar basit referans kurallarıdır; Gayem'in ayrı model sonuçları olarak adlandırılmaz.", ""]
    if len(common):
        sections += ["## Aynı eski test örneklerinde geniş veriyle eğitim", "",
                     "Old-trained model kendi eski testinde, full-trained model de aynı örnek alt kümesinde değerlendirilir. Bu, eğitim ve validation kapsamının genişletildiği protokolün etkisidir; farklı test kapsamlarından doğan skor farkı ayıklanır.", "",
                     markdown_table(common), ""]
    sections += ["## Grafikler", ""]
    for path in sorted(plots.glob("*.png")):
        sections += [f"![{path.stem}](grafikler/{path.name})", ""]
    sections += ["## Çıktı dosyaları", "",
                 "- tum_sonuclar.csv: model ve sınıf metrikleri.",
                 "- window_katkisi.csv: aynı örneklerde none/mean farkları ve iki modelden yalnız birinin doğru bildiği örnek sayıları.",
                 "- ortak_test_veri_katkisi.csv: geniş veriyle eğitimin ortak test karşılaştırması.",
                 "- baseline_sonuclari.csv: train-only referanslar.",
                 "- Her koşunun klasöründe best.pt, config.json, history.json, results.json ve örnek düzeyi val/test_predictions.tsv.",
                 "- plan.json: ortam, kod ve veri hash'leri; selection_before_test.json: testten önce sabitlenen checkpointler.", "",
                 "## Ortam", "", "```json", json.dumps(data["plan"]["environment"], ensure_ascii=False, indent=2), "```", ""]
    (root / "RAPOR.md").write_text("\n".join(sections), encoding="utf-8")
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", type=Path)
    report_suite(p.parse_args().suite)
