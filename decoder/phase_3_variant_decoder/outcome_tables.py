"""Readable prediction-outcome tables; counts and scores keep distinct meanings.

For random controls, counts are repeat means and F1 stays the mean of repeat F1
scores. Never recompute nonlinear scores from the averaged confusion counts.
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np

from phase_3_variant_decoder.config import LABELS
from phase_3_variant_decoder.presentation_terms import save_figure


RULES_TR = {
    "always_lof": "Her örneğe LOF der.",
    "always_gof": "Her örneğe GOF der.",
    "always_neutral": "Her örneğe NEUTRAL der.",
    "train_majority": "Train'de en sık görülen sınıfı herkese söyler.",
    "uniform_random": "Üç sınıftan birini eşit olasılıkla seçer.",
    "train_prior_random": "Train sınıf oranlarına göre rastgele seçer.",
    "go_lookup": "Sorulan GO için train'deki en sık etiketi söyler; görülmeyen GO'da train majority kullanır.",
}


def outcome_rows(result: dict, split: str) -> tuple[list[dict], list[dict]]:
    summaries, classes = [], []
    family = result.get("data", {}).get("family", "unspecified")
    for record in result["results"].values():
        if record["split"] != split:
            continue
        cm = np.asarray(record["mean_confusion"], dtype=float)
        total = int(record["n"])
        if cm.shape != (3, 3) or not np.isfinite(cm).all() or (cm < 0).any():
            raise ValueError("Invalid saved prediction counts")
        if not np.isclose(cm.sum(), total):
            raise ValueError("Saved prediction counts disagree with the evaluation population")
        correct = float(np.trace(cm))
        if not np.isclose(correct / total, record["accuracy"]):
            raise ValueError("Saved accuracy disagrees with correct / total")
        base = {"family": family, "partition": split, "method": record["method"],
                "rule": record["description"], "count_type": "repeat mean" if record["repeats"] > 1 else "exact",
                "repeats": record["repeats"], "total_n": total}
        summaries.append({**base, "correct_n": correct, "incorrect_n": total - correct,
                          "accuracy": record["accuracy"], "macro_f1": record["macro_f1"],
                          **{f"{label.lower()}_f1": record[f"{label.lower()}_f1"] for label in LABELS}})
        for index, label in enumerate(LABELS):
            support = float(cm[index].sum())
            tp = float(cm[index, index])
            classes.append({**base, "true_class": label, "actual_n": int(round(support)),
                            "correct_n": tp, "incorrect_n": support - tp,
                            "recall": tp / support if support else None,
                            **{f"predicted_{lab.lower()}_n": float(cm[index, col]) for col, lab in enumerate(LABELS)}})
    return summaries, classes


def _number(value: float, mean: bool) -> str:
    return f"{value:,.1f}" if mean else f"{value:,.0f}"


def _table(headers, rows, path, title, description, *, widths=None, cell_colors=None, config=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, max(3.7, .44 * len(rows) + 1.8)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center", colWidths=widths,
                     bbox=[0, 0, 1, .9])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6dee8")
        if row == 0:
            cell.set_facecolor("#17324d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 == 0 else "white")
            cell.set_text_props(color="#172b42")
            if cell_colors and (row - 1, col) in cell_colors:
                cell.set_facecolor(cell_colors[row - 1, col])
    ax.set_title(title, fontsize=16, weight="bold", pad=25, color="#17324d")
    fig.tight_layout()
    save_figure(fig, path, config=config, description=description)
    plt.close(fig)


def plot_outcome_summary(result: dict, split: str, path: Path) -> None:
    summaries, _ = outcome_rows(result, split)
    rows = []
    for row in summaries:
        random = row["count_type"] == "repeat mean"
        name = row["rule"] + (f"\n(always {result['train_majority_label']})" if row["method"] == "train_majority" else "")
        rows.append([name, _number(row["correct_n"], random), _number(row["incorrect_n"], random),
                     f"{row['accuracy']:.1%}", f"{row['macro_f1']:.3f}"])
    family = result.get("data", {}).get("family", "unspecified").upper()
    n = summaries[0]["total_n"]
    _table(["Prediction rule", "Correct (n)", "Incorrect (n)", "Accuracy\ncorrect / total", "Macro-F1\n0 to 1"], rows, path,
           f"{family} / {split.upper()} — prediction outcomes | Total: {n:,} rows",
           result.get("outcome_description", "Correct doğru tahmin, Incorrect yanlış tahmin sayısıdır; toplamları Total'a eşittir. Accuracy doğru tahmin oranıdır. Macro-F1 üç sınıfa eşit ağırlık verir. Random satırlardaki ondalıklı sayılar tekrar ortalamasıdır."),
           widths=[.38, .14, .14, .17, .17], config=result.get("configuration"))


def plot_class_recall(result: dict, split: str, path: Path) -> None:
    summaries, classes = outcome_rows(result, split)
    rows = []
    colors = {}
    for i, summary in enumerate(summaries):
        group = [row for row in classes if row["method"] == summary["method"]]
        cells = [summary["rule"]]
        for j, row in enumerate(group, 1):
            mean = row["count_type"] == "repeat mean"
            cells.append(f"{_number(row['correct_n'], mean)} / {row['actual_n']:,}")
            colors[i, j] = "#fce8e6" if row["correct_n"] == 0 and row["actual_n"] else "#e5f1e9"
        rows.append(cells)
    family = result.get("data", {}).get("family", "unspecified").upper()
    _table(["Prediction rule", "LOF\ncorrect / actual", "GOF\ncorrect / actual", "NEUTRAL\ncorrect / actual"], rows, path,
           f"{family} / {split.upper()} — how many examples of each class are recovered?",
           result.get("class_recall_description", "Her hücre doğru bulunan / o sınıfta gerçekte mevcut örnek sayısıdır. Örneğin 0 / 273, 273 örneğin hiçbirinin doğru bulunmadığını söyler. Bu tablo yüzde veya F1 değil, örnek sayısı gösterir. Random değerler tekrar ortalamasıdır."),
           widths=[.4, .2, .2, .2], cell_colors=colors, config=result.get("configuration"))


def plot_prediction_counts(items: dict[str, dict], path: Path, *, title: str, config=None, random=False) -> None:
    """A flat count table instead of row-normalized heatmaps."""
    rows, colors = [], {}
    for name, metric in items.items():
        cm = np.asarray(metric.get("mean_confusion", metric.get("confusion")), dtype=float)
        mean = metric.get("repeats", 1) > 1
        for i, label in enumerate(LABELS):
            r = len(rows)
            rows.append([name if i == 0 else "", label, _number(cm[i].sum(), False),
                         *[_number(value, mean) for value in cm[i]]])
            for j in range(3):
                colors[r, j + 3] = "#d9eddf" if i == j else "#fff0ed"
    description = "True class gerçek sınıftır; Actual o sınıftaki toplam örnek sayısıdır. Predicted sütunları yöntemin verdiği cevabı gösterir. Yeşil hücreler doğru, pembe hücreler yanlış tahmin sayılarıdır. Satır yüzdesi kullanılmıyor."
    if random:
        description += " Random sayılar tekrar ortalamasıdır; kesirli örnek yoktur. Görünen sayılar yuvarlanmıştır."
    _table(["Method", "True class", "Actual (n)", "Predicted\nLOF (n)", "Predicted\nGOF (n)", "Predicted\nNEUTRAL (n)"],
           rows, path, title, description, widths=[.32, .12, .12, .145, .145, .145], cell_colors=colors, config=config)


def _html_table(headers, rows) -> str:
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>")


def outcome_html(result: dict, split: str) -> str:
    summaries, classes = outcome_rows(result, split)
    family = result.get("data", {}).get("family", "unspecified").upper()
    rows = [[row["rule"], RULES_TR[row["method"]], _number(row["correct_n"], row["repeats"] > 1),
             _number(row["incorrect_n"], row["repeats"] > 1), f"{row['accuracy']:.1%}", f"{row['macro_f1']:.3f}"] for row in summaries]
    class_rows = []
    for row in summaries:
        group = [item for item in classes if item["method"] == row["method"]]
        cells = [row["rule"]]
        for item in group:
            cells.append(f"{_number(item['correct_n'], item['repeats'] > 1)} / {item['actual_n']:,}")
        class_rows.append(cells)
    return (f"<h2>{family} / {split.upper()} — Total: {summaries[0]['total_n']:,} rows</h2>"
            "<p>Correct = doğru, Incorrect = yanlış tahmin sayısı. Random satırlar tekrar ortalamasıdır. F1 yüzde değildir.</p>" +
            _html_table(["Prediction rule", "Kısa açıklama", "Correct (n)", "Incorrect (n)", "Accuracy", "Macro-F1"], rows) +
            "<h3>Class recovery — correct / actual</h3><p>Her sütun o sınıfta doğru bulunan / gerçekte mevcut örnek sayısını gösterir. 0 / 273, 273 örneğin hiçbirinin doğru bulunmadığı demektir. Burada yüzde veya F1 gösterilmiyor.</p>" +
            _html_table(["Prediction rule", "LOF: correct / actual", "GOF: correct / actual", "NEUTRAL: correct / actual"], class_rows))


def write_outcome_reports(result: dict, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    summary_rows, class_rows = [], []
    for split in ("val", "test"):
        for key, plot in (("outcomes", plot_outcome_summary), ("class_recall", plot_class_recall)):
            path = output / f"{split}_{key}.png"
            plot(result, split, path)
            paths[f"{split}_{key}"] = str(path)
        summary, classes = outcome_rows(result, split)
        summary_rows.extend(summary)
        class_rows.extend(classes)
    for key, rows in (("outcomes_csv", summary_rows), ("class_outcomes_csv", class_rows)):
        path = output / f"{key}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths[key] = str(path)
    return paths
