"""Combine completed V5 CPU runs while leaving each input run directory read-only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from .util import LABELS, WORKSPACE, run_conditions, sha256_file, write_json

SEEDS = (42, 43, 44)
FACTORS = ("cohort", "family", "window_mode", "head", "pair_mode")
COMPARISONS = (
    ("encoder", "cohort", "mean", "distance"),
    ("pair_mode", "pair_mode", "joint", "score_difference"),
    ("window_context", "window_mode", "none", "mean"),
    ("window_comparison", "window_mode", "pair_mean", "esim"),
    ("head", "head", "parallel", "multiclass"),
)
SCORES = ("macro_f1", "mcc", "accuracy", "balanced_accuracy")
SCOPE_FIELDS = ("n", *SCORES)
REFERENCE_PROVENANCE = (
    "prepared_manifest_sha256", "source_feature_sha256", "source_cohorts", "ontology",
    "offline_runner_sha256", "torch_version", "numpy_version", "gpu",
)
V5_CONDITIONS = run_conditions(("mean", "distance"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(path, seed):
    summary = read_json(path / "summary.json")
    plan = read_json(path / "plan.json")
    lock = read_json(path / "selection_before_test.json")
    qa = read_json(path / "final_qa.json")
    if (summary["plan"] != plan or plan["settings"]["seed"] != seed or plan["device"] != "cpu"
            or plan["format"] != "contvar_mf_window7_v5_suite" or plan["conditions"] != V5_CONDITIONS
            or summary.get("seed", seed) != seed or lock.get("seed", seed) != seed
            or qa.get("seed", seed) != seed or qa["passed"] is not True or qa["checked_models"] != 48
            or any(item["test_used_for_selection"] for item in (summary, plan, lock))):
        raise ValueError(f"Expected a completed and verified V5 CPU suite for seed {seed}: {path}")
    names = {condition["name"] for condition in plan["conditions"]}
    if set(summary["models"]) != names or set(lock["checkpoints"]) != names:
        raise ValueError(f"Incomplete model or selection records: {path}")
    expected_selected = {}
    for condition in plan["conditions"]:
        name = condition["name"]
        result = summary["models"][name]
        locked = lock["checkpoints"][name]
        if (result["condition"] != condition or result.get("seed", seed) != seed
                or result["best_epoch"] != locked["best_epoch"]
                or result["validation"]["macro_f1"] != locked["validation_macro_f1"]
                or sha256_file(path / name / "best.pt") != locked["checkpoint_sha256"]):
            raise ValueError(f"Model record or checkpoint disagrees with the lock: {path / name}")
        group = f"{condition['cohort']}/{condition['family']}"
        earlier = expected_selected.setdefault(group, name)
        if result["validation"]["macro_f1"] > summary["models"][earlier]["validation"]["macro_f1"]:
            expected_selected[group] = name
    if summary["best_by_validation"] != expected_selected or lock["best_by_validation"] != expected_selected:
        raise ValueError(f"Validation selections disagree: {path}")
    return summary, read_json(path / "baselines.json")


def verify_comparability(runs, baselines):
    reference = runs[42]["plan"]
    old_code = reference["provenance"]["source_code_sha256"]
    for seed in SEEDS[1:]:
        plan = runs[seed]["plan"]
        if dict(plan["settings"], seed=42) != reference["settings"]:
            raise ValueError(f"Seed {seed} changed a training setting")
        if any(plan["provenance"][key] != reference["provenance"][key] for key in REFERENCE_PROVENANCE):
            raise ValueError(f"Seed {seed} changed data or runtime provenance")
        new_code = plan["provenance"]["source_code_sha256"]
        if set(old_code) != set(new_code) or any(old_code[name] != new_code[name]
                                                for name in old_code.keys() - {"suite.py", "train.py"}):
            raise ValueError(f"Seed {seed} changed model or input processing source")
        if baselines[seed] != baselines[42]:
            raise ValueError(f"Deterministic baselines differ between seeds 42 and {seed}")
    if runs[43]["plan"]["provenance"] != runs[44]["plan"]["provenance"]:
        raise ValueError("New seeds were produced with different code or provenance")


def score_fields(result):
    values = {}
    for phase in ("validation", "test"):
        metrics = result[phase]
        domains = [("", metrics)]
        domains.extend((f"_{domain}", metrics[domain])
                       for domain in ("single_substitution", "combinatorial") if domain in metrics)
        for suffix, scope in domains:
            prefix = phase + suffix
            values.update({f"{prefix}_{key}": scope[key] for key in SCOPE_FIELDS})
            for label in LABELS:
                label = label.lower()
                values.update({f"{prefix}_{label}_{key}": scope[label][key]
                               for key in ("precision", "recall", "f1", "support")})
    return values


def stats(values):
    if len(values) != len(SEEDS):
        raise ValueError("Every aggregate must have one value per seed")
    return {"mean": mean(values), "sd": stdev(values), "min": min(values), "max": max(values)}


def aggregate_group(records, metadata, metric_keys):
    if tuple(row["seed"] for row in records) != SEEDS:
        raise ValueError("Group does not contain seeds 42, 43 and 44 in order")
    row = dict(metadata)
    for record in records:
        seed = record["seed"]
        for key in ("validation_macro_f1", "test_macro_f1", "best_epoch", "condition"):
            row[f"{key}_seed{seed}"] = record[key]
    for key in metric_keys:
        row.update({f"{key}_{stat}": value
                    for stat, value in stats([entry[key] for entry in records]).items()})
    return row


def seed_rows(runs):
    rows = []
    for seed in SEEDS:
        summary = runs[seed]
        selected = set(summary["best_by_validation"].values())
        for condition in V5_CONDITIONS:
            name = condition["name"]
            result = summary["models"][name]
            rows.append(dict(seed=seed, condition=name, **{k: condition[k] for k in FACTORS},
                             selected_by_validation=name in selected, best_epoch=result["best_epoch"],
                             epochs_run=result["epochs_run"], **score_fields(result)))
    return rows


def paired_rows(all_rows):
    lookup = {(r["seed"], *(r[k] for k in FACTORS)): r for r in all_rows}
    per_seed, aggregates = [], []
    metrics = ("validation_macro_f1", "test_macro_f1", "test_mcc")
    for name, factor, left, right in COMPARISONS:
        for condition in V5_CONDITIONS:
            if condition[factor] != left:
                continue
            left_key = tuple(condition[k] for k in FACTORS)
            right_key = tuple(right if k == factor else condition[k] for k in FACTORS)
            if (SEEDS[0], *right_key) not in lookup:
                continue
            fixed = {k: condition[k] if k != factor else "" for k in FACTORS}
            group = []
            for seed in SEEDS:
                a, b = (lookup[seed, *key] for key in (left_key, right_key))
                group.append(dict(seed=seed, comparison=name, factor=factor, left=left, right=right,
                                  **fixed, left_condition=a["condition"], right_condition=b["condition"],
                                  **{f"{metric}_{column}": value
                                     for metric in metrics for column, value in
                                     (("left", a[metric]), ("right", b[metric]), ("delta", b[metric] - a[metric]))}))
            per_seed.extend(group)
            aggregates.append({**{k: group[0][k] for k in
                                   ("comparison", "factor", "left", "right", *FACTORS,
                                    "left_condition", "right_condition")},
                               **{f"{metric}_delta_seed{r['seed']}": r[f"{metric}_delta"]
                                  for metric in metrics for r in group},
                               **{f"{metric}_delta_{stat}": value
                                  for metric in metrics for stat, value in
                                  stats([r[f"{metric}_delta"] for r in group]).items()}})
    return per_seed, aggregates


def comparison_overview(per_seed):
    rows = []
    for name, _, left, right in COMPARISONS:
        for family in ("easy", "hard"):
            groups = [[r for r in per_seed if r["comparison"] == name and r["seed"] == seed
                       and r["family"] == family] for seed in SEEDS]
            rows.append(dict(comparison=name, family=family, left=left, right=right,
                             matches_per_seed=len(groups[0]),
                             **{f"{metric}_delta_{stat}": value
                                for metric in ("validation_macro_f1", "test_macro_f1", "test_mcc")
                                for stat, value in stats(
                                    [mean(r[f"{metric}_delta"] for r in group) for group in groups]).items()}))
    return rows


def aggregate(runs):
    rows = seed_rows(runs)
    selected = [r for r in rows if r["selected_by_validation"]]
    metric_keys = tuple(score_fields(runs[42]["models"][V5_CONDITIONS[0]["name"]]))
    fixed_summary = []
    for condition in V5_CONDITIONS:
        name = condition["name"]
        fixed_summary.append(aggregate_group(
            [r for r in rows if r["condition"] == name],
            dict(condition=name, **{k: condition[k] for k in FACTORS}), metric_keys))
    selected_summary = []
    for encoder in ("mean", "distance"):
        for family in ("easy", "hard"):
            selected_summary.append(aggregate_group(
                [r for r in selected if r["cohort"] == encoder and r["family"] == family],
                dict(cohort=encoder, family=family), metric_keys))
    paired, pairs = paired_rows(rows)
    return dict(conditions_by_seed=rows, condition_summary=fixed_summary, selection_by_seed=selected,
                selection_summary=selected_summary, paired_differences_by_seed=paired,
                paired_differences_summary=pairs, comparison_overview=comparison_overview(paired))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(tables):
    lines = [
        "# V5 decoder deneyleri üç seed sonuçları", "",
        "Seed 42, 43 ve 44 için aynı V5 MF verisi, dondurulmuş Mean ve Distance encoder çıktıları, "
        "96 yeni eğitim ve CPU üzerinde koşul başına aynı ayarlar kullanıldı. Seed 42 daha önceki "
        "doğrulanmış çalışmadan alındı. Her seed kendi 48 validation seçimini testten önce kilitledi.", "",
        "SS, üç seed üzerinden örneklem standart sapmasıdır (n−1). Eşleştirilmiş farkların işareti "
        "sağdaki seçeneğin skoru eksi soldaki seçeneğin skorudur.", "",
        "## Validation ile her seed için seçilen decoder'lar", "",
        "| Encoder | Split | Seed | Decoder | En iyi epoch | Validation Macro-F1 | Test Macro-F1 | Test MCC |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in tables["selection_by_seed"]:
        lines.append(f"| {row['cohort']} | {row['family']} | {row['seed']} | "
                     f"`{row['condition'].rsplit('/', 1)[1]}` | {row['best_epoch']} | "
                     f"{row['validation_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | {row['test_mcc']:.4f} |")
    lines += ["", "| Encoder | Split | Validation Macro-F1 ort ± SS | Test Macro-F1 ort ± SS | Test MCC ort ± SS |",
              "|---|---|---:|---:|---:|"]
    for row in tables["selection_summary"]:
        lines.append(f"| {row['cohort']} | {row['family']} | " + " | ".join(
            f"{row[k + '_mean']:.4f} ± {row[k + '_sd']:.4f}"
            for k in ("validation_macro_f1", "test_macro_f1", "test_mcc")) + " |")
    lines += [
        "", "## Sabit decoder koşulları", "",
        "| Encoder | Split | Decoder | Validation F1 (42 / 43 / 44) | Test F1 (42 / 43 / 44) | Test F1 ort ± SS | Test MCC ort ± SS |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in tables["condition_summary"]:
        values = [" / ".join(f"{row[f'{k}_seed{s}']:.4f}" for s in SEEDS)
                  for k in ("validation_macro_f1", "test_macro_f1")]
        lines.append(f"| {row['cohort']} | {row['family']} | `{row['condition'].rsplit('/', 1)[1]}` | "
                     f"{values[0]} | {values[1]} | {row['test_macro_f1_mean']:.4f} ± {row['test_macro_f1_sd']:.4f} | "
                     f"{row['test_mcc_mean']:.4f} ± {row['test_mcc_sd']:.4f} |")
    lines += [
        "", "## Eşleştirilmiş karşılaştırmalar", "",
        "Özet satırları, her seed içinde diğer koşulları aynı olan eşleşmelerin farklarını "
        "ortalayıp ardından üç seed'in ortalamasını ve SS'sini hesaplar.", "",
        "| Karşılaştırma (sağ − sol) | Split | Seed başına eşleşme | Test Macro-F1 farkı ort ± SS | Test MCC farkı ort ± SS |",
        "|---|---|---:|---:|---:|",
    ]
    for row in tables["comparison_overview"]:
        lines.append(f"| {row['right']} − {row['left']} | {row['family']} | {row['matches_per_seed']} | "
                     f"{row['test_macro_f1_delta_mean']:+.4f} ± {row['test_macro_f1_delta_sd']:.4f} | "
                     f"{row['test_mcc_delta_mean']:+.4f} ± {row['test_mcc_delta_sd']:.4f} |")
    lines += [
        "", "## Baseline ve deneysel sınırlılıklar", "",
        "- Üç seed'in genel çoğunluk, GO çoğunluk ve protein–GO çoğunluk baseline dosyaları içerik olarak birebir aynı.",
        "- Danışmanın hazırladığı V5 split'leri korundu. Denetim, her iki encoder'da Easy testte 5 ve Hard testte "
        "9 satırın sayısal girdisinin train'de de bulunduğunu gösterdi. Validation da hesaba katılınca Hard "
        "testte 10 benzersiz satır etkilendi. Seed 42'nin bu test satırları dışarıda tutularak yapılan "
        "duyarlılık hesabında Macro-F1 değişiminin mutlak değeri 0,001'in altında kaldı.",
        "- Arşiv Phase 0 ve Phase 1 listelerinde potansiyel protein maruziyeti var. Listeler iki V5 encoder "
        "checkpoint'inin gerçek eğitim üyeliğine kesin olarak bağlanamadı.",
        "", "## Ayrıntılı tablolar", "",
        "- [Koşul ve seed bazında metrikler, alt kümeler ve epoch](conditions_by_seed.csv)",
        "- [Sabit koşul ortalamaları, SS, min, max ve sınıf metrikleri](condition_summary.csv)",
        "- [Her seed'in validation ile seçtiği modeller](selection_by_seed.csv)",
        "- [Validation seçimi üzerinden üç seed özeti](selection_summary.csv)",
        "- [Seed bazında eşleştirilmiş farklar](paired_differences_by_seed.csv)",
        "- [Her sabit eşleşmenin üç seed özeti](paired_differences_summary.csv)",
        "- [Karşılaştırma özeti](comparison_overview.csv)",
        "- [Baseline eşitliği](baselines_equal.json)",
        "- [Seed 42 denetim raporu](../v5_seed42_audit/DENETIM_V5_SEED42.md)", "",
    ]
    return "\n".join(lines)


def main():
    data = WORKSPACE.parents[2] / "data"
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}", type=Path, default=data / f"v5_mf_window7_seed{seed}")
    parser.add_argument("--out", type=Path, default=data / "v5_three_seed")
    args = parser.parse_args()
    roots = {seed: getattr(args, f"seed{seed}").resolve() for seed in SEEDS}
    out = args.out.resolve()
    if any(out == path or path in out.parents for path in roots.values()):
        parser.error("--out must be separate from the input run directories")
    runs, baselines = {}, {}
    for seed, path in roots.items():
        runs[seed], baselines[seed] = read_run(path, seed)
    verify_comparability(runs, baselines)
    tables = aggregate(runs)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
    write_json(out / "baselines_equal.json", dict(seeds=list(SEEDS), exact_equality=True,
                                                 baselines=baselines[42]))
    write_json(out / "three_seed_results.json", dict(
        seeds=list(SEEDS), new_runs=96, sample_standard_deviation=True, baseline_exact_equality=True,
        input_run_roots={seed: str(path) for seed, path in roots.items()},
        input_summary_sha256={seed: sha256_file(path / "summary.json") for seed, path in roots.items()},
        validation_selection_only=True, **tables))
    (out / "THREE_SEED_RESULTS.md").write_text(report_text(tables), encoding="utf-8")
    print(f"THREE_SEED_RESULTS_COMPLETE {out}", flush=True)


if __name__ == "__main__":
    main()
