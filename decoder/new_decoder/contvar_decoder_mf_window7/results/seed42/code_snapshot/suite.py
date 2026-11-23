"""Run the approved 48 fits, lock validation selections, then evaluate tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset
from .model import load_decoder
from .train import evaluate, fit, metrics
from .util import DATA_ROOT, LABELS, WORKSPACE, run_conditions, sha256_file, utc_now, write_json

SETTINGS = dict(seed=42, epochs=20, patience=5, batch_size=128, lr=3e-4,
                weight_decay=1e-4, width=128, attention_heads=4, dropout=.1,
                gradient_clip=5., class_balancing=False, reverse_penalty=False)


def verify_preparation(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["policy"]["window_size"] != 7 or manifest["policy"]["namespace"] != "molecular_function":
        raise ValueError("Wrong preparation protocol")
    for name, item in manifest["tables"].items():
        if "final" not in item:
            continue
        cohort, family, split = name.split("/")
        path = prepared / cohort / f"{family}_{split}.tsv"
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Prepared table changed: {path}")
    if sha256_file(prepared / manifest["feature_cache"]) != manifest["feature_cache_sha256"]:
        raise ValueError("Feature cache changed")
    return manifest


def baseline_predictions(train, test):
    count_columns = list(LABELS)
    global_counts = train.label.value_counts().reindex(count_columns, fill_value=0).to_numpy(dtype=float)
    global_prior = global_counts / global_counts.sum()
    mappings = {}
    for name, columns in [("go_majority", ["matched_go_id"]),
                          ("protein_go_majority", ["protein_id", "matched_go_id"])]:
        counts = train.groupby(columns + ["label"], sort=False).size().unstack(fill_value=0)
        counts = counts.reindex(columns=count_columns, fill_value=0)
        mappings[name] = {key if isinstance(key, tuple) else (key,): row.to_numpy(dtype=float) / row.sum()
                          for key, row in counts.iterrows()}
    result = {"majority": np.repeat(global_prior[None, :], len(test), axis=0)}
    result["go_majority"] = np.stack([mappings["go_majority"].get((g,), global_prior) for g in test.matched_go_id])
    result["protein_go_majority"] = np.stack([
        mappings["protein_go_majority"].get((p, g), mappings["go_majority"].get((g,), global_prior))
        for p, g in test[["protein_id", "matched_go_id"]].itertuples(index=False, name=None)])
    return result


def write_baselines(prepared, output):
    results = {}
    for cohort in ("old", "full"):
        for family in ("easy", "hard"):
            train = pd.read_csv(prepared / cohort / f"{family}_train.tsv", sep="\t", keep_default_na=False)
            test = pd.read_csv(prepared / cohort / f"{family}_test.tsv", sep="\t", keep_default_na=False)
            y = test.label_idx.to_numpy()
            common = test.common_test.astype(str).str.lower().eq("true").to_numpy()
            results[f"{cohort}/{family}"] = {}
            for name, scores in baseline_predictions(train, test).items():
                pred = scores.argmax(1)
                results[f"{cohort}/{family}"][name] = dict(
                    native=metrics(y, pred, scores, "baseline"),
                    common=metrics(y[common], pred[common], scores[common], "baseline"),
                    fitted_with="this cohort/family training labels only",
                    scores="empirical training class proportions; majority decision; GO/global fallback")
    write_json(output / "baselines.json", results)


def paired_common_comparison(models, output):
    rows = []
    by_config = {}
    for name, result in models.items():
        c = result["condition"]
        by_config[(c["cohort"], c["family"], c["window_mode"], c["head"], c["pair_mode"])] = result
    for family in ("easy", "hard"):
        for c in run_conditions():
            if c["cohort"] != "old" or c["family"] != family:
                continue
            left = by_config[("old", family, c["window_mode"], c["head"], c["pair_mode"])]
            right = by_config[("full", family, c["window_mode"], c["head"], c["pair_mode"])]
            lp = pd.read_csv(output / left["condition"]["name"] / "test_predictions.tsv", sep="\t", keep_default_na=False)
            rp = pd.read_csv(output / right["condition"]["name"] / "test_predictions.tsv", sep="\t", keep_default_na=False)
            lp = lp.loc[lp.common_test.astype(str).str.lower().eq("true")].set_index("common_key").sort_index()
            rp = rp.loc[rp.common_test.astype(str).str.lower().eq("true")].set_index("common_key").sort_index()
            if list(lp.index) != list(rp.index) or not lp.label.equals(rp.label):
                raise ValueError("Common-test identities/labels differ between paired models")
            a, b = left["test"]["common_test"], right["test"]["common_test"]
            row = dict(family=family, window_mode=c["window_mode"], head=c["head"], pair_mode=c["pair_mode"],
                       n_common=len(lp), old_macro_f1=a["macro_f1"], full_macro_f1=b["macro_f1"],
                       delta_macro_f1=b["macro_f1"] - a["macro_f1"],
                       old_mcc=a["mcc"], full_mcc=b["mcc"], delta_mcc=b["mcc"] - a["mcc"])
            for label in LABELS:
                row[f"delta_{label.lower()}_f1"] = b[label.lower()]["f1"] - a[label.lower()]["f1"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "common_scope_comparisons.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=WORKSPACE / "prepared")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=WORKSPACE / "results/seed42")
    parser.add_argument("--phase", choices=("all", "fit", "evaluate"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    prepared = verify_preparation(args.prepared)
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = ["model.py", "data.py", "train.py", "suite.py", "util.py", "go_embeddings.py"]
    provenance = dict(prepared_manifest_sha256=sha256_file(args.prepared / "manifest.json"),
        source_code_sha256={name: sha256_file(Path(__file__).parent / name) for name in source_files},
        offline_runner_sha256=sha256_file(WORKSPACE / "offline_run.py"),
        feature_cache_sha256=prepared["feature_cache_sha256"],
        source_feature_sha256=prepared["source_feature_sha256"],
        source_cohorts=prepared["source_cohorts"], ontology=prepared["ontology"],
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None)
    plan = dict(format="contvar_mf_window7_suite_v1", settings=SETTINGS, device=args.device,
                conditions=run_conditions(), provenance=provenance,
                test_used_for_selection=False, historical_results_used=False)
    plan_path = args.out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing suite has a different pinned plan")
    write_json(plan_path, plan)
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in source_files:
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    store = FeatureStore(args.prepared, args.data)
    fitted = {}
    started = time.monotonic()
    for condition in plan["conditions"]:
        destination = args.out / condition["name"]
        if args.phase == "evaluate":
            fitted[condition["name"]] = json.loads((destination / "fit_results.json").read_text(encoding="utf-8"))
            continue
        folder = args.prepared / condition["cohort"]
        datasets = {split: PairDataset(folder / f"{condition['family']}_{split}.tsv",
                                       store, condition["window_mode"]) for split in ("train", "val")}
        fitted[condition["name"]] = fit(condition, datasets, torch.device(args.device), destination,
                                        SETTINGS, provenance, args.resume)
        write_json(args.out / "training_progress.json", dict(completed=len(fitted), total=48, models=fitted))
    if len(fitted) != 48:
        raise RuntimeError("All 48 fits must finish before test evaluation")
    lock = dict(created=utc_now(), test_used_for_selection=False, checkpoints={
        name: dict(best_epoch=r["best_epoch"], validation_macro_f1=r["validation"]["macro_f1"],
                   checkpoint_sha256=sha256_file(args.out / name / "best.pt")) for name, r in fitted.items()})
    selected = {}
    for cohort in ("old", "full"):
        for family in ("easy", "hard"):
            candidates = [r for r in fitted.values()
                          if r["condition"]["cohort"] == cohort and r["condition"]["family"] == family]
            winner = max(candidates, key=lambda r: r["validation"]["macro_f1"])
            selected[f"{cohort}/{family}"] = winner["condition"]["name"]
    lock["best_by_validation"] = selected
    lock_path = args.out / "selection_before_test.json"
    if lock_path.exists():
        old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if old_lock["checkpoints"] != lock["checkpoints"] or old_lock["best_by_validation"] != selected:
            raise ValueError("Locked checkpoints changed")
    else:
        write_json(lock_path, lock)
    if args.phase == "fit":
        print("ALL_48_FITS_COMPLETE: validation selections locked, tests not evaluated", flush=True)
        return
    models = {}
    for condition in plan["conditions"]:
        name, destination = condition["name"], args.out / condition["name"]
        path = destination / "results.json"
        if args.resume and path.exists() and (destination / "test_predictions.tsv").exists():
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            model, checkpoint = load_decoder(destination / "best.pt", args.device)
            if checkpoint["provenance"] != provenance:
                raise ValueError("Checkpoint provenance mismatch")
            dataset = PairDataset(args.prepared / condition["cohort"] / f"{condition['family']}_test.tsv",
                                  store, condition["window_mode"])
            loader = DataLoader(dataset, batch_size=SETTINGS["batch_size"], num_workers=0,
                                pin_memory=args.device == "cuda")
            test, prediction = evaluate(model, loader, args.device)
            prediction.to_csv(destination / "test_predictions.tsv", sep="\t", index=False)
            prediction.loc[prediction.common_test.astype(str).str.lower().eq("true")].to_csv(
                destination / "common_test_predictions.tsv", sep="\t", index=False)
            result = {**fitted[name], "test": test}
            write_json(path, result)
        models[name] = result
        print(f"TEST {condition['number']:02d}/48 {name}: F1={result['test']['macro_f1']:.4f} "
              f"common_F1={result['test']['common_test']['macro_f1']:.4f}", flush=True)
    write_baselines(args.prepared, args.out)
    comparisons = paired_common_comparison(models, args.out)
    summary = dict(plan=plan, models=models, distinct_training_runs=48,
                   best_by_validation=selected, paired_common_test=comparisons,
                   preparation_manifest=str((args.prepared / "manifest.json").resolve()),
                   test_used_for_selection=False, historical_results_used=False,
                   total_training_seconds=sum(r["seconds"] for r in models.values()),
                   current_invocation_seconds=time.monotonic() - started, completed=utc_now())
    write_json(args.out / "summary.json", summary)
    print("SUITE_COMPLETE: 48 fits, 48 native tests and paired common-test evaluations", flush=True)


if __name__ == "__main__":
    main()
