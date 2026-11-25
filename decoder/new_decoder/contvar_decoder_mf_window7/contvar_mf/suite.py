"""Run the approved 48 fits, lock validation selections, then evaluate tests."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset
from .model import load_decoder
from .train import evaluate, fit, metrics, write_predictions
from .util import DATA_ROOT, LABELS, WORKSPACE, run_conditions, sha256_file, utc_now, write_json
from .v5 import ENCODERS, V5_FORMAT, V5Bundle, V5FeatureStore

SETTINGS = dict(seed=42, epochs=20, patience=5, batch_size=128, lr=3e-4,
                weight_decay=1e-4, width=128, attention_heads=4, dropout=.1,
                gradient_clip=5., class_balancing=False, reverse_penalty=False)


def parse_seed(value):
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("Seed must be between 0 and 4294967295")
    return seed


def output_path(explicit, prepared, is_v5, seed):
    if explicit is not None:
        return explicit.resolve()
    if is_v5:
        return prepared.resolve().with_name(f"{prepared.resolve().name}_seed{seed}")
    return WORKSPACE / f"results/seed{seed}"


def check_output_directory(output, plan, resume):
    if any((parent / "plan.json").is_file() for parent in output.parents):
        raise ValueError(f"Output directory is nested inside an existing suite: {output}")
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError(f"Existing suite has a different pinned plan: {output}")
        if not resume:
            raise FileExistsError(f"Existing suite requires --resume: {output}")
        return
    if output.exists():
        entries = {path.name for path in output.iterdir()}
        if entries and not (resume and entries == {"plan.json.tmp"}):
            raise FileExistsError(f"Nonempty output directory without a pinned plan: {output}")


@contextmanager
def output_lock(output):
    """Lock as a sibling so an output directory is checked before any writes to it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Output directory is in use: {output}") from exc
        yield


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
    if manifest["format"] == V5_FORMAT:
        if sha256_file(prepared / "go_terms_embeddings_anc2vec.npz") != manifest["go_sha256"]:
            raise ValueError("Prepared GO embeddings changed")
    elif sha256_file(prepared / manifest["feature_cache"]) != manifest["feature_cache_sha256"]:
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


def write_baselines(prepared, output, cohorts=("old", "full"), shared_tables=None):
    results = {}
    for cohort in cohorts:
        folder = prepared / (shared_tables or cohort)
        for family in ("easy", "hard"):
            train = pd.read_csv(folder / f"{family}_train.tsv", sep="\t", keep_default_na=False)
            test = pd.read_csv(folder / f"{family}_test.tsv", sep="\t", keep_default_na=False)
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


def paired_common_comparison(models, output, cohorts=("old", "full")):
    rows = []
    by_config = {}
    for name, result in models.items():
        c = result["condition"]
        by_config[(c["cohort"], c["family"], c["window_mode"], c["head"], c["pair_mode"])] = result
    left_source, right_source = cohorts
    for family in ("easy", "hard"):
        for c in run_conditions(cohorts):
            if c["cohort"] != left_source or c["family"] != family:
                continue
            left = by_config[(left_source, family, c["window_mode"], c["head"], c["pair_mode"])]
            right = by_config[(right_source, family, c["window_mode"], c["head"], c["pair_mode"])]
            lp = pd.read_csv(output / left["condition"]["name"] / "test_predictions.tsv", sep="\t", keep_default_na=False)
            rp = pd.read_csv(output / right["condition"]["name"] / "test_predictions.tsv", sep="\t", keep_default_na=False)
            lp = lp.loc[lp.common_test.astype(str).str.lower().eq("true")].set_index("common_key").sort_index()
            rp = rp.loc[rp.common_test.astype(str).str.lower().eq("true")].set_index("common_key").sort_index()
            if list(lp.index) != list(rp.index) or not lp.label.equals(rp.label):
                raise ValueError("Common-test identities/labels differ between paired models")
            a, b = left["test"]["common_test"], right["test"]["common_test"]
            row = dict(family=family, window_mode=c["window_mode"], head=c["head"], pair_mode=c["pair_mode"],
                       n_common=len(lp), delta_macro_f1=b["macro_f1"] - a["macro_f1"],
                       delta_mcc=b["mcc"] - a["mcc"])
            for source, stats in ((left_source, a), (right_source, b)):
                for metric in ("macro_f1", "mcc"):
                    row[f"{source}_{metric}"] = stats[metric]
            for label in LABELS:
                row[f"delta_{label.lower()}_f1"] = b[label.lower()]["f1"] - a[label.lower()]["f1"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "common_scope_comparisons.csv", index=False)
    return rows


def write_v5_results(models, output, selected):
    rows = []
    for name, result in models.items():
        condition, test = result["condition"], result["test"]
        combo = test.get("combinatorial", {})
        rows.append(dict(seed=result["seed"], encoder=condition["cohort"], family=condition["family"],
                         window_mode=condition["window_mode"], head=condition["head"],
                         pair_mode=condition["pair_mode"], selected_by_validation=name in selected.values(),
                         best_epoch=result["best_epoch"], val_macro_f1=result["validation"]["macro_f1"],
                         test_rows=test["n"], test_macro_f1=test["macro_f1"], test_mcc=test["mcc"],
                         test_lof_f1=test["lof"]["f1"], test_gof_f1=test["gof"]["f1"],
                         test_neutral_f1=test["neutral"]["f1"], combinatorial_rows=combo.get("n", 0),
                         combinatorial_macro_f1=combo.get("macro_f1"),
                         single_substitution_macro_f1=test.get("single_substitution", {}).get("macro_f1")))
    pd.DataFrame(rows).to_csv(output / "all_runs.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=WORKSPACE / "prepared")
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--v5-bundle", type=Path, help="Directory with the V5 ZIPs and the two corrected window H5s")
    parser.add_argument("--seed", type=parse_seed, default=SETTINGS["seed"])
    parser.add_argument("--out", type=Path, help="Defaults to a seed-specific directory")
    parser.add_argument("--phase", choices=("all", "fit", "evaluate"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    prepared = verify_preparation(args.prepared)
    is_v5 = prepared["format"] == V5_FORMAT
    if is_v5 != (args.v5_bundle is not None):
        parser.error("Pass --v5-bundle exactly when --prepared was produced by contvar_mf.v5")
    bundle = V5Bundle(args.v5_bundle) if is_v5 else None
    cohorts = ENCODERS if is_v5 else ("old", "full")
    shared_tables = "v5" if is_v5 else None
    args.out = output_path(args.out, args.prepared, is_v5, args.seed)
    settings = {**SETTINGS, "seed": args.seed}
    if bundle is not None and bundle.hashes() != prepared["source_feature_sha256"]:
        raise ValueError("V5 bundle changed since preparation")
    source_files = ["model.py", "data.py", "train.py", "suite.py", "util.py", "go_embeddings.py"]
    if is_v5:
        source_files += ["v5.py", "prepare.py"]
    provenance = dict(prepared_manifest_sha256=sha256_file(args.prepared / "manifest.json"),
        source_code_sha256={name: sha256_file(Path(__file__).parent / name) for name in source_files},
        offline_runner_sha256=sha256_file(WORKSPACE / "offline_run.py"),
        feature_cache_sha256=prepared.get("feature_cache_sha256"),
        source_feature_sha256=prepared["source_feature_sha256"],
        source_cohorts=prepared["source_cohorts"], ontology=prepared["ontology"],
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None)
    plan = dict(format="contvar_mf_window7_v5_suite" if is_v5 else "contvar_mf_window7_suite_v1",
                settings=settings, device=args.device, conditions=run_conditions(cohorts), provenance=provenance,
                test_used_for_selection=False, historical_results_used=False)
    check_output_directory(args.out, plan, args.resume)
    with output_lock(args.out):
        check_output_directory(args.out, plan, args.resume)
        if (args.out / "summary.json").exists():
            completed = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))
            if completed.get("plan") != plan or len(completed.get("models", {})) != len(plan["conditions"]):
                raise ValueError(f"Completed summary disagrees with the pinned plan: {args.out}")
            print(f"SUITE_ALREADY_COMPLETE seed={args.seed}: {args.out}", flush=True)
            return
        run_suite(args, is_v5, bundle, cohorts, shared_tables, plan, source_files)


def run_suite(args, is_v5, bundle, cohorts, shared_tables, plan, source_files):
    settings, provenance, conditions = (plan[key] for key in ("settings", "provenance", "conditions"))
    seed, total = settings["seed"], len(conditions)
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / "plan.json"
    if not plan_path.exists():
        write_json(plan_path, plan)
    snapshot = args.out / "code_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in source_files:
        (snapshot / name).write_bytes((Path(__file__).parent / name).read_bytes())
    legacy_store = None if is_v5 else FeatureStore(args.prepared, args.data)
    fitted = {}
    started = time.monotonic()
    for cohort in cohorts:
        store = V5FeatureStore(args.prepared, bundle, cohort) if is_v5 and args.phase != "evaluate" else legacy_store
        folder = args.prepared / (shared_tables or cohort)
        for condition in conditions:
            if condition["cohort"] != cohort:
                continue
            name = condition["name"]
            destination = args.out / name
            if args.phase == "evaluate":
                fitted[name] = json.loads((destination / "fit_results.json").read_text(encoding="utf-8"))
                continue
            datasets = {split: PairDataset(folder / f"{condition['family']}_{split}.tsv",
                                           store, condition["window_mode"]) for split in ("train", "val")}
            fitted[name] = fit(condition, datasets, torch.device(args.device), destination,
                               settings, provenance, args.resume)
            del datasets
            write_json(args.out / "training_progress.json", dict(seed=seed, completed=len(fitted), total=total, models=fitted))
        del store
    if len(fitted) != total:
        raise RuntimeError(f"All {total} fits must finish before test evaluation")
    if any(r["seed"] != seed for r in fitted.values()):
        raise ValueError("Fit result seed mismatch")
    lock = dict(seed=seed, created=utc_now(), test_used_for_selection=False, checkpoints={
        name: dict(best_epoch=r["best_epoch"], validation_macro_f1=r["validation"]["macro_f1"],
                   checkpoint_sha256=sha256_file(args.out / name / "best.pt")) for name, r in fitted.items()})
    selected = {}
    for cohort in cohorts:
        for family in ("easy", "hard"):
            candidates = [r for r in fitted.values()
                          if r["condition"]["cohort"] == cohort and r["condition"]["family"] == family]
            winner = max(candidates, key=lambda r: r["validation"]["macro_f1"])
            selected[f"{cohort}/{family}"] = winner["condition"]["name"]
    lock["best_by_validation"] = selected
    lock_path = args.out / "selection_before_test.json"
    if lock_path.exists():
        old_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if (old_lock.get("seed") != seed or old_lock["checkpoints"] != lock["checkpoints"]
                or old_lock["best_by_validation"] != selected):
            raise ValueError("Locked checkpoints changed")
    else:
        write_json(lock_path, lock)
    if args.phase == "fit":
        print(f"ALL_{total}_FITS_COMPLETE: validation selections locked, tests not evaluated", flush=True)
        return
    models = {}
    for cohort in cohorts:
        store = V5FeatureStore(args.prepared, bundle, cohort) if is_v5 else legacy_store
        folder = args.prepared / (shared_tables or cohort)
        for condition in conditions:
            if condition["cohort"] != cohort:
                continue
            name, destination = condition["name"], args.out / condition["name"]
            path = destination / "results.json"
            if args.resume and path.exists() and (destination / "test_predictions.tsv").exists():
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                model, checkpoint = load_decoder(destination / "best.pt", args.device)
                if (checkpoint["provenance"] != provenance or checkpoint["settings"] != settings
                        or checkpoint["seed"] != seed or checkpoint["condition"] != condition):
                    raise ValueError("Checkpoint provenance or settings mismatch")
                dataset = PairDataset(folder / f"{condition['family']}_test.tsv", store, condition["window_mode"])
                loader = DataLoader(dataset, batch_size=settings["batch_size"], num_workers=0,
                                    pin_memory=args.device == "cuda")
                test, prediction = evaluate(model, loader, args.device)
                write_predictions(prediction, destination / "test_predictions.tsv", store)
                if not is_v5:
                    prediction.loc[prediction.common_test.astype(str).str.lower().eq("true")].to_csv(
                        destination / "common_test_predictions.tsv", sep="\t", index=False)
                result = {**fitted[name], "test": test}
                write_json(path, result)
                del loader, dataset
            models[name] = result
            print(f"TEST {condition['number']:02d}/{total} {name}: F1={result['test']['macro_f1']:.4f} "
                  f"common_F1={result['test']['common_test']['macro_f1']:.4f}", flush=True)
        del store
    write_baselines(args.prepared, args.out, cohorts, shared_tables)
    comparisons = paired_common_comparison(models, args.out, cohorts)
    if is_v5:
        write_v5_results(models, args.out, selected)
    summary = dict(seed=seed, plan=plan, models=models, distinct_training_runs=total,
                   best_by_validation=selected, paired_common_test=comparisons,
                   preparation_manifest=str((args.prepared / "manifest.json").resolve()),
                   test_used_for_selection=False, historical_results_used=False,
                   total_training_seconds=sum(r["seconds"] for r in models.values()),
                   current_invocation_seconds=time.monotonic() - started, completed=utc_now())
    write_json(args.out / "summary.json", summary)
    print(f"SUITE_COMPLETE: {total} fits, {total} native tests and paired common-test evaluations", flush=True)


if __name__ == "__main__":
    main()
