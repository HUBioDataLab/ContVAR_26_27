"""Export tables from a completed suite without fitting or selecting on test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import DEFAULT_RUNS_DIR


def generate_report(suite: str | Path, out: str | Path) -> list[dict[str, Any]]:
    """Write all conditions and the candidates frozen before test evaluation.

    Historical records may be supplied as a JSON file; a live run is supplied
    as its directory. This command never loads model weights or refits a rule.
    """
    suite, out = Path(suite), Path(out)
    summary_path = suite / "summary.json" if suite.is_dir() else suite
    summary = json.loads(summary_path.read_text())
    if out.exists():
        raise FileExistsError(f"Choose a new report directory: {out}")
    rows = []
    for family, runs in summary["models"].items():
        for name, result in runs.items():
            base = f"{result['head']}_{result['pair_mode']}"
            selected = summary["selection_before_test"][family][base] == name
            rows.append(
                dict(
                    family=family,
                    model=base,
                    lambda_reverse=result["lambda_reverse"],
                    selected_by_validation=selected,
                    best_epoch=result["best_epoch"],
                    epochs_run=result["epochs_run"],
                    n_train=result["n_train"],
                    n_val=result["n_val"],
                    n_test=result["test"]["n"],
                    val_macro_f1=result["val"]["macro_f1"],
                    test_macro_f1=result["test"]["macro_f1"],
                    test_accuracy=result["test"]["accuracy"],
                    test_lof_f1=result["test"]["lof"]["f1"],
                    test_gof_f1=result["test"]["gof"]["f1"],
                    test_neutral_f1=result["test"]["neutral"]["f1"],
                    test_reverse_mse=result["test"]["reverse_mse"],
                    parameters=result["parameter_count"],
                )
            )
    if not rows:
        raise ValueError("No completed suite results found")
    out.mkdir(parents=True)
    for name, selected_only in (("all_runs.csv", False), ("selected_models.csv", True)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                row for row in rows if not selected_only or row["selected_by_validation"]
            )
    text = [
        "# Decoder experiment results",
        "",
        f"Seed: {summary['plan']['seed']}. Smoke: {summary['plan']['smoke']}. "
        f"Distinct fits: {summary['distinct_training_runs']}.",
        "",
        "Candidates below were selected using validation Macro-F1 before test evaluation. "
        "Joint lambda controls are in all_runs.csv. Score-difference reversal is structural; "
        "a small reverse error alone is not evidence of useful classification.",
        "",
        "| Family | Model | Lambda | Validation Macro-F1 | Test Macro-F1 | Test reverse MSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["selected_by_validation"]:
            text.append(
                f"| {row['family']} | {row['model']} | {row['lambda_reverse']:g} | "
                f"{row['val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} | "
                f"{row['test_reverse_mse']:.6g} |"
            )
    text.extend(
        [
            "",
            "This table describes the supplied record, not a new training run. "
            "One seed does not establish statistical superiority. Compare the train-only "
            "GO/protein-GO reference rules and mutation-sensitivity controls before attributing "
            "performance to mutation information. Parallel neutral scores are auxiliary "
            "decision scores, not a learned third probability.",
            "",
        ]
    )
    (out / "RESULTS.md").write_text("\n".join(text))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_RUNS_DIR / "full_seed42")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = generate_report(args.suite, args.out)
    print(f"Wrote {len(rows)} recorded conditions to {args.out}")


if __name__ == "__main__":
    main()
