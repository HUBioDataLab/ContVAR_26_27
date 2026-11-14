"""
Report the single best run per (aspect, hidden_dims, embedding_type) group.

Primary source is a CSV produced by `grid_search.py --output <path>`. If
--csv is omitted or the file is missing/empty, falls back to querying WandB
directly (parsing run names with the same pattern wandb_tag_runs.py uses).

Within each group, the winner is the row with the highest val_mAP across
ALL stages (stage1/stage2/stage3/final) — no stage filtering.

Usage:
    python -m phase_2_decoder.best_runs_report --csv grid_runs/results.csv
    python -m phase_2_decoder.best_runs_report --csv grid_runs/results.csv --output best.csv
    python -m phase_2_decoder.best_runs_report --wandb-project ContVAR-Sekans --no-link
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from .config import DecoderConfig
    from .grid_search import CSV_FIELDS, RunParams, _run_id
    from .wandb_tag_runs import RUN_NAME_RE
    from .benchmark_eval import BM_METRIC_KEYS
    from .evaluate import FINAL_METRIC_KEYS
except ImportError:
    from config import DecoderConfig
    from grid_search import CSV_FIELDS, RunParams, _run_id
    from wandb_tag_runs import RUN_NAME_RE
    from benchmark_eval import BM_METRIC_KEYS
    from evaluate import FINAL_METRIC_KEYS

REPORT_FIELDS = [
    "aspect", "hidden_dims", "embedding_type", "stage", "best_epoch",
    "val_mAP", "test_mAP", "test_MCC",
    *[f"bm_val_{key}" for key in BM_METRIC_KEYS],
    *[f"bm_test_{key}" for key in BM_METRIC_KEYS],
    *[f"final_val_{key}" for key in FINAL_METRIC_KEYS],
    *[f"final_test_{key}" for key in FINAL_METRIC_KEYS],
    "lr", "dropout", "weight_decay", "pos_weight_clamp",
    "wandb_url",
]

GROUP_KEYS = ("aspect", "hidden_dims", "embedding_type")


def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows_from_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_rows_from_wandb(api, project: str, entity: str | None) -> list[dict]:
    entity = entity or api.default_entity
    path = f"{entity}/{project}"
    rows = []
    for run in api.runs(path):
        m = RUN_NAME_RE.match(run.name)
        if not m:
            continue
        g = m.groupdict()
        summary = run.summary
        val_map = summary.get("best_mAP")
        if val_map is None:
            continue
        rows.append({
            "tag": g["tag"],
            "stage": g["stage"],
            "aspect": g["aspect"].upper(),
            "embedding_type": g["embedding"],
            "hidden_dims": g["hidden"],
            "lr": summary.get("train/lr", ""),
            "dropout": "",
            "weight_decay": "",
            "pos_weight_clamp": "",
            "best_epoch": summary.get("best_epoch", ""),
            "val_mAP": val_map,
            # Best-effort only: these are the last-logged values, not
            # necessarily from the best (val_mAP-maximizing) epoch, since
            # train.py only pins val metrics into run.summary at best-epoch.
            "test_mAP": summary.get("test/mAP", ""),
            "test_F1_macro": summary.get("test/F1_macro", ""),
            "test_MCC": summary.get("test/MCC", ""),
            # Same last-epoch caveat as above — train.py only writes
            # benchmark_*/... via wandb.log() each epoch, never pins it
            # into run.summary at the best epoch.
            **{f"bm_val_{key}": summary.get(f"benchmark_val/{key}", "") for key in BM_METRIC_KEYS},
            **{f"bm_test_{key}": summary.get(f"benchmark_test/{key}", "") for key in BM_METRIC_KEYS},
            # Same caveat again — final_val/final_test/... is logged every
            # epoch via wandb.log(), never pinned into run.summary.
            **{f"final_val_{key}": summary.get(f"final_val/{key}", "") for key in FINAL_METRIC_KEYS},
            **{f"final_test_{key}": summary.get(f"final_test/{key}", "") for key in FINAL_METRIC_KEYS},
            "_wandb_url": run.url,
        })
    return rows


def pick_winners(rows: list[dict]) -> dict[tuple, dict]:
    """Winner per (aspect, hidden_dims, embedding_type) group.

    A stage="final" row (the full-epoch rerun with benchmark/final-metric
    eval) always wins over any stage1/2/3 row in the same group, since it's
    the authoritative, fully-evaluated run — even if an earlier stage
    happened to log a marginally higher val_mAP. Only when a group has no
    final row do we fall back to the highest val_mAP among its other stages.
    """
    best_final: dict[tuple, dict] = {}
    best_other: dict[tuple, dict] = {}
    for row in rows:
        val_map = _to_float(row.get("val_mAP"))
        if val_map is None:
            continue
        key = tuple(row[k] for k in GROUP_KEYS)
        bucket = best_final if row.get("stage") == "final" else best_other
        current = bucket.get(key)
        if current is None or val_map > _to_float(current["val_mAP"]):
            bucket[key] = row
    winners = dict(best_other)
    winners.update(best_final)
    return winners


def resolve_wandb_url(api, project: str, entity: str | None, row: dict) -> tuple[str, str | None]:
    """Returns (url, failure_reason). failure_reason is None on success."""
    if "_wandb_url" in row:
        return row["_wandb_url"] or "", None
    try:
        params = RunParams(
            lr=float(row["lr"]),
            dropout=float(row["dropout"]),
            hidden_dims=tuple(int(x) for x in row["hidden_dims"].split("-")),
            weight_decay=float(row["weight_decay"]),
            pos_weight_clamp=float(row["pos_weight_clamp"]),
        )
        run_id = _run_id(row["stage"], params)
        name = f"decoder-grid-{row['tag']}-{row['aspect'].lower()}-{row['embedding_type']}-{run_id}"
    except (KeyError, ValueError) as e:
        return "", f"couldn't rebuild run name ({e})"

    entity = entity or api.default_entity
    path = f"{entity}/{project}"
    try:
        matches = api.runs(path, filters={"displayName": name})
        for run in matches:
            return run.url, None
    except Exception as e:
        return "", f"WandB API error: {e}"
    return "", f"no run named {name!r} found in {path}"


def _fmt(value) -> str:
    f = _to_float(value)
    return f"{f:.4f}" if f is not None else "-"


_BM_INT_KEYS = {"n_total", "n_lof", "n_gof"}
_BM_THRESH_KEYS = {"gof_threshold"}


def _fmt_bm(value, key: str) -> str:
    f = _to_float(value)
    if f is None:
        return "-"
    if key in _BM_INT_KEYS:
        return str(int(f))
    if key in _BM_THRESH_KEYS:
        return f"{f:.2f}"
    return f"{f:.4f}"


_FINAL_THRESH_KEYS = {"threshold"}


def _fmt_final(value, key: str) -> str:
    f = _to_float(value)
    if f is None:
        return "-"
    if key in _FINAL_THRESH_KEYS:
        return f"{f:.2f}"
    return f"{f:.4f}"


def _arch_label(row) -> str:
    """Architecture the run belongs to: aspect + embedding_type + hidden layer
    sizes, e.g. 'c_concat_phase0_512-512'."""
    return (
        f"{str(row.get('aspect', '')).lower()}_"
        f"{row.get('embedding_type', '')}_"
        f"{row.get('hidden_dims', '')}"
    )


# Deliberately compact. val_mAP is the ONLY val-side point metric (it's the
# selection criterion); everything else is test-only.
_BASE_COLS = [("architecture", 30), ("epoch", 5), ("val_mAP", 8), ("test_mAP", 8), ("test_MCC", 8)]
_URL_COL = [("wandb_url", 78)]


def _bm_test_cols():
    """bm_{key}_test — measured on the curated LOF/GOF variant benchmark
    dataset (variant embeddings), NOT the main GOA val/test split. Test split
    only."""
    cols = []
    for key in BM_METRIC_KEYS:
        label = f"bm_{key}_test"
        cols.append((label, max(10, len(label) + 1)))
    return cols


def _final_cols():
    """Main GOA test split at the F1-optimized threshold (see
    find_best_f1_threshold), not the fixed eval_threshold used by
    val_mAP/test_mAP/test_MCC. final_threshold is shared by val/test, so it's a
    single column; the rest are test-only."""
    cols = [("final_threshold", 16)]
    for key in FINAL_METRIC_KEYS:
        if key == "threshold":
            continue
        label = f"final_{key}_test"
        cols.append((label, max(10, len(label) + 1)))
    return cols


def _print_table(winners: list[dict]):
    # Only the full final-stage reruns carry benchmark/final-threshold metrics;
    # a stage="final" run counts even if early stopping cut it short of the full
    # epoch budget. Groups whose winner is a stage1/2/3 run (no final rerun
    # exists) are omitted from the printed table.
    winners = [r for r in winners if r.get("stage") == "final"]
    if not winners:
        print("No final-stage winning rows found.")
        return
    columns = _BASE_COLS + _bm_test_cols() + _final_cols() + _URL_COL
    header = " ".join(f"{label:<{width}}" for label, width in columns)
    print(header)
    print("-" * len(header))
    for row in winners:
        best_epoch = row.get("best_epoch", "")
        values = {
            "architecture": _arch_label(row),
            "epoch": str(best_epoch) if best_epoch != "" else "?",
            "val_mAP": _fmt(row.get("val_mAP")),
            "test_mAP": _fmt(row.get("test_mAP")),
            "test_MCC": _fmt(row.get("test_MCC")),
            "wandb_url": row.get("wandb_url", ""),
        }
        for key in BM_METRIC_KEYS:
            values[f"bm_{key}_test"] = _fmt_bm(row.get(f"bm_test_{key}"), key)
        # final_threshold is the same for val and test (test reuses the
        # val-optimized threshold); prefer test's, fall back to val's.
        thr = row.get("final_test_threshold", "") or row.get("final_val_threshold", "")
        values["final_threshold"] = _fmt_final(thr, "threshold")
        for key in FINAL_METRIC_KEYS:
            if key == "threshold":
                continue
            values[f"final_{key}_test"] = _fmt_final(row.get(f"final_test_{key}"), key)
        print(" ".join(f"{values[label]:<{width}}" for label, width in columns))


def _write_csv(path: Path, winners: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in winners:
            writer.writerow({k: row.get(k, "") for k in REPORT_FIELDS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None,
                         help="grid_search.py --output CSV (primary source)")
    parser.add_argument("--wandb-project", default=DecoderConfig().wandb_project,
                         help="WandB project (used as fallback source, and for --link lookups)")
    parser.add_argument("--entity", default=None, help="WandB entity (default: your default entity)")
    parser.add_argument("--output", type=Path, default=None, help="Write winners to this CSV path")
    parser.add_argument("--no-link", action="store_true", help="Skip WandB URL lookup for winners")
    args = parser.parse_args()

    rows = load_rows_from_csv(args.csv) if args.csv else []
    source = "csv"
    api = None
    if not rows:
        if args.csv:
            print(f"[WARN] {args.csv} missing/empty, falling back to WandB.")
        import wandb
        api = wandb.Api()
        rows = load_rows_from_wandb(api, args.wandb_project, args.entity)
        source = "wandb"
    print(f"[INFO] Loaded {len(rows)} candidate rows from {source}.")

    winners_by_key = pick_winners(rows)
    all_groups = {tuple(r[k] for k in GROUP_KEYS) for r in rows}
    missing = all_groups - set(winners_by_key)
    if missing:
        print(f"[WARN] {len(missing)} group(s) had no usable val_mAP and were skipped:")
        for key in sorted(missing):
            print(f"    - aspect={key[0]} hidden_dims={key[1]} embedding_type={key[2]}")

    winners = sorted(winners_by_key.values(), key=lambda r: _to_float(r["val_mAP"]), reverse=True)

    if not args.no_link:
        if api is None:
            import wandb
            api = wandb.Api()
        link_failures = []
        for row in winners:
            url, reason = resolve_wandb_url(api, args.wandb_project, args.entity, row)
            row["wandb_url"] = url
            if reason:
                link_failures.append((row["aspect"], row["hidden_dims"], row["embedding_type"], reason))
        if link_failures:
            print(f"[WARN] Couldn't resolve a WandB link for {len(link_failures)}/{len(winners)} winner(s):")
            for aspect, hd, emb, reason in link_failures[:10]:
                print(f"    - aspect={aspect} hidden_dims={hd} embedding_type={emb}: {reason}")
            if len(link_failures) > 10:
                print(f"    ... and {len(link_failures) - 10} more")
    else:
        for row in winners:
            row["wandb_url"] = ""

    _print_table(winners)
    if args.output:
        _write_csv(args.output, winners)
        print(f"\n[INFO] Wrote {len(winners)} rows to {args.output}")


if __name__ == "__main__":
    main()
