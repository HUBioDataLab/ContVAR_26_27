"""
Per-run persistent result log (append-only CSV).

Design: phase_3_variant_decoder/DESIGN.md — ADDENDUM 2 / R2

Same pattern as phase 2's grid_search._append_csv: one row is written the moment
a run finishes. Crash-resistant — if a sweep is interrupted, everything up to that
point is already on disk. The schema lives in one place (RESULT_FIELDS), mirroring
phase 2's CSV_FIELDS.

train.py, sweep_train.py and apply_best.py all write the same row through fit().
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

from phase_3_variant_decoder.config import LABELS

SPLITS = ("val", "test")
CLASSES = tuple(l.lower() for l in LABELS)          # lof / gof / neutral
BREAKDOWNS = ("seen", "seen_not", "ala", "ala_not")

IDENTITY = [
    "run_id", "timestamp", "family", "embedding_type", "input_mode", "source",
    "preset", "sweep_id", "wandb_run_id", "wandb_url", "checkpoint",
]
HPARAMS = [
    # `margin_ratio`/`margin_neutral` existed while the loss used a separate
    # NEUTRAL margin; removed at the user's request (single shared `margin` now).
    # Old rows (pre-revision) still have those two columns; new rows won't.
    "margin", "w_lof", "w_gof", "w_neutral",
    "lr", "weight_decay", "dropout", "batch_size", "hidden_dims", "epochs",
    "patience", "seed",
]
DATA = ["n_train", "n_val", "n_test", "n_classes", "input_dim"]
SELECTION = ["best_epoch", "center", "width"]


def _metric_fields() -> list[str]:
    out = []
    for split in SPLITS:
        out += [f"{split}_{k}" for k in
                ("macro_f1", "macro_auprc", "macro_auroc", "accuracy", "lof_gof_auroc")]
        for cls in CLASSES:
            out += [f"{split}_{cls}_{m}" for m in
                    ("precision", "recall", "f1", "auprc", "auroc", "support")]
        for bd in BREAKDOWNS:
            out += [f"{split}_{bd}_{m}" for m in
                    ("n", "macro_f1", "macro_auprc", "macro_auroc", "lof_gof_auroc")]
        for cls in CLASSES:
            out.append(f"{split}_{cls}_abs_d_median")
        out += [f"{split}_lof_beyond_margin", f"{split}_gof_beyond_margin",
                f"{split}_neutral_within_margin"]
    return out


RESULT_FIELDS = IDENTITY + HPARAMS + DATA + SELECTION + _metric_fields()

# Renamed for the single-margin revision — the old results.csv was written under
# the two-margin schema (margin_ratio/margin_neutral columns) and append() now
# refuses to write the new schema's rows into it (see append()'s docstring).
DEFAULT_PATH = Path("phase_3_variant_decoder/runs/results_single_margin.csv")


def make_run_id(cfg, best: dict) -> str:
    """Deterministic short id — identical settings produce the same id twice."""
    payload = "|".join(str(x) for x in (
        cfg.family, cfg.embedding_type, cfg.input_mode, cfg.margin,
        cfg.w_gof, cfg.w_neutral, cfg.lr, cfg.weight_decay,
        cfg.dropout, cfg.batch_size, cfg.hidden_dims, cfg.seed,
        best.get("epoch", ""),
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def build_row(cfg, best: dict, n_rows: dict, n_classes: int,
              checkpoint: str = "", wandb_run=None) -> dict:
    """Build the full CSV row from fit()'s `best` dict.

    Any field absent from `best` stays an empty string — the schema is fixed and
    every row has the same number of columns.
    """
    row = {k: "" for k in RESULT_FIELDS}

    row.update({
        "run_id": make_run_id(cfg, best),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "family": cfg.family,
        "embedding_type": cfg.embedding_type,
        "input_mode": cfg.input_mode,
        "source": getattr(cfg, "source", "train"),
        "preset": getattr(cfg, "preset", ""),
        "sweep_id": getattr(cfg, "sweep_id", ""),
        "checkpoint": checkpoint,
        "margin": cfg.margin,
        "w_lof": cfg.w_lof, "w_gof": cfg.w_gof, "w_neutral": cfg.w_neutral,
        "lr": cfg.lr, "weight_decay": cfg.weight_decay, "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "hidden_dims": "-".join(str(x) for x in cfg.hidden_dims),
        "epochs": cfg.epochs, "patience": cfg.early_stop_patience, "seed": cfg.seed,
        "n_train": n_rows.get("train", ""), "n_val": n_rows.get("val", ""),
        "n_test": n_rows.get("test", ""),
        "n_classes": n_classes, "input_dim": cfg.input_dim,
        "best_epoch": best.get("epoch", ""),
        "center": best.get("center", ""), "width": best.get("width", ""),
    })

    if wandb_run is not None:
        row["wandb_run_id"] = getattr(wandb_run, "id", "")
        try:
            row["wandb_url"] = wandb_run.get_url() or ""
        except Exception:
            row["wandb_url"] = ""

    for key in _metric_fields():
        if key in best:
            row[key] = best[key]
    return row


def append(row: dict, path: str | Path = DEFAULT_PATH) -> Path:
    """Append the row; create the file with a header if it does not exist.

    BUG FOUND & FIXED HERE: DictWriter writes columns in the *fieldnames* order
    passed to it (RESULT_FIELDS) — it does not read the file's own existing
    header row. If RESULT_FIELDS ever changes (a column added/removed/reordered
    — e.g. the single-margin revision dropping `margin_ratio`/`margin_neutral`)
    and you append to a file written under the *old* schema, the header line
    stays old but the new rows' values land in the new column order: every
    downstream reader misaligns those rows against the wrong header, silently.
    (This already happened once — the live results.csv's header predates the
    `preset` column, so `preset` has never actually been populated in it.)
    Refusing beats corrupting: if the file's own header doesn't match
    RESULT_FIELDS, stop and ask for a fresh path instead of appending blind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        with path.open(newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != RESULT_FIELDS:
            raise SystemExit(
                f"[ERROR] {path} was written under a different results.csv schema "
                f"({len(existing_header)} columns vs {len(RESULT_FIELDS)} now) — "
                "appending would silently misalign columns. Point --results_csv at "
                "a fresh path for this schema (e.g. runs/results_v2.csv) instead."
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def load(path: str | Path = DEFAULT_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
