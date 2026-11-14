"""
Phase 3 final analysis report (terminal).

Design: phase_3_variant_decoder/DESIGN.md — ADDENDUM 2 / R4, R6

Reads results.csv (or the wandb API with --from_wandb) and prints, in order:

    1  Leaderboard
    2  Grid table (embedding type x family x input_mode)
    3  ABLATION: input_mode — both vs global   <- the most important table
    4  easy vs hard generalisation gap
    5  Diagnostic breakdowns (unseen GO term / alanine)
    6  Class balance and the degenerate-solution indicator
    7  Margin health
    8  Hyperparameter sensitivity
    9  Phase 2 baseline comparison

With --scores it also runs the delta-signal diagnostic (phase 2's delta_analysis.py).

Missing data does not crash it: if only `both` runs exist, the ablation table shows
empty cells.

Usage:
    python -m phase_3_variant_decoder.report
    python -m phase_3_variant_decoder.report --scores runs/scores_hard_raw_esm_both.csv
    python -m phase_3_variant_decoder.report --from_wandb --project contvar_gridsearch
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from phase_3_variant_decoder.config import (
    EMBEDDING_TYPES, FAMILIES, INPUT_MODES, WANDB_ENTITY, WANDB_PROJECT,
)
from phase_3_variant_decoder.results import DEFAULT_PATH, load
from phase_3_variant_decoder.presentation_terms import ENCODER_LABELS, INPUT_LABELS

# Baseline for separating LOF/GOF via delta on phase 2's best F decoder.
PHASE2_BASELINE_AUROC = 0.54

HPARAM_COLS = ["margin", "w_gof", "w_neutral", "lr",
               "weight_decay", "dropout", "batch_size"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(row: dict, key: str):
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def _fmt(v, nd=4, width=8):
    return f"{'—':>{width}s}" if v is None else f"{v:>{width}.{nd}f}"


_SECTION = [0]


def _header(title: str):
    _SECTION[0] += 1
    print()
    print(f"{_SECTION[0]}. {title}")
    print("─" * max(60, len(title) + 6))


def _best(rows: list[dict], metric: str):
    scored = [(r, _f(r, metric)) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    return max(scored, key=lambda t: t[1])[0] if scored else None


def _group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k, "") for k in keys)].append(r)
    return out


# ── sections ─────────────────────────────────────────────────────────────────

def sec_leaderboard(rows, metric: str, limit: int):
    _header(f"LEADERBOARD — top {limit} runs ({metric})")
    scored = [(r, _f(r, metric)) for r in rows]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        print("  (no data)")
        return
    print(f"  {'#':>2s} {'family':<5s} {'embedding':<14s} {'mode':<7s} {'source':<10s} "
          f"{'ranking':>8s} {'test_F1':>8s} {'LOFGOF':>8s} {'mAUROC':>8s} {'AUPRC':>8s} {'neu_rec':>8s}  hyperparams")
    for i, (r, v) in enumerate(sorted(scored, key=lambda t: -t[1])[:limit], 1):
        hp = (f"m={_f(r,'margin') or 0:.2f} "
              f"wG={_f(r,'w_gof') or 0:.1f} wN={_f(r,'w_neutral') or 0:.1f} "
              f"lr={_f(r,'lr') or 0:.1e}")
        print(f"  {i:>2d} {r['family']:<5s} {r['embedding_type']:<14s} "
              f"{r['input_mode']:<7s} {r.get('source',''):<10s} "
              f"{_fmt(v)} {_fmt(_f(r,'test_macro_f1'))} {_fmt(_f(r,'test_lof_gof_auroc'))} "
              f"{_fmt(_f(r,'test_macro_auroc'))} "
              f"{_fmt(_f(r,'test_macro_auprc'))} {_fmt(_f(r,'test_neutral_recall'))}  {hp}")


def sec_grid(rows, metric: str):
    _header(f"GRID — best per cell ({metric})")
    modes = [m for m in INPUT_MODES if any(r["input_mode"] == m for r in rows)]
    cells = _group(rows, "embedding_type", "family", "input_mode")
    head = "  " + f"{'embedding':<15s}" + "".join(
        f"{fam[:4] + '/' + m[:4]:>12s}" for fam in FAMILIES for m in modes)
    print(head)
    for emb in EMBEDDING_TYPES:
        line = f"  {emb:<15s}"
        for fam in FAMILIES:
            for m in modes:
                b = _best(cells.get((emb, fam, m), []), metric)
                line += f"{_fmt(_f(b, metric) if b else None, 4, 12)}"
        print(line)


def sec_ablation(rows, metrics=("test_lof_gof_auroc", "test_macro_f1"), selection_metric="val_macro_f1"):
    _header("ABLATION: input_mode — the contribution of the local embedding")
    print("  Local denotes the single mutation-site residue embedding.")
    print(f"  Rows are selected by {selection_metric}; both - global is a descriptive paired contrast.")
    print()
    print("  Macro-F1 depends on the decision rule; LOF/GOF AUROC measures binary ranking.")
    print("  Neither alone diagnoses encoder quality. Compare dataset-specific no-training controls.")
    print()
    cells = _group(rows, "family", "embedding_type", "input_mode")
    hdr = f"  {'family':<5s} {'embedding':<15s}"
    for m in metrics:
        hdr += f"{m.replace('test_', '')[:13]:>27s}"
    print(hdr)
    print(f"  {'':<5s} {'':<15s}" + "".join(
        f"{'both':>9s}{'global':>9s}{'diff':>9s}" for _ in metrics))

    verdicts = []
    for fam in FAMILIES:
        for emb in EMBEDDING_TYPES:
            line = f"  {fam:<5s} {emb:<15s}"
            has = False
            for m in metrics:
                b = _best(cells.get((fam, emb, "both"), []), selection_metric)
                g = _best(cells.get((fam, emb, "global"), []), selection_metric)
                vb = _f(b, m) if b else None
                vg = _f(g, m) if g else None
                diff = (vb - vg) if (vb is not None and vg is not None) else None
                if vb is not None or vg is not None:
                    has = True
                if m == metrics[0] and diff is not None:
                    verdicts.append((fam, emb, diff))
                line += _fmt(vb, 4, 9) + _fmt(vg, 4, 9) + _fmt(diff, 4, 9)
            if has:
                print(line)

    print()
    if not verdicts:
        print("  (no comparable pair yet — runs are needed in both modes)")
        return
    mean_diff = float(np.mean([d for _, _, d in verdicts]))
    print(f"  Descriptive mean difference on {metrics[0]}: {mean_diff:+.4f} "
          f"({len(verdicts)} pairs)")
    print("  Confirm differences with matched configurations, repeated seeds and uncertainty estimates.")


def sec_family_gap(rows, metric: str):
    _header(f"easy vs hard — generalisation gap ({metric})")
    print("  Hard is protein-disjoint; easy permits protein overlap. Evaluation populations")
    print("  and label distributions differ; the gap alone does not quantify memorization.")
    print()
    cells = _group(rows, "embedding_type", "input_mode", "family")
    print(f"  {'embedding':<15s} {'mode':<7s} {'easy':>9s} {'hard':>9s} {'gap':>9s}")
    for emb in EMBEDDING_TYPES:
        for m in INPUT_MODES:
            e = _best(cells.get((emb, m, "easy"), []), metric)
            h = _best(cells.get((emb, m, "hard"), []), metric)
            ve, vh = (_f(e, metric) if e else None), (_f(h, metric) if h else None)
            if ve is None and vh is None:
                continue
            gap = (ve - vh) if (ve is not None and vh is not None) else None
            print(f"  {emb:<15s} {m:<7s} {_fmt(ve,4,9)} {_fmt(vh,4,9)} {_fmt(gap,4,9)}")


def best_per_architecture(rows, metric: str) -> list[dict]:
    """One row per (family, embedding_type, input_mode) — its best trial by `metric`.

    Sections that characterise an architecture (breakdowns, class balance, margin
    health) should read this, not the raw trial population: with a Bayesian sweep
    that's hundreds or thousands of trials per architecture, printing one line per
    trial is neither readable nor informative — the diagnostic is about the
    architecture's best result, same as sec_ablation/sec_grid already do via
    `_best()`. sec_hparam_sensitivity is the one exception that needs every trial.
    """
    out = []
    for grp in _group(rows, "family", "embedding_type", "input_mode").values():
        b = _best(grp, metric)
        if b is not None:
            out.append(b)
    return out


def sec_breakdowns(rows, metric: str):
    _header("DIAGNOSTIC BREAKDOWNS")
    base = metric.replace("test_", "")
    print("  seen: target GO ID occurs in train. Interpretation depends on decoder architecture:")
    print("  legacy has a separate GO output; cascade uses shared GO-conditioned weights.")
    print("  ala : is the mutant residue alanine (the alanine-scan shortcut)")
    print()
    print(f"  {'family':<5s} {'embedding':<14s} {'mode':<7s} "
          f"{'seen':>8s} {'seen_not':>9s} {'Δ':>8s} {'ala':>8s} {'ala_not':>8s} {'Δ':>8s}")
    for r in sorted(rows, key=lambda r: -(_f(r, metric) or -1)):
        vals = [_f(r, f"test_{bd}_{base}") for bd in ("seen", "seen_not", "ala", "ala_not")]
        if all(v is None for v in vals):
            continue
        d1 = (vals[0] - vals[1]) if None not in vals[:2] else None
        d2 = (vals[2] - vals[3]) if None not in vals[2:] else None
        print(f"  {r['family']:<5s} {r['embedding_type']:<14s} {r['input_mode']:<7s} "
              f"{_fmt(vals[0])} {_fmt(vals[1],4,9)} {_fmt(d1)} "
              f"{_fmt(vals[2])} {_fmt(vals[3])} {_fmt(d2)}")


def sec_class_balance(rows):
    _header("CLASS BALANCE — degenerate-solution indicator")
    print("  Inspect all class recalls and F1 values. Majority-class prediction can yield high")
    print("  accuracy while missing minority classes. Class proportions depend on the filtered split.")
    print()
    print(f"  {'family':<5s} {'embedding':<14s} {'mode':<7s} "
          f"{'lof_rec':>8s} {'gof_rec':>8s} {'neu_rec':>8s} {'lof_F1':>8s} "
          f"{'gof_F1':>8s} {'neu_F1':>8s}")
    for r in sorted(rows, key=lambda r: -(_f(r, "test_macro_f1") or -1)):
        print(f"  {r['family']:<5s} {r['embedding_type']:<14s} {r['input_mode']:<7s} "
              + " ".join(_fmt(_f(r, f"test_{c}_{m}"))
                         for m in ("recall", "f1") for c in ("lof", "gof", "neutral")))


def sec_margin(rows):
    _header("MARGIN HEALTH (DESIGN §8.7, single-margin revision)")
    print("  Training enforces |d| <= m for NEUTRAL and |d| >= m for LOF/GOF — one shared m")
    print("  (the earlier separate m_n was removed at the user's request). The half-width")
    print("  w/2 found on val is compared directly to m: since both targets meet exactly at")
    print("  d = m, w/2 landing close to m is the healthy case; far from it either way means")
    print("  the model's real separation doesn't match what the shared margin asked for.")
    print()
    print(f"  {'family':<5s} {'embedding':<14s} {'mode':<7s} {'w/2':>7s} "
          f"{'m':>7s} {'w/2÷m':>7s}  {'status':<28s} {'lof>m':>7s} {'neu<m':>7s}")
    for r in sorted(rows, key=lambda r: -(_f(r, "test_macro_f1") or -1)):
        m, w = _f(r, "margin"), _f(r, "width")
        if None in (m, w) or not m:
            continue
        half = w / 2
        ratio = half / m
        status = ("close to m (healthy)" if 0.7 <= ratio <= 1.3
                  else "band much narrower than m" if ratio < 0.7
                  else "band much wider than m")
        print(f"  {r['family']:<5s} {r['embedding_type']:<14s} {r['input_mode']:<7s} "
              f"{half:>7.3f} {m:>7.3f} {ratio:>7.2f}  {status:<28s} "
              f"{_fmt(_f(r,'test_lof_beyond_margin'),3,7)} "
              f"{_fmt(_f(r,'test_neutral_within_margin'),3,7)}")


def sec_hparam_sensitivity(rows, metric: str, min_rows: int = 12):
    _header("HYPERPARAMETER SENSITIVITY")
    if len(rows) < min_rows:
        print(f"  (only {len(rows)} runs; >={min_rows} needed for a meaningful correlation)")
        return
    from scipy.stats import spearmanr

    y = np.array([_f(r, metric) for r in rows], dtype=float)
    ok = ~np.isnan(y)
    print("  Spearman correlation + the range of the top 10%.")
    print("  The Bayesian counterpart of phase 2's staged leaderboard:")
    print("  which axis actually makes a difference.")
    print()
    top_n = max(3, int(ok.sum() * 0.10))
    top_idx = np.argsort(-np.nan_to_num(y, nan=-1))[:top_n]
    print(f"  {'parameter':<15s} {'rho':>7s} {'p':>8s}  {'top-10% range':<26s}")
    for col in HPARAM_COLS:
        x = np.array([_f(r, col) if _f(r, col) is not None else np.nan for r in rows])
        m = ok & ~np.isnan(x)
        if m.sum() < min_rows or len(set(x[m])) < 3:
            continue
        rho, pval = spearmanr(x[m], y[m])
        sel = x[top_idx]
        sel = sel[~np.isnan(sel)]
        rng = f"{sel.min():.4g} – {sel.max():.4g}" if len(sel) else "—"
        flag = " ←" if pval < 0.05 else ""
        print(f"  {col:<15s} {rho:>7.3f} {pval:>8.3g}  {rng:<26s}{flag}")


def sec_phase2(rows, selection_metric="val_macro_f1"):
    _header("HISTORICAL PHASE 2 CONTEXT (not a controlled comparison)")
    best = _best(rows, selection_metric)
    if best is None:
        print("  (no data)")
        return
    v = _f(best, "test_lof_gof_auroc")
    print(f"  phase 2 delta baseline               {PHASE2_BASELINE_AUROC:.4f}")
    print("  Historical score uses a different protocol/population; do not infer improvement.")
    if v is None:
        print("  No test LOF/GOF AUROC for the validation-selected row.")
        return
    print(f"  phase 3 validation-selected row      {v:.4f}   "
          f"({best['family']}/{best['embedding_type']}/{best['input_mode']})")


def sec_delta_signal(scores_path: str):
    """R6 — the counterpart of phase 2's delta_analysis.py."""
    import pandas as pd

    _header(f"DELTA SIGNAL DIAGNOSTIC — {Path(scores_path).name}")
    d = pd.read_csv(scores_path)
    for split in ("val", "test"):
        sub = d[d["split"] == split]
        if sub.empty:
            continue
        print(f"\n  [{split.upper()}]  n={len(sub):,}")
        print(f"  {'class':<9s} {'n':>6s} {'d median':>10s} {'IQR':>18s} "
              f"{'|d|<1e-6':>9s}")
        for lab in ("LOF", "GOF", "NEUTRAL"):
            s = sub[sub["label"] == lab]["d"]
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            dead = float((s.abs() < 1e-6).mean())
            print(f"  {lab:<9s} {len(s):>6,} {s.median():>10.4f} "
                  f"{f'[{q1:+.3f}, {q3:+.3f}]':>18s} {dead:>9.3f}")
        dead_all = float((sub["d"].abs() < 1e-6).mean())
        if dead_all > 0.10:
            print(f"  WARNING: |d|<1e-6 for {dead_all*100:.1f}% of rows — the signal is "
                  "largely DEAD: the model does not distinguish WT from variant.")
        # overlap between class pairs
        lof = sub[sub["label"] == "LOF"]["d"]
        gof = sub[sub["label"] == "GOF"]["d"]
        if len(lof) and len(gof):
            ov = float(((gof > lof.quantile(0.25)) & (gof < lof.quantile(0.75))).mean())
            print(f"  fraction of GOF falling inside LOF's IQR: {ov:.3f}  (lower = better separation)")


def load_from_wandb(project: str, entity: str | None) -> list[dict]:
    """Pull from wandb if results.csv was deleted (phase 2's best_runs_report fallback)."""
    import wandb

    api = wandb.Api()
    entity = entity or api.default_entity
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"tags": "phase3"}):
        row = {**{k: v for k, v in run.config.items() if not k.startswith("_")}}
        row.update({k.replace("best_", "", 1): v for k, v in run.summary.items()
                    if k.startswith("best_")})
        row.setdefault("family", ""); row.setdefault("embedding_type", "")
        row.setdefault("input_mode", "both"); row.setdefault("source", "sweep")
        row["wandb_url"] = run.url
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description="Phase 3 analysis report")
    p.add_argument("--results", default=str(DEFAULT_PATH))
    p.add_argument("--from_wandb", action="store_true")
    p.add_argument("--project", default=WANDB_PROJECT)
    p.add_argument("--entity", default=WANDB_ENTITY)
    p.add_argument("--metric", default="val_macro_f1",
                   help="selection/ranking metric; defaults to validation macro-F1")
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--scores", default=None, help="per-row score dump (R6 diagnostic)")
    p.add_argument("--preset", default=None,
                   help="only rows from this make_sweeps.py preset (lite/full/manual/collect). "
                        "results.csv is one cumulative file shared by every batch of runs ever "
                        "done, so without this a report can silently mix an old quick scan in "
                        "with the real search. Rows written before the preset column existed "
                        "have preset='' — use --since for those instead.")
    p.add_argument("--since", default=None,
                   help="only rows with timestamp >= this (YYYY-MM-DD[ HH:MM:SS]); "
                        "for filtering out pre-preset-column rows by date")
    args = p.parse_args()

    rows = (load_from_wandb(args.project, args.entity) if args.from_wandb
            else load(args.results))
    if not rows:
        raise SystemExit(f"[ERROR] no results found: "
                         f"{'wandb' if args.from_wandb else args.results}")

    if args.preset is not None:
        rows = [r for r in rows if r.get("preset", "") == args.preset]
    if args.since is not None:
        rows = [r for r in rows if r.get("timestamp", "") >= args.since]
    if not rows:
        raise SystemExit("[ERROR] no rows left after --preset/--since filtering")

    src = "wandb" if args.from_wandb else args.results
    filt = "".join(f" · {k}={v}" for k, v in
                   (("preset", args.preset), ("since", args.since)) if v is not None)
    print("=" * 78)

    print("Encoder identifiers (training history; '+' means feature concatenation):")
    for key, label in ENCODER_LABELS.items():
        print(f"  {key}: {label}")
    print("Input identifiers:")
    for key, label in INPUT_LABELS.items():
        print(f"  {key}: {label}")
    if args.metric.startswith("test_"):
        print("[DESCRIPTIVE ONLY] Ranking by test scores is not valid model selection.")
    print(f"  PHASE 3 ANALYSIS REPORT   ·   {len(rows)} runs   ·   source: {src}{filt}")
    print("=" * 78)

    sec_leaderboard(rows, args.metric, args.limit)
    sec_grid(rows, args.metric)
    sec_ablation(rows, selection_metric=args.metric)
    sec_family_gap(rows, args.metric)
    arch_best = best_per_architecture(rows, args.metric)
    sec_breakdowns(arch_best, args.metric)
    sec_class_balance(arch_best)
    sec_margin(arch_best)
    sec_hparam_sensitivity(rows, args.metric)
    sec_phase2(rows, selection_metric=args.metric)
    if args.scores:
        sec_delta_signal(args.scores)
    print()


if __name__ == "__main__":
    main()
