"""
Phase 3 evaluation: val/test metrics from a checkpoint + family comparison.

Design: phase_3_variant_decoder/DESIGN.md §5

Thresholds are searched on VAL and applied unchanged to TEST. The checkpoint carries
the (centre, width) found during training; --research_thresholds re-searches them
(still on val only).

Usage:
    python -m phase_3_variant_decoder.evaluate \\
        --checkpoint phase_3_variant_decoder/runs/decoder_hard_raw_esm.pt \\
        --global_h5 ... --local_h5 ... --cuda 0

    # several runs side by side:
    python -m phase_3_variant_decoder.evaluate --compare phase_3_variant_decoder/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase_2_decoder.config import pick_device
from phase_2_decoder.models.ffn_decoder import FFNDecoder
from phase_3_variant_decoder.config import SECONDARY_REQUIRED, VariantDecoderConfig
from phase_3_variant_decoder.data.pair_dataset import build_pair_tensors
from phase_3_variant_decoder.data.variant_dataset import prepare
from phase_3_variant_decoder.metrics import search_thresholds
from phase_3_variant_decoder.metrics import predict_three_class
from phase_3_variant_decoder.train import (
    compute_deltas, compute_pair_scores, evaluate_split,
)

REPORT_KEYS = [
    "macro_f1", "macro_auprc", "lof_gof_auroc", "accuracy",
    "lof_f1", "gof_f1", "neutral_f1",
    "lof_recall", "gof_recall", "neutral_recall",
]


def run(args) -> dict:
    device = torch.device(pick_device(args.cuda))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("format") == "contvar_go_scalar_cascade_v1":
        from phase_3_variant_decoder.cascade import evaluate_checkpoint
        return evaluate_checkpoint(args)

    cfg = VariantDecoderConfig(family=ckpt["family"], embedding_type=ckpt["embedding_type"])
    cfg.input_mode = ckpt.get("input_mode", "both")
    for name in ("dataset_dir", "sequences_json"):
        if getattr(args, name, None):
            setattr(cfg, name, getattr(args, name))
    cfg.global_h5 = args.global_h5
    cfg.local_h5 = args.local_h5
    cfg.global_h5_secondary = args.global_h5_secondary
    cfg.local_h5_secondary = args.local_h5_secondary
    cfg.cuda = args.cuda
    cfg.validate()

    df, go_vocab, report = prepare(cfg)
    if len(go_vocab) != ckpt["n_classes"]:
        raise ValueError(
            f"vocabulary size mismatch: data {len(go_vocab)} vs checkpoint {ckpt['n_classes']}. "
            "It may have been trained on a different family or filter setting."
        )
    print(report.render())

    tensors = build_pair_tensors(df, cfg).to(device)
    views = {s: tensors.split_view(s) for s in ("val", "test")}

    model = FFNDecoder(
        input_dim=ckpt["input_dim"], n_classes=ckpt["n_classes"],
        hidden_dims=ckpt["hidden_dims"], dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.research_thresholds or ckpt.get("center") is None:
        d_val = compute_deltas(model, tensors, views["val"])
        y_val = views["val"]["label_idx"].cpu().numpy()
        center, width, _ = search_thresholds(d_val, y_val)
        print(f"[INFO] thresholds re-searched on val: c={center:+.4f} w={width:.4f}")
    else:
        center, width = ckpt["center"], ckpt["width"]
        print(f"[INFO] checkpoint thresholds: c={center:+.4f} w={width:.4f}")

    out = {"family": cfg.family, "embedding_type": cfg.embedding_type,
           "center": center, "width": width}
    for split in ("val", "test"):
        out.update({f"{split}_{k}": v for k, v in
                    evaluate_split(model, tensors, views[split], center, width, cfg).items()})

    print()
    print(f"{'metric':<20s} {'val':>10s} {'test':>10s}")
    print("-" * 42)
    for k in REPORT_KEYS:
        print(f"{k:<20s} {out.get(f'val_{k}', float('nan')):>10.4f} "
              f"{out.get(f'test_{k}', float('nan')):>10.4f}")

    print()
    print("diagnostic breakdowns (DESIGN §5.2) — test:")
    for pref, desc in (("seen", "GO term seen in train"), ("ala", "alanine mutation")):
        a = out.get(f"test_{pref}_macro_f1")
        b = out.get(f"test_{pref}_not_macro_f1")
        na = out.get(f"test_{pref}_n", 0)
        nb = out.get(f"test_{pref}_not_n", 0)
        if a is not None and b is not None:
            print(f"  {desc:<28s} yes={a:.4f} (n={na:,})   no={b:.4f} (n={nb:,})   "
                  f"diff={a - b:+.4f}")

    if args.dump_scores:
        _dump_scores(model, tensors, views, center, width, args.dump_scores)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[DONE] → {args.out_json}")
    return out


def _dump_scores(model, tensors, views, center, width, path: str):
    """Per-row score dump (R5) — the counterpart of phase 2's benchmark_*_scores.csv.

    For error analysis: which GO terms / proteins we get wrong. report.py --scores
    also uses it for the delta-signal diagnostic.
    """
    import csv as _csv

    from phase_3_variant_decoder.config import LABELS

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["split", "protein_id", "snp", "matched_go_id", "matched_go_name", "label",
            "pred", "s_wt", "s_var", "d", "correct",
            "seen_in_train", "is_alanine", "is_combinatorial"]
    n_total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for split, view in views.items():
            s_wt, s_var = compute_pair_scores(model, tensors, view)
            d = s_wt - s_var
            pred = predict_three_class(d, center, width)
            sub = tensors.df[tensors.df["split"] == split].reset_index(drop=True)
            for i, r in enumerate(sub.itertuples()):
                w.writerow({
                    "split": split, "protein_id": r.protein_id, "snp": r.snp,
                    "matched_go_id": r.matched_go_id, "matched_go_name": r.matched_go_name,
                    "label": r.label, "pred": LABELS[int(pred[i])],
                    "s_wt": round(float(s_wt[i]), 6), "s_var": round(float(s_var[i]), 6),
                    "d": round(float(d[i]), 6),
                    "correct": int(LABELS[int(pred[i])] == r.label),
                    "seen_in_train": int(r.seen_in_train),
                    "is_alanine": int(r.is_alanine),
                    "is_combinatorial": int(bool(r.is_combinatorial)),
                })
                n_total += 1
    print(f"[DONE] dumped {n_total:,} scored rows -> {out}")


def compare(run_dir: str):
    """Collect the best_*.json files under runs/ into a single table."""
    rows = []
    for p in sorted(Path(run_dir).glob("best_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append((p.stem.replace("best_", ""), d))
    if not rows:
        print(f"[WARN] no best_*.json under {run_dir}")
        return
    print(f"{'run':<28s} {'ep':>4s} {'val_F1':>8s} {'test_F1':>8s} "
          f"{'test_AUPRC':>11s} {'test_LOFGOF':>12s} {'neu_rec':>8s}")
    print("-" * 84)
    for name, d in rows:
        print(f"{name:<28s} {d.get('epoch', -1):>4d} "
              f"{d.get('val_macro_f1', float('nan')):>8.4f} "
              f"{d.get('test_macro_f1', float('nan')):>8.4f} "
              f"{d.get('test_macro_auprc', float('nan')):>11.4f} "
              f"{d.get('test_lof_gof_auroc', float('nan')):>12.4f} "
              f"{d.get('test_neutral_recall', float('nan')):>8.3f}")


def main():
    p = argparse.ArgumentParser(description="Phase 3 evaluation")
    p.add_argument("--compare", default=None, help="print the runs/ directory as a table and exit")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--sequences_json", default=None)
    p.add_argument("--global_h5", default=None)
    p.add_argument("--local_h5", default=None)
    p.add_argument("--global_h5_secondary", default=None)
    p.add_argument("--local_h5_secondary", default=None)
    p.add_argument("--research_thresholds", action="store_true",
                   help="re-search thresholds on val (test is still never touched)")
    p.add_argument("--dump_scores", default=None,
                   help="path for the per-row score CSV (error analysis + report.py --scores)")
    p.add_argument("--out_json", default=None)
    p.add_argument("--cuda", default=None)
    args = p.parse_args()

    if args.compare:
        compare(args.compare)
        return
    missing = [n for n in ("checkpoint", "global_h5", "local_h5") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"required unless --compare is given: {', '.join(missing)}")
    run(args)


if __name__ == "__main__":
    main()
