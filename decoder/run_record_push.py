#!/usr/bin/env python3
"""ContVAR Fast Record Push (6 runs total, ~7-8 min on RTX 4060).

Combines the exact winning ingredients discovered in Phase 1 + Phase 2-3:
  - Phase 1 Winner: trained_distance + ESIM (without rel_pos, which beat rel_pos by +0.86)
  - Sweet-spot light regularization (adapter_drop=0.15, attn_drop=0.10, drop=0.15, WD=3e-3, LS=0.05, Cosine)
  - Nihat Encoder 2: 15 protein-invariant biophysical features
  - Nihat Antisymmetry: Reverse-Consistency Loss (lambda=0.15)
  - Post-hoc Val-Tuned Logit Adjustment (for 7.5% GOF imbalance)

Only 2 focused configs x 3 seeds [42, 43, 44] = 6 runs!
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DECODER_DIR = Path(__file__).resolve().parent
DATA_DIR = DECODER_DIR.parent / "data"
RESULTS_ROOT = DECODER_DIR / "tournament_results" / "record_push"

sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(DECODER_DIR / "new_decoder" / "contvar_decoder_mf_window7"))

from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402
from run_phase1_tournament import (  # noqa: E402
    SOURCE_DIRS,
    PreloadedFeatureStore,
    format_seconds,
    load_prepared_split,
    valid_key_mask,
)
from run_phase2_3_experiments import (  # noqa: E402
    FastAuxDataset,
    build_phase2_3_report,
    extract_biophysical_features,
    run_experiment,
)

RECORD_EXPERIMENTS = [
    {
        "name": "rec1_biophys_sweetspot",
        "desc": "ESIM + Sweet-Spot Light Reg (Drop 0.15/0.10, WD 3e-3, LS 0.05) + 15 Biophysical",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": False,
        "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "rec2_biophys_reverse_sweetspot",
        "desc": "ESIM + Sweet-Spot Light Reg + 15 Biophysical + Reverse-Consistency Loss (lambda=0.15)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": False,
        "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.15, "class_weight": "none", "use_aux": True,
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("hard", "easy"), default="hard")
    p.add_argument("--source", choices=("trained_distance", "trained_mean"), default="trained_distance")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = len(RECORD_EXPERIMENTS) * len(args.seeds)

    print("=" * 78, flush=True)
    print(f"ContVAR RECORD PUSH | {args.family.upper()} | {args.source} | {total_runs} runs (~7-8 min)", flush=True)
    print("=" * 78, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    go_table = GOEmbeddingTable(DATA_DIR / "embeddings" / "go_terms_embeddings_anc2vec.npz")

    raw_train = load_prepared_split(args.family, "train")
    raw_val = load_prepared_split(args.family, "val")
    raw_test = load_prepared_split(args.family, "test")

    common_test_ids = set(raw_test["sample_id"])
    for src in ("baseline", "trained_mean", "trained_distance"):
        m = valid_key_mask(raw_test, SOURCE_DIRS[src])
        common_test_ids &= set(raw_test.loc[m, "sample_id"])

    with open(DATA_DIR / "sequences" / "uniprot_dict.json", "r", encoding="utf-8") as fh:
        seq_data = json.load(fh)
    seq_lengths = (
        {k: len(v) for k, v in seq_data.items()}
        if isinstance(seq_data, dict)
        else {item["Entry"]: len(item["Sequence"]) for item in seq_data if "Entry" in item and "Sequence" in item}
    )
    del seq_data

    aux_tr = extract_biophysical_features(raw_train, seq_lengths)
    aux_va = extract_biophysical_features(raw_val, seq_lengths)
    aux_te = extract_biophysical_features(raw_test, seq_lengths)
    mu = aux_tr.mean(axis=0, keepdims=True)
    sd = np.where(aux_tr.std(axis=0, keepdims=True) > 1e-6, aux_tr.std(axis=0, keepdims=True), 1.0)
    aux_tr, aux_va, aux_te = (aux_tr - mu) / sd, (aux_va - mu) / sd, (aux_te - mu) / sd

    store = PreloadedFeatureStore(args.source, [raw_train, raw_val, raw_test], go_table, device)
    train_ds = FastAuxDataset(raw_train, store, aux_tr)
    val_ds = FastAuxDataset(raw_val, store, aux_va)
    test_ds = FastAuxDataset(raw_test, store, aux_te)

    all_results = []
    run_idx = 0
    start_time = time.perf_counter()
    for exp in RECORD_EXPERIMENTS:
        for seed in args.seeds:
            run_idx += 1
            res = run_experiment(
                run_idx=run_idx, total_runs=total_runs, exp=exp,
                family=args.family, source=args.source, seed=seed,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                common_test_ids=common_test_ids, device=device,
                out_dir=args.out_dir, args=args, start_time=start_time,
            )
            all_results.append(res)

    build_phase2_3_report(all_results, args.out_dir, args.source, args.family)
    print(f"\nRECORD PUSH COMPLETED IN {format_seconds(time.perf_counter() - start_time)}!", flush=True)


if __name__ == "__main__":
    main()
