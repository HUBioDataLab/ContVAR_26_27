#!/usr/bin/env python3
"""ContVAR Official MF-Only (Molecular Function) Controlled Experiments.

Uses the exact official Gene Ontology `molecular_function` filter (`data/mf_go_ids.json`),
matching the team's V5 MF-only evaluation split down to the exact row:
  - hard_train: 33,852 rows (LOF=22,897, NEUTRAL=9,078, GOF=1,877)
  - hard_val  :  3,772 rows (augmented removed)
  - hard_test :  3,875 rows (LOF=2,892, GOF=255, NEUTRAL=728; augmented removed)

5 Focused Experiments x 3 seeds [42, 43, 44] = 15 runs (~9-11 min total on RTX 4060):
  1. mf1_ref_esim                : Raw ESIM reference (matches team's ~0.5025 baseline)
  2. mf2_biophys_reg             : ESIM + Light Reg + 15 Biophysical Features
  3. mf3_biophys_reverse         : ESIM + Light Reg + 15 Biophysical + Reverse Loss (λ=0.15)
  4. mf4_cnn1d_biophys_reverse   : Multi-Scale 1D CNN (k=3,5) + Light Reg + 15 Biophysical + Reverse Loss
  5. mf5_imbalance_biophys_rev   : ESIM + Light Reg + 15 Biophysical + Reverse Loss + Mild 1/sqrt(N) Class Weighting

All runs also report post-hoc validation-tuned logit adjustment (+LogitAdj F1) and ECE calibration.
Outputs saved neatly under decoder/tournament_results/mf_only/ (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

DECODER_DIR = Path(__file__).resolve().parent
DATA_DIR = DECODER_DIR.parent / "data"
RESULTS_ROOT = DECODER_DIR / "tournament_results" / "mf_only"

sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(DECODER_DIR / "new_decoder" / "contvar_decoder_mf_window7"))

from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402
from run_phase1_tournament import (  # noqa: E402
    PreloadedFeatureStore,
    compute_metrics,
    format_seconds,
    is_true_series,
    load_prepared_split,
    seed_everything,
)
from run_phase2_3_experiments import (  # noqa: E402
    EnhancedPairedGODecoder,
    FastAuxDataset,
    evaluate_dataset,
    extract_biophysical_features,
    forward_model,
    iter_batches,
    reverse_consistency_loss,
    tune_val_logit_bias,
)
from run_cnn1d_experiment import CNNPairedGODecoder  # noqa: E402

MF_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "mf1_ref_esim",
        "desc": "Raw ESIM reference (no reg, no aux, no reverse loss — matches ~0.5025 MF baseline)",
        "arch": "esim", "adapter_dropout": 0.0, "attn_dropout": 0.0, "dropout": 0.1,
        "weight_decay": 1e-4, "label_smoothing": 0.0, "cosine_lr": False,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": False,
    },
    {
        "name": "mf2_biophys_reg",
        "desc": "ESIM + Light Reg (Drop 0.15/0.10, WD 3e-3, LS 0.05) + 15 Biophysical Features",
        "arch": "esim", "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "mf3_biophys_reverse",
        "desc": "ESIM + Light Reg + 15 Biophysical + Reverse-Consistency Loss (λ=0.15)",
        "arch": "esim", "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.15, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "mf4_cnn1d_biophys_reverse",
        "desc": "Multi-Scale 1D CNN (k=3,5) + Light Reg + 15 Biophysical + Reverse Loss (λ=0.15)",
        "arch": "cnn1d", "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.15, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "mf5_imbalance_biophys_rev",
        "desc": "ESIM + Light Reg + 15 Biophysical + Reverse Loss (λ=0.15) + Mild 1/sqrt(N) Weighting",
        "arch": "esim", "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.15, "class_weight": "sqrt", "use_aux": True,
    },
]


def run_one(
    *,
    run_idx: int,
    total_runs: int,
    exp: dict[str, Any],
    family: str,
    source: str,
    seed: int,
    train_ds: FastAuxDataset,
    val_ds: FastAuxDataset,
    test_ds: FastAuxDataset,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    start_time: float,
) -> dict[str, Any]:
    run_name = f"mf_{family}__{source}__{exp['name']}__seed{seed}"
    run_dir = out_dir / "runs" / run_name
    metrics_path = run_dir / "metrics.json"

    if args.resume and metrics_path.exists():
        print(f"[{run_idx:02d}/{total_runs:02d}] SKIP (resume) {run_name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    eval_bs = args.batch_size * 4

    class_weights = None
    if exp["class_weight"] == "sqrt":
        counts = np.bincount(train_ds.frame["label_idx"].to_numpy(dtype=np.int64), minlength=3).astype(np.float64)
        w = np.sqrt(counts.sum() / np.maximum(1.0, counts))
        w = w / w.mean()
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    model_cls = CNNPairedGODecoder if exp["arch"] == "cnn1d" else EnhancedPairedGODecoder
    model = model_cls(
        input_dim=1280,
        go_dim=train_ds.store.go.dimension,
        width=args.width,
        heads=args.heads,
        pair_mode="joint",
        window_mode="esim",
        use_rel_pos=False,
        adapter_dropout=exp["adapter_dropout"],
        attn_dropout=exp["attn_dropout"],
        dropout=exp["dropout"],
        aux_dim=15 if exp["use_aux"] else 0,
        window_size=7,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=exp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5) if exp["cosine_lr"] else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []
    run_t0 = time.perf_counter()

    print(f"\n[{run_idx:02d}/{total_runs:02d}] START {run_name} ({exp['desc']})", flush=True)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for batch in iter_batches(train_ds, batch_size=args.batch_size, shuffle=True, generator=g):
            optimizer.zero_grad(set_to_none=True)
            out_fwd, labels = forward_model(model, batch, device, swap_wt_var=False)
            loss = F.cross_entropy(
                out_fwd["logits"], labels,
                weight=class_weights,
                label_smoothing=exp["label_smoothing"],
            )
            if exp["lambda_reverse"] > 0.0:
                out_rev, _ = forward_model(model, batch, device, swap_wt_var=True)
                loss = loss + exp["lambda_reverse"] * reverse_consistency_loss(out_fwd["logits"], out_rev["logits"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss_sum += float(loss.item()) * len(labels)
            tr_n += len(labels)

        if scheduler is not None:
            scheduler.step()

        tr_loss = tr_loss_sum / max(1, tr_n)
        v_labels, v_preds, v_probs, _, v_loss = evaluate_dataset(model, val_ds, eval_bs, device)
        v_metrics = compute_metrics(v_labels, v_preds, v_probs)
        val_f1 = v_metrics["macro_f1"]
        val_conf = v_metrics["confidence"]["mean_confidence"]
        val_ece = v_metrics["confidence"]["ece_10bin"]
        ep_dt = time.perf_counter() - ep_t0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            star = "*"
        else:
            no_improve += 1
            star = " "

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss,
            "val_macro_f1": val_f1, "val_gof_f1": v_metrics["gof"]["f1"],
            "val_conf": val_conf, "val_ece": val_ece, "seconds": ep_dt,
        })
        print(
            f"  Ep {epoch:02d}/{args.epochs:02d} {star} | tr_loss={tr_loss:.4f} | val_loss={v_loss:.4f} "
            f"| val_F1={val_f1:.4f} (best={best_val_f1:.4f}@{best_epoch}) "
            f"| GOF={v_metrics['gof']['f1']:.4f} | conf={val_conf:.3f} | ECE={val_ece:.3f} | {ep_dt:.1f}s",
            flush=True,
        )
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})", flush=True)
            break

    assert best_state is not None
    torch.save({"exp": exp, "model_spec": model.spec, "state_dict": best_state, "best_epoch": best_epoch}, run_dir / "best_model.pt")
    model.load_state_dict(best_state)

    v_labels, v_preds, v_probs, v_logits, _ = evaluate_dataset(model, val_ds, eval_bs, device)
    t_labels, t_preds, t_probs, t_logits, _ = evaluate_dataset(model, test_ds, eval_bs, device)

    val_bias = tune_val_logit_bias(v_logits, v_labels)
    t_logits_adj = t_logits + val_bias
    t_probs_adj = F.softmax(torch.tensor(t_logits_adj), dim=-1).numpy()
    t_preds_adj = t_logits_adj.argmax(axis=1)

    test_frame = test_ds.frame
    is_comb = is_true_series(test_frame["is_combinatorial"]).to_numpy(dtype=bool) if "is_combinatorial" in test_frame.columns else np.zeros(len(test_frame), dtype=bool)
    is_single = ~is_comb

    val_m = compute_metrics(v_labels, v_preds, v_probs)
    test_m = compute_metrics(t_labels, t_preds, t_probs)
    test_adj_m = compute_metrics(t_labels, t_preds_adj, t_probs_adj)
    single_m = compute_metrics(t_labels[is_single], t_preds[is_single], t_probs[is_single])
    comb_m = compute_metrics(t_labels[is_comb], t_preds[is_comb], t_probs[is_comb]) if is_comb.any() else {"n": 0}

    run_sec = time.perf_counter() - run_t0
    payload = {
        "run_name": run_name, "exp_name": exp["name"], "exp_desc": exp["desc"],
        "family": family, "source": source, "seed": seed, "best_epoch": best_epoch,
        "duration_seconds": run_sec, "val_tuned_logit_bias": val_bias.tolist(),
        "val": val_m, "test": test_m, "test_logit_adj": test_adj_m,
        "single_test": single_m, "combinatorial_test": comb_m,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    eta = (elapsed / run_idx) * (total_runs - run_idx)
    print(
        f"  DONE {run_name} ({run_sec:.1f}s) | Test F1={test_m['macro_f1']:.4f} "
        f"(LOF={test_m['lof']['f1']:.4f}, GOF={test_m['gof']['f1']:.4f}, NEU={test_m['neutral']['f1']:.4f}, ECE={test_m['confidence']['ece_10bin']:.4f}) "
        f"| +LogitAdj={test_adj_m['macro_f1']:.4f} (GOF={test_adj_m['gof']['f1']:.4f}) "
        f"| Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}",
        flush=True,
    )
    return payload


def build_mf_report(all_results: list[dict[str, Any]], out_dir: Path, source: str, family: str) -> None:
    rows = []
    for r in all_results:
        tm, ta = r["test"], r["test_logit_adj"]
        rows.append({
            "exp_name": r["exp_name"], "seed": r["seed"], "best_epoch": r["best_epoch"],
            "val_macro_f1": r["val"]["macro_f1"],
            "test_macro_f1": tm["macro_f1"],
            "test_lof_f1": tm["lof"]["f1"],
            "test_gof_f1": tm["gof"]["f1"],
            "test_neu_f1": tm["neutral"]["f1"],
            "test_mcc": tm["mcc"],
            "test_conf": tm["confidence"]["mean_confidence"],
            "test_conf_gap": tm["confidence"]["confidence_gap"],
            "test_ece": tm["confidence"]["ece_10bin"],
            "adj_macro_f1": ta["macro_f1"],
            "adj_gof_f1": ta["gof"]["f1"],
            "single_f1": r["single_test"]["macro_f1"],
            "comb_f1": r["combinatorial_test"].get("macro_f1", float("nan")),
        })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_dir / "summary_all_runs.csv", index=False)

    mcols = [c for c in df_all.columns if c not in ("exp_name", "seed", "best_epoch")]
    agg = []
    for ename, grp in df_all.groupby("exp_name", sort=False):
        rec: dict[str, Any] = {"exp_name": ename, "n_seeds": len(grp)}
        for c in mcols:
            rec[f"{c}_mean"] = float(grp[c].mean())
            rec[f"{c}_std"] = float(grp[c].std(ddof=1)) if len(grp) > 1 else 0.0
        agg.append(rec)
    df_agg = pd.DataFrame(agg).sort_values("test_macro_f1_mean", ascending=False)
    df_agg.to_csv(out_dir / "summary_3seed_mean_std.csv", index=False)

    def fmt(m: float, s: float) -> str:
        return f"{m:.4f} ± {s:.4f}"

    lines = [
        f"# ContVAR V5 MF-Only Controlled Experiments ({family.upper()} | {source})",
        "",
        f"- **Generated (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`",
        "- **Scope:** Official `molecular_function` only (hard_test n=3,875: LOF=2,892, GOF=255, NEUTRAL=728), 3 seeds `[42, 43, 44]`.",
        "",
        "| Experiment | Test Macro-F1 | +LogitAdj F1 | LOF-F1 | GOF-F1 | +Adj GOF | NEUTRAL-F1 | MCC | ECE ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_agg.iterrows():
        lines.append(
            f"| `{r['exp_name']}` | **{fmt(r['test_macro_f1_mean'], r['test_macro_f1_std'])}** "
            f"| {fmt(r['adj_macro_f1_mean'], r['adj_macro_f1_std'])} "
            f"| {fmt(r['test_lof_f1_mean'], r['test_lof_f1_std'])} "
            f"| {fmt(r['test_gof_f1_mean'], r['test_gof_f1_std'])} "
            f"| {fmt(r['adj_gof_f1_mean'], r['adj_gof_f1_std'])} "
            f"| {fmt(r['test_neu_f1_mean'], r['test_neu_f1_std'])} "
            f"| {fmt(r['test_mcc_mean'], r['test_mcc_std'])} "
            f"| **{r['test_ece_mean']:.4f}** |"
        )
    rpath = out_dir / "MF_ONLY_REPORT.md"
    rpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== MF-ONLY REPORT SAVED TO: {rpath} ===", flush=True)


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
    p.add_argument("--exps", nargs="+", default=None, help="Optional subset of experiment names to run")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    selected_exps = [e for e in MF_EXPERIMENTS if (args.exps is None or e["name"] in args.exps)]
    if not selected_exps:
        raise ValueError(f"No matching experiments for --exps {args.exps}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_runs = len(selected_exps) * len(args.seeds)

    mf_ids = set(json.loads((DATA_DIR / "mf_go_ids.json").read_text(encoding="utf-8")))

    print("=" * 78, flush=True)
    print(f"ContVAR V5 MF-ONLY EXPERIMENTS | {args.family.upper()} | {args.source} | {total_runs} runs (~9-11 min)", flush=True)
    print("=" * 78, flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    go_table = GOEmbeddingTable(DATA_DIR / "embeddings" / "go_terms_embeddings_anc2vec.npz")

    raw_train = load_prepared_split(args.family, "train")
    raw_val = load_prepared_split(args.family, "val")
    raw_test = load_prepared_split(args.family, "test")

    # Apply strict MF-only filter
    raw_train = raw_train[raw_train["matched_go_id"].isin(mf_ids)].reset_index(drop=True)
    raw_val = raw_val[raw_val["matched_go_id"].isin(mf_ids)].reset_index(drop=True)
    raw_test = raw_test[raw_test["matched_go_id"].isin(mf_ids)].reset_index(drop=True)
    print(f"  [MF-Only] train={len(raw_train)}, val={len(raw_val)}, test={len(raw_test)}", flush=True)

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
    for exp in selected_exps:
        for seed in args.seeds:
            run_idx += 1
            res = run_one(
                run_idx=run_idx, total_runs=total_runs, exp=exp,
                family=args.family, source=args.source, seed=seed,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                device=device, out_dir=args.out_dir, args=args, start_time=start_time,
            )
            all_results.append(res)

    build_mf_report(all_results, args.out_dir, args.source, args.family)
    print(f"\nALL {total_runs} MF-ONLY RUNS COMPLETED IN {format_seconds(time.perf_counter() - start_time)}!", flush=True)


if __name__ == "__main__":
    main()
