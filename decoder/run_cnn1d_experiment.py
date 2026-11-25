#!/usr/bin/env python3
"""ContVAR Multi-Scale 1D CNN Local-Window Feature Extractor Experiment.

Replaces the TransformerEncoderLayer in the local ESIM branch with a
Multi-Scale 1D CNN (parallel Conv1d k=3 + Conv1d k=5 + residual + LayerNorm)
to capture local residue triplets and alpha-helix turns with built-in
relative-position awareness and fewer parameters.

Controlled apples-to-apples comparison against rec1 / rec2 (HARD | V5 Distance | 3 seeds):
  1. cnn1_multiscale_biophys         (vs. rec1_biophys_sweetspot: 0.5723)
  2. cnn2_multiscale_biophys_reverse (vs. rec2_biophys_reverse_sweetspot: 0.5801 / +Adj 0.5844)

Total: 2 configs x 3 seeds [42, 43, 44] = 6 runs (~5-6 min on RTX 4060).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DECODER_DIR = Path(__file__).resolve().parent
DATA_DIR = DECODER_DIR.parent / "data"
RESULTS_ROOT = DECODER_DIR / "tournament_results" / "cnn1d"

sys.path.insert(0, str(DECODER_DIR))
sys.path.insert(0, str(DECODER_DIR / "new_decoder" / "contvar_decoder_mf_window7"))

from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable  # noqa: E402
from run_phase1_tournament import (  # noqa: E402
    SOURCE_DIRS,
    PreloadedFeatureStore,
    compute_metrics,
    format_seconds,
    is_true_series,
    load_prepared_split,
    seed_everything,
    valid_key_mask,
)
from run_phase2_3_experiments import (  # noqa: E402
    EnhancedPairedGODecoder,
    FastAuxDataset,
    build_phase2_3_report,
    evaluate_dataset,
    extract_biophysical_features,
    forward_model,
    iter_batches,
    reverse_consistency_loss,
    tune_val_logit_bias,
)


class MultiScale1DCNNReader(nn.Module):
    """Multi-scale 1D CNN (k=3 triplet + k=5 helix-turn) followed by GO cross-attention."""

    def __init__(self, width: int, heads: int, attn_dropout: float = 0.1) -> None:
        super().__init__()
        half = width // 2
        self.conv3 = nn.Conv1d(width, half, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(width, width - half, kernel_size=5, padding=2)
        self.conv_act = nn.GELU()
        self.conv_drop = nn.Dropout(attn_dropout)
        self.conv_norm = nn.LayerNorm(width)

        self.attention = nn.MultiheadAttention(width, heads, dropout=attn_dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(2 * width, width),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Zero out padded positions before 1D convolutions
        clean = torch.where(valid.unsqueeze(-1), memory, torch.zeros_like(memory))
        x = clean.transpose(1, 2)  # [B, width, 7]
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        conv_out = self.conv_drop(self.conv_act(torch.cat([c3, c5], dim=1))).transpose(1, 2)  # [B, 7, width]
        mem = self.conv_norm(clean + conv_out)
        mem = torch.where(valid.unsqueeze(-1), mem, torch.zeros_like(mem))

        q = query.unsqueeze(1)
        read, weights = self.attention(q, mem, mem, key_padding_mask=~valid)
        h = self.norm1(q + read)
        return self.norm2(h + self.ffn(h)).squeeze(1), weights.squeeze(1)


class CNNPairedGODecoder(EnhancedPairedGODecoder):
    """Identical to EnhancedPairedGODecoder except local_reader uses MultiScale1DCNNReader."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.local_reader = MultiScale1DCNNReader(
            width=kwargs.get("width", 128),
            heads=kwargs.get("heads", 4),
            attn_dropout=kwargs.get("attn_dropout", 0.1),
        )


CNN_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "cnn1_multiscale_biophys",
        "desc": "Multi-Scale 1D CNN (k=3,5) + Sweet-Spot Reg + 15 Biophysical (vs. rec1)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": False,
        "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.0, "class_weight": "none", "use_aux": True,
    },
    {
        "name": "cnn2_multiscale_biophys_reverse",
        "desc": "Multi-Scale 1D CNN (k=3,5) + Sweet-Spot Reg + 15 Biophysical + Reverse Loss (λ=0.15, vs. rec2)",
        "pair_mode": "joint", "window_mode": "esim", "use_rel_pos": False,
        "adapter_dropout": 0.15, "attn_dropout": 0.10, "dropout": 0.15,
        "weight_decay": 3e-3, "label_smoothing": 0.05, "cosine_lr": True,
        "lambda_reverse": 0.15, "class_weight": "none", "use_aux": True,
    },
]


def run_cnn_experiment(
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
    common_test_ids: set[str],
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    start_time: float,
) -> dict[str, Any]:
    run_name = f"{family}__{source}__{exp['name']}__seed{seed}"
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

    model = CNNPairedGODecoder(
        input_dim=1280,
        go_dim=train_ds.store.go.dimension,
        width=args.width,
        heads=args.heads,
        pair_mode=exp["pair_mode"],
        window_mode=exp["window_mode"],
        use_rel_pos=exp["use_rel_pos"],
        adapter_dropout=exp["adapter_dropout"],
        attn_dropout=exp["attn_dropout"],
        dropout=exp["dropout"],
        aux_dim=15 if exp["use_aux"] else 0,
        window_size=7,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=exp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

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
            loss = F.cross_entropy(out_fwd["logits"], labels, label_smoothing=exp["label_smoothing"])
            if exp["lambda_reverse"] > 0.0:
                out_rev, _ = forward_model(model, batch, device, swap_wt_var=True)
                loss = loss + exp["lambda_reverse"] * reverse_consistency_loss(out_fwd["logits"], out_rev["logits"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss_sum += float(loss.item()) * len(labels)
            tr_n += len(labels)

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

    test_frame = test_ds.frame.copy()
    is_common = test_frame["sample_id"].isin(common_test_ids).to_numpy(dtype=bool)
    is_comb = is_true_series(test_frame["is_combinatorial"]).to_numpy(dtype=bool) if "is_combinatorial" in test_frame.columns else np.zeros(len(test_frame), dtype=bool)
    is_single = ~is_comb

    val_m = compute_metrics(v_labels, v_preds, v_probs)
    full_m = compute_metrics(t_labels, t_preds, t_probs)
    common_m = compute_metrics(t_labels[is_common], t_preds[is_common], t_probs[is_common])
    single_m = compute_metrics(t_labels[is_single], t_preds[is_single], t_probs[is_single])
    comb_m = compute_metrics(t_labels[is_comb], t_preds[is_comb], t_probs[is_comb]) if is_comb.any() else {"n": 0}
    full_adj_m = compute_metrics(t_labels, t_preds_adj, t_probs_adj)
    common_adj_m = compute_metrics(t_labels[is_common], t_preds_adj[is_common], t_probs_adj[is_common])

    run_sec = time.perf_counter() - run_t0
    payload = {
        "run_name": run_name, "exp_name": exp["name"], "exp_desc": exp["desc"],
        "family": family, "source": source, "seed": seed, "best_epoch": best_epoch,
        "duration_seconds": run_sec, "val_tuned_logit_bias": val_bias.tolist(),
        "val": val_m, "full_test": full_m, "common_test": common_m,
        "single_test": single_m, "combinatorial_test": comb_m,
        "full_test_logit_adj": full_adj_m, "common_test_logit_adj": common_adj_m,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    eta = (elapsed / run_idx) * (total_runs - run_idx)
    print(
        f"  DONE {run_name} ({run_sec:.1f}s) | Full F1={full_m['macro_f1']:.4f} (GOF={full_m['gof']['f1']:.4f}, ECE={full_m['confidence']['ece_10bin']:.4f}) "
        f"| Common F1={common_m['macro_f1']:.4f} | +LogitAdj={full_adj_m['macro_f1']:.4f} "
        f"| Elapsed={format_seconds(elapsed)} | ETA={format_seconds(eta)}",
        flush=True,
    )
    return payload


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
    total_runs = len(CNN_EXPERIMENTS) * len(args.seeds)

    print("=" * 78, flush=True)
    print(f"ContVAR 1D CNN EXPERIMENT | {args.family.upper()} | {args.source} | {total_runs} runs (~5-6 min)", flush=True)
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
    for exp in CNN_EXPERIMENTS:
        for seed in args.seeds:
            run_idx += 1
            res = run_cnn_experiment(
                run_idx=run_idx, total_runs=total_runs, exp=exp,
                family=args.family, source=args.source, seed=seed,
                train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                common_test_ids=common_test_ids, device=device,
                out_dir=args.out_dir, args=args, start_time=start_time,
            )
            all_results.append(res)

    build_phase2_3_report(all_results, args.out_dir, args.source, args.family)
    print(f"\n1D CNN EXPERIMENTS COMPLETED IN {format_seconds(time.perf_counter() - start_time)}!", flush=True)


if __name__ == "__main__":
    main()
