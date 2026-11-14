"""
Variant-Specific GO Benchmark Evaluation.

Evaluates the decoder on the LOF/GOF benchmark dataset using variant protein embeddings.
For each benchmark entry (protein_id, snp, matched_go_id, label):
  - Decoder runs on variant embedding → var_score
  - var_score >= threshold → GOF prediction
  - var_score <  threshold → LOF prediction

Usage:
    python -m phase_2_decoder.benchmark_eval \\
        --aspect F \\
        --embeddings_h5 artifacts/pipeline_run/decoder/raw_embeddings.h5 \\
        --variant_h5 artifacts/pipeline_run/decoder/variant_embeddings.h5 \\
        --checkpoint artifacts/pipeline_run/decoder/decoder_best_f_raw.pt \\
        --go_vocab_json artifacts/pipeline_run/decoder/go_vocab_f_raw.json
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

try:
    from phase_2_decoder.models.ffn_decoder import FFNDecoder
except ModuleNotFoundError:
    from models.ffn_decoder import FFNDecoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Canonical order for the full LOF/GOF metric set (used by grid_search.py's CSV
# columns and best_runs_report.py, per-split as bm_{val,test}_{key}).
BM_METRIC_KEYS = [
    "n_total", "n_lof", "n_gof", "gof_threshold", "overall_acc",
    "lof_recall", "gof_recall", "precision", "recall", "f1", "auroc", "auprc",
]


def compute_lof_gof_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Full LOF/GOF benchmark metric set for one (scores, labels) split.

    scores: var_score (decoder sigmoid output) — higher = more GOF-like.
    labels: 1 = GOF (positive class), 0 = LOF (negative class).

    Returns a dict with exactly BM_METRIC_KEYS plus tp/tn/fp/fn (used by
    callers that also want raw confusion-matrix counts).
    """
    n_gof = int(labels.sum())
    n_lof = int(len(labels) - labels.sum())

    best_acc, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.96, 0.05):
        acc = ((scores >= thr).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    preds = (scores >= best_thr).astype(int)

    gof_mask = labels == 1
    lof_mask = labels == 0
    gof_recall = preds[gof_mask].mean() if gof_mask.sum() > 0 else 0.0
    lof_recall = (1 - preds[lof_mask]).mean() if lof_mask.sum() > 0 else 0.0
    overall_acc = (preds == labels).mean()

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    return {
        "n_total": len(labels),
        "n_lof": n_lof,
        "n_gof": n_gof,
        "gof_threshold": round(float(best_thr), 2),
        "overall_acc": round(float(overall_acc), 4),
        "lof_recall": round(float(lof_recall), 4),
        "gof_recall": round(float(gof_recall), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark evaluation run."""
    aspect: str                        # "F", "P", or "C"

    # WT embeddings — single H5 file, keyed by protein_id
    embeddings_h5: Optional[str] = None
    # Variant embeddings — keyed by "{protein_id}_{snp}"
    variant_h5: Optional[str] = None
    # concat_phase0/concat_phase1 checkpoint'leri icin ikinci varyant embedding H5'i (ayni key formati)
    variant_h5_secondary: Optional[str] = None

    # Decoder
    checkpoint: Optional[str] = None
    go_vocab_json: Optional[str] = None
    dropout: float = 0.3
    min_go_freq: int = 10

    # go_vocab_json bulunamayip sifirdan kurulmasi gerektiginde kullanilir.
    # grid_search.py varsayilan olarak propagation'i KAPALI egitir (--propagation
    # verilmedigi surece), o yuzden varsayilan burada da False - aksi halde
    # sifirdan kurulan vocab, checkpoint'in egitildigi vocab'la boyutca uyusmaz.
    use_go_propagation: bool = False
    go_obo_path: Optional[str] = None

    # Data paths — filled with repo-root defaults in __post_init__
    benchmark_tsv: Optional[str] = None
    uniref_tsv: Optional[str] = None
    split_json: Optional[str] = None
    goa_tsv: Optional[str] = None

    def __post_init__(self):
        from pathlib import Path as _P
        asp  = self.aspect.lower()
        root = _P(__file__).resolve().parents[1]
        dec  = root / "phase_2_decoder"
        art  = root / "artifacts" / "pipeline_run" / "decoder"

        if self.benchmark_tsv is None:
            self.benchmark_tsv = str(root / "data" / "raw" / "variant_specific_go_benchmark.tsv")
        if self.goa_tsv is None:
            self.goa_tsv = str(root / "data" / "raw" / "goa_2025-12-04_swissprot_noiea.tsv")
        if self.go_obo_path is None:
            self.go_obo_path = str(root / "data" / "raw" / "go-basic.obo")
        if self.uniref_tsv is None:
            self.uniref_tsv = str(root / "phase_0_go_pretrained" / "protein_uniref50.tsv")
        if self.split_json is None:
            self.split_json = str(root / "phase0_protein_split_removed_graphless.json")

        if self.embeddings_h5 is None:
            candidates = [
                dec / f"embeddings_{asp}.h5",
                art / "raw_embeddings.h5",
            ]
            self.embeddings_h5 = str(next((p for p in candidates if p.exists()), candidates[0]))

        if self.checkpoint is None:
            candidates = [
                dec / f"decoder_best_{asp}.pt",
                art / f"decoder_best_{asp}.pt",
            ]
            self.checkpoint = str(next((p for p in candidates if p.exists()), candidates[0]))

        if self.go_vocab_json is None:
            candidates = [
                dec / f"go_vocab_{asp}.json",
                art / f"go_vocab_{asp}.json",
                art / f"go_vocab_{asp}_raw.json",
                art / f"go_vocab_{asp}_phase_1.json",
                art / f"go_vocab_{asp}_phase_0.json",
            ]
            self.go_vocab_json = str(next((p for p in candidates if p.exists()), candidates[0]))


# ── Split Loading ────────────────────────────────────────────────────────────

def load_protein_splits(uniref_tsv: str, split_json: str) -> dict:
    """Returns protein_id → split mapping.

    Supports two JSON formats:
      - {"protein_to_split": {"P12345": "train", ...}}   ← phase0_protein_split_removed_graphless.json
      - {"group_to_split": {...}}  + uniref TSV (legacy)
    """
    with open(split_json) as f:
        phase0 = json.load(f)

    if "protein_to_split" in phase0:
        return phase0["protein_to_split"]

    # Legacy: group_to_split + uniref TSV
    group_to_split = phase0["group_to_split"]
    protein_to_split = {}
    with open(uniref_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            grp = row["group_id"]
            split = group_to_split.get(grp)
            if split is not None:
                protein_to_split[row["protein_id"]] = split
    return protein_to_split


# ── Model Loading ────────────────────────────────────────────────────────────

def auto_detect_architecture(checkpoint_path: str) -> dict:
    """Auto-detect model architecture from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # Find input dim from first linear layer
    first_weight = sd.get("net.0.weight")
    if first_weight is None:
        raise ValueError("Cannot find net.0.weight in checkpoint")
    input_dim = first_weight.shape[1]

    # Find all hidden dims (linear layers before output) — sort numerically by layer index
    linear_keys = sorted(
        [k for k in sd if k.endswith(".weight") and k.startswith("net.") and len(sd[k].shape) == 2],
        key=lambda k: int(k.split(".")[1])
    )
    # Last one is output layer
    hidden_dims = [sd[k].shape[0] for k in linear_keys[:-1]]
    output_dim = sd[linear_keys[-1]].shape[0]

    has_rescaling = "input_scale" in sd

    return {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "has_rescaling": has_rescaling,
        "state_dict": sd,
    }


def load_decoder_model(checkpoint_path: str, n_classes: int,
                       input_dim: int = None, hidden_dims: list = None,
                       dropout: float = 0.3) -> torch.nn.Module:
    """Load decoder model, auto-detecting architecture if not specified."""
    arch = auto_detect_architecture(checkpoint_path)
    sd = arch["state_dict"]

    actual_input = input_dim or arch["input_dim"]
    actual_hidden = hidden_dims or arch["hidden_dims"]

    decoder = FFNDecoder(actual_input, n_classes, actual_hidden, dropout)

    # Handle learnable rescaling parameters
    if arch["has_rescaling"]:
        decoder.input_scale = torch.nn.Parameter(torch.ones(actual_input))
        decoder.input_bias = torch.nn.Parameter(torch.zeros(actual_input))
        # Override forward to apply rescaling
        original_forward = decoder.forward

        def rescaled_forward(x):
            x = x * decoder.input_scale + decoder.input_bias
            return decoder.net(x)

        decoder.forward = rescaled_forward

    decoder.load_state_dict(sd, strict=False)
    decoder.eval().to(DEVICE)
    return decoder


# ── Embedding Loading ────────────────────────────────────────────────────────

def load_embeddings(keys: list, h5_path: str) -> dict:
    """Load embeddings from H5 file. Returns {key: tensor[D]}."""
    embeddings = {}
    with h5py.File(h5_path, "r") as h5f:
        available = set(h5f.keys())
        for key in keys:
            if key in available:
                emb = torch.FloatTensor(h5f[key][:])
                if emb.dim() == 2:
                    emb = emb.mean(dim=0)
                embeddings[key] = emb
    return embeddings


def load_variant_embeddings(protein_snp_pairs: list,
                            variant_h5: str,
                            secondary_variant_h5: str | None = None) -> dict:
    """
    Load variant embeddings from H5.

    Key format in H5: "{protein_id}_{snp}"  (e.g. "A0A061I403_E231G")
    Returns: {(protein_id, snp): tensor[D]}

    secondary_variant_h5 verilirse (concat_phase0/concat_phase1 checkpoint'leri
    icin), ayni key ikinci dosyadan da okunup [primary | secondary] concat edilir;
    key ikinci dosyada yoksa o (protein_id, snp) ciftindeki sonuc dict'e eklenmez.
    """
    embeddings = {}
    with h5py.File(variant_h5, "r") as h5f:
        available = set(h5f.keys())
        h5f_sec = h5py.File(secondary_variant_h5, "r") if secondary_variant_h5 else None
        try:
            secondary_available = set(h5f_sec.keys()) if h5f_sec is not None else None
            for pid, snp in protein_snp_pairs:
                key = f"{pid}_{snp}"
                if key not in available:
                    continue
                if h5f_sec is not None and key not in secondary_available:
                    continue
                emb = torch.FloatTensor(h5f[key][:])
                if emb.dim() == 2:
                    emb = emb.mean(dim=0)
                if h5f_sec is not None:
                    emb_sec = torch.FloatTensor(h5f_sec[key][:])
                    if emb_sec.dim() == 2:
                        emb_sec = emb_sec.mean(dim=0)
                    emb = torch.cat([emb, emb_sec], dim=-1)
                embeddings[(pid, snp)] = emb
        finally:
            if h5f_sec is not None:
                h5f_sec.close()
    return embeddings


# ── In-memory Benchmark (used during training) ───────────────────────────────

def evaluate_benchmark_in_memory(
    decoder: torch.nn.Module,
    go_vocab: dict,
    benchmark_tsv: str,
    embeddings_h5: str,
    split_json: str,
    uniref_tsv: str,
    aspect: str,
    device: torch.device,
    variant_h5: str | None = None,
    variant_h5_secondary: str | None = None,
) -> dict:
    """
    Benchmark evaluation with a live model — no checkpoint save/load.
    Returns {"val": {metrics}, "test": {metrics}}.
    Called after each training epoch.

    variant_h5 verilmezse boş dict döner (benchmark atlanır).

    Decoder varyant embedding üzerinde çalışır:
      var_score >= threshold → GOF tahmin
      var_score <  threshold → LOF tahmin
    Signal olarak var_score doğrudan kullanılır (yüksek = GOF, pozitif sınıf).
    """
    if not variant_h5:
        return {}

    vocab_gos = set(go_vocab.keys()) - {"NULL_FUNCTION"}

    benchmark = pd.read_csv(benchmark_tsv, sep="\t")
    benchmark = benchmark[benchmark["matched_go_id"].isin(vocab_gos)]
    if len(benchmark) == 0:
        return {}

    protein_to_split = load_protein_splits(uniref_tsv, split_json)
    benchmark = benchmark.copy()
    benchmark["split"] = benchmark["protein_id"].map(protein_to_split)

    # Load variant embeddings
    all_pairs = list(
        benchmark[["protein_id", "snp"]].drop_duplicates().itertuples(index=False, name=None)
    )
    var_embeddings = load_variant_embeddings(all_pairs, variant_h5, secondary_variant_h5=variant_h5_secondary)
    benchmark = benchmark[
        benchmark.apply(lambda r: (r["protein_id"], r["snp"]) in var_embeddings, axis=1)
    ]
    if len(benchmark) == 0:
        return {}

    # Variant scores
    decoder.eval()
    var_scores: dict[tuple, np.ndarray] = {}
    with torch.no_grad():
        unique_pairs = list(var_embeddings.keys())
        batch_size = 256
        for i in range(0, len(unique_pairs), batch_size):
            batch_pairs = unique_pairs[i:i+batch_size]
            batch_embs = torch.stack([var_embeddings[p] for p in batch_pairs]).to(device)
            batch_probs = torch.sigmoid(decoder(batch_embs)).cpu().numpy()
            for j, pair in enumerate(batch_pairs):
                var_scores[pair] = batch_probs[j]

    results = {}
    for split_name in ["val", "test"]:
        split_df = benchmark[benchmark["split"] == split_name]
        if len(split_df) == 0:
            continue

        scores, labels = [], []
        for _, row in split_df.iterrows():
            pid, snp, go_id = row["protein_id"], row["snp"], row["matched_go_id"]
            if go_id not in go_vocab:
                continue
            pair = (pid, snp)
            if pair not in var_scores:
                continue
            go_idx = go_vocab[go_id]
            var_s = float(var_scores[pair][go_idx])
            # var_score kendisi pozitif (GOF) sinyalidir:
            #   high var_score → GOF (label=1)
            #   low var_score  → LOF (label=0)
            scores.append(var_s)
            labels.append(1 if row["label"] == "GOF" else 0)

        if not scores:
            continue

        scores = np.array(scores)
        labels = np.array(labels)

        results[split_name] = compute_lof_gof_metrics(scores, labels)

    return results


# ── Main Evaluation ──────────────────────────────────────────────────────────

def evaluate_benchmark(cfg: BenchmarkConfig) -> dict:
    """
    Run benchmark evaluation for a single configuration.
    Returns metrics dict.
    """
    asp = cfg.aspect.lower()
    print(f"\n{'='*70}")
    print(f"  Benchmark Evaluation | Aspect={cfg.aspect}")
    print(f"  Variant H5: {cfg.variant_h5}")
    print(f"  Checkpoint: {cfg.checkpoint}")
    print(f"{'='*70}")

    if cfg.variant_h5 is None or not Path(cfg.variant_h5).exists():
        print(f"  [SKIP] variant_h5 not provided or not found: {cfg.variant_h5}")
        return None

    # 1. Load GO vocabulary
    if not Path(cfg.go_vocab_json).exists():
        print(f"  Building GO vocab for aspect {cfg.aspect}...")
        from phase_2_decoder.data.parse_goa import parse_goa_tsv, build_go_vocab
        _obo = cfg.go_obo_path if (cfg.use_go_propagation and Path(cfg.go_obo_path).exists()) else None
        annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect, obo_path=_obo)
        go_vocab = build_go_vocab(annotations, min_freq=cfg.min_go_freq,
                                  save_path=cfg.go_vocab_json)
    else:
        with open(cfg.go_vocab_json) as f:
            go_vocab = json.load(f)
    n_classes = len(go_vocab)
    vocab_gos = set(go_vocab.keys()) - {"NULL_FUNCTION"}
    print(f"  GO vocab: {n_classes} terms (aspect {cfg.aspect})")

    # 2. Detect checkpoint architecture
    arch = auto_detect_architecture(cfg.checkpoint)

    # 3. Load decoder
    decoder = load_decoder_model(
        cfg.checkpoint, n_classes,
        input_dim=arch["input_dim"],
        hidden_dims=arch["hidden_dims"],
        dropout=cfg.dropout,
    )
    print(f"  Decoder loaded: input={arch['input_dim']}, hidden={arch['hidden_dims']}, "
          f"output={arch['output_dim']}")

    # 4. Load benchmark data
    benchmark = pd.read_csv(cfg.benchmark_tsv, sep="\t")
    benchmark = benchmark[benchmark["matched_go_id"].isin(vocab_gos)]
    print(f"  Benchmark rows with GO in vocab: {len(benchmark)}")

    if len(benchmark) == 0:
        print("  [SKIP] No benchmark rows match this aspect's GO vocab!")
        return None

    # 5. Load splits
    protein_to_split = load_protein_splits(cfg.uniref_tsv, cfg.split_json)
    benchmark["split"] = benchmark["protein_id"].map(protein_to_split)

    # 6. Load variant embeddings
    all_pairs = list(benchmark[["protein_id", "snp"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"  Loading variant embeddings for {len(all_pairs)} (protein, snp) pairs...")
    var_embeddings = load_variant_embeddings(all_pairs, cfg.variant_h5,
                                              secondary_variant_h5=cfg.variant_h5_secondary)
    print(f"  Variant embeddings loaded: {len(var_embeddings)}/{len(all_pairs)} pairs")

    benchmark = benchmark[
        benchmark.apply(lambda r: (r["protein_id"], r["snp"]) in var_embeddings, axis=1)
    ]
    print(f"  Benchmark rows with variant embeddings: {len(benchmark)}")

    if len(benchmark) == 0:
        print("  [SKIP] No benchmark rows have variant embeddings!")
        return None

    # 7. Run decoder on variant embeddings
    print("  Running decoder on variant embeddings...")
    var_scores = {}
    batch_size = 256
    with torch.no_grad():
        unique_pairs = list(var_embeddings.keys())
        for i in range(0, len(unique_pairs), batch_size):
            batch_pairs = unique_pairs[i:i+batch_size]
            batch_embs = torch.stack([var_embeddings[p] for p in batch_pairs])
            batch_probs = torch.sigmoid(decoder(batch_embs.to(DEVICE))).cpu().numpy()
            for j, pair in enumerate(batch_pairs):
                var_scores[pair] = batch_probs[j]

    # 8. Evaluate per split
    # Signal: var_score doğrudan kullanılır, pozitif sınıf = GOF
    #   var_score >= threshold → GOF (label=1)
    #   var_score <  threshold → LOF (label=0)
    results = {}
    for split_name in ["val", "test"]:
        split_df = benchmark[benchmark["split"] == split_name].copy()
        if len(split_df) == 0:
            continue

        scores = []
        labels = []

        for _, row in split_df.iterrows():
            pid   = row["protein_id"]
            snp   = row["snp"]
            go_id = row["matched_go_id"]
            go_idx = go_vocab[go_id]
            var_s = float(var_scores[(pid, snp)][go_idx])
            scores.append(var_s)
            labels.append(1 if row["label"] == "GOF" else 0)

        scores = np.array(scores)
        labels = np.array(labels)

        m = compute_lof_gof_metrics(scores, labels)
        split_results = {"split": split_name, **m}
        results[split_name] = split_results

        print(f"\n  [{split_name.upper()}] n={m['n_total']} (LOF={m['n_lof']}, GOF={m['n_gof']})")
        print(f"    GOF threshold (var_score >= ): {m['gof_threshold']:.2f}")
        print(f"    Overall Accuracy:  {m['overall_acc']:.4f}")
        print(f"    LOF Recall:        {m['lof_recall']:.4f}")
        print(f"    GOF Recall:        {m['gof_recall']:.4f}")
        print(f"    Precision/Recall:  {m['precision']:.4f} / {m['recall']:.4f}")
        print(f"    F1:                {m['f1']:.4f}")
        print(f"    AUROC:             {m['auroc']:.4f}")
        print(f"    AUPRC:             {m['auprc']:.4f}")

    # Per-entry score rows
    score_rows = []
    for _, row in benchmark.iterrows():
        pid   = row["protein_id"]
        snp   = row["snp"]
        go_id = row["matched_go_id"]
        if go_id not in go_vocab or (pid, snp) not in var_scores:
            continue
        go_idx = go_vocab[go_id]
        var_s  = float(var_scores[(pid, snp)][go_idx])
        score_rows.append({
            "protein_id":      pid,
            "snp":             snp,
            "label":           row["label"],
            "matched_go_id":   go_id,
            "matched_go_name": row.get("matched_go_name", ""),
            "var_score":       var_s,
            "split":           row.get("split", ""),
            "aspect":          cfg.aspect,
            "variant_h5":      Path(cfg.variant_h5).name,
        })

    return {
        "aspect":     cfg.aspect,
        "variant_h5": Path(cfg.variant_h5).name,
        "checkpoint": Path(cfg.checkpoint).name,
        "results":    results,
        "score_rows": score_rows,
    }


# ── Summary Table ────────────────────────────────────────────────────────────

def print_summary_table(all_results: list):
    """Print a formatted summary table of all evaluations."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        for split_name in ["val", "test"]:
            if split_name not in res["results"]:
                continue
            sr = res["results"][split_name]
            rows.append({
                "Variant H5":  res["variant_h5"],
                "Checkpoint":  res["checkpoint"],
                "Aspect":      res["aspect"],
                "Split":       split_name,
                "N":           sr["n_total"],
                "LOF":         sr["n_lof"],
                "GOF":         sr["n_gof"],
                "GOF_Thr":     sr["gof_threshold"],
                "Acc":         sr["overall_acc"],
                "LOF_Rec":     sr["lof_recall"],
                "GOF_Rec":     sr["gof_recall"],
                "F1":          sr["f1"],
                "AUROC":       sr["auroc"],
                "AUPRC":       sr["auprc"],
            })

    if not rows:
        print("\nNo results to display.")
        return

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("  BENCHMARK EVALUATION SUMMARY -- LOF/GOF Prediction (variant embedding)")
    print("=" * 90)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print()

    return df


def save_scores_csv(all_score_rows: list, output_path: str = "benchmark_scores.csv"):
    """Save per-entry variant decoder scores.
    Columns: protein_id, snp, label, matched_go_id, matched_go_name,
             var_score, split, aspect, variant_h5
    """
    if not all_score_rows:
        return
    df = pd.DataFrame(all_score_rows)
    df = df.sort_values(["aspect", "split", "protein_id", "snp", "label"])
    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Per-entry scores saved to {output_path}  ({len(df)} rows)")


def save_results_csv(all_results: list, output_path: str = "benchmark_results.csv"):
    """Save detailed results to CSV."""
    rows = []
    for res in all_results:
        if res is None:
            continue
        for split_name in ["val", "test"]:
            if split_name not in res["results"]:
                continue
            sr = res["results"][split_name]
            rows.append({
                "variant_h5": res["variant_h5"],
                "checkpoint": res["checkpoint"],
                "aspect":     res["aspect"],
                "split":      split_name,
                **sr,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate decoder on variant-specific GO benchmark"
    )
    parser.add_argument("--aspect", default=None, choices=["F", "P", "C"],
                        help="GO aspect — F=MF, P=BP, C=CC (default: all three)")
    parser.add_argument("--embeddings_h5", default=None,
                        help="WT protein embeddings H5, keyed by protein_id")
    parser.add_argument("--variant_h5", default=None, required=True,
                        help="Variant embeddings H5, keyed by {protein_id}_{snp}")
    parser.add_argument("--variant_h5_secondary", default=None,
                        help="concat_phase0/concat_phase1 checkpoint'leri icin ikinci varyant embedding H5'i "
                             "(--variant_h5 ile ayni key formati)")
    parser.add_argument("--checkpoint", default=None,
                        help="Decoder checkpoint .pt")
    parser.add_argument("--go_vocab_json", default=None,
                        help="GO vocab JSON (must match checkpoint)")
    parser.add_argument("--propagation", action="store_true",
                        help="go_vocab_json bulunamayip sifirdan kurulmasi gerektiginde GO ancestor "
                             "propagation kullan (varsayilan kapali, grid_search.py'nin varsayilanini yansitir)")
    parser.add_argument("--output", default="benchmark_results.csv",
                        help="Output CSV path (default: benchmark_results.csv)")
    args = parser.parse_args()

    aspects = [args.aspect] if args.aspect else ["F", "P", "C"]

    all_results = []
    for aspect in aspects:
        cfg = BenchmarkConfig(aspect=aspect, use_go_propagation=args.propagation)
        if args.embeddings_h5:
            cfg.embeddings_h5 = args.embeddings_h5
        if args.variant_h5:
            cfg.variant_h5 = args.variant_h5
        if args.variant_h5_secondary:
            cfg.variant_h5_secondary = args.variant_h5_secondary
        if args.checkpoint:
            cfg.checkpoint = args.checkpoint
        if args.go_vocab_json:
            cfg.go_vocab_json = args.go_vocab_json

        if not Path(cfg.checkpoint).exists():
            print(f"\n[WARN] Checkpoint not found: {cfg.checkpoint} -- skipping aspect={aspect}")
            continue

        result = evaluate_benchmark(cfg)
        if result is not None:
            all_results.append(result)

    # Summary
    df = print_summary_table(all_results)
    save_results_csv(all_results, args.output)

    all_score_rows = [row for r in all_results for row in r.get("score_rows", [])]
    scores_path = args.output.replace(".csv", "_scores.csv")
    save_scores_csv(all_score_rows, scores_path)


if __name__ == "__main__":
    main()
