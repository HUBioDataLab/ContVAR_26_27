"""
LOF/GOF delta sinyali teşhis analizi.

Her pipeline varyantı (raw / phase_0 / phase_1) için:
  - delta = wt_score - var_score  dağılımını LOF vs GOF için ayrı çıkarır
  - örtüşmeyi (overlap), AUROC'u ve sinyalin "ölü" olup olmadığını raporlar
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from benchmark_eval import (
    load_protein_splits,
    load_embeddings,
    load_variant_embeddings,
    load_decoder_model,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENCHMARK_TSV = ROOT / "data" / "raw" / "variant_specific_go_benchmark.tsv"
SPLIT_JSON = ROOT / "phase0_protein_split_removed_graphless.json"
UNIREF_TSV = ROOT / "phase_0_go_pretrained" / "protein_uniref50.tsv"

VARIANTS = {
    "raw": {
        "ckpt":  ROOT / "artifacts/pipeline_run/decoder/decoder_best_f_raw.pt",
        "emb":   ROOT / "artifacts/pipeline_run/decoder/raw_embeddings.h5",
        "var":   ROOT / "data/processed/variant_embeddings_raw.h5",
        "vocab": ROOT / "artifacts/pipeline_run/decoder/go_vocab_f_raw.json",
    },
    "phase_0": {
        "ckpt":  ROOT / "artifacts/pipeline_run/decoder/decoder_best_f_phase_0.pt",
        "emb":   ROOT / "artifacts/pipeline_run/decoder/phase_0_embeddings.h5",
        "var":   ROOT / "data/processed/variant_embeddings_phase_0.h5",
        "vocab": ROOT / "artifacts/pipeline_run/decoder/go_vocab_f_phase_0.json",
    },
    "phase_1": {
        "ckpt":  ROOT / "artifacts/pipeline_run/decoder/decoder_best_f_phase_1.pt",
        "emb":   ROOT / "artifacts/pipeline_run/decoder/phase1_650M_UR50D_phase1_embeddings.h5",
        "var":   ROOT / "data/processed/variant_embeddings_phase_1.h5",
        "vocab": ROOT / "artifacts/pipeline_run/decoder/go_vocab_f_phase_1.json",
    },
}


def stat_line(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return f"    {name:5s}: (boş)"
    return (f"    {name:5s}: n={len(arr):4d}  mean={arr.mean():+.4f}  "
            f"std={arr.std():.4f}  median={np.median(arr):+.4f}  "
            f"min={arr.min():+.4f}  max={arr.max():+.4f}")


def analyze(tag, paths):
    print(f"\n{'='*72}\n  VARYANT: {tag}\n{'='*72}")
    for k, p in paths.items():
        if not Path(p).exists():
            print(f"  [SKIP] eksik dosya: {k} -> {p}")
            return

    with open(paths["vocab"]) as f:
        go_vocab = json.load(f)
    vocab_gos = set(go_vocab.keys()) - {"NULL_FUNCTION"}
    n_classes = len(go_vocab)

    bm = pd.read_csv(BENCHMARK_TSV, sep="\t")
    bm = bm[bm["matched_go_id"].isin(vocab_gos)]
    p2s = load_protein_splits(str(UNIREF_TSV), str(SPLIT_JSON))
    bm = bm.copy()
    bm["split"] = bm["protein_id"].map(p2s)

    proteins = bm["protein_id"].unique().tolist()
    wt_emb = load_embeddings(proteins, str(paths["emb"]))
    bm = bm[bm["protein_id"].isin(wt_emb)]

    pairs = list(bm[["protein_id", "snp"]].drop_duplicates().itertuples(index=False, name=None))
    var_emb = load_variant_embeddings(pairs, str(paths["var"]))
    bm = bm[bm.apply(lambda r: (r["protein_id"], r["snp"]) in var_emb, axis=1)]

    if len(bm) == 0:
        print("  [SKIP] eşleşen satır yok")
        return

    decoder = load_decoder_model(str(paths["ckpt"]), n_classes)

    # skorlar
    wt_scores = {}
    with torch.no_grad():
        pids = list(wt_emb.keys())
        for i in range(0, len(pids), 256):
            b = pids[i:i+256]
            x = torch.stack([wt_emb[p] for p in b]).to(DEVICE)
            pr = torch.sigmoid(decoder(x)).cpu().numpy()
            for j, p in enumerate(b):
                wt_scores[p] = pr[j]
    var_scores = {}
    with torch.no_grad():
        vp = list(var_emb.keys())
        for i in range(0, len(vp), 256):
            b = vp[i:i+256]
            x = torch.stack([var_emb[p] for p in b]).to(DEVICE)
            pr = torch.sigmoid(decoder(x)).cpu().numpy()
            for j, p in enumerate(b):
                var_scores[p] = pr[j]

    for split in ["val", "test"]:
        sdf = bm[bm["split"] == split]
        if len(sdf) == 0:
            continue
        deltas, labels, wt_v, var_v, emb_l2 = [], [], [], [], []
        for _, row in sdf.iterrows():
            pid, snp, go = row["protein_id"], row["snp"], row["matched_go_id"]
            pair = (pid, snp)
            if go not in go_vocab or pid not in wt_scores or pair not in var_scores:
                continue
            gi = go_vocab[go]
            ws = float(wt_scores[pid][gi]); vs = float(var_scores[pair][gi])
            deltas.append(ws - vs)          # benchmark_eval signal = wt - var
            wt_v.append(ws); var_v.append(vs)
            labels.append(1 if row["label"] == "LOF" else 0)
            emb_l2.append(float(torch.norm(wt_emb[pid] - var_emb[pair]).item()))

        deltas = np.array(deltas); labels = np.array(labels)
        wt_v = np.array(wt_v); var_v = np.array(var_v); emb_l2 = np.array(emb_l2)
        lof = labels == 1; gof = labels == 0

        print(f"\n  [{split.upper()}]  n={len(labels)}  (LOF={lof.sum()}, GOF={gof.sum()})")
        print("  -- delta = wt_score - var_score (LOF'da pozitif beklenir) --")
        print(stat_line("LOF", deltas[lof]))
        print(stat_line("GOF", deltas[gof]))
        # ham skorlar
        print("  -- ham wt_score --")
        print(stat_line("LOF", wt_v[lof]))
        print(stat_line("GOF", wt_v[gof]))
        # embedding hareketi
        print("  -- ||wt_emb - var_emb||_2 (embedding mutasyona ne kadar duyarlı) --")
        print(stat_line("LOF", emb_l2[lof]))
        print(stat_line("GOF", emb_l2[gof]))

        # ne kadar delta tam sıfır (sinyal yok)
        zero_frac = float((np.abs(deltas) < 1e-6).mean())
        print(f"  -- |delta|<1e-6 oranı: {zero_frac:.3f}  (yüksekse sinyal ölü)")

        if len(np.unique(labels)) == 2:
            try:
                print(f"  -- AUROC(delta): {roc_auc_score(labels, deltas):.4f}")
                print(f"  -- AUROC(wt_score): {roc_auc_score(labels, wt_v):.4f}")
            except ValueError:
                pass


if __name__ == "__main__":
    for tag, paths in VARIANTS.items():
        analyze(tag, paths)
