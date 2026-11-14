"""
Toplu varyant fonksiyon değişim tahmini.

embeddings_variable.h5'teki her varyant için:
    1. WT UniProt ID'yi key'den parse et
    2. WT GO anotasyonlarını GOA'dan al
    3. Varyant embedding'i mean-pool yap: (L, 1280) → (1280,)
    4. predict_function_change() çağır
    5. Sonuçları CSV'ye yaz

Kullanım:
    python -m Sekans_decoder.predict_variants
    python -m Sekans_decoder.predict_variants --var_h5 embeddings_variable.h5
"""

import argparse
import csv
import re
from contextlib import ExitStack

import h5py
import torch
import numpy as np
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from .config import DecoderConfig, pick_device
    from .data.parse_goa import parse_goa_tsv
    from .inference import load_decoder, predict_function_change
except ImportError:
    from config import DecoderConfig, pick_device
    from data.parse_goa import parse_goa_tsv
    from inference import load_decoder, predict_function_change


DEVICE = torch.device(pick_device())

# UniProt accession regex
_UNIPROT_RE  = re.compile(
    r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$'
)
_MUTATION_RE = re.compile(r'^[A-Z]\d+[A-Z*]$')


def mean_pool(h5f, key: str, secondary_h5f=None) -> torch.Tensor:
    """
    (L, 1280) → (1280,) mean pooling. Zaten (1280,) ise olduğu gibi döner.

    secondary_h5f verilirse (concat_phase0/concat_phase1 çalıştırmaları için),
    aynı key ikinci dosyadan da okunup [primary | secondary] concat edilir.
    """
    emb = torch.FloatTensor(h5f[key][:])
    if emb.dim() == 2:
        emb = emb.mean(dim=0)
    if secondary_h5f is not None:
        emb_sec = torch.FloatTensor(secondary_h5f[key][:])
        if emb_sec.dim() == 2:
            emb_sec = emb_sec.mean(dim=0)
        emb = torch.cat([emb, emb_sec], dim=-1)
    return emb


def parse_wt_id(variant_key: str, goa_ids: set) -> str | None:
    """
    Key içindeki UniProt accession adaylarını bulur,
    GOA'da olanı döndürür. Bulunamazsa None.
    """
    for part in variant_key.split("_"):
        if _UNIPROT_RE.match(part) and part in goa_ids:
            return part
    return None


def run(var_h5: str, out_csv: str, cfg: DecoderConfig, secondary_var_h5: str | None = None):
    # GO annotations
    print("Loading GO annotations...")
    annotations = parse_goa_tsv(cfg.goa_tsv, aspect=cfg.aspect)

    # Decoder + vocab
    print("Loading decoder...")
    decoder, go_vocab = load_decoder(
        checkpoint=cfg.decoder_checkpoint,
        go_vocab_json=cfg.go_vocab_json,
        encoder_output_dim=cfg.encoder_output_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )

    # WandB
    use_wandb = HAS_WANDB
    if use_wandb:
        try:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"predict-variants-{cfg.aspect.lower()}",
                config={"var_h5": var_h5, "out_csv": out_csv,
                        "aspect": cfg.aspect},
            )
        except Exception:
            use_wandb = False

    # CSV header
    fieldnames = ["variant_id", "wt_id", "exp_label", "ContVAR_fitness",
                  "n_loss", "n_gain", "n_neutral", "wt_go_count"]

    skipped = 0
    all_fitness: list[float] = []
    all_labels:  list[int]   = []

    with ExitStack() as stack:
        h5f = stack.enter_context(h5py.File(var_h5, "r"))
        h5f_sec = (stack.enter_context(h5py.File(secondary_var_h5, "r"))
                   if secondary_var_h5 else None)
        fout = stack.enter_context(open(out_csv, "w", newline=""))

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        keys = list(h5f.keys())
        print(f"Total variants: {len(keys)}")

        for key in tqdm(keys, desc="Predicting"):
            parts = key.split("_")
            if not any(_MUTATION_RE.match(p) for p in parts):
                skipped += 1
                continue

            wt_id = parse_wt_id(key, set(annotations.keys()))
            if wt_id is None:
                skipped += 1
                continue

            if h5f_sec is not None and key not in h5f_sec:
                skipped += 1
                continue

            try:
                exp_label = int(parts[-1])
            except ValueError:
                exp_label = -1

            wt_go_terms = annotations[wt_id]
            var_emb = mean_pool(h5f, key, secondary_h5f=h5f_sec)
            result  = predict_function_change(wt_go_terms, var_emb, decoder, go_vocab)

            fitness   = result.pop("ContVAR_fitness")
            n_loss    = sum(1 for v in result.values() if v["status"] == "LOSS")
            n_gain    = sum(1 for v in result.values() if v["status"] == "GAIN")
            n_neutral = sum(1 for v in result.values() if v["status"] == "NEUTRAL")

            writer.writerow({
                "variant_id":      key,
                "wt_id":           wt_id,
                "exp_label":       exp_label,
                "ContVAR_fitness": fitness,
                "n_loss":          n_loss,
                "n_gain":          n_gain,
                "n_neutral":       n_neutral,
                "wt_go_count":     len(wt_go_terms),
            })

            if exp_label in (0, 1):
                all_fitness.append(fitness)
                all_labels.append(exp_label)

    print(f"\nDone → {out_csv}")
    print(f"Skipped (no GO annotation): {skipped}")

    if HAS_SKLEARN and len(all_labels) >= 2 and len(set(all_labels)) == 2:
        fitness_arr = np.array(all_fitness)
        label_arr   = np.array(all_labels)

        auroc = roc_auc_score(label_arr, fitness_arr)
        auprc = average_precision_score(label_arr, fitness_arr)

        print(f"AUROC (fitness vs exp_label): {auroc:.4f}")
        print(f"AUPRC (fitness vs exp_label): {auprc:.4f}")

        if use_wandb:
            wandb.log({
                "n_predicted": len(all_fitness),
                "n_skipped":   skipped,
                "AUROC":       auroc,
                "AUPRC":       auprc,
            })

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--var_h5", default="embeddings_variable.h5")
    parser.add_argument("--secondary_var_h5", default=None,
                        help="concat_phase0/concat_phase1 checkpoint'leri için ikinci varyant embedding H5'i "
                             "(--var_h5 ile aynı key formatı: {protein_id}_{snp})")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--aspect", default="F", choices=["F", "P", "C"])
    args = parser.parse_args()

    cfg = DecoderConfig(aspect=args.aspect)
    out = args.out or f"predictions_{args.aspect.lower()}.csv"
    run(args.var_h5, out, cfg, secondary_var_h5=args.secondary_var_h5)
