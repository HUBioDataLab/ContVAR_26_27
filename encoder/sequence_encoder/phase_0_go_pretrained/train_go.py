# go_pretrain/train_go.py
# GO Pretraining: ESM2 backbone + 3 GO head ile triplet loss
#
# Kullanım:
#   python train_go.py \
#       --dataset data/processed/go_pretrain_dataset.pkl \
#       --split_file data/splits/my_split.json \
#       --esm_model esm2_t6_8M_UR50D \
#       --epochs 30 \
#       --batch_size 8 \
#       --out models/go_pretrained
#
# Split JSON formatı:
#   {
#     "version": 1,
#     "seed": 42,
#     "train_ratio": 0.8,
#     "val_ratio": 0.1,
#     "test_ratio": 0.1,
#     "group_to_split": {
#       "UniRef50_Q9VHS2": "train",
#       "UniRef50_P39715": "val",
#       "UniRef50_O59732": "test",
#       ...
#     }
#   }
#
# Stage 1 tamamlandıktan sonra:
#   → models/go_pretrained/{run_name}_full_model_best.pt  (ESM2 + 3 head — en iyi val loss)
#   → models/go_pretrained/{run_name}_full_model_last.pt  (ESM2 + 3 head — son epoch)
#   → models/go_pretrained/{run_name}_latest_checkpoint.pt (tam eğitim durumu, resume için)
#   → models/go_pretrained/checkpoints/{run_name}_epoch_XXXX.pt (periyodik checkpoint'ler)
#
# Downstream (Phase1, Decoder) için ESM2 ağırlıkları full_model_best.pt'den çekilir:
#   esm_state_dict = torch.load("{run_name}_full_model_best.pt")["esm_state_dict"]

import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from .metrics import compute_detailed_metrics, compute_retrieval_mrr, InterAndIntraClassSimilarity
    from .go_model import GOPretrainModel, load_esm2_model, tokenize_sequences
except ImportError:
    from metrics import compute_detailed_metrics, compute_retrieval_mrr, InterAndIntraClassSimilarity
    from go_model import GOPretrainModel, load_esm2_model, tokenize_sequences
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =============================================
# CONFIG
# =============================================
EPOCHS = 30
BATCH_SIZE = 4        # ESM2 fine-tuning bellek yoğun, küçük batch
ACCUM_STEPS = 4       # Gradient accumulation: effective batch = BATCH_SIZE * ACCUM_STEPS (4*4=16)
LR = 5e-6             # ESM2 backbone için daha konservatif fine-tune LR
LR_HEADS = 1e-4       # Head'ler için daha yüksek LR
MARGIN = 0.3
OUT_DIM = 256
VAL_EVERY_SAMPLES = 2000


# =============================================
# Checkpoint Naming Helper
# =============================================

def ts():
    """Returns a compact datetime string for checkpoint filenames, e.g. '20250407_153022'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================
# Checkpoint Save / Load / Cleanup
# =============================================

def save_checkpoint(path, model, optimizer, epoch, global_step,
                    best_val_loss, best_train_loss):
    """Save full resumable training checkpoint (model + optimizer + training state)."""
    torch.save({
        'epoch':                epoch,
        'global_step':          global_step,
        'best_val_loss':        best_val_loss,
        'best_train_loss':      best_train_loss,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"[CKPT] Training state saved → {path}")


def load_checkpoint(path, model, optimizer, device):
    """
    Load a resumable checkpoint into model and optimizer.
    Returns (start_epoch, global_step, best_val_loss, best_train_loss).
    start_epoch = last completed epoch + 1.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch     = ckpt['epoch'] + 1
    global_step     = ckpt['global_step']
    best_val_loss   = ckpt.get('best_val_loss',   float('inf'))
    best_train_loss = ckpt.get('best_train_loss', float('inf'))
    print(
        f"[RESUME] Checkpoint yüklendi → epoch {start_epoch}'dan devam, "
        f"step={global_step}, best_val={best_val_loss:.4f}"
    )
    return start_epoch, global_step, best_val_loss, best_train_loss


def cleanup_old_checkpoints(ckpt_dir, keep_n, run_name=""):
    """Delete oldest periodic checkpoints, keeping the last keep_n files."""
    if keep_n <= 0:
        return
    prefix = f"{run_name}_epoch_" if run_name else "epoch_"
    ckpts = sorted(ckpt_dir.glob(f"{prefix}*.pt"))
    for old in ckpts[:-keep_n]:
        old.unlink()
        print(f"[CKPT] Eski checkpoint silindi: {old.name}")


# =============================================
# Split File Utilities
# =============================================

def load_split_file(split_path):
    """
    Split JSON dosyasını yükler ve protein_id → split eşlemesini döndürür.

    Args:
        split_path: JSON dosya yolu

    Returns:
        protein_to_split: dict {protein_id: "train" | "val" | "test"}
        meta: dict (version, seed, ratios, ...)
    """
    with open(split_path, 'r') as f:
        data = json.load(f)

    if "protein_to_split" not in data:
        raise ValueError(
            "Split dosyasi 'protein_to_split' alanini icermeli. "
            "Bu kod artik sadece protein-level split formatini destekliyor."
        )

    split_mapping = data.get("protein_to_split", {})

    # Tüm değerlerin geçerli olduğunu kontrol et
    valid_splits = {"train", "val", "test"}
    invalid = {k: v for k, v in split_mapping.items() if v not in valid_splits}
    if invalid:
        raise ValueError(
            f"Split dosyasında geçersiz split değerleri var: {list(invalid.items())[:5]} ..."
        )

    meta = {
        k: v for k, v in data.items()
        if k not in {"group_to_split", "protein_to_split"}
    }
    meta["split_level"] = "protein"

    counts = {"train": 0, "val": 0, "test": 0}
    for v in split_mapping.values():
        counts[v] += 1

    print(f"[Split] Loaded from: {split_path}")
    print(f"  level=protein, version={meta.get('version')}, seed={meta.get('seed')}")
    print(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return split_mapping, meta


def resolve_cluster_map_path(split_path, meta):
    """
    Split JSON içindeki cluster map yolunu lokal dosya sisteminde çöz.

    Öncelik sırası:
        1. JSON içindeki path doğrudan mevcutsa
        2. JSON path'inin basename'i split dosyasıyla aynı klasörde varsa
        3. split dosyası klasöründeki 'protein_uniref50.tsv'
    """
    split_dir = Path(split_path).resolve().parent

    raw_cluster_map = None
    if isinstance(meta.get("meta"), dict):
        raw_cluster_map = meta["meta"].get("cluster_map_path")

    candidates = []
    if raw_cluster_map:
        raw_path = Path(raw_cluster_map)
        candidates.append(raw_path)
        candidates.append(split_dir / Path(str(raw_cluster_map).replace("\\", "/")).name)

    candidates.append(split_dir / "protein_uniref50.tsv")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def expand_group_to_split_via_cluster_map(group_to_split, cluster_map_path):
    """
    group_id -> split eşleşmesini protein_id -> split formatına genişlet.
    """
    protein_to_split = {}
    conflicts = 0

    with open(cluster_map_path, "r") as f:
        header = f.readline().strip().split("\t")
        if header[:2] != ["protein_id", "group_id"]:
            raise ValueError(
                f"Beklenmeyen cluster map başlığı: {header[:2]}. "
                "Beklenen: ['protein_id', 'group_id']"
            )

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            protein_id, group_id = line.split("\t")[:2]
            split_name = group_to_split.get(group_id)
            if split_name is None:
                continue

            prev = protein_to_split.get(protein_id)
            if prev is not None and prev != split_name:
                conflicts += 1
                continue
            protein_to_split[protein_id] = split_name

    return protein_to_split, conflicts


def _strip_prefix(protein_id):
    """
    ID prefix normalizasyonu:
        'UniRef50_Q9VHS2'  → 'Q9VHS2'
        'UniRef90_Q9VHS2'  → 'Q9VHS2'
        'UniRef100_Q9VHS2' → 'Q9VHS2'
        'sp|Q9VHS2|NAME'   → 'Q9VHS2'
        'tr|Q9VHS2|NAME'   → 'Q9VHS2'
        'Q9VHS2'           → 'Q9VHS2'   (zaten temiz)
    """
    import re
    m = re.match(r'^UniRef\d+_(.+)$', protein_id, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r'^(?:sp|tr)\|([^|]+)\|', protein_id)
    if m:
        return m.group(1)
    return protein_id


def normalize_group_to_split(group_to_split, dataset_proteins):
    """
    Split dosyasındaki protein ID'lerini dataset formatına normalize eder.

    Sorun: split dosyası  → 'UniRef50_Q9VHS2'
           dataset         → 'Q9VHS2'
    (veya tam tersi)

    Strateji:
        1. Ham haliyle %5'ten fazlası eşleşiyorsa → dönüşüm gerekmez.
        2. Aksi halde her iki tarafı prefix'ten arındırarak eşleştir.
        3. Yine de eşleşme bulunamazsa orijinal key'i koru.

    Returns:
        normalized:     dict {dataset_format_id: split_label}
        was_stripped:   bool — dönüşüm yapıldıysa True
    """
    dataset_ids = set(dataset_proteins)

    direct_hits = sum(1 for k in group_to_split if k in dataset_ids)
    total       = len(group_to_split)

    if total > 0 and direct_hits / total >= 0.05:
        return dict(group_to_split), False

    # Dataset tarafını normalize et: stripped_id → orijinal_id
    stripped_dataset = {_strip_prefix(d): d for d in dataset_ids}

    normalized = {}
    for key, label in group_to_split.items():
        stripped_key = _strip_prefix(key)
        if stripped_key in stripped_dataset:
            normalized[stripped_dataset[stripped_key]] = label
        else:
            normalized[key] = label  # Eşleşme bulunamadı, orijinali koru

    return normalized, True


def check_split_dataset_consistency(group_to_split, dataset_proteins, max_print=20):
    """
    Split dosyası ile pickle dataset arasındaki protein uyumsuzluklarını raporlar.
    Prefix farkını (UniRef50_ vs düz accession) otomatik tespit edip normalize eder.

    İki yönlü kontrol:
        1. Split dosyasında olup dataset'te OLMAYAN proteinler (sekans yok → kullanılamaz)
        2. Dataset'te olup split dosyasında OLMAYAN proteinler (split atanmamış → dışlanır)

    Args:
        group_to_split:   dict {protein_id: split_label}
        dataset_proteins: iterable — pickle'dan gelen tüm protein id'leri
        max_print:        Listelenecek maksimum örnek sayısı

    Returns:
        normalized_group_to_split: prefix normalize edilmiş dict (train_go'da kullanılır)
    """
    dataset_proteins = list(dataset_proteins)
    dataset_ids      = set(dataset_proteins)

    # Önce normalize et
    normalized, was_stripped = normalize_group_to_split(group_to_split, dataset_proteins)

    print("\n[Split Consistency Check]")
    if was_stripped:
        sample_raw  = next(iter(group_to_split))
        sample_norm = next(iter(normalized))
        print(f"  [!] ID format uyuşmazlığı tespit edildi — prefix otomatik normalize edildi.")
        print(f"      Split örneği   : '{sample_raw}'")
        print(f"      Dataset örneği : '{sample_norm}'")
    else:
        print(f"  [OK] ID formatları uyuşuyor, normalizasyon gerekmedi.")

    split_ids       = set(normalized.keys())
    only_in_split   = split_ids - dataset_ids
    only_in_dataset = dataset_ids - split_ids

    print(f"\n  Split dosyasındaki protein sayısı  : {len(group_to_split)}")
    print(f"  Dataset'teki protein sayısı        : {len(dataset_ids)}")
    print(f"  Normalize sonrası kesişim          : {len(split_ids & dataset_ids)}")

    if only_in_split:
        print(f"\n  [!] Split'te olup dataset'te OLMAYAN: {len(only_in_split)} protein")
        print(      "      → Sekansları yok, eğitimde kullanılamayacak.")
        for p in sorted(only_in_split)[:max_print]:
            print(f"      - {p}  [{normalized[p]}]")
        if len(only_in_split) > max_print:
            print(f"      ... ve {len(only_in_split) - max_print} tane daha.")
    else:
        print("\n  [OK] Split dosyasındaki tüm proteinler dataset'te mevcut.")

    if only_in_dataset:
        print(f"\n  [!] Dataset'te olup split'e ATANMAMIŞ: {len(only_in_dataset)} protein")
        print(      "      → Hiçbir split'e dahil edilmeyecek, eğitimde kullanılmayacak.")
        for p in sorted(only_in_dataset)[:max_print]:
            print(f"      - {p}")
        if len(only_in_dataset) > max_print:
            print(f"      ... ve {len(only_in_dataset) - max_print} tane daha.")
    else:
        print("\n  [OK] Dataset'teki tüm proteinler split dosyasına atanmış.")

    print()
    return normalized


def filter_proteins_by_split(all_proteins, group_to_split, split):
    """
    Protein listesini verilen split'e göre filtreler.
    group_to_split'in normalize_group_to_split() ile işlenmiş olması beklenir.

    Args:
        all_proteins:   list of protein ids
        group_to_split: dict {protein_id: split_label} — normalize edilmiş
        split:          "train" | "val" | "test"

    Returns:
        filtered: list of protein ids
    """
    if not group_to_split:
        return list(all_proteins)

    filtered = [p for p in all_proteins if group_to_split.get(p) == split]
    return filtered


# =============================================
# GO Triplet Dataset
# =============================================


NEG_BUCKETS = [(0.1, 0.2), (0.0, 0.1)]  # hard first, easy second


class GOMultiAspectDataset(Dataset):

    def __init__(self, dataset_path, samples_per_epoch=10000,
             group_to_split=None, split='train',
             aspect_weights=None):

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        self.protein_seqs = data['protein_seqs']
        self.raw_semantic_pairs = data['semantic_pairs']
        self.samples_per_epoch = samples_per_epoch
        self.split = split
        # ---- Validate dataset has score fields ----
        _sample_asp = next(iter(self.raw_semantic_pairs))
        _sample_anchor = next(iter(self.raw_semantic_pairs[_sample_asp]))
        _sample_cands = self.raw_semantic_pairs[_sample_asp][_sample_anchor]

        if 'neg_scores' not in _sample_cands:
            raise RuntimeError(
                "Dataset is missing 'neg_scores' — balanced bucket sampling requires "
                "a regenerated dataset. Run build_go_dataset.py and rebuild the pickle."
            )
        group_to_split = group_to_split or {}
        all_proteins = list(self.protein_seqs.keys())
        self.all_proteins = filter_proteins_by_split(all_proteins, group_to_split, split)
        self.all_proteins_set = set(self.all_proteins)

        self.aspects = []
        self.valid_anchors = {'F': [], 'P': [], 'C': []}
        self.split_pairs   = {'F': {}, 'P': {}, 'C': {}}

        for asp in ('F', 'P', 'C'):
            for anchor, cands in self.raw_semantic_pairs[asp].items():
                if anchor not in self.all_proteins_set:
                    continue

                valid_pos = [p for p in cands['pos'] if p in self.all_proteins_set]

                # ── Bucket negatives by similarity score ──────────────────
                neg_ids    = cands['neg']
                neg_scores = cands.get('neg_scores', [0.0] * len(neg_ids))

                bucketed_neg = {bucket: [] for bucket in NEG_BUCKETS}
                for nid, score in zip(neg_ids, neg_scores):
                    if nid not in self.all_proteins_set:
                        continue
                    for (lo, hi) in NEG_BUCKETS:
                        if lo <= score < hi:
                            bucketed_neg[(lo, hi)].append(nid)
                            break

                non_empty_buckets = {k: v for k, v in bucketed_neg.items() if v}
                # ─────────────────────────────────────────────────────────

                if len(valid_pos) > 0 and len(non_empty_buckets) > 0:
                    self.valid_anchors[asp].append(anchor)
                    self.split_pairs[asp][anchor] = {
                        'pos':         valid_pos,
                        'neg_buckets': non_empty_buckets,
                        'bucket_keys': list(non_empty_buckets.keys()),
                    }

            if len(self.valid_anchors[asp]) > 0:
                self.aspects.append(asp)
        
        default_weights = {'F': 0.5, 'P': 0.3, 'C': 0.2}
        
        raw = aspect_weights or default_weights
        total = sum(raw[a] for a in self.aspects)
        self.aspect_probs = np.array(
            [raw[a] / total for a in self.aspects]
        )
        
        print(f"[GOMultiAspectDataset/{split}] Proteins in split: {len(self.all_proteins)}")
        if not self.aspects:
            print(f"  [WARNING] No valid semantic triplets found for split '{split}'!")
        else:
            for asp in self.aspects:
                label = {'F': 'MF', 'P': 'BP', 'C': 'CC'}[asp]
                n_anchors = len(self.valid_anchors[asp])
                # Count how many anchors have both buckets populated
                both = sum(
                    1 for a in self.valid_anchors[asp]
                    if len(self.split_pairs[asp][a]['bucket_keys']) == len(NEG_BUCKETS)
                )
                print(f"  {label}: {n_anchors} valid anchors "
                      f"({both} with both neg buckets populated)")
    
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if not self.aspects:
            raise RuntimeError(
                f"Cannot sample: No valid triplets in split '{self.split}'. "
                "Check your dataset or split ratios."
            )

        aspect = np.random.choice(self.aspects, p=self.aspect_probs)
        anchor_id = np.random.choice(self.valid_anchors[aspect])
        entry     = self.split_pairs[aspect][anchor_id]

        positive_id = np.random.choice(entry['pos'])

        # ── Balanced negative sampling ────────────────────────────────────
        # 1. Pick a bucket uniformly (each similarity range equally likely)
        bucket_keys  = entry['bucket_keys']
        chosen_bucket = bucket_keys[np.random.randint(len(bucket_keys))]
        # 2. Pick a negative uniformly within that bucket
        negative_id  = np.random.choice(entry['neg_buckets'][chosen_bucket])
        # ─────────────────────────────────────────────────────────────────

        return (
            self.protein_seqs[anchor_id],
            self.protein_seqs[positive_id],
            self.protein_seqs[negative_id],
            aspect,
        )
    
def collate_go_batch(batch, alphabet, max_len=1022):
    """
    Custom collate: string sekansları tokenize eder.

    Args:
        batch:    list of (anchor_seq, pos_seq, neg_seq, aspect)
        alphabet: ESM2 alphabet
        max_len:  Max sekans uzunluğu

    Returns:
        anchor_tokens, pos_tokens, neg_tokens: (B, L) padded tensors
        aspects: list of aspect strings
    """
    anchor_seqs = [(f"a_{i}", b[0][:max_len]) for i, b in enumerate(batch)]
    pos_seqs    = [(f"p_{i}", b[1][:max_len]) for i, b in enumerate(batch)]
    neg_seqs    = [(f"n_{i}", b[2][:max_len]) for i, b in enumerate(batch)]
    aspects     = [b[3] for b in batch]

    converter = alphabet.get_batch_converter()

    _, _, anchor_tokens = converter(anchor_seqs)
    _, _, pos_tokens    = converter(pos_seqs)
    _, _, neg_tokens    = converter(neg_seqs)

    return anchor_tokens, pos_tokens, neg_tokens, aspects


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss       = 0.0
    loss_per_aspect  = {'F': 0.0, 'P': 0.0, 'C': 0.0}
    count_per_aspect = {'F': 0,   'P': 0,   'C': 0}

    # Her aspect için metrik biriktiriciler — loss'tan tamamen bağımsız
    metric_sums  = {asp: {} for asp in ('F', 'P', 'C')}
    metric_counts = {'F': 0, 'P': 0, 'C': 0}
    retrieval_pools = {
        asp: {"anchor": [], "pos": [], "neg": []}
        for asp in ('F', 'P', 'C')
    }
    # Epoch-level inter/intra class similarity — embeddings collected across all batches,
    # computed once at epoch end (same pattern as MRR_FullValPool)
    class_sim_pools = {
        asp: {"preds": [], "labels": [], "next_class_id": 0}
        for asp in ('F', 'P', 'C')
    }

    for anchor_tokens, pos_tokens, neg_tokens, aspects in dataloader:
        anchor_tokens = anchor_tokens.to(device)
        pos_tokens    = pos_tokens.to(device)
        neg_tokens    = neg_tokens.to(device)

        _, z_mf_a, z_bp_a, z_cc_a = model(anchor_tokens)
        _, z_mf_p, z_bp_p, z_cc_p = model(pos_tokens)
        _, z_mf_n, z_bp_n, z_cc_n = model(neg_tokens)

        batch_loss = torch.tensor(0.0, device=device)
        n_losses   = 0

        mf_mask = [i for i, a in enumerate(aspects) if a == 'F']
        bp_mask = [i for i, a in enumerate(aspects) if a == 'P']
        cc_mask = [i for i, a in enumerate(aspects) if a == 'C']

        for mask, z_a, z_p, z_n, asp in [
            (mf_mask, z_mf_a, z_mf_p, z_mf_n, 'F'),
            (bp_mask, z_bp_a, z_bp_p, z_bp_n, 'P'),
            (cc_mask, z_cc_a, z_cc_p, z_cc_n, 'C'),
        ]:
            if len(mask) == 0:
                continue

            idx = torch.tensor(mask, device=device)
            za, zp, zn = z_a[idx], z_p[idx], z_n[idx]

            # --- Loss (değişmedi) ---
            loss = loss_fn(za, zp, zn)
            batch_loss = batch_loss + loss
            n_losses   += 1
            loss_per_aspect[asp]  += loss.item()
            count_per_aspect[asp] += 1

            # --- Metrikler: normalize edilmiş embedding'lerle ---
            za_n = nn.functional.normalize(za, dim=-1)
            zp_n = nn.functional.normalize(zp, dim=-1)
            zn_n = nn.functional.normalize(zn, dim=-1)

            batch_metrics = compute_detailed_metrics(za_n, zp_n, zn_n)

            for k, v in batch_metrics.items():
                metric_sums[asp][k] = metric_sums[asp].get(k, 0.0) + v
            metric_counts[asp] += 1

            retrieval_pools[asp]["anchor"].append(za_n.detach().cpu())
            retrieval_pools[asp]["pos"].append(zp_n.detach().cpu())
            retrieval_pools[asp]["neg"].append(zn_n.detach().cpu())

            # Accumulate for epoch-level inter/intra class similarity.
            # Each triplet i gets unique class id (anchor_i & pos_i share it, neg_i gets a separate one).
            B = za_n.shape[0]
            start = class_sim_pools[asp]["next_class_id"]
            pos_ids = torch.arange(start, start + B)
            neg_ids = torch.arange(start + B, start + 2 * B)
            class_sim_pools[asp]["preds"].append(torch.cat([za_n.cpu(), zp_n.cpu(), zn_n.cpu()]))
            class_sim_pools[asp]["labels"].append(torch.cat([pos_ids, pos_ids, neg_ids]))
            class_sim_pools[asp]["next_class_id"] = start + 2 * B

        if n_losses > 0:
            batch_loss = batch_loss / n_losses
        total_loss += batch_loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    avg_per_aspect = {
        asp: loss_per_aspect[asp] / max(1, count_per_aspect[asp])
        for asp in ('F', 'P', 'C')
    }

    # Her aspect için metrikleri ortala
    avg_metrics = {
        asp: {
            k: v / max(1, metric_counts[asp])
            for k, v in metric_sums[asp].items()
        }
        for asp in ('F', 'P', 'C')
    }

    for asp in ('F', 'P', 'C'):
        if retrieval_pools[asp]["anchor"]:
            anchor_pool = torch.cat(retrieval_pools[asp]["anchor"], dim=0)
            pos_pool = torch.cat(retrieval_pools[asp]["pos"], dim=0)
            neg_pool = torch.cat(retrieval_pools[asp]["neg"], dim=0)
            avg_metrics[asp]["MRR_FullValPool"] = compute_retrieval_mrr(
                anchor_pool, pos_pool, neg_pool
            )
        else:
            avg_metrics[asp]["MRR_FullValPool"] = 0.0

        if class_sim_pools[asp]["preds"]:
            all_preds  = torch.cat(class_sim_pools[asp]["preds"],  dim=0)
            all_labels = torch.cat(class_sim_pools[asp]["labels"], dim=0)
            metric = InterAndIntraClassSimilarity()
            metric.update(all_preds, all_labels)
            inter_sim, intra_sim = metric.compute()
            avg_metrics[asp]["InterClassSim"] = inter_sim.item()
            avg_metrics[asp]["IntraClassSim"] = intra_sim.item()
        else:
            avg_metrics[asp]["InterClassSim"] = 0.0
            avg_metrics[asp]["IntraClassSim"] = 0.0

    return avg_loss, avg_per_aspect, avg_metrics


# =============================================
# Training Loop
# =============================================

def train_go(args):
    device = args.device

    # -------------------------------------------
    # 1. Model yükle
    # -------------------------------------------
    print("=" * 60)
    print("STAGE 1: GO PRETRAINING")
    print("=" * 60)

    esm_model, alphabet = load_esm2_model(args.esm_model)
    model = GOPretrainModel(
        esm_model, alphabet,
        out_dim=OUT_DIM,
        freeze_layers=args.freeze_layers
    ).to(device)

    accum_steps    = args.accum_steps
    effective_batch = BATCH_SIZE * accum_steps
    print(f"[INFO] Batch size: {BATCH_SIZE} | Accum steps: {accum_steps} | "
          f"Effective batch: {effective_batch}")

    # -------------------------------------------
    # 2. Split dosyasını yükle (opsiyonel)
    # -------------------------------------------
    if args.split_file:
        group_to_split, split_meta = load_split_file(args.split_file)

        # Pickle dataset'i geçici olarak aç — sadece protein id'leri için
        with open(args.dataset, 'rb') as _f:
            _dataset_proteins = list(pickle.load(_f)['protein_seqs'].keys())
        group_to_split = check_split_dataset_consistency(group_to_split, _dataset_proteins)
    else:
        group_to_split = {}
        split_meta     = {}
        print("[Split] split_file verilmedi — tüm proteinler train olarak kullanılıyor.")

    # -------------------------------------------
    # 3. Dataset & DataLoader
    # -------------------------------------------

    aspect_weights = {'F': args.mf_weight, 'P': args.bp_weight, 'C': args.cc_weight}
    
    train_dataset = GOMultiAspectDataset(
        args.dataset,
        samples_per_epoch=args.samples_per_epoch,
        group_to_split=group_to_split,
        split='train',
        aspect_weights=aspect_weights
    )

    # Validation dataset (split_file yoksa boş bırak)
    val_dataset = None
    if group_to_split:
        val_dataset = GOMultiAspectDataset(
            args.dataset,
            samples_per_epoch=max(500, args.samples_per_epoch // 10),
            group_to_split=group_to_split,
            split='val',
            aspect_weights=aspect_weights
        )

    def collate_fn(batch):
        return collate_go_batch(batch, alphabet, max_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = None
    if val_dataset and len(val_dataset.all_proteins) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        print("[Split] Val seti boş veya split_file yok — val atlanıyor.")

    # -------------------------------------------
    # 4. Optimizer & Loss
    # -------------------------------------------
    optimizer = torch.optim.AdamW([
        {'params': model.esm_model.parameters(), 'lr': LR},
        {'params': model.head_mf.parameters(),   'lr': LR_HEADS},
        {'params': model.head_bp.parameters(),   'lr': LR_HEADS},
        {'params': model.head_cc.parameters(),   'lr': LR_HEADS},
    ], weight_decay=0.01)

    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)

    # -------------------------------------------
    # 5. WandB
    # -------------------------------------------
    if HAS_WANDB and not args.no_wandb:
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={
                "stage":             "go_pretraining",
                "task":              "GO-semantic encoder pretraining",
                "encoder_label":     "Phase 0 (GO)",
                "encoder_initialization": "Pretrained ESM-2",
                "esm_model":         args.esm_model,
                "epochs":            EPOCHS,
                "batch_size":        BATCH_SIZE,
                "lr_backbone":       LR,
                "lr_heads":          LR_HEADS,
                "margin":            MARGIN,
                "out_dim":           OUT_DIM,
                "freeze_layers":     args.freeze_layers,
                "samples_per_epoch": args.samples_per_epoch,
                "accum_steps":       accum_steps,
                "effective_batch":   effective_batch,
                "split_file":        args.split_file or "none",
                **{f"split_{k}": v for k, v in split_meta.items()
                   if k in ("seed", "train_ratio", "val_ratio", "test_ratio")},
            }
        )

    # -------------------------------------------
    # 6. Training
    # -------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    # -------------------------------------------
    # 6. Resume (isteğe bağlı)
    # -------------------------------------------
    start_epoch     = 0
    global_step     = 0
    best_val_loss   = float("inf")
    best_train_loss = float("inf")

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"--resume: checkpoint bulunamadı: {args.resume}")
        start_epoch, global_step, best_val_loss, best_train_loss = \
            load_checkpoint(args.resume, model, optimizer, device)
        if start_epoch >= EPOCHS:
            print(f"[RESUME] Eğitim zaten tamamlandı "
                  f"({start_epoch} epoch / hedef {EPOCHS}). Çıkılıyor.")
            return

    for epoch in range(start_epoch, EPOCHS):
        model.train()

        next_val_at_samples = args.val_every_samples

        total_loss       = 0.0
        loss_per_aspect  = {'F': 0.0, 'P': 0.0, 'C': 0.0}
        count_per_aspect = {'F': 0,   'P': 0,   'C': 0}
        train_metric_sums = {'F': {}, 'P': {}, 'C': {}}
        train_metric_counts = {'F': 0, 'P': 0, 'C': 0}
        val_loss         = None
        seen_train_samples = 0

        # Zero gradients at the start of each epoch
        optimizer.zero_grad()

        # Accumulation window tracking for averaged update-level WandB logging.
        accum_loss_sum    = 0.0
        accum_loss_count  = 0
        accum_loss_mf_sum = accum_loss_bp_sum = accum_loss_cc_sum = 0.0
        accum_loss_mf_cnt = accum_loss_bp_cnt = accum_loss_cc_cnt = 0
        accum_metric_sums = {'F': {}, 'P': {}, 'C': {}}
        accum_metric_counts = {'F': 0, 'P': 0, 'C': 0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")

        for batch_idx, (anchor_tokens, pos_tokens, neg_tokens, aspects) in enumerate(pbar):
            anchor_tokens = anchor_tokens.to(device)
            pos_tokens    = pos_tokens.to(device)
            neg_tokens    = neg_tokens.to(device)
            batch_size_actual = anchor_tokens.size(0)

            _, z_mf_a, z_bp_a, z_cc_a = model(anchor_tokens)
            _, z_mf_p, z_bp_p, z_cc_p = model(pos_tokens)
            _, z_mf_n, z_bp_n, z_cc_n = model(neg_tokens)

            batch_loss = torch.tensor(0.0, device=device)
            n_losses   = 0

            mf_mask = [i for i, a in enumerate(aspects) if a == 'F']
            bp_mask = [i for i, a in enumerate(aspects) if a == 'P']
            cc_mask = [i for i, a in enumerate(aspects) if a == 'C']

            batch_loss_mf = batch_loss_bp = batch_loss_cc = None

            if len(mf_mask) > 0:
                idx     = torch.tensor(mf_mask, device=device)
                z_a = z_mf_a[idx]
                z_p = z_mf_p[idx]
                z_n = z_mf_n[idx]
                loss_mf = loss_fn(z_a, z_p, z_n)
                batch_loss = batch_loss + loss_mf
                n_losses  += 1
                loss_per_aspect['F']  += loss_mf.item()
                count_per_aspect['F'] += 1
                batch_loss_mf = loss_mf.item()
                batch_metrics = compute_detailed_metrics(
                    nn.functional.normalize(z_a, dim=-1),
                    nn.functional.normalize(z_p, dim=-1),
                    nn.functional.normalize(z_n, dim=-1),
                )
                for k, v in batch_metrics.items():
                    train_metric_sums['F'][k] = train_metric_sums['F'].get(k, 0.0) + v
                    accum_metric_sums['F'][k] = accum_metric_sums['F'].get(k, 0.0) + v
                train_metric_counts['F'] += 1
                accum_metric_counts['F'] += 1

            if len(bp_mask) > 0:
                idx     = torch.tensor(bp_mask, device=device)
                z_a = z_bp_a[idx]
                z_p = z_bp_p[idx]
                z_n = z_bp_n[idx]
                loss_bp = loss_fn(z_a, z_p, z_n)
                batch_loss = batch_loss + loss_bp
                n_losses  += 1
                loss_per_aspect['P']  += loss_bp.item()
                count_per_aspect['P'] += 1
                batch_loss_bp = loss_bp.item()
                batch_metrics = compute_detailed_metrics(
                    nn.functional.normalize(z_a, dim=-1),
                    nn.functional.normalize(z_p, dim=-1),
                    nn.functional.normalize(z_n, dim=-1),
                )
                for k, v in batch_metrics.items():
                    train_metric_sums['P'][k] = train_metric_sums['P'].get(k, 0.0) + v
                    accum_metric_sums['P'][k] = accum_metric_sums['P'].get(k, 0.0) + v
                train_metric_counts['P'] += 1
                accum_metric_counts['P'] += 1

            if len(cc_mask) > 0:
                idx     = torch.tensor(cc_mask, device=device)
                z_a = z_cc_a[idx]
                z_p = z_cc_p[idx]
                z_n = z_cc_n[idx]
                loss_cc = loss_fn(z_a, z_p, z_n)
                batch_loss = batch_loss + loss_cc
                n_losses  += 1
                loss_per_aspect['C']  += loss_cc.item()
                count_per_aspect['C'] += 1
                batch_loss_cc = loss_cc.item()
                batch_metrics = compute_detailed_metrics(
                    nn.functional.normalize(z_a, dim=-1),
                    nn.functional.normalize(z_p, dim=-1),
                    nn.functional.normalize(z_n, dim=-1),
                )
                for k, v in batch_metrics.items():
                    train_metric_sums['C'][k] = train_metric_sums['C'].get(k, 0.0) + v
                    accum_metric_sums['C'][k] = accum_metric_sums['C'].get(k, 0.0) + v
                train_metric_counts['C'] += 1
                accum_metric_counts['C'] += 1

            if n_losses > 0:
                batch_loss = batch_loss / n_losses

            # Scale loss by accum_steps so the effective gradient magnitude
            # stays the same regardless of how many micro-steps we accumulate.
            scaled_loss = batch_loss / accum_steps
            scaled_loss.backward()

            is_update_step = (batch_idx + 1) % accum_steps == 0 or \
                             (batch_idx + 1) == len(train_loader)

            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += batch_loss.item()
            seen_train_samples += batch_size_actual
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}",
                             eff=f"{(batch_idx % accum_steps) + 1}/{accum_steps}")

            # Accumulate per-aspect losses for averaging over the accumulation window
            if batch_loss_mf is not None:
                accum_loss_mf_sum += batch_loss_mf
                accum_loss_mf_cnt += 1
            if batch_loss_bp is not None:
                accum_loss_bp_sum += batch_loss_bp
                accum_loss_bp_cnt += 1
            if batch_loss_cc is not None:
                accum_loss_cc_sum += batch_loss_cc
                accum_loss_cc_cnt += 1
            accum_loss_sum += batch_loss.item()
            accum_loss_count += 1

            # ---- WandB logging: only on optimizer update steps ----
            if is_update_step:
                global_step += 1

                if HAS_WANDB and not args.no_wandb:
                    update_log = {
                        "train_update/loss":  accum_loss_sum / max(1, accum_loss_count),
                        "train_update/epoch": epoch + 1,
                    }
                    if accum_loss_mf_cnt > 0:
                        update_log["train_update/loss_mf"] = accum_loss_mf_sum / accum_loss_mf_cnt
                    if accum_loss_bp_cnt > 0:
                        update_log["train_update/loss_bp"] = accum_loss_bp_sum / accum_loss_bp_cnt
                    if accum_loss_cc_cnt > 0:
                        update_log["train_update/loss_cc"] = accum_loss_cc_sum / accum_loss_cc_cnt
                    asp_labels = {'F': 'mf', 'P': 'bp', 'C': 'cc'}
                    for asp, label in asp_labels.items():
                        if accum_metric_counts[asp] > 0 and "MRR" in accum_metric_sums[asp]:
                            update_log[f"train_update/mrr_{label}"] = (
                                accum_metric_sums[asp]["MRR"] / accum_metric_counts[asp]
                            )
                    wandb.log(update_log, step=global_step)

                    accum_loss_sum = 0.0
                    accum_loss_count = 0
                    accum_loss_mf_sum = accum_loss_bp_sum = accum_loss_cc_sum = 0.0
                    accum_loss_mf_cnt = accum_loss_bp_cnt = accum_loss_cc_cnt = 0
                    accum_metric_sums = {'F': {}, 'P': {}, 'C': {}}
                    accum_metric_counts = {'F': 0, 'P': 0, 'C': 0}
                else:
                    accum_loss_sum = 0.0
                    accum_loss_count = 0
                    accum_loss_mf_sum = accum_loss_bp_sum = accum_loss_cc_sum = 0.0
                    accum_loss_mf_cnt = accum_loss_bp_cnt = accum_loss_cc_cnt = 0
                    accum_metric_sums = {'F': {}, 'P': {}, 'C': {}}
                    accum_metric_counts = {'F': 0, 'P': 0, 'C': 0}

                # ---- Mid-epoch validation ----
                if val_loader is not None and seen_train_samples >= next_val_at_samples:
                    val_loss, val_per_asp, val_metrics = \
                        evaluate(model, val_loader, loss_fn, device)
                    next_val_at_samples += args.val_every_samples

                    asp_labels = {'F': 'mf', 'P': 'bp', 'C': 'cc'}
                    print(
                        f"  [VAL step={global_step}] loss={val_loss:.4f} | "
                        + " | ".join(
                            f"{asp_labels[a].upper()} "
                            f"loss={val_per_asp.get(a, 0.):.3f} "
                            f"align={val_metrics[a].get('Alignment', 0.):.3f} "
                            f"unif={val_metrics[a].get('Uniformity', 0.):.3f} "
                            f"mrr={val_metrics[a].get('MRR', 0.):.3f} "
                            f"pool_mrr={val_metrics[a].get('MRR_FullValPool', 0.):.3f} "
                            f"acc={val_metrics[a].get('Simple_Acc', 0.):.3f} "
                            f"intra_sim={val_metrics[a].get('IntraClassSim', 0.):.3f} "
                            f"inter_sim={val_metrics[a].get('InterClassSim', 0.):.3f}"
                            for a in ('F', 'P', 'C')
                        )
                    )

                    if HAS_WANDB and not args.no_wandb:
                        log_dict = {"val/loss": val_loss}
                        for asp, label in asp_labels.items():
                            log_dict[f"val/loss_{label}"] = val_per_asp.get(asp, 0.)
                            for metric_name, val in val_metrics[asp].items():
                                safe_name = metric_name.replace("@", "_at_").lower()
                                log_dict[f"val/{safe_name}_{label}"] = val
                        wandb.log(log_dict, step=global_step)

                    # Best model checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.save_full(os.path.join(args.out, f"{args.run_name}_full_model_best.pt"))
                        print(f"  [BEST] val_loss={val_loss:.4f} → best checkpoint saved")

                    model.train()  # evaluate() sonrası train moduna geri dön
        # ---- Epoch train istatistikleri ----
        avg_loss = total_loss / max(1, len(train_loader))
        avg_mf   = loss_per_aspect['F'] / max(1, count_per_aspect['F'])
        avg_bp   = loss_per_aspect['P'] / max(1, count_per_aspect['P'])
        avg_cc   = loss_per_aspect['C'] / max(1, count_per_aspect['C'])
        train_avg_metrics = {
            asp: {
                k: v / max(1, train_metric_counts[asp])
                for k, v in train_metric_sums[asp].items()
            }
            for asp in ('F', 'P', 'C')
        }

        print(
            f"[EPOCH {epoch+1:02d}/{EPOCHS}] TRAIN "
            f"loss={avg_loss:.4f} | "
            f"MF={avg_mf:.4f} BP={avg_bp:.4f} CC={avg_cc:.4f} | "
            f"MRR mf={train_avg_metrics['F'].get('MRR', 0.):.3f} "
            f"bp={train_avg_metrics['P'].get('MRR', 0.):.3f} "
            f"cc={train_avg_metrics['C'].get('MRR', 0.):.3f}"
        )

        if HAS_WANDB and not args.no_wandb:
            epoch_log = {
                "epoch":         epoch + 1,
                "train/loss":    avg_loss,
                "train/loss_mf": avg_mf,
                "train/loss_bp": avg_bp,
                "train/loss_cc": avg_cc,
            }
            for asp, label in {'F': 'mf', 'P': 'bp', 'C': 'cc'}.items():
                if "MRR" in train_avg_metrics[asp]:
                    epoch_log[f"train/mrr_{label}"] = train_avg_metrics[asp]["MRR"]
            wandb.log(epoch_log, step=global_step)


        # ---- Overwrite last checkpoint every epoch ----
        model.save_full(os.path.join(args.out, f"{args.run_name}_full_model_last.pt"))
        print(f"[CKPT] Epoch {epoch+1} → last checkpoint overwritten")

        # ---- Her epoch sonunda resumable checkpoint ----
        save_checkpoint(
            os.path.join(args.out, f"{args.run_name}_latest_checkpoint.pt"),
            model, optimizer, epoch, global_step,
            best_val_loss, best_train_loss,
        )

        # ---- Periyodik checkpoint (her checkpoint_every epoch'ta bir) ----
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt_dir = Path(args.out) / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"{args.run_name}_epoch_{epoch+1:04d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, global_step,
                            best_val_loss, best_train_loss)
            cleanup_old_checkpoints(ckpt_dir, args.keep_checkpoints, args.run_name)

        # ---- Overwrite best checkpoint only on improvement ----
        monitor_loss = val_loss if val_loss is not None else avg_loss
        ref_best     = best_val_loss if val_loss is not None else best_train_loss

        if monitor_loss < ref_best:
            if val_loss is not None:
                best_val_loss = monitor_loss
            else:
                best_train_loss = monitor_loss

            model.save_full(os.path.join(args.out, f"{args.run_name}_full_model_best.pt"))
            print(f"[BEST] New best (monitor_loss={monitor_loss:.4f}) → best checkpoint overwritten")

    best = best_val_loss if val_loader else best_train_loss
    print(f"\n[DONE] GO Pretraining complete. Best loss: {best:.4f}")
    print(f"  Best model  : {os.path.join(args.out, f'{args.run_name}_full_model_best.pt')}")
    print(f"  Last model  : {os.path.join(args.out, f'{args.run_name}_full_model_last.pt')}")
    print(f"  Resume ckpt : {os.path.join(args.out, f'{args.run_name}_latest_checkpoint.pt')}")


# =============================================
# Main
# =============================================

def pick_device(cuda_id=None):
    """
    Device seçimi:
        --cuda 0   → cuda:0
        --cuda 1   → cuda:1
        --cuda mps → mps
        --cuda cpu → cpu
        (belirtilmezse) → mps > cuda:0 > cpu sırasıyla otomatik seçim
    """
    if cuda_id is not None:
        cuda_id = str(cuda_id).strip()
        if cuda_id.lower() == "cpu":
            return "cpu"
        if cuda_id.lower() == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MPS istendi fakat bu sistemde mevcut değil.")
        # Sayısal CUDA index veya "cuda:N" formatı
        idx = cuda_id.replace("cuda:", "")
        if not idx.isdigit():
            raise ValueError(
                f"Geçersiz --cuda değeri: '{cuda_id}'. "
                "Beklenen: bir tam sayı (0, 1, 2...), 'mps', veya 'cpu'."
            )
        idx = int(idx)
        n_gpus = torch.cuda.device_count()
        if idx >= n_gpus:
            raise RuntimeError(
                f"CUDA:{idx} istendi fakat sistemde yalnızca {n_gpus} GPU var "
                f"(geçerli indeksler: 0–{n_gpus - 1})."
            )
        return f"cuda:{idx}"

    # Otomatik seçim
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main():
    global EPOCHS, BATCH_SIZE, ACCUM_STEPS, LR, LR_HEADS

    parser = argparse.ArgumentParser(description="GO Pretraining - Stage 1")
    parser.add_argument("--dataset",    required=True, help="Path to go_pretrain_dataset.pkl")
    parser.add_argument("--split_file", default=None,
                        help="(Opsiyonel) JSON split dosyası — protein → train/val/test eşlemesi")
    parser.add_argument("--esm_model",  default="esm2_t6_8M_UR50D",
                        help="ESM2 model name",choices=["esm2_t6_8M_UR50D","esm2_t33_650M_UR50D"])
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--accum_steps", type=int,  default=ACCUM_STEPS,
                        help=(
                            "Gradient accumulation adımı. "
                            "Efektif batch = batch_size * accum_steps. "
                            "Varsayılan: 4 → efektif batch 16."
                        ))
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--lr_heads",   type=float, default=LR_HEADS)
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="İlk N ESM2 katmanını dondur (0=hepsini eğit)")
    parser.add_argument("--samples_per_epoch", type=int, default=10000)
    parser.add_argument("--max_seq_len",       type=int, default=1022)
    parser.add_argument("--val_every_samples", type=int, default=VAL_EVERY_SAMPLES,
                        help="Validation'i yaklasik her N train sample sonrasinda calistir")
    parser.add_argument("--out",           default="models/go_pretrained")
    parser.add_argument("--project_name", default="protein-go-pretrain")
    parser.add_argument("--run_name",     default="run_phase_0")
    parser.add_argument("--no_wandb",     action="store_true")
    parser.add_argument("--cuda", default=None, metavar="ID",
                        help=(
                            "Kullanılacak cihaz. Örnekler: "
                            "0 veya cuda:0 (ilk GPU), 1 veya cuda:1 (ikinci GPU), "
                            "mps (Apple Silicon), cpu. "
                            "Belirtilmezse otomatik seçilir: mps > cuda:0 > cpu."
                        ))
    parser.add_argument("--mf_weight", type=float, default=0.5)
    parser.add_argument("--bp_weight", type=float, default=0.3)
    parser.add_argument("--cc_weight", type=float, default=0.2)
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="Eğitimi checkpoint'ten devam ettir "
                             "(örn: models/go_pretrained/latest_checkpoint.pt)")
    parser.add_argument("--checkpoint_every", type=int, default=5,
                        help="Her N epoch'ta bir periyodik checkpoint kaydet (0=devre dışı)")
    parser.add_argument("--keep_checkpoints", type=int, default=3,
                        help="Saklanacak maksimum periyodik checkpoint sayısı (0=hepsini sakla)")
    args = parser.parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    ACCUM_STEPS = args.accum_steps
    LR         = args.lr
    LR_HEADS   = args.lr_heads

    args.device = pick_device(args.cuda)
    print(f"[INFO] Device: {args.device}")
    if "cuda" in args.device:
        idx = int(args.device.split(":")[-1])
        print(f"[INFO] GPU: {torch.cuda.get_device_name(idx)}")

    train_go(args)


if __name__ == "__main__":
    main()
