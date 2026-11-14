"""
Phase1 DMS backbone fine-tuning.

Amaç:
  Phase0 GO-pretrained ESM backbone'u yükleyip DMS triplet/local losses ile
  gerçekten backbone seviyesinde fine-tune etmek.

Kurgu:
  - son N layer trainable (default: 8)
  - global head: WT / positive / negative pooled embedding üstünde triplet loss
  - local head: mutasyon pozisyonundaki WT ve mutant residue embedding'leri üstünde
    contrastive-style local loss
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from go_model import load_esm2_model, tokenize_sequences
    from dms_data import (
        DMSTripletSequenceDataset,
        build_phase1_families,
        build_validation_records,
    )
except ImportError:
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PHASE0_DIR = REPO_ROOT / "phase_0_go_pretrained"
    PHASE1_DIR = REPO_ROOT / "phase_1_dms_fine_tuning"
    for path in (PHASE0_DIR, PHASE1_DIR):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    from go_model import load_esm2_model, tokenize_sequences
    from dms_data import (
        DMSTripletSequenceDataset,
        build_phase1_families,
        build_validation_records,
    )


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, deep: bool = False):
        super().__init__()
        if deep:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def pick_device(cuda_id=None):
    if cuda_id is not None:
        cuda_id = str(cuda_id).strip()
        if cuda_id.lower() == "cpu":
            return "cpu"
        if cuda_id.lower() == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MPS istendi fakat bu sistemde mevcut değil.")
        idx = cuda_id.replace("cuda:", "")
        if not idx.isdigit():
            raise ValueError(f"Invalid --cuda: {cuda_id}")
        idx = int(idx)
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA:{idx} requested but only {torch.cuda.device_count()} visible")
        return f"cuda:{idx}"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def freeze_esm_except_last_n(esm_model, last_trainable_layers: int):
    n_layers = esm_model.num_layers
    freeze_before = max(0, n_layers - last_trainable_layers)
    for param in esm_model.parameters():
        param.requires_grad = False

    for name, param in esm_model.named_parameters():
        if "layers." in name:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
            if layer_idx >= freeze_before:
                param.requires_grad = True
        elif any(x in name for x in ("embed_tokens", "contact_head", "lm_head")):
            # leave these frozen; decoder use-case doesn't need LM head
            continue

    # final layer norm often useful to train with upper layers
    for name, param in esm_model.named_parameters():
        if "emb_layer_norm_after" in name:
            param.requires_grad = True

    print(f"[INFO] ESM fine-tuning: train last {last_trainable_layers}/{n_layers} layers")
    print(f"[INFO] Froze first {freeze_before} layers")


def load_phase0_backbone(backbone_path: str, esm_model_name: str, device: str, last_trainable_layers: int):
    model, alphabet = load_esm2_model(esm_model_name)
    ckpt = torch.load(backbone_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["esm_state_dict"])
    freeze_esm_except_last_n(model, last_trainable_layers)
    model = model.to(device)
    return model, alphabet


def collate_triplets(batch):
    return {
        "family_key": [x["family_key"] for x in batch],
        "wt_sequence": [x["wt_sequence"] for x in batch],
        "pos_sequence": [x["pos_sequence"] for x in batch],
        "neg_sequence": [x["neg_sequence"] for x in batch],
        "pos_position": [x["pos_position"] for x in batch],
        "neg_position": [x["neg_position"] for x in batch],
        "pos_mutation": [x["pos_mutation"] for x in batch],
        "neg_mutation": [x["neg_mutation"] for x in batch],
    }


def encode_sequences(esm_model, alphabet, sequences, device, max_seq_len):
    labeled = [(str(i), seq) for i, seq in enumerate(sequences)]
    tokens = tokenize_sequences(labeled, alphabet, max_len=max_seq_len).to(device)
    results = esm_model(tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
    token_emb = results["representations"][esm_model.num_layers]

    padding_mask = (tokens != alphabet.padding_idx).float()
    padding_mask[:, 0] = 0
    residue_list = []
    for i in range(tokens.shape[0]):
        valid_positions = (tokens[i] != alphabet.padding_idx).nonzero(as_tuple=True)[0]
        last_valid = valid_positions[-1].item()
        padding_mask[i, last_valid] = 0
        residue_list.append(token_emb[i, 1:last_valid, :])

    pooled = (token_emb * padding_mask.unsqueeze(-1)).sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).clamp(min=1)
    return pooled, residue_list


def gather_local_pairs(local_head, wt_residues, pos_residues, neg_residues, pos_positions, neg_positions):
    wt_pos_vecs, pos_vecs = [], []
    wt_neg_vecs, neg_vecs = [], []

    for i in range(len(wt_residues)):
        if pos_positions[i] < len(wt_residues[i]) and pos_positions[i] < len(pos_residues[i]):
            wt_pos_vecs.append(wt_residues[i][pos_positions[i]])
            pos_vecs.append(pos_residues[i][pos_positions[i]])
        if neg_positions[i] < len(wt_residues[i]) and neg_positions[i] < len(neg_residues[i]):
            wt_neg_vecs.append(wt_residues[i][neg_positions[i]])
            neg_vecs.append(neg_residues[i][neg_positions[i]])

    if wt_pos_vecs:
        z_wt_pos = local_head(torch.stack(wt_pos_vecs))
        z_pos = local_head(torch.stack(pos_vecs))
    else:
        z_wt_pos = z_pos = None

    if wt_neg_vecs:
        z_wt_neg = local_head(torch.stack(wt_neg_vecs))
        z_neg = local_head(torch.stack(neg_vecs))
    else:
        z_wt_neg = z_neg = None

    return z_wt_pos, z_pos, z_wt_neg, z_neg


def local_contrastive_loss(z_wt_pos, z_pos, z_wt_neg, z_neg, margin, fallback_device):
    losses = []
    if z_wt_pos is not None and z_pos is not None:
        d_pos = torch.sqrt(((z_wt_pos - z_pos) ** 2).sum(dim=-1) + 1e-8)
        losses.append((d_pos ** 2).mean())
    if z_wt_neg is not None and z_neg is not None:
        d_neg = torch.sqrt(((z_wt_neg - z_neg) ** 2).sum(dim=-1) + 1e-8)
        losses.append((F.relu(margin - d_neg) ** 2).mean())
    if losses:
        return sum(losses) / len(losses)
    return torch.tensor(0.0, device=fallback_device)


def evaluate_phase1(esm_model, alphabet, global_head, local_head, val_records, device, max_seq_len, batch_size, margin):
    if not val_records:
        return {}

    esm_model.eval()
    global_head.eval()
    local_head.eval()

    d_global_all = []
    d_local_all = []
    labels_all = []
    family_all = []
    per_family = {}
    z_wt_g_list = []
    z_mut_g_list = []

    with torch.no_grad():
        for start in range(0, len(val_records), batch_size):
            batch = val_records[start:start + batch_size]
            wt_seqs = [x["wt_sequence"] for x in batch]
            mut_seqs = [x["mut_sequence"] for x in batch]
            positions = [x["position"] for x in batch]

            pooled, residues = encode_sequences(
                esm_model, alphabet, wt_seqs + mut_seqs, device, max_seq_len
            )
            B = len(batch)
            wt_pooled, mut_pooled = pooled[:B], pooled[B:]
            wt_residues, mut_residues = residues[:B], residues[B:]

            z_wt_g = global_head(wt_pooled)
            z_mut_g = global_head(mut_pooled)
            z_wt_g_list.append(z_wt_g.cpu().numpy())
            z_mut_g_list.append(z_mut_g.cpu().numpy())

            wt_local_vecs, mut_local_vecs, valid_meta = [], [], []
            for i, pos0 in enumerate(positions):
                if pos0 < len(wt_residues[i]) and pos0 < len(mut_residues[i]):
                    wt_local_vecs.append(wt_residues[i][pos0])
                    mut_local_vecs.append(mut_residues[i][pos0])
                    valid_meta.append(i)

            z_wt_l = z_mut_l = None
            if wt_local_vecs:
                z_wt_l = local_head(torch.stack(wt_local_vecs))
                z_mut_l = local_head(torch.stack(mut_local_vecs))

            d_global = torch.sqrt(((z_mut_g - z_wt_g) ** 2).sum(dim=-1) + 1e-8).cpu().numpy()
            d_local = np.full(B, np.nan, dtype=np.float32)
            if z_wt_l is not None:
                d_local_valid = torch.sqrt(((z_mut_l - z_wt_l) ** 2).sum(dim=-1) + 1e-8).cpu().numpy()
                for j, orig_i in enumerate(valid_meta):
                    d_local[orig_i] = d_local_valid[j]

            for i, rec in enumerate(batch):
                label = int(rec["label"])
                family = rec["family_key"]
                dg = float(d_global[i])
                dl = float(d_local[i]) if not np.isnan(d_local[i]) else 0.0
                d_global_all.append(dg)
                d_local_all.append(dl)
                labels_all.append(label)
                family_all.append(family)
                fam_stats = per_family.setdefault(family, {"pos_g": [], "neg_g": [], "pos_l": [], "neg_l": []})
                if label == 1:
                    fam_stats["pos_g"].append(dg)
                    fam_stats["pos_l"].append(dl)
                else:
                    fam_stats["neg_g"].append(dg)
                    fam_stats["neg_l"].append(dl)

    labels = np.asarray(labels_all, dtype=np.int64)
    d_global = np.asarray(d_global_all, dtype=np.float32)
    d_local = np.asarray(d_local_all, dtype=np.float32)
    pos_mask = labels == 1
    neg_mask = labels == 0

    metrics = {}
    if len(np.unique(labels)) >= 2:
        from sklearn.metrics import roc_auc_score
        metrics["auroc_g"] = float(roc_auc_score(labels, -d_global))
        metrics["auroc_l"] = float(roc_auc_score(labels, -d_local))
    else:
        metrics["auroc_g"] = 0.5
        metrics["auroc_l"] = 0.5

    metrics["d_pos_g"] = float(d_global[pos_mask].mean()) if pos_mask.any() else 0.0
    metrics["d_neg_g"] = float(d_global[neg_mask].mean()) if neg_mask.any() else 0.0
    metrics["margin_g"] = metrics["d_neg_g"] - metrics["d_pos_g"]
    metrics["d_pos_l"] = float(d_local[pos_mask].mean()) if pos_mask.any() else 0.0
    metrics["d_neg_l"] = float(d_local[neg_mask].mean()) if neg_mask.any() else 0.0
    metrics["margin_l"] = metrics["d_neg_l"] - metrics["d_pos_l"]

    triplet_losses_g, triplet_losses_l = [], []
    for fam_stats in per_family.values():
        for dp in fam_stats["pos_g"]:
            for dn in fam_stats["neg_g"]:
                triplet_losses_g.append(max(0.0, dp - dn + margin))
        for dp in fam_stats["pos_l"]:
            for dn in fam_stats["neg_l"]:
                triplet_losses_l.append(max(0.0, dp - dn + margin))

    metrics["val_loss_g"] = float(np.mean(triplet_losses_g)) if triplet_losses_g else 0.0
    metrics["val_loss_l"] = float(np.mean(triplet_losses_l)) if triplet_losses_l else 0.0
    metrics["val_loss"] = metrics["val_loss_g"] + metrics["val_loss_l"]

    # --- additional ContVAR metrics ---
    z_wt_np = np.concatenate(z_wt_g_list, axis=0)   # [N, D]
    z_mut_np = np.concatenate(z_mut_g_list, axis=0)  # [N, D]

    # alignment: E[||z_wt - z_pos||^2] (positive pairs only)
    if pos_mask.any():
        diff_pos = z_wt_np[pos_mask] - z_mut_np[pos_mask]
        metrics["alignment"] = float((diff_pos ** 2).sum(axis=-1).mean())
        metrics["cos_pos"] = float((z_wt_np[pos_mask] * z_mut_np[pos_mask]).sum(axis=-1).mean())
    else:
        metrics["alignment"] = 0.0
        metrics["cos_pos"] = 0.0

    if neg_mask.any():
        metrics["cos_neg"] = float((z_wt_np[neg_mask] * z_mut_np[neg_mask]).sum(axis=-1).mean())
    else:
        metrics["cos_neg"] = 0.0

    # uniformity: log E[exp(-2*||z_i - z_j||^2)] over WT embeddings
    # For L2-normalized vectors: ||z_i - z_j||^2 = 2 - 2*(z_i·z_j)
    gram = z_wt_np @ z_wt_np.T
    sq_dists_u = np.clip(2.0 - 2.0 * gram, 0.0, None)
    metrics["uniformity"] = float(np.log(np.mean(np.exp(-2.0 * sq_dists_u)) + 1e-30))

    # emb_std: mean feature-wise std of WT validation embeddings
    metrics["emb_std"] = float(z_wt_np.std(axis=0).mean())

    # acc_g: fraction of proteins where mean_d_pos < mean_d_neg
    acc_g_correct = 0
    acc_g_total = 0
    for fam_stats in per_family.values():
        if fam_stats["pos_g"] and fam_stats["neg_g"]:
            acc_g_correct += int(float(np.mean(fam_stats["pos_g"])) < float(np.mean(fam_stats["neg_g"])))
            acc_g_total += 1
    metrics["acc_g"] = acc_g_correct / acc_g_total if acc_g_total > 0 else 0.5

    return metrics


def save_phase1_checkpoint(path, esm_model, global_head, local_head, args):
    torch.save(
        {
            "esm_state_dict": esm_model.state_dict(),
            "global_head_state_dict": global_head.state_dict(),
            "local_head_state_dict": local_head.state_dict(),
            "esm_dim": esm_model.embed_dim,
            "out_dim": args.out_dim,
            "esm_model_name": args.esm_model,
            "last_trainable_layers": args.last_trainable_layers,
        },
        path,
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Phase1 DMS backbone fine-tuning")
    parser.add_argument("--backbone", required=True, help="Phase0 backbone_best.pt")
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--triplet_json", default=str(repo_root / ".." / "triplet_data.json"))
    parser.add_argument(
        "--reference_csv",
        default=str(repo_root / "data" / "raw" / "DMS_ProteinGym_substitutions" / "0_DMS_substitutions_reference_file.csv"),
    )
    parser.add_argument(
        "--proteins_csv",
        default=None,
        help="Variant/WT sequence table, e.g. updated_version/data/proteins 2.csv. If provided, this is the primary Phase1 sequence source.",
    )
    parser.add_argument("--split_json", default=None,required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=5e-6)
    parser.add_argument("--lr_global_head", type=float, default=1e-4)
    parser.add_argument("--lr_local_head", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--last_trainable_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--limit_families", type=int, default=None)
    parser.add_argument("--cuda", default=None)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Cosine annealing eta_min for all param groups")
    parser.add_argument("--project_name", default="contvar-phase1-dms-backbone")
    parser.add_argument("--run_name", default="phase1_dms_backbone_finetune")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.cuda)
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families, skipped = build_phase1_families(
        args.triplet_json,
        args.reference_csv,
        args.split_json,
        args.limit_families,
        proteins_csv=args.proteins_csv,
    )
    print(f"[INFO] Phase1 families loaded: {len(families)}")
    if skipped:
        print(f"[WARN] Skipped families: {len(skipped)}")

    train_ds = DMSTripletSequenceDataset(families)
    val_records = build_validation_records(families)
    print(f"[INFO] Train triplets: {len(train_ds)}")
    print(f"[INFO] Val records: {len(val_records)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_triplets
    )

    esm_model, alphabet = load_phase0_backbone(
        args.backbone, args.esm_model, device, args.last_trainable_layers
    )
    global_head = ProjectionHead(esm_model.embed_dim, args.out_dim, deep=True).to(device)
    local_head = ProjectionHead(esm_model.embed_dim, args.out_dim, deep=False).to(device)

    backbone_params = [p for p in esm_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": global_head.parameters(), "lr": args.lr_global_head},
        {"params": local_head.parameters(), "lr": args.lr_local_head},
    ])
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**vars(args),
                    "task": "DMS-based encoder adaptation",
                    "encoder_label": "Phase 0 (GO) -> Phase 1 (DMS)",
                    "encoder_initialization": "Phase 0 GO-pretrained backbone; see backbone path",
                    "training_history_note": "Cumulative GO then DMS adaptation, not a DMS-only control."},
        )

    best_score = -1.0
    best_backbone_path = out_dir / "phase1_backbone_best.pt"
    best_full_path = out_dir / "phase1_full_best.pt"

    for epoch in range(1, args.epochs + 1):
        esm_model.train()
        global_head.train()
        local_head.train()

        total_loss = total_loss_g = total_loss_l = 0.0
        total_easy_g = total_semi_hard_g = total_hard_g = 0
        total_easy_l = total_semi_hard_l = total_hard_l = 0
        total_emb_std = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            wt_seqs = batch["wt_sequence"]
            pos_seqs = batch["pos_sequence"]
            neg_seqs = batch["neg_sequence"]
            pos_positions = batch["pos_position"]
            neg_positions = batch["neg_position"]

            pooled, residues = encode_sequences(
                esm_model,
                alphabet,
                wt_seqs + pos_seqs + neg_seqs,
                device,
                args.max_seq_len,
            )
            B = len(wt_seqs)
            wt_pooled, pos_pooled, neg_pooled = pooled[:B], pooled[B:2*B], pooled[2*B:]
            wt_residues, pos_residues, neg_residues = residues[:B], residues[B:2*B], residues[2*B:]

            z_wt_g = global_head(wt_pooled)
            z_pos_g = global_head(pos_pooled)
            z_neg_g = global_head(neg_pooled)
            loss_g = triplet_loss_fn(z_wt_g, z_pos_g, z_neg_g)

            z_wt_pos, z_pos, z_wt_neg, z_neg = gather_local_pairs(
                local_head, wt_residues, pos_residues, neg_residues, pos_positions, neg_positions
            )
            loss_l = local_contrastive_loss(
                z_wt_pos, z_pos, z_wt_neg, z_neg, args.margin, fallback_device=wt_pooled.device
            )
            loss = (loss_g + loss_l) / 2.0

            # --- per-batch hardness & emb_std (no extra grad graph) ---
            with torch.no_grad():
                d_ap_g = torch.sqrt(((z_wt_g - z_pos_g) ** 2).sum(dim=-1) + 1e-8)
                d_an_g = torch.sqrt(((z_wt_g - z_neg_g) ** 2).sum(dim=-1) + 1e-8)
                easy_g_b  = int((d_an_g >= d_ap_g + args.margin).sum())
                hard_g_b  = int((d_an_g <= d_ap_g).sum())
                total_easy_g      += easy_g_b
                total_hard_g      += hard_g_b
                total_semi_hard_g += B - easy_g_b - hard_g_b

                easy_l_b = hard_l_b = n_l = 0
                if z_wt_pos is not None and z_pos is not None:
                    d_ap_l = torch.sqrt(((z_wt_pos - z_pos) ** 2).sum(dim=-1) + 1e-8)
                    # attract: easy if nearly collapsed, hard if beyond margin
                    easy_l_b += int((d_ap_l < args.margin / 3).sum())
                    hard_l_b += int((d_ap_l >= args.margin).sum())
                    n_l += d_ap_l.shape[0]
                if z_wt_neg is not None and z_neg is not None:
                    d_an_l = torch.sqrt(((z_wt_neg - z_neg) ** 2).sum(dim=-1) + 1e-8)
                    # repel: easy if d >= margin (zero loss), hard if d < margin/3
                    easy_l_b += int((d_an_l >= args.margin).sum())
                    hard_l_b += int((d_an_l < args.margin / 3).sum())
                    n_l += d_an_l.shape[0]
                total_easy_l      += easy_l_b
                total_hard_l      += hard_l_b
                total_semi_hard_l += max(0, n_l - easy_l_b - hard_l_b)

                all_g_emb = torch.cat([z_wt_g, z_pos_g, z_neg_g], dim=0)
                total_emb_std += float(all_g_emb.std(dim=0).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_loss_g = total_loss_g / max(1, n_batches)
        avg_loss_l = total_loss_l / max(1, n_batches)
        avg_emb_std = total_emb_std / max(1, n_batches)
        lr_global = optimizer.param_groups[1]["lr"]
        lr_local  = optimizer.param_groups[2]["lr"]

        val_metrics = evaluate_phase1(
            esm_model, alphabet, global_head, local_head,
            val_records, device, args.max_seq_len, args.batch_size, args.margin
        )
        val_score = val_metrics.get("auroc_g", 0.0)

        print(
            f"[EPOCH {epoch:02d}/{args.epochs}] "
            f"loss={avg_loss:.4f} (G={avg_loss_g:.4f}, L={avg_loss_l:.4f}) | "
            f"val_auroc_g={val_metrics.get('auroc_g', 0.):.4f} "
            f"val_auroc_l={val_metrics.get('auroc_l', 0.):.4f} "
            f"val_margin_g={val_metrics.get('margin_g', 0.):.4f} "
            f"val_margin_l={val_metrics.get('margin_l', 0.):.4f}"
        )

        scheduler.step()

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/loss_g": avg_loss_g,
                "train/loss_l": avg_loss_l,
                "train/easy_g": total_easy_g,
                "train/semi_hard_g": total_semi_hard_g,
                "train/hard_g": total_hard_g,
                "train/easy_l": total_easy_l,
                "train/semi_hard_l": total_semi_hard_l,
                "train/hard_l": total_hard_l,
                "train/emb_std": avg_emb_std,
                "train/lr_global": lr_global,
                "train/lr_local": lr_local,
            }
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            wandb.log(log_dict)

        latest_backbone_path = out_dir / "phase1_backbone_latest.pt"
        latest_full_path = out_dir / "phase1_full_latest.pt"
        save_phase1_checkpoint(latest_backbone_path, esm_model, global_head, local_head, args)
        save_phase1_checkpoint(latest_full_path, esm_model, global_head, local_head, args)

        if val_score > best_score:
            best_score = val_score
            save_phase1_checkpoint(best_backbone_path, esm_model, global_head, local_head, args)
            save_phase1_checkpoint(best_full_path, esm_model, global_head, local_head, args)
            print(f"[SAVE] New best Phase1 checkpoint -> {best_backbone_path} (val_auroc_g={best_score:.4f})")

    if use_wandb:
        wandb.finish()

    summary = {
        "best_val_auroc_g": best_score,
        "best_backbone": str(best_backbone_path),
        "best_full": str(best_full_path),
        "train_triplets": len(train_ds),
        "val_records": len(val_records),
    }
    with open(out_dir / "phase1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[DONE] Phase1 backbone fine-tuning complete")


if __name__ == "__main__":
    main()
