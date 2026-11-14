# src/scripts_yeni/validation.py
# ContVAR-inspired validation metrics + split.json integration

import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# =============================================
# split.json Parsing
# =============================================

def parse_mutation_from_cif(cif_name):
    """Extract mutation string from CIF filename.
    E.g., '..._k117q_1_model.cif' → 'K117Q'
    """
    m = re.search(r'_([a-z]\d+[a-z])_[01]_model\.cif$', cif_name.lower())
    if m:
        return m.group(1).upper()
    return None


def build_val_mask(split_json_path, data_path):
    """Build boolean mask identifying validation samples from split.json.
    
    Returns:
        val_mask: (N,) boolean array — True = validation sample
    """
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    data = np.load(data_path, allow_pickle=True)
    mutants = data['mutants']
    prefixes = data['protein_prefix']
    y_bin = data['y_bin']
    N = len(mutants)

    # Build validation entries from split.json
    val_entries = []
    for family_key, splits in split_data.get('families', {}).items():
        family_base = family_key.replace('_wt_model', '').lower()

        for cif in splits.get('val_positives', []):
            mut = parse_mutation_from_cif(cif)
            if mut:
                val_entries.append((family_base, mut.lower(), 1))

        for cif in splits.get('val_negatives', []):
            mut = parse_mutation_from_cif(cif)
            if mut:
                val_entries.append((family_base, mut.lower(), 0))

    # Match against dataset
    val_mask = np.zeros(N, dtype=bool)
    matched = 0
    for i in range(N):
        mut_i = str(mutants[i]).lower()
        prefix_i = str(prefixes[i]).lower()
        label_i = int(y_bin[i])

        for family_base, val_mut, val_label in val_entries:
            if mut_i == val_mut and label_i == val_label and family_base.startswith(prefix_i):
                val_mask[i] = True
                matched += 1
                break

    n_pos = int((y_bin[val_mask] == 1).sum())
    n_neg = int((y_bin[val_mask] == 0).sum())
    print(f"[VAL] split.json → {len(val_entries)} entries, matched {matched}/{N} samples | val_pos={n_pos}, val_neg={n_neg}")
    return val_mask


def load_val_data(data_path, val_mask):
    """Preload validation data once (avoid re-reading NPZ each epoch)."""
    data = np.load(data_path, allow_pickle=True)
    return {
        'X': data['X'][val_mask].astype(np.float32),
        'X_local': data['X_local'][val_mask].astype(np.float32),
        'y_bin': data['y_bin'][val_mask].astype(np.int64),
        'y_dms': data['y_dms_id'][val_mask].astype(np.int64),
        'positions': data['mutant_positions'][val_mask].astype(np.int64),
        'wt_global': data['wt_embeddings_global'].item(),
        'wt_full': data['wt_embeddings_full'].item(),
    }


# =============================================
# Validation Metrics (ContVAR-inspired)
# =============================================

def validate_epoch(model_global, model_local, val_data, device):
    """Compute comprehensive validation metrics.

    Metrics:
      - auroc_g / auroc_l: Distance-based AUC (global / local)
      - acc_g:             Per-protein accuracy (d_pos < d_neg)
      - d_pos_g, d_neg_g:  Mean distance to WT (positive / negative)
      - margin_g / margin_l: d_neg - d_pos (global / local)
      - alignment:         Mean ||z_wt − z_pos||² (lower = better)
      - uniformity:        log(mean(exp(−2·||z_i−z_j||²))) (lower = better)
      - cos_pos / cos_neg: Mean cosine similarity (WT vs pos/neg)
      - emb_std:           Embedding std (collapse detection, should be > 0)
    """
    X_val = val_data['X']
    X_local_val = val_data['X_local']
    y_bin_val = val_data['y_bin']
    y_dms_val = val_data['y_dms']
    positions_val = val_data['positions']
    wt_global_dict = val_data['wt_global']
    wt_full_dict = val_data['wt_full']

    if len(X_val) == 0:
        return {}

    was_training_g = model_global.training
    was_training_l = model_local.training
    model_global.eval()
    model_local.eval()

    with torch.no_grad():
        Z_g = model_global(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy()
        Z_l = model_local(torch.tensor(X_local_val, dtype=torch.float32, device=device)).cpu().numpy()

    # Project WT global embeddings
    wt_proj = {}
    for dms_id, emb in wt_global_dict.items():
        emb_np = np.asarray(emb, dtype=np.float32)
        with torch.no_grad():
            z = model_global(torch.tensor(emb_np, dtype=torch.float32, device=device).unsqueeze(0))
        wt_proj[dms_id] = z.squeeze(0).cpu().numpy()

    # Project WT local at each val sample's mutation position
    wt_local_proj = np.zeros_like(Z_l)
    for i in range(len(X_val)):
        dms_id = int(y_dms_val[i])
        pos = int(positions_val[i])
        wt_full = wt_full_dict.get(dms_id)
        if wt_full is not None:
            residue = np.asarray(wt_full[pos], dtype=np.float32)
            with torch.no_grad():
                z = model_local(torch.tensor(residue, dtype=torch.float32, device=device).unsqueeze(0))
            wt_local_proj[i] = z.squeeze(0).cpu().numpy()

    # --- Masks ---
    pos_mask = y_bin_val == 1
    neg_mask = y_bin_val == 0
    metrics = {}

    # --- GLOBAL DISTANCES ---
    d_global = np.array([
        np.sqrt(np.sum((Z_g[i] - wt_proj.get(int(y_dms_val[i]), Z_g[i])) ** 2))
        for i in range(len(Z_g))
    ])
    metrics['d_pos_g'] = float(d_global[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
    metrics['d_neg_g'] = float(d_global[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
    metrics['margin_g'] = metrics['d_neg_g'] - metrics['d_pos_g']

    # --- LOCAL DISTANCES ---
    d_local = np.sqrt(np.sum((Z_l - wt_local_proj) ** 2, axis=1))
    metrics['d_pos_l'] = float(d_local[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
    metrics['d_neg_l'] = float(d_local[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
    metrics['margin_l'] = metrics['d_neg_l'] - metrics['d_pos_l']

    # --- AUROC (distance-based) ---
    if len(np.unique(y_bin_val)) >= 2:
        try:
            metrics['auroc_g'] = float(roc_auc_score(y_bin_val, -d_global))
        except Exception:
            metrics['auroc_g'] = 0.5
        try:
            metrics['auroc_l'] = float(roc_auc_score(y_bin_val, -d_local))
        except Exception:
            metrics['auroc_l'] = 0.5
    else:
        metrics['auroc_g'] = 0.5
        metrics['auroc_l'] = 0.5

    # --- PER-PROTEIN ACCURACY ---
    accs = []
    for dms_id in np.unique(y_dms_val):
        dm = y_dms_val == dms_id
        dp = d_global[dm & pos_mask]
        dn = d_global[dm & neg_mask]
        if len(dp) > 0 and len(dn) > 0:
            accs.append(float(dp.mean() < dn.mean()))
    metrics['acc_g'] = float(np.mean(accs)) if accs else 0.0

    # --- ALIGNMENT: mean ||z_wt - z_pos||² ---
    align_vals = []
    for i in np.where(pos_mask)[0]:
        dms_id = int(y_dms_val[i])
        if dms_id in wt_proj:
            align_vals.append(np.sum((Z_g[i] - wt_proj[dms_id]) ** 2))
    metrics['alignment'] = float(np.mean(align_vals)) if align_vals else 0.0

    # --- UNIFORMITY: log(mean(exp(-2·||z_i-z_j||²))) ---
    if len(Z_g) > 1:
        pw_sq = pdist(Z_g, 'sqeuclidean')
        metrics['uniformity'] = float(np.log(np.mean(np.exp(-2 * pw_sq)) + 1e-10))
    else:
        metrics['uniformity'] = 0.0

    # --- COSINE SIMILARITY ---
    cos_pos, cos_neg = [], []
    for i in range(len(X_val)):
        dms_id = int(y_dms_val[i])
        if dms_id in wt_proj:
            wt_norm = np.linalg.norm(wt_proj[dms_id])
            z_norm = np.linalg.norm(Z_g[i])
            if wt_norm > 1e-8 and z_norm > 1e-8:
                cos = np.dot(Z_g[i], wt_proj[dms_id]) / (z_norm * wt_norm)
                if y_bin_val[i] == 1:
                    cos_pos.append(cos)
                else:
                    cos_neg.append(cos)
    metrics['cos_pos'] = float(np.mean(cos_pos)) if cos_pos else 0.0
    metrics['cos_neg'] = float(np.mean(cos_neg)) if cos_neg else 0.0

    # --- EMBEDDING STD (collapse detection) ---
    metrics['emb_std'] = float(Z_g.std(axis=0).mean())

    # --- VALIDATION TRIPLET LOSS ---
    # For each protein with both pos & neg val samples, form triplets and compute loss
    MARGIN = 0.3
    triplet_losses_g = []
    triplet_losses_l = []
    for dms_id in np.unique(y_dms_val):
        dm = y_dms_val == dms_id
        pos_idx = np.where(dm & pos_mask)[0]
        neg_idx = np.where(dm & neg_mask)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        if dms_id not in wt_proj:
            continue
        wt_g = wt_proj[dms_id]  # (256,)
        # All (pos, neg) pairs for this protein
        for pi in pos_idx:
            for ni in neg_idx:
                d_pos_trip = np.sqrt(np.sum((Z_g[pi] - wt_g) ** 2))
                d_neg_trip = np.sqrt(np.sum((Z_g[ni] - wt_g) ** 2))
                triplet_losses_g.append(max(0.0, d_pos_trip - d_neg_trip + MARGIN))
                # Local
                d_pos_l = np.sqrt(np.sum((Z_l[pi] - wt_local_proj[pi]) ** 2))
                d_neg_l = np.sqrt(np.sum((Z_l[ni] - wt_local_proj[ni]) ** 2))
                triplet_losses_l.append(max(0.0, d_pos_l - d_neg_l + MARGIN))

    metrics['val_loss_g'] = float(np.mean(triplet_losses_g)) if triplet_losses_g else 0.0
    metrics['val_loss_l'] = float(np.mean(triplet_losses_l)) if triplet_losses_l else 0.0
    metrics['val_loss'] = metrics['val_loss_g'] + metrics['val_loss_l']

    # Restore training mode
    if was_training_g:
        model_global.train()
    if was_training_l:
        model_local.train()

    return metrics


def print_val_metrics(metrics, epoch=None):
    """Pretty-print validation metrics."""
    prefix = f"  [VAL Epoch {epoch}]" if epoch else "  [VAL]"
    print(
        f"{prefix} "
        f"AUROC_g={metrics.get('auroc_g', 0):.4f} AUROC_l={metrics.get('auroc_l', 0):.4f} | "
        f"Acc={metrics.get('acc_g', 0):.4f} | "
        f"d_pos={metrics.get('d_pos_g', 0):.4f} d_neg={metrics.get('d_neg_g', 0):.4f} "
        f"margin_g={metrics.get('margin_g', 0):.4f} margin_l={metrics.get('margin_l', 0):.4f}"
    )
    print(
        f"         "
        f"cos_pos={metrics.get('cos_pos', 0):.4f} cos_neg={metrics.get('cos_neg', 0):.4f} | "
        f"align={metrics.get('alignment', 0):.4f} unif={metrics.get('uniformity', 0):.4f} | "
        f"emb_std={metrics.get('emb_std', 0):.4f} | "
        f"val_loss={metrics.get('val_loss', 0):.4f} (G={metrics.get('val_loss_g', 0):.4f}, L={metrics.get('val_loss_l', 0):.4f})"
    )


