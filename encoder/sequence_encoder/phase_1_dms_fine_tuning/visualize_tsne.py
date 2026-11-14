import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
# =============================================
# Model Architecture (Must match training script)
# =============================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, deep=False):
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
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z

def pick_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"
# =============================================
# t-SNE Visualization
# =============================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

def plot_tsne(data_path, model_global, model_local, device,
              save_dir="models", max_samples=3000, perplexity=30, seed=42, n_proteins=10):
    """Generate t-SNE plots comparing baseline vs projected embeddings.
    Filtered by top N proteins. Includes both Global and Local WT visibility.
    """
    print(f"\n[t-SNE] Loading data and filtering to Top {n_proteins} proteins...")

    data = np.load(data_path, allow_pickle=True)
    X_raw = data['X'].astype(np.float32)
    X_local_raw = data['X_local'].astype(np.float32)
    y_bin = data['y_bin'].astype(np.int64)
    y_dms = data['y_dms_id'].astype(np.int64)
    
    # Load WT dictionaries
    wt_global_dict = data['wt_embeddings_global'].item()
    # Attempt to load local WT embeddings; fallback to empty if not found
    wt_local_dict = data.get('wt_embeddings_local', {}).item() if 'wt_embeddings_local' in data.files else {}

    # --- Filter by Top K Proteins ---
    unique_ids, counts = np.unique(y_dms, return_counts=True)
    top_k_ids = unique_ids[np.argsort(-counts)[:n_proteins]]
    
    mask = np.isin(y_dms, top_k_ids)
    X_raw, X_local_raw = X_raw[mask], X_local_raw[mask]
    y_bin, y_dms = y_bin[mask], y_dms[mask]

    # Subsample if too large
    if len(X_raw) > max_samples:
        np.random.seed(seed)
        idx = np.random.choice(len(X_raw), max_samples, replace=False)
        X_raw, X_local_raw, y_bin, y_dms = X_raw[idx], X_local_raw[idx], y_bin[idx], y_dms[idx]
        print(f"[t-SNE] Subsampled to {max_samples} points")

    # --- Marker Assignment Logic ---
    filled_markers = ["o", "s", "D", "v", "^", "<", "p", "P", "*", "X", "h", "H", "8", "d"]
    unfilled_markers = ["1", "2", "3", "4", "+", "|", "_"]
    all_markers = filled_markers + unfilled_markers
    
    random.seed(seed)
    random.shuffle(all_markers)
    marker_mapping = {dms_id: all_markers[i % len(all_markers)] for i, dms_id in enumerate(top_k_ids)}

    # --- Project through models ---
    model_global.eval()
    model_local.eval()

    # Data structures for Wild-Types
    wt_payload = {'g_raw': [], 'g_proj': [], 'l_raw': [], 'l_proj': [], 'g_ids': [], 'l_ids': []}

    with torch.no_grad():
        # Project Mutants
        X_proj_g = model_global(torch.tensor(X_raw, device=device)).cpu().numpy()
        X_proj_l = model_local(torch.tensor(X_local_raw, device=device)).cpu().numpy()

        # Project ONLY relevant WT embeddings (Global & Local)
        for dms_id in top_k_ids:
            if dms_id in wt_global_dict:
                wt_payload['g_ids'].append(dms_id)
                
                # Global WT
                g_emb = np.asarray(wt_global_dict[dms_id], dtype=np.float32)
                wt_payload['g_raw'].append(g_emb)
                z_g = model_global(torch.tensor(g_emb, device=device).unsqueeze(0))
                wt_payload['g_proj'].append(z_g.squeeze(0).cpu().numpy())
                
                # A missing WT embedding must not be replaced by a mutant point
                # and displayed as WT. Local WT markers require actual WT data.
                if dms_id in wt_local_dict:
                    l_emb = np.asarray(wt_local_dict[dms_id], dtype=np.float32)
                    wt_payload['l_ids'].append(dms_id)
                    wt_payload['l_raw'].append(l_emb)
                    z_l = model_local(torch.tensor(l_emb, device=device).unsqueeze(0))
                    wt_payload['l_proj'].append(z_l.squeeze(0).cpu().numpy())

    # ============================
    # 1. Comparison Plot (2x2)
    # ============================
    def with_wt(mutants, wt_key, ids_key):
        wt = wt_payload[wt_key]
        return {'X': np.vstack([mutants, wt]) if wt else mutants,
                'n_wt': len(wt), 'wt_ids': wt_payload[ids_key]}

    datasets = {
        'Unprojected Global input': with_wt(X_raw, 'g_raw', 'g_ids'),
        'DMS projection: Global': with_wt(X_proj_g, 'g_proj', 'g_ids'),
        'Unprojected Local input': with_wt(X_local_raw, 'l_raw', 'l_ids'),
        'DMS projection: Local': with_wt(X_proj_l, 'l_proj', 'l_ids'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle(f't-SNE Embedding Visualization (Top {n_proteins} DMS assay IDs)\nUnprojected input vs DMS projection · Global vs Local',
                  fontsize=18, fontweight='bold', y=0.98)
    legend = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color, label=label)
              for color, label in [('#e74c3c', 'DMS binary label 0'), ('#2ecc71', 'DMS binary label 1'), ('gold', 'WT')]]
    fig.legend(handles=legend, loc='lower center', ncol=3)
    fig.text(.5, .035, 'Independent t-SNE fits: axes and distances are not aligned across panels. DMS binary labels are not GO-specific LOF/GOF labels.', ha='center')

    for ax, (title, dset) in zip(axes.flatten(), datasets.items()):
        print(f"[t-SNE] Computing: {title}...")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(dset['X']) - 1),
                    random_state=seed, max_iter=1000, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(dset['X'])

        n_wt = dset['n_wt']
        n_mut = len(dset['X']) - n_wt
        mut_2d, wt_2d = emb_2d[:n_mut], emb_2d[n_mut:]

        # Plot Mutants per protein
        for dms_id in top_k_ids:
            marker = marker_mapping[dms_id]
            is_filled = marker not in unfilled_markers
            ec = 'none' if is_filled else None
            mask_p = y_dms == dms_id
            
            # Bad (Red) / Good (Green)
            for label_val, color in [(0, '#e74c3c'), (1, '#2ecc71')]:
                mask_m = (y_bin == label_val) & mask_p
                if np.any(mask_m):
                    ax.scatter(mut_2d[mask_m, 0], mut_2d[mask_m, 1],
                               c=color, marker=marker, s=40, alpha=0.4, edgecolors=ec,linewidth=1.2 if not is_filled else 0.5)

        # Plot WT as gold stars/shapes (Visible in ALL 4 plots)
        for i, dms_id in enumerate(dset['wt_ids']):
            marker = marker_mapping[dms_id]
            is_filled = marker not in unfilled_markers
            ax.scatter(wt_2d[i, 0], wt_2d[i, 1],
                       c='gold', s=180, marker=marker, 
                       edgecolors='black' if is_filled else None,
                       linewidth=1.2, zorder=10)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE coordinate 1 (arbitrary units)')
        ax.set_ylabel('t-SNE coordinate 2 (arbitrary units)')
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, .07, 1, 0.95])
    save_path = os.path.join(save_dir, f"tsne_comparison_top_{n_proteins}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    # ============================
    # 2. Per-protein Clustering Plot
    # ============================
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 9))
    fig2.suptitle(f't-SNE — DMS assay grouping (Top {n_proteins} assay IDs)',
                  fontsize=16, fontweight='bold')
    fig2.text(.5, -.01, 'Independent t-SNE fits; visual separation is exploratory and does not establish improved downstream prediction.', ha='center')

    cmap = plt.cm.get_cmap('tab10' if n_proteins <= 10 else 'tab20', n_proteins)

    for ax, (title, X_emb) in zip(axes2, [('Unprojected Global input', X_raw), ('DMS projection: Global', X_proj_g)]):
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X_emb) - 1),
                    random_state=seed, max_iter=1000, init='pca', learning_rate='auto')
        emb_2d = tsne.fit_transform(X_emb)

        for ci, dms_id in enumerate(top_k_ids):
            marker = marker_mapping[dms_id]
            is_filled = marker not in unfilled_markers
            mask_prot = y_dms == dms_id
            
            ax.scatter(emb_2d[mask_prot, 0], emb_2d[mask_prot, 1],
                        c=[cmap(ci)], marker=marker, s=35, alpha=0.7,
                        label=f'DMS assay {dms_id}',
                        edgecolors='none' if is_filled else None)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE coordinate 1 (arbitrary units)')
        ax.set_ylabel('t-SNE coordinate 2 (arbitrary units)')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8, ncol=2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path2 = os.path.join(save_dir, f"tsne_per_protein_top_{n_proteins}.png")
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved Comparison → {save_path}")
    print(f"[SUCCESS] Saved Per-Protein → {save_path2}")
    return save_path, save_path2
# =============================================
# Main Visualization Script
# =============================================
def main():
    parser = argparse.ArgumentParser(description="Visualize pre-trained protein triplet models using t-SNE.")
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--model_path", default="models/proj_triplet.pt", help="Path to saved .pt model")
    parser.add_argument("--n_proteins", type=int, default=10, help="Number of top proteins to include in t-SNE")
    parser.add_argument("--max_samples", type=int, default=3000, help="Max mutant samples for t-SNE")
    parser.add_argument("--save_dir", default="visualizations", help="Where to save plots")
    
    args = parser.parse_args()
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    # 1. Load Checkpoint
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # 2. Instantiate Models with dimensions from checkpoint
    in_dim = checkpoint.get('in_dim', 1280) # Default to ESM-2 dimension if missing
    out_dim = checkpoint.get('out_dim', 256)
    
    model_global = ProjectionHead(in_dim=in_dim, out_dim=out_dim, deep=True).to(device)
    model_local = ProjectionHead(in_dim=in_dim, out_dim=out_dim, deep=False).to(device)

    # 3. Load weights
    model_global.load_state_dict(checkpoint['model_state_dict'])
    model_local.load_state_dict(checkpoint['local_model_state_dict'])
    print(f"[INFO] Loaded pre-trained models from {args.model_path}")

    # 4. Generate t-SNE plots
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"[INFO] Starting t-SNE for top {args.n_proteins} proteins...")
    p1, p2 = plot_tsne(
        data_path=args.data,
        model_global=model_global,
        model_local=model_local,
        device=device,
        save_dir=args.save_dir,
        max_samples=args.max_samples,
        n_proteins=args.n_proteins
    )

    print("\n" + "="*30)
    print(f"[SUCCESS] Visualization complete!")
    print(f"Comparison Plot: {p1}")
    print(f"Per-Protein Plot: {p2}")
    print("="*30)

if __name__ == "__main__":
    main()
