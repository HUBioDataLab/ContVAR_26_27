# src/scripts/04_train_triplet.py
#
#   python src/scripts/04_train_triplet.py --data data/processed/tpmt_dataset.npz

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from build_dataset import get_triplet_dataloader
from tqdm import tqdm
from validation import build_val_mask, load_val_data, validate_epoch, print_val_metrics, plot_tsne


# TRAINING CONFIG 
EPOCHS = 400
BATCH_SIZE = 32
LR_GLOBAL = 1e-4 
LR_LOCAL = 1e-4   # Faster for local
MARGIN = 0.3
OUT_DIM = 256

MODEL_OUT = "models/proj_triplet.pt"
PLOT_OUT = "models/triplet_mining_stats.png"
LOSS_PLOT_OUT = "models/triplet_loss_curve.png"
VAL_PLOT_OUT = "models/triplet_val_curve.png" # EKLEME: Val Plot yolu


DEVICE = None 

def pick_device():
    if DEVICE is not None:
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"



# Projection Head
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


# Training

def train_triplet(data_path, device: str, mining_strategy="semi-hard", n_neg_candidates=5, group_by_protein=False, split_json_path=None):
    print(f"[INFO] device: {device}, mining: {mining_strategy}, n_neg: {n_neg_candidates}, group_by_protein: {group_by_protein}")

    # WandB initialization is done in main(), so we assume it's active.

    # =====================
    # Validation Setup
    # =====================
    val_data = None
    if split_json_path and os.path.exists(split_json_path):
        print(f"[INFO] Loading validation split from {split_json_path}")
        val_mask = build_val_mask(split_json_path, data_path)
        if val_mask.sum() > 0:
            val_data = load_val_data(data_path, val_mask)
            print(f"[INFO] Validation set: {len(val_data['X'])} samples ({(val_data['y_bin']==1).sum()} pos, {(val_data['y_bin']==0).sum()} neg)")
        else:
            print("[WARN] No validation samples matched! Continuing without validation.")
    else:
        print("[INFO] No split.json provided, training without validation.")

    best_val_auroc = 0.0

    n_neg = n_neg_candidates if mining_strategy == "semi-hard" else 1
    dl, embed_dim = get_triplet_dataloader(
        data_path, BATCH_SIZE, n_neg_candidates=n_neg, group_by_protein=group_by_protein
    )

    model_global = ProjectionHead(in_dim=embed_dim, out_dim=OUT_DIM, deep=True).to(device)
    model_local = ProjectionHead(in_dim=embed_dim, out_dim=OUT_DIM, deep=False).to(device)

    # Optimizer (both models with different LRs)
    opt = torch.optim.AdamW([
        {'params': model_global.parameters(), 'lr': LR_GLOBAL},
        {'params': model_local.parameters(), 'lr': LR_LOCAL}
    ])
    
    # --- YENİ: LR Scheduler ---
    # T_max: Kaç epoch'ta minimum LR'a ulaşılacağı. 
    # eta_min: İneceği en düşük LR seviyesi.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    # --------------------------
    loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)


    history = {
        "loss": [],
        "loss_g": [],
        "loss_l": [],
        # Global stats
        "easy_g": [],
        "semi_hard_g": [],
        "hard_g": [],
        # Local stats
        "easy_l": [],
        "semi_hard_l": [],
        "hard_l": [],
        # EKLEME: Validation history
        "val_auroc_g": [],
        "val_auroc_l": [],
    }


    model_global.train()
    model_local.train()


    for ep in range(EPOCHS):
        total_loss = 0.0
        total_loss_g = 0.0
        total_loss_l = 0.0
        
        # Separate counters
        c_easy_g = c_semi_g = c_hard_g = 0
        c_easy_l = c_semi_l = c_hard_l = 0

        # Collect embeddings for std tracking
        epoch_embeddings_g = []


        for (anchor_data, positive_data, negative_data, local_pair_data) in tqdm(dl):
            # =============================================
            # Unpack new structure
            # =============================================
            (anchor,) = anchor_data                     # WT global
            (positive,) = positive_data                 # Positive mutant global
            (negatives,) = negative_data                # Negative mutant(s) global
            
            wt_local, mut_local, label = local_pair_data  # Local contrastive pair

            # -----------------------------
            # GLOBAL Data
            # -----------------------------
            a_g = anchor.to(device)             
            p_g = positive.to(device)           
            n_g = negatives.to(device)       

            B_size = a_g.shape[0]
            # Handle K check
            if len(n_g.shape) == 3:
                K = n_g.shape[1]
            else:
                K = 1
                n_g = n_g.unsqueeze(1)

            # Forward Global
            za_g = model_global(a_g)
            zp_g = model_global(p_g)
            
            # Flatten negatives for global [B*K, D]
            zn_g = model_global(n_g.view(B_size*K, -1))
            zn_g = zn_g.view(B_size, K, -1)

            # Collect for std tracking
            epoch_embeddings_g.append(za_g.detach().cpu())
            epoch_embeddings_g.append(zp_g.detach().cpu())

            # -----------------------------
            # LOCAL Data (her sample kendi pozisyonunda)
            # -----------------------------
            wt_l = wt_local.to(device)      # [B, D]  WT residue at sample's own position
            mut_l = mut_local.to(device)    # [B, D]  Mutant residue at same position
            lbl = label.to(device).float()  # [B]     1.0=iyi, 0.0=kötü

            # Forward Local
            z_wt_l = model_local(wt_l)      # [B, out_dim]
            z_mut_l = model_local(mut_l)    # [B, out_dim]


            # -----------------------------
            # Mining (GLOBAL embeddings)
            # -----------------------------
            # -----------------------------
            # Mining Helper
            # -----------------------------
            def dist_sq(x, y):
                return torch.sum((x - y) ** 2, dim=-1)

            def mine_negatives(za, zp, zn, margin, strategy):
                """
                za: [B, D]
                zp: [B, D]
                zn: [B, K, D]
                """
                # Distances
                d_p = torch.sqrt(dist_sq(za, zp) + 1e-8)             # [B]
                d_n = torch.sqrt(dist_sq(za.unsqueeze(1), zn) + 1e-8) # [B, K]
                
                d_p_np = d_p.detach().cpu().numpy()
                d_n_np = d_n.detach().cpu().numpy()
                
                final_zn = []
                c_easy = c_semi = c_hard = 0
                
                best_k_indices = [] 

                for i in range(B_size):
                    dp_i = d_p_np[i]
                    dn_i = d_n_np[i]
                    
                    semi_mask = (dn_i > dp_i) & (dn_i < dp_i + margin)
                    hard_mask = dn_i <= dp_i
                    valid_mask = semi_mask | hard_mask
                    
                    if np.any(valid_mask):
                        candidates = np.where(valid_mask)[0]
                        if strategy == "random":
                             best_k = np.random.choice(len(dn_i))
                        else:
                             best_k = np.random.choice(candidates)
                    else:
                        best_k = np.argmin(dn_i)
                        
                    best_k_indices.append(best_k)
                    final_zn.append(zn[i, best_k])
                    
                    sel_dn = dn_i[best_k]
                    if sel_dn > dp_i + margin:
                        c_easy += 1
                    elif sel_dn < dp_i:
                        c_hard += 1
                    else:
                        c_semi += 1
                        
                return torch.stack(final_zn), best_k_indices, (c_easy, c_semi, c_hard)

            # -----------------------------
            # Mining (GLOBAL)
            # -----------------------------
            hardest_zn_g, best_k_indices_g, (eg, sg, hg) = mine_negatives(za_g, zp_g, zn_g, MARGIN, mining_strategy)
            
            c_easy_g += eg
            c_semi_g += sg
            c_hard_g += hg

            # Calculate Global Loss (Triplet)
            loss_g = loss_fn(za_g, zp_g, hardest_zn_g)


            # -----------------------------
            # Loss Local (Contrastive Loss - Pozisyon Bazlı)
            # -----------------------------
            # Her sample kendi pozisyonundaki WT ile karşılaştırılıyor
            # Farklı pozisyonlar arası karşılaştırma YOK
            #
            # label=1 (iyi mutant) → WT'ye yakınlaştır (d²)
            # label=0 (kötü mutant) → WT'den uzaklaştır (max(0, m-d)²)
            
            d_local = torch.sqrt(dist_sq(z_wt_l, z_mut_l) + 1e-8)  # [B]
            
            # Contrastive Loss: label bazlı
            loss_attract = lbl * (d_local ** 2)                                  # İyi mutantlar → yakınlaştır
            loss_repel   = (1.0 - lbl) * (F.relu(MARGIN - d_local) ** 2) # Kötü mutantlar → uzaklaştır
            loss_l = (loss_attract + loss_repel).mean()
            
            # Stats (contrastive perspektifinden)
            with torch.no_grad():
                neg_mask = (lbl == 0)
                pos_mask = (lbl == 1)
                if neg_mask.sum() > 0:
                    d_neg = d_local[neg_mask]
                    c_easy_l += int((d_neg > MARGIN).sum().item())
                    c_hard_l += int((d_neg < MARGIN * 0.5).sum().item())  # Çok yakın olanlar
                    c_semi_l += int(((d_neg >= MARGIN * 0.5) & (d_neg <= MARGIN)).sum().item())


            # -----------------------------
            # Combined Loss
            # -----------------------------
            loss = (loss_g + loss_l) / 2

            opt.zero_grad()
            
            loss.backward()
            
            opt.step()

            total_loss += loss.item()
            total_loss_g += loss_g.item()
            total_loss_l += loss_l.item()


        avg_loss = total_loss / max(1, len(dl))
        avg_loss_g = total_loss_g / max(1, len(dl))
        avg_loss_l = total_loss_l / max(1, len(dl))

        history["loss"].append(avg_loss)
        history["loss_g"].append(avg_loss_g)
        history["loss_l"].append(avg_loss_l)
        
        history["easy_g"].append(c_easy_g)
        history["semi_hard_g"].append(c_semi_g)
        history["hard_g"].append(c_hard_g)
        
        history["easy_l"].append(c_easy_l)
        history["semi_hard_l"].append(c_semi_l)
        history["hard_l"].append(c_hard_l)


        # Compute train embedding std
        all_emb_g = torch.cat(epoch_embeddings_g, dim=0).numpy()
        train_emb_std = float(all_emb_g.std(axis=0).mean())

        current_lr_global = opt.param_groups[0]['lr']
        current_lr_local = opt.param_groups[1]['lr']

        print(
            f"[EPOCH {ep+1:02d}/{EPOCHS}] "
            f"loss={avg_loss:.4f} (G={avg_loss_g:.4f}, L={avg_loss_l:.4f}) | "
            f"loss={avg_loss:.4f} | LR_G={current_lr_global:.2e} |  LR_L={current_lr_local:.2e} | "
            f"Global[Easy:{c_easy_g}, Semi:{c_semi_g}, Hard:{c_hard_g}] | "
            f"Local[Easy:{c_easy_l}, Semi:{c_semi_l}, Hard:{c_hard_l}] | "
            f"train_emb_std={train_emb_std:.4f}"
        )
        scheduler.step()
        # =====================
        # Validation
        # =====================
        val_metrics = {}
        if val_data is not None:
            val_metrics = validate_epoch(model_global, model_local, val_data, device)
            print_val_metrics(val_metrics, epoch=ep+1)
            # EKLEME: Log metrics to history
            history["val_auroc_g"].append(val_metrics.get('auroc_g', 0))
            history["val_auroc_l"].append(val_metrics.get('auroc_l', 0))
        else:
            # Sync history length even if no val
            history["val_auroc_g"].append(0)
            history["val_auroc_l"].append(0)

        # Log to WandB
        log_dict = {
            "epoch": ep + 1,
            "train_loss": avg_loss,
            "train_loss_global": avg_loss_g,
            "train_loss_local": avg_loss_l,
            # Global mining
            "global_easy": c_easy_g,
            "global_semi": c_semi_g,
            "global_hard": c_hard_g,
            # Local mining
            "local_easy": c_easy_l,
            "local_semi": c_semi_l,
            "local_hard": c_hard_l,
            "train_emb_std": train_emb_std,
        }
        # Add validation metrics to WandB
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
        
        log_dict.update({
            "lr_global": current_lr_global,
            "lr_local": current_lr_local
        })

        wandb.log(log_dict)

        # Save best model (prefer val AUROC if available, else train loss)
        save_model = False
        if val_metrics and 'auroc_g' in val_metrics:
            if val_metrics['auroc_g'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc_g']
                save_model = True
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model = True

        if save_model:
            torch.save(
                {
                    "model_state_dict": model_global.state_dict(),
                    "local_model_state_dict": model_local.state_dict(),
                    "in_dim": embed_dim,
                    "local_in_dim": embed_dim,
                    "out_dim": OUT_DIM,
                    "margin": MARGIN,
                },
                MODEL_OUT,
            )
            if val_metrics:
                print(f"[SAVE] best model -> {MODEL_OUT} (val_auroc_g={best_val_auroc:.4f})")
            else:
                print(f"[SAVE] best model -> {MODEL_OUT} (train_loss={best_loss:.4f})")

    print("[DONE] Training complete. Best loss:", best_loss)
    # -----------------------------
    # Plotting - Mining Stats (GLOBAL)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)

    plt.plot(epochs_range, history["easy_g"], label="Easy", marker="o", linestyle=":")
    plt.plot(epochs_range, history["semi_hard_g"], label="Semi-Hard", marker="o")
    plt.plot(epochs_range, history["hard_g"], label="Hard", marker="o")

    plt.title("GLOBAL Mining Stats")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plot_out_g = PLOT_OUT.replace(".png", "_global.png")
    plt.savefig(plot_out_g)
    print(f"[PLOT] Saved global mining stats to {plot_out_g}")
    
    # -----------------------------
    # Plotting - Mining Stats (LOCAL)
    # -----------------------------
    plt.figure(figsize=(10, 6))

    plt.plot(epochs_range, history["easy_l"], label="Easy", marker="o", linestyle=":")
    plt.plot(epochs_range, history["semi_hard_l"], label="Semi-Hard", marker="o")
    plt.plot(epochs_range, history["hard_l"], label="Hard", marker="o")

    plt.title("LOCAL Mining Stats")
    plt.xlabel("Epoch")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plot_out_l = PLOT_OUT.replace(".png", "_local.png")
    plt.savefig(plot_out_l)
    print(f"[PLOT] Saved local mining stats to {plot_out_l}")

    # -----------------------------
    # Plotting - Loss Curve
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs_range,
        history["loss"],
        label="Combined Loss",
        marker="o",
        color="purple",
    )
    plt.plot(
        epochs_range,
        history["loss_g"],
        label="Global Loss",
        linestyle="--",
        color="blue",
    )
    plt.plot(
        epochs_range,
        history["loss_l"],
        label="Local Loss",
        linestyle="--",
        color="orange",
    )


    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(LOSS_PLOT_OUT)
    print(f"[PLOT] Saved loss curve to {LOSS_PLOT_OUT}")

    # -----------------------------
    # EKLEME: Plotting - Validation Curves
    # -----------------------------
    if val_data is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history["val_auroc_g"], label="Val AUROC (Global)", marker="s", color="blue")
        plt.plot(epochs_range, history["val_auroc_l"], label="Val AUROC (Local)", marker="s", color="green")
        plt.title("Validation AUROC over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("AUROC")
        plt.legend()
        plt.grid(True)
        plt.savefig(VAL_PLOT_OUT)
        print(f"[PLOT] Saved validation curve to {VAL_PLOT_OUT}")

    # =====================
    # t-SNE Visualization
    # =====================
    print("\n" + "="*60)
    print("t-SNE VISUALIZATION")
    print("="*60)

    # Load best model for visualization
    best_ckpt = torch.load(MODEL_OUT, map_location=device, weights_only=False)
    model_global.load_state_dict(best_ckpt['model_state_dict'])
    model_local.load_state_dict(best_ckpt['local_model_state_dict'])
    print(f"[t-SNE] Loaded best model from {MODEL_OUT}")

    tsne_path1, tsne_path2 = plot_tsne(
        data_path, model_global, model_local, device,
        save_dir=os.path.dirname(MODEL_OUT),
        max_samples=3000
    )

    # Log t-SNE plots to WandB
    try:
        wandb.log({
            "tsne/embedding_comparison": wandb.Image(tsne_path1),
            "tsne/per_protein": wandb.Image(tsne_path2),
        })
        print("[t-SNE] Logged to WandB")
    except Exception as e:
        print(f"[t-SNE] WandB log skipped: {e}")

    return model_global, model_local



# =====================
# Main
# =====================
def main():
    global MODEL_OUT, PLOT_OUT, LOSS_PLOT_OUT, VAL_PLOT_OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to npz dataset")
    parser.add_argument("--mining", default="semi-hard", choices=["semi-hard", "random"], help="Mining strategy")
    parser.add_argument("--neg_candidates", type=int, default=5, help="Number of negative candidates to sample per anchor")
    parser.add_argument("--group_by_protein", action="store_true", help="Her batch sadece tek proteinden örnekler içerir")
    parser.add_argument("--project_name", default="protein-triplet-v1", help="WandB project name")
    parser.add_argument("--run_name", default=None, help="WandB run name")
    parser.add_argument("--split_json", default=None, help="Path to split.json for validation")
    parser.add_argument("--out", default=MODEL_OUT, help="Where to save best triplet checkpoint")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    
    args = parser.parse_args()

    MODEL_OUT = args.out
    out_dir = os.path.dirname(MODEL_OUT) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(MODEL_OUT))[0]
    PLOT_OUT = os.path.join(out_dir, f"{stem}_mining_stats.png")
    LOSS_PLOT_OUT = os.path.join(out_dir, f"{stem}_loss_curve.png")
    VAL_PLOT_OUT = os.path.join(out_dir, f"{stem}_val_curve.png")

    # Init WandB
    if not args.no_wandb:
        wandb.init(
            project=args.project_name, 
            name=args.run_name,
            config={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr_global": LR_GLOBAL,
                "lr_local": LR_LOCAL,
                "margin": MARGIN,
                "out_dim": OUT_DIM,
                "n_neg": args.neg_candidates,
                "group_by_protein": args.group_by_protein,
                "model": "SeparateHeads_NoBatchNorm",
                "validation": args.split_json is not None,
            }
        )
    else:
        wandb.init(mode="disabled")

    device = pick_device()
    train_triplet(args.data, device=device, mining_strategy=args.mining, n_neg_candidates=args.neg_candidates, group_by_protein=args.group_by_protein, split_json_path=args.split_json)



if __name__ == "__main__":
    main()
