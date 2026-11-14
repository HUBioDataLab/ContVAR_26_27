# go_pretrain/extract_embeddings.py
# Stage 1 sonrası: Fine-tuned ESM2 backbone ile DMS proteinleri için
# yeni embedding'ler çıkart. Çıktı mevcut NPZ formatıyla uyumlu.
#
# Kullanım:
#   python extract_embeddings.py \
#       --backbone models/go_pretrained/backbone.pt \
#       --esm_model esm2_t6_8M_UR50D \
#       --fasta data/raw/fasta_files_proteinGym_assays_100variant_per_protein \
#       --h5 data/raw/embeddings_variable.h5 \
#       --json triplet_data.json \
#       --out data/processed/go_finetuned_dms_dataset.npz

"""
Bu script, GO pretraining'den çıkan fine-tuned ESM2 backbone'u kullanarak
DMS proteinlerinin embedding'lerini yeniden çıkartır.

Mevcut pipeline (Stage 2) bu yeni embedding'lerle çalışabilir çünkü
çıktı formatı mevcut combined_dms_dataset.npz ile aynıdır.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    from .go_model import load_esm2_model
except ImportError:
    from go_model import load_esm2_model


def load_finetuned_backbone(backbone_path, esm_model_name, device):
    """
    GO pretrained backbone'u yükle.
    
    Args:
        backbone_path: backbone.pt dosya yolu
        esm_model_name: Orijinal ESM2 model adı
        device: torch device
    
    Returns:
        model: ESM2 model (fine-tuned ağırlıklarla)
        alphabet: ESM2 alphabet
    """
    # Önce base model'i yükle
    model, alphabet = load_esm2_model(esm_model_name)
    
    # Fine-tuned ağırlıkları yükle
    checkpoint = torch.load(backbone_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['esm_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Loaded fine-tuned backbone from {backbone_path}")
    print(f"  ESM dim: {checkpoint['esm_dim']}")
    
    return model, alphabet


def extract_embedding_from_sequence(model, alphabet, sequence, device, max_len=1022):
    """
    Tek bir protein sekansı için embedding çıkart.
    
    Returns:
        global_emb: (D,) — mean-pooled protein embedding
        full_emb:   (L, D) — her residue için embedding
    """
    # Tokenize
    batch_converter = alphabet.get_batch_converter()
    seq_truncated = sequence[:max_len]
    _, _, tokens = batch_converter([("protein", seq_truncated)])
    tokens = tokens.to(device)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    
    # Son katman representation
    token_emb = results["representations"][model.num_layers]  # (1, L+2, D) — BOS/EOS dahil
    
    # BOS ve EOS'u çıkar
    residue_emb = token_emb[0, 1:-1, :]  # (L, D) — sadece residue'ler
    
    # Global: mean pool
    global_emb = residue_emb.mean(dim=0)  # (D,)
    
    return global_emb.cpu().numpy(), residue_emb.cpu().numpy()


def extract_dms_embeddings(model, alphabet, json_path, device, max_len=1022):
    """
    DMS proteinleri için fine-tuned ESM2 embedding'leri çıkart.
    
    Mevcut build_dataset.py'deki yapıyı takip eder:
      - WT global embedding
      - WT full embedding (tüm residue'ler)
      - Mutant global embedding
      - Mutant local embedding (mutasyon pozisyonundaki residue)
    
    Returns:
        Mevcut NPZ formatıyla uyumlu dict
    """
    import json
    
    with open(json_path, 'r') as f:
        data_json = json.load(f)
    
    X, X_local, y_bin, y_score, y_dms = [], [], [], [], []
    mutants, names, prefixes, dms_ids = [], [], [], []
    mutant_positions = []
    wt_embeddings_global = {}
    wt_embeddings_full = {}
    
    dms_map = {}
    for wt_name_search in data_json.keys():
        wt_name = "_".join(wt_name_search.split("_")[:-1])
        if wt_name not in dms_map:
            dms_map[wt_name] = len(dms_map)
    
    print(f"[INFO] Processing {len(data_json)} protein families...")
    
    for wt_name_search, content in tqdm(data_json.items()):
        wt_name = "_".join(wt_name_search.split("_")[:-1])
        dms_idx = dms_map[wt_name]

        wt_sequence = content.get("wt_sequence", None)
        if wt_sequence is None:
            print(f"  [WARN] No WT sequence for {wt_name_search}, skipping")
            continue
        
        wt_global, wt_full = extract_embedding_from_sequence(
            model, alphabet, wt_sequence, device, max_len
        )
        wt_embeddings_global[dms_idx] = wt_global
        wt_embeddings_full[dms_idx] = wt_full
        
        for label_type in ["positives", "negatives"]:
            bin_label = 1 if label_type == "positives" else 0
            
            for entry in content[label_type]:
                mut = entry["mutation"]
                position = int(entry["position"])
                pos_0 = position - 1 
                
                
                mut_from = mut[0]    
                mut_to = mut[-1]     
                mut_pos = int(mut[1:-1]) - 1  
                
                mut_sequence = list(wt_sequence)
                mut_sequence[mut_pos] = mut_to
                mut_sequence = ''.join(mut_sequence)
                
                # Mutant embedding çıkart
                mut_global, mut_full = extract_embedding_from_sequence(
                    model, alphabet, mut_sequence, device, max_len
                )
                
                X.append(mut_global)
                X_local.append(mut_full[pos_0] if pos_0 < len(mut_full) else mut_global)
                mutant_positions.append(pos_0)
                
                y_bin.append(bin_label)
                y_score.append(float(bin_label))
                y_dms.append(dms_idx)
                
                mutants.append(mut)
                names.append(f"{wt_name}_{mut}")
                prefixes.append(wt_name)
                dms_ids.append(wt_name)
    
    print(f"[INFO] Total samples: {len(X)}")
    print(f"[INFO] Total WTs: {len(wt_embeddings_global)}")
    
    return {
        'X': np.asarray(X, np.float32),
        'X_local': np.asarray(X_local, np.float32),
        'y_bin': np.asarray(y_bin, np.int64),
        'y_score': np.asarray(y_score, np.float32),
        'y_dms_id': np.asarray(y_dms, np.int64),
        'dms_id_str': np.asarray(dms_ids, dtype=object),
        'mutants': np.asarray(mutants, dtype=object),
        'names': np.asarray(names, dtype=object),
        'protein_prefix': np.asarray(prefixes, dtype=object),
        'mutant_positions': np.asarray(mutant_positions, dtype=np.int64),
        'wt_embeddings_global': np.asarray(wt_embeddings_global, dtype=object),
        'wt_embeddings_full': np.asarray(wt_embeddings_full, dtype=object),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings using fine-tuned ESM2")
    parser.add_argument("--backbone", required=True, help="Path to backbone.pt")
    parser.add_argument("--esm_model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--json", required=True, help="Path to triplet_data.json")
    parser.add_argument("--out", required=True, help="Output NPZ path")
    parser.add_argument("--max_seq_len", type=int, default=1022)
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Device: {device}")
    
    # Load fine-tuned backbone
    model, alphabet = load_finetuned_backbone(
        args.backbone, args.esm_model, device
    )
    
    # Extract embeddings
    result = extract_dms_embeddings(
        model, alphabet, args.json, device, args.max_seq_len
    )
    
    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **result)
    print(f"[DONE] Saved → {args.out}")


if __name__ == "__main__":
    main()
