import os
import glob
import h5py
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

def pool(x):
    return x.mean(axis=0)


def get_prefix_from_csv_path(p):
    return os.path.splitext(os.path.basename(p))[0]


def find_wt_key(keys, prefix):
    for k in keys:
        if k.lower().startswith(prefix) and k.split("_")[-1]== "WT":
            return k
    return None


def find_mut_key(keys, prefix, mutant):
    start = f"{prefix}_{mutant}"
    for k in keys:
        if k.lower().startswith(prefix.lower()) and mutant.lower() in k.lower():
            return k
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/raw/embeddings_variable.h5")
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import json

    with open(args.json, "r") as f_json:
        data_json = json.load(f_json)

    f = h5py.File(args.h5, "r")
    keys = list(f.keys())

    X, X_local, y_bin, y_score, y_dms = [], [], [], [], []
    mutants, names, prefixes, dms_ids = [], [], [], []
    mutant_positions = [] # Store 0-based mutation position for each sample

    # Store WT embeddings separately (Mapped by dms_idx)
    wt_embeddings_global = {}
    wt_embeddings_full = {}

    dms_map = {}
    for wt_name_search in data_json.keys():
        wt_name = "_".join(wt_name_search.split("_")[:-1])
        if wt_name not in dms_map:
            dms_map[wt_name] = len(dms_map)

    print(f"Toplam unique wild-type: {len(dms_map)}")
    missing_mutants = []
    
    for wt_name_search, content in tqdm(data_json.items()):

        wt_name = "_".join(wt_name_search.split("_")[:-1])
        if wt_name not in dms_map:
            dms_map[wt_name] = len(dms_map)
        dms_idx = dms_map[wt_name]

        # WT key bul ve Embeddingleri kaydet
        wt_key = find_wt_key(keys, wt_name)
        if wt_key is None:
            continue
        
        # Load WT embeddings
        wt_full_emb = f[wt_key][()]      # (L, 1280)
        wt_global_emb = pool(wt_full_emb) # (1280,)

        # Save to dictionary
        wt_embeddings_full[dms_idx] = wt_full_emb
        wt_embeddings_global[dms_idx] = wt_global_emb

        prefix_full = wt_key[:-3]

        for label_type in ["positives", "negatives"]:
            bin_label = 1 if label_type == "positives" else 0

            for entry in content[label_type]:

                mut = entry["mutation"]  # örn: A36S
                mut_key = find_mut_key(keys, prefix_full, mut)
                if mut_key is None:
                    missing_mutants.append(prefix_full+"_"+mut)
                    continue

                h_mut = pool(f[mut_key][()])

                varying_pos = int(entry["position"])
                varying_pos_0 = varying_pos - 1 # 0-based

                X.append(h_mut)
                X_local.append(f[mut_key][varying_pos_0])
                mutant_positions.append(varying_pos_0) # Save position

                y_bin.append(bin_label)
                y_score.append(float(bin_label))

                y_dms.append(dms_idx)

                mutants.append(mut)
                names.append(mut_key)
                prefixes.append(wt_name)
                dms_ids.append(wt_name)
    f.close()
    
    print(f"Total Mutants Saved: {len(X)}")
    print(f"Total WTs Saved: {len(wt_embeddings_global)}")

    np.savez(
        args.out,
        X=np.asarray(X, np.float32),
        X_local=np.asarray(X_local, np.float32),
        y_bin=np.asarray(y_bin, np.int64),
        y_score=np.asarray(y_score, np.float32),
        y_dms_id=np.asarray(y_dms, np.int64),
        dms_id_str=np.asarray(dms_ids, dtype=object),
        mutants=np.asarray(mutants, dtype=object),
        names=np.asarray(names, dtype=object),
        protein_prefix=np.asarray(prefixes, dtype=object),
        mutant_positions=np.asarray(mutant_positions, dtype=np.int64),
        # Save WT dicts
        wt_embeddings_global=np.asarray(wt_embeddings_global, dtype=object),
        wt_embeddings_full=np.asarray(wt_embeddings_full, dtype=object)
    )

    print("[DONE]", args.out)


# Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, data_path, n_neg_candidates=50):
        
        data = np.load(data_path, allow_pickle=True)

        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.X_local = torch.tensor(data["X_local"], dtype=torch.float32)
        self.y_bin = torch.tensor(data["y_bin"], dtype=torch.long)
        self.mutant_positions = torch.tensor(data["mutant_positions"], dtype=torch.long)
        
        # Load WT dictionary 
        self.wt_embeddings_global = data["wt_embeddings_global"].item()
        self.wt_embeddings_full = data["wt_embeddings_full"].item() 

        self.N = self.X.shape[0]

        # Handle legacy dataset without y_dms_id
        if "y_dms_id" in data:
            self.y_dms = torch.tensor(data["y_dms_id"], dtype=torch.long)
        elif "protein_prefix" in data:
            prefixes = data["protein_prefix"]
            if prefixes.shape == (1,) and self.N > 1:
                 inner = prefixes[0]
                 if hasattr(inner, "__len__") and len(inner) == self.N and not isinstance(inner, str):
                     prefixes = inner
                 else:
                     prefixes = np.repeat(prefixes, self.N)

            if len(prefixes) != self.N:
                 ids = np.zeros(self.N, dtype=np.int64)
            else:
                 unique_prefixes, ids = np.unique(prefixes, return_inverse=True)

            self.y_dms = torch.tensor(ids, dtype=torch.long)
        else:
            raise KeyError("Dataset must contain 'y_dms_id' or 'protein_prefix'")

        self.n_neg_candidates = n_neg_candidates
        self.embed_dim = self.X.shape[1]
        
        print("[INFO] Mutants X:", self.X.shape)
        print("[INFO] WTs Available:", len(self.wt_embeddings_global))

        self.protein_label_groups = {}

        for i in range(self.N):
            key = (int(self.y_dms[i]), int(self.y_bin[i]))
            self.protein_label_groups.setdefault(key, []).append(i)

    def __len__(self):
        return self.N

    def _sample_positive(self, idx):
        dms_i = int(self.y_dms[idx])
        pos_candidates = self.protein_label_groups.get((dms_i, 1), [])
        if len(pos_candidates) == 0: return idx 
        return np.random.choice(pos_candidates)

    def _sample_negative(self, idx):
        dms_i = int(self.y_dms[idx])
        neg_candidates = self.protein_label_groups.get((dms_i, 0), [])
        if len(neg_candidates) == 0: return idx 
        return np.random.choice(neg_candidates)

    def __getitem__(self, i):
        # Identify which Protein (DMS) this index belongs to
        dms_i = int(self.y_dms[i])
        
        # GLOBAL ANCHOR
        anchor_global = self.wt_embeddings_global[dms_i] # (D,)
        if isinstance(anchor_global, np.ndarray):
             anchor_global = torch.from_numpy(anchor_global).float()

        # FULL WT EMBEDDING (for local extraction)
        wt_full = self.wt_embeddings_full[dms_i] # (L, D)
        if isinstance(wt_full, np.ndarray):
             wt_full = torch.from_numpy(wt_full).float()

        # =============================================
        # LOCAL CONTRASTIVE PAIR (sample i'nin KENDİ pozisyonu)
        # =============================================
        # Her sample kendi pozisyonundaki WT residue ile çift oluşturur
        self_pos_idx = self.mutant_positions[i]
        self_wt_local = wt_full[self_pos_idx]      # WT[pos_i]  → (D,)
        self_mut_local = self.X_local[i]            # Mutant[pos_i] → (D,)
        self_label = self.y_bin[i]                  # 1=iyi, 0=kötü

        # =============================================
        # GLOBAL TRIPLET (positive/negative sampling)
        # =============================================
        # 1. Sample Positive
        pos_i = self._sample_positive(i)
        positive = self.X[pos_i]
        
        # 2. Sample Negatives (Deleterious Mutant)
        neg_indices = [self._sample_negative(i) for _ in range(self.n_neg_candidates)]
        negatives = torch.stack([self.X[n] for n in neg_indices])

        if self.n_neg_candidates == 1:
            negatives = negatives.squeeze(0)

        # Structure: 
        # tuple 0: (Anchor Global,)  → global triplet anchor
        # tuple 1: (Positive Global,) → global triplet positive
        # tuple 2: (Negatives Global,) → global triplet negatives
        # tuple 3: (self_wt_local, self_mut_local, self_label) → local contrastive pair
        return (anchor_global,), (positive,), (negatives,), (self_wt_local, self_mut_local, self_label)

    def get_embed_dim(self):
        return self.embed_dim
    
class ProteinGroupedBatchSampler:
    """
    Her batch'in sadece TEK bir proteinden (DMS) örnekler içermesini sağlar.
    Protein içi sıralama ve batch sırası her epoch'ta karıştırılır.
    """
    def __init__(self, y_dms, batch_size):
        self.batch_size = batch_size
        # İndeksleri protein bazlı grupla
        self.protein_groups = {}
        for i in range(len(y_dms)):
            dms_id = int(y_dms[i])
            self.protein_groups.setdefault(dms_id, []).append(i)
        
        print(f"[ProteinGroupedBatchSampler] {len(self.protein_groups)} protein, "
              f"batch_size={batch_size}")
        for dms_id, indices in self.protein_groups.items():
            n_batches = (len(indices) + batch_size - 1) // batch_size
            print(f"  Protein {dms_id}: {len(indices)} örnek → {n_batches} batch")

    def __iter__(self):
        all_batches = []
        for dms_id, indices in self.protein_groups.items():
            # Her protein içinde karıştır
            shuffled = list(indices)
            np.random.shuffle(shuffled)
            # Batch'lere böl
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                all_batches.append(batch)
        # Batch sırasını karıştır (ama her batch hâlâ tek protein)
        np.random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.protein_groups.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def get_triplet_dataloader(data_path, batch_size, n_neg_candidates=1, group_by_protein=False):
    dataset = TripletDataset(data_path, n_neg_candidates=n_neg_candidates)
    
    if group_by_protein:
        sampler = ProteinGroupedBatchSampler(dataset.y_dms, batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    return loader, dataset.get_embed_dim()

if __name__ == "__main__":
    main()