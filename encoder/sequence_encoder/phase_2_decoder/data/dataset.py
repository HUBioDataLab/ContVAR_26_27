"""
GOAnnotationDataset — Sekans tabanlı decoder için dataset.

Her sample:
    emb   [encoder_output_dim]  — ESM-2 mean-pooled protein embedding
    label [N_go_terms]          — multi-hot binary vektör

label[i] = 1  →  protein bu GO terimine sahip
label[i] = 0  →  protein bu GO terimine sahip değil

Split: UniRef50 cluster-based — aynı cluster'daki proteinler aynı split'e düşer.
Split atamaları phase0_go_split.json'dan okunur (train 80% / val 10% / test 10%).
Protein → cluster eşlemesi protein_uniref50.tsv'den yüklenir.

Tüm embedding'ler ve label'lar __init__'te RAM'e yüklenir;
__getitem__ saf tensor indeksleme yapar (disk I/O yok).
"""

import csv
import json

import h5py
import torch
from torch.utils.data import Dataset


def _load_uniref_split(uniref_tsv: str, split_json: str) -> dict[str, str]:
    """
    Protein ID → split ("train" | "val" | "test") eşlemesi döndürür.

    Args:
        uniref_tsv:  protein_id <TAB> group_id satırları içeren dosya
        split_json:  phase0_go_split.json (group_id → split atamaları)

    Returns:
        {protein_id: "train" | "val" | "test"}
    """
    with open(split_json) as f:
        phase0 = json.load(f)

    # Hem group_to_split hem protein_to_split formatını destekle
    if "protein_to_split" in phase0:
        return phase0["protein_to_split"]

    group_to_split: dict[str, str] = phase0["group_to_split"]

    protein_to_split: dict[str, str] = {}
    with open(uniref_tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid   = row["protein_id"]
            grp   = row["group_id"]
            split = group_to_split.get(grp)
            if split is not None:
                protein_to_split[pid] = split

    return protein_to_split


def _resolve_protein_id(h5_key: str) -> str:
    """
    H5 key'inden protein accession çıkar.

    Desteklenen örnekler:
      - 'P12345'                                      -> 'P12345'
      - 'A0A1I9GEU1_NEIME_..._A109V_1'                -> 'A0A1I9GEU1'
    """
    return h5_key.split("_")[0]


class GOAnnotationDataset(Dataset):
    """
    H5 dosyasından ESM-2 embedding'leri + GO anotasyonlarından
    multi-hot label vektörleri oluşturur.
    """

    def __init__(self, h5_path: str, annotations: dict[str, set],
                 go_vocab: dict[str, int], split: str = "train",
                 uniref_tsv: str = "protein_uniref50.tsv",
                 split_json: str = "phase0_go_split.json",
                 secondary_h5_path: str | None = None,
                 ancestor_map: dict[str, frozenset] | None = None):
        """
        Args:
            h5_path:     ESM-2 embedding H5 dosyası — {uniprot_id: array[D]}
            annotations: parse_goa_tsv() çıktısı
            go_vocab:    build_go_vocab() çıktısı
            split:       "train" | "val" | "test"
            uniref_tsv:  Protein → UniRef50 cluster eşleme dosyası
            split_json:  UniRef50 cluster → split atamaları
            secondary_h5_path: Verilirse (concat_phase0/concat_phase1), her protein için
                bu dosyadaki embedding h5_path'teki embedding'e concat edilir. Protein
                bazında eşleştirilir (secondary dosya protein-level, tek satır varsayılır).
            ancestor_map: {GO_id: frozenset(ancestor GO_ids)} — verilirse VE split
                "val" veya "test" ise, pozitif (1) etiketli GO terimlerinin
                ancestor'ları da 1 olarak işaretlenir. split="train" için HİÇBİR
                ZAMAN uygulanmaz (bilerek — model eğitim sırasında propagate
                edilmemiş, doğrudan anote edilmiş terimlerle eğitilir).
        """
        n_classes = len(go_vocab)

        protein_to_split = _load_uniref_split(uniref_tsv, split_json)

        secondary_by_protein = None
        if secondary_h5_path:
            secondary_by_protein = {}
            with h5py.File(secondary_h5_path, "r") as h5_sec:
                for h5_key in h5_sec.keys():
                    protein_id = h5_key if h5_key in annotations else _resolve_protein_id(h5_key)
                    emb = torch.from_numpy(h5_sec[h5_key][:]).float()
                    if emb.dim() == 2:
                        emb = emb.mean(dim=0)
                    secondary_by_protein[protein_id] = emb

        with h5py.File(h5_path, "r") as h5:
            samples = []
            for h5_key in h5.keys():
                protein_id = h5_key if h5_key in annotations else _resolve_protein_id(h5_key)
                if protein_id not in annotations:
                    continue
                if protein_to_split.get(protein_id) != split:
                    continue
                if secondary_by_protein is not None and protein_id not in secondary_by_protein:
                    continue
                samples.append((h5_key, protein_id))

            print(f"[{split}] Loading {len(samples)} samples...", flush=True)

            embs_list = []
            protein_ids = []
            for h5_key, protein_id in samples:
                emb = torch.from_numpy(h5[h5_key][:]).float()
                # Per-residue / per-token embedding ise mean-pool yap.
                if emb.dim() == 2:
                    emb = emb.mean(dim=0)
                if secondary_by_protein is not None:
                    emb = torch.cat([emb, secondary_by_protein[protein_id]], dim=-1)
                embs_list.append(emb)
                protein_ids.append(protein_id)

            embs = torch.stack(embs_list)  # [N, D]

        propagate_labels = ancestor_map is not None and split in ("val", "test")

        labels = torch.zeros(len(protein_ids), n_classes)
        for i, pid in enumerate(protein_ids):
            go_terms = annotations.get(pid, set())
            if propagate_labels:
                expanded = set(go_terms)
                for go in go_terms:
                    expanded.update(ancestor_map.get(go, frozenset()))
                go_terms = expanded
            for go in go_terms:
                if go in go_vocab:
                    labels[i, go_vocab[go]] = 1.0

        self.embs   = embs    # [N, D]  — float32
        self.labels = labels  # [N, C]  — float32 multi-hot

    def __len__(self) -> int:
        return len(self.embs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embs[idx], self.labels[idx]
