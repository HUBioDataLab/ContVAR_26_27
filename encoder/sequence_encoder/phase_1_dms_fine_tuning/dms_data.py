from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

from torch.utils.data import Dataset


_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$",
    re.IGNORECASE,
)
_VAL_MUT_RE = re.compile(r"_([a-z]\d+[a-z])_[01]_model\.cif$", re.IGNORECASE)
_MUT_RE = re.compile(r"^[A-Z]\d+[A-Z]$", re.IGNORECASE)


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_mutation_from_cif(path: str) -> str | None:
    m = _VAL_MUT_RE.search(path)
    if m:
        return m.group(1).upper()
    return None


def apply_mutation(sequence: str, mutation: str):
    src = mutation[0]
    dst = mutation[-1]
    pos0 = int(mutation[1:-1]) - 1
    if pos0 < 0 or pos0 >= len(sequence):
        raise IndexError(f"Mutation position out of bounds: {mutation} for len={len(sequence)}")
    seq_list = list(sequence)
    if seq_list[pos0] != src:
        # Bazı kayıtlar offset / annotation farkı içerebiliyor.
        # Hard fail yerine uyarı verip yine de mutasyonu uygula.
        print(
            f"[WARN] WT residue mismatch for {mutation}: expected {src}, found {seq_list[pos0]}. "
            "Mutation will still be applied."
        )
    seq_list[pos0] = dst
    return "".join(seq_list), pos0


def mutation_to_position0(mutation: str) -> int:
    pos0 = int(mutation[1:-1]) - 1
    if pos0 < 0:
        raise IndexError(f"Mutation position out of bounds: {mutation}")
    return pos0


def infer_numbering_offset(wt_sequence: str, mut_sequences: dict):
    """
    Some ProteinGym constructs use full-length mutation numbering on a shorter
    assay sequence. Infer a constant per-family offset by comparing each mutant
    sequence against the WT sequence and locating the actual changed residue.
    """
    offsets = []
    for mutation, mut_seq in mut_sequences.items():
        if len(mut_seq) != len(wt_sequence):
            continue
        diff_positions = [i for i, (a, b) in enumerate(zip(wt_sequence, mut_seq)) if a != b]
        if len(diff_positions) != 1:
            continue
        diff_pos0 = diff_positions[0]
        stated_pos1 = int(mutation[1:-1])
        offsets.append(stated_pos1 - (diff_pos0 + 1))

    if not offsets:
        return 0

    counts = {}
    for off in offsets:
        counts[off] = counts.get(off, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def build_reference_index(reference_csv: str):
    rows = list(csv.DictReader(open(reference_csv, newline="", encoding="utf-8")))
    index = {}
    for row in rows:
        index[normalize_name(row["DMS_id"])] = row
        index[normalize_name(Path(row["DMS_filename"]).stem)] = row
    return index


def load_proteins_index(proteins_csv: str):
    """
    proteins 2.csv format:
      - mutant rows: FAMILY_ACCESSION_A25H_0 / _1
      - WT rows:     FAMILY_ACCESSION_WT
    We normalize them back to triplet family keys by dropping accession + suffix.
    """
    rows = list(csv.DictReader(open(proteins_csv, newline="", encoding="utf-8")))
    index = {}

    for row in rows:
        file_name = row["File_Name"].strip()
        seq = row["Sequence"].strip()
        if not file_name or not seq:
            continue

        parts = file_name.split("_")
        if len(parts) >= 4 and parts[-1] in {"0", "1"} and _MUT_RE.match(parts[-2]):
            family_key = "_".join(parts[:-3]).lower()
            mutation = parts[-2].upper()
            fam = index.setdefault(family_key, {"wt_sequence": None, "mut_sequences": {}})
            fam["mut_sequences"][mutation] = seq
            continue

        if len(parts) >= 2 and parts[-1].upper() == "WT":
            if len(parts) >= 3 and _ACCESSION_RE.match(parts[-2]):
                family_key = "_".join(parts[:-2]).lower()
            else:
                family_key = "_".join(parts[:-1]).lower()
            fam = index.setdefault(family_key, {"wt_sequence": None, "mut_sequences": {}})
            fam["wt_sequence"] = seq

    for fam in index.values():
        wt_seq = fam.get("wt_sequence")
        mut_seqs = fam.get("mut_sequences", {})
        fam["numbering_offset"] = infer_numbering_offset(wt_seq, mut_seqs) if wt_seq else 0

    return index


def resolve_family_row(family_key: str, index):
    candidates = [normalize_name(family_key)]
    parts = family_key.split("_")
    if parts and _ACCESSION_RE.match(parts[-1]):
        candidates.append(normalize_name("_".join(parts[:-1])))

    for cand in candidates:
        row = index.get(cand)
        if row is not None:
            return row

    prefix_hits = []
    for cand in candidates:
        matches = [
            row for key, row in index.items()
            if key.startswith(cand) or cand.startswith(key)
        ]
        uniq = []
        seen = set()
        for row in matches:
            rid = row["DMS_id"]
            if rid not in seen:
                uniq.append(row)
                seen.add(rid)
        if len(uniq) == 1:
            return uniq[0]
        prefix_hits.extend(uniq)

    uniq_hits = []
    seen = set()
    for row in prefix_hits:
        rid = row["DMS_id"]
        if rid not in seen:
            uniq_hits.append(row)
            seen.add(rid)
    if len(uniq_hits) == 1:
        return uniq_hits[0]
    return None


def load_split_json(split_json: str | None):
    if split_json is None:
        return {}
    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _resolve_protein_split(family_key: str, protein_to_split: dict) -> str | None:
    """
    family_key içindeki token'lardan UniProt ID benzeri bir şey bulup
    protein_to_split'te arar. Bulamazsa None döner (→ train).
    """
    tokens = family_key.split("_")
    # Son token'dan başa doğru tara — çoğunlukla UniProt ID sonda
    for tok in reversed(tokens):
        candidate = tok.upper()
        if candidate in protein_to_split:
            return protein_to_split[candidate]
    return None


def split_family_entries(family_key: str, content: dict, split_data: dict):
    """
    Üç split formatını destekler:
      1. Yeni family-level split:
         {"family_to_split": {family_wt_model: "train"|"val"|"test", ...}}
      2. Protein-level split (phase0_protein_split_removed_graphless.json formatı):
         {"protein_to_split": {"P12345": "train"|"val"|"test", ...}}
         family_key içindeki token'lar üzerinden UniProt ID aranır.
      3. Legacy mutation-level split:
         {"families": {family_wt_model: {"val_positives": [...], "val_negatives": [...]}}}
    """
    split_key = f"{normalize_name(family_key)}_wt_model"

    # Format 1: family-level split
    if "family_to_split" in split_data:
        family_to_split = split_data["family_to_split"]
        split_name = family_to_split.get(split_key)

        if split_name == "train" or split_name is None:
            return content["positives"], content["negatives"], [], []
        if split_name == "val":
            return [], [], content["positives"], content["negatives"]
        if split_name == "test":
            return [], [], [], []

        raise ValueError(f"Unknown family split '{split_name}' for {split_key}")

    # Format 2: protein-level split (phase0 formatı)
    if "protein_to_split" in split_data:
        split_name = _resolve_protein_split(family_key, split_data["protein_to_split"])
        if split_name == "val":
            return [], [], content["positives"], content["negatives"]
        if split_name == "test":
            return [], [], [], []
        # train veya eşleşme yok → train
        return content["positives"], content["negatives"], [], []

    # Format 3: legacy mutation-level split
    split_families = split_data.get("families", {})
    family_split = split_families.get(split_key, {})

    val_pos = {
        parse_mutation_from_cif(cif)
        for cif in family_split.get("val_positives", [])
        if parse_mutation_from_cif(cif) is not None
    }
    val_neg = {
        parse_mutation_from_cif(cif)
        for cif in family_split.get("val_negatives", [])
        if parse_mutation_from_cif(cif) is not None
    }

    train_pos, eval_pos = [], []
    train_neg, eval_neg = [], []

    for entry in content["positives"]:
        (eval_pos if entry["mutation"].upper() in val_pos else train_pos).append(entry)
    for entry in content["negatives"]:
        (eval_neg if entry["mutation"].upper() in val_neg else train_neg).append(entry)

    return train_pos, train_neg, eval_pos, eval_neg


def _prepare_entry(entry: dict, wt_sequence: str, mut_sequences: dict, numbering_offset: int = 0):
    prepared = dict(entry)
    mutation = entry["mutation"].upper()
    pos0 = mutation_to_position0(mutation) - numbering_offset

    mut_sequence = mut_sequences.get(mutation)
    if mut_sequence is None:
        mut_sequence, pos0 = apply_mutation(wt_sequence, mutation)
    else:
        if pos0 >= len(mut_sequence):
            raise IndexError(f"Mutation position out of bounds: {mutation} for len={len(mut_sequence)}")

    prepared["mutation"] = mutation
    prepared["position0"] = pos0
    prepared["sequence"] = mut_sequence
    return prepared


def build_phase1_families(
    triplet_json: str,
    reference_csv: str | None = None,
    split_json: str | None = None,
    limit_families: int | None = None,
    proteins_csv: str | None = None,
):
    with open(triplet_json, "r", encoding="utf-8") as f:
        triplet_data = json.load(f)

    reference_index = build_reference_index(reference_csv) if reference_csv else {}
    proteins_index = load_proteins_index(proteins_csv) if proteins_csv else {}
    split_data = load_split_json(split_json)

    family_keys = list(triplet_data.keys())
    if limit_families is not None:
        family_keys = family_keys[:limit_families]

    families = {}
    skipped = []
    for family_key in family_keys:
        proteins_row = proteins_index.get(family_key.lower())
        if proteins_row is not None and proteins_row.get("wt_sequence"):
            wt_sequence = proteins_row["wt_sequence"]
            mut_sequences = proteins_row.get("mut_sequences", {})
            numbering_offset = int(proteins_row.get("numbering_offset", 0))
            dms_id = family_key
        else:
            row = resolve_family_row(family_key, reference_index)
            if row is None:
                skipped.append(family_key)
                continue
            wt_sequence = row["target_seq"]
            if not wt_sequence:
                skipped.append(family_key)
                continue
            mut_sequences = {}
            numbering_offset = 0
            dms_id = row["DMS_id"]

        train_pos, train_neg, val_pos, val_neg = split_family_entries(
            family_key, triplet_data[family_key], split_data
        )
        try:
            train_pos = [_prepare_entry(entry, wt_sequence, mut_sequences, numbering_offset) for entry in train_pos]
            train_neg = [_prepare_entry(entry, wt_sequence, mut_sequences, numbering_offset) for entry in train_neg]
            val_pos = [_prepare_entry(entry, wt_sequence, mut_sequences, numbering_offset) for entry in val_pos]
            val_neg = [_prepare_entry(entry, wt_sequence, mut_sequences, numbering_offset) for entry in val_neg]
        except Exception as exc:
            print(f"[WARN] Skipping family {family_key}: {exc}")
            skipped.append(family_key)
            continue

        families[family_key] = {
            "wt_sequence": wt_sequence,
            "train_positives": train_pos,
            "train_negatives": train_neg,
            "val_positives": val_pos,
            "val_negatives": val_neg,
            "dms_id": dms_id,
        }

    return families, skipped


class DMSTripletSequenceDataset(Dataset):
    """
    Her sample:
      - WT anchor sequence
      - positive mutant sequence
      - negative mutant sequence
      - positive mutation position
      - negative mutation position
    """

    def __init__(self, families: dict, seed: int = 42):
        self.families = families
        self.rng = random.Random(seed)
        self.records = []
        for family_key, fam in families.items():
            if not fam["train_positives"] or not fam["train_negatives"]:
                continue
            for pos_entry in fam["train_positives"]:
                self.records.append((family_key, pos_entry))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        family_key, pos_entry = self.records[idx]
        fam = self.families[family_key]
        neg_entry = self.rng.choice(fam["train_negatives"])

        wt_seq = fam["wt_sequence"]
        pos_seq = pos_entry.get("sequence")
        pos_pos0 = pos_entry.get("position0")
        if pos_seq is None or pos_pos0 is None:
            pos_seq, pos_pos0 = apply_mutation(wt_seq, pos_entry["mutation"])
        neg_seq = neg_entry.get("sequence")
        neg_pos0 = neg_entry.get("position0")
        if neg_seq is None or neg_pos0 is None:
            neg_seq, neg_pos0 = apply_mutation(wt_seq, neg_entry["mutation"])

        return {
            "family_key": family_key,
            "wt_sequence": wt_seq,
            "pos_sequence": pos_seq,
            "neg_sequence": neg_seq,
            "pos_position": pos_pos0,
            "neg_position": neg_pos0,
            "pos_mutation": pos_entry["mutation"],
            "neg_mutation": neg_entry["mutation"],
        }


def build_validation_records(families: dict):
    records = []
    for family_key, fam in families.items():
        wt_seq = fam["wt_sequence"]
        for entry in fam["val_positives"]:
            mut_seq = entry.get("sequence")
            pos0 = entry.get("position0")
            if mut_seq is None or pos0 is None:
                mut_seq, pos0 = apply_mutation(wt_seq, entry["mutation"])
            records.append({
                "family_key": family_key,
                "wt_sequence": wt_seq,
                "mut_sequence": mut_seq,
                "position": pos0,
                "label": 1,
                "mutation": entry["mutation"],
            })
        for entry in fam["val_negatives"]:
            mut_seq = entry.get("sequence")
            pos0 = entry.get("position0")
            if mut_seq is None or pos0 is None:
                mut_seq, pos0 = apply_mutation(wt_seq, entry["mutation"])
            records.append({
                "family_key": family_key,
                "wt_sequence": wt_seq,
                "mut_sequence": mut_seq,
                "position": pos0,
                "label": 0,
                "mutation": entry["mutation"],
            })
    return records
