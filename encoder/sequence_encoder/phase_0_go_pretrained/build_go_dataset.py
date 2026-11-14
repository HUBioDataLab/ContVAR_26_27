# go_pretrain/build_go_dataset.py
"""
Kullanım:
    python go_pretrain/build_go_dataset.py \
        --sim_bp data/raw/semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv \
        --sim_cc data/raw/semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv \
        --sim_mf data/raw/semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv \
        --json data/raw/uniprot_dict.json \
        --out data/processed/go_pretrain_dataset.pkl
"""

import os
import json
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm

def safe_float(val_str):
    """Handles edge cases in numerical formatting like '1.000.000' -> 1.0"""
    val_str = val_str.strip()
    if val_str.count('.') > 1:
        parts = val_str.split('.')
        val_str = parts[0] + "." + "".join(parts[1:])
    return float(val_str)

def parse_similarity_file(filepath, aspect_label, valid_proteins):
    print(f"[INFO] Parsing {aspect_label} annotations from {filepath} ...")
    
    pairs = defaultdict(lambda: {'pos': [], 'pos_scores': [], 'neg': [], 'neg_scores': []})
    
    if not filepath or not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}. Skipping {aspect_label}.")
        return pairs

    with open(filepath, 'r') as f:
        header = f.readline()
        for line in tqdm(f, desc=f"Parsing {aspect_label}"):
            cols = line.strip().split('\t')
            if len(cols) < 3:
                continue
            
            anchor, candidate, sim_str = cols[0], cols[1], cols[2]
            
            if anchor not in valid_proteins or candidate not in valid_proteins:
                continue
                
            try:
                sim = safe_float(sim_str)
            except ValueError:
                continue
            
            if sim >= 0.8:
                pairs[anchor]['pos'].append(candidate)
                pairs[anchor]['pos_scores'].append(sim)
            elif sim <= 0.2:
                pairs[anchor]['neg'].append(candidate)
                pairs[anchor]['neg_scores'].append(sim)
                
    usable_pairs = {
        anchor: data for anchor, data in pairs.items()
        if len(data['pos']) > 0 and len(data['neg']) > 0
    }

    print(f"  [RESULT] {aspect_label}: {len(usable_pairs)} usable anchors with both +/- candidates.")
    return usable_pairs, pairs

def load_sequences_from_json(json_path):
    print(f"[INFO] Loading sequences from {json_path} ...")
    sequences = {}
    try:
        with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
            for item in data:
                uniprot_id = item.get("Entry")
                sequence = item.get("Sequence")
                if uniprot_id and sequence:
                    sequences[uniprot_id] = sequence
    except Exception as e:
        print(f"[ERROR] JSON read error: {e}")
    print(f"[INFO] Loaded {len(sequences)} sequences")
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Build GO pretraining dataset using Semantic Similarity")
    parser.add_argument("--sim_bp", required=True, help="Path to BP similarity TSV")
    parser.add_argument("--sim_cc", required=True, help="Path to CC similarity TSV")
    parser.add_argument("--sim_mf", required=True, help="Path to MF similarity TSV")
    parser.add_argument("--json", required=True, help="Path to protein sequences JSON file")
    parser.add_argument("--out", required=True, help="Output pickle file path")
    parser.add_argument("--max_seq_len", type=int, default=1022, help="Reference length for downstream truncation info")
    parser.add_argument("--split_json", default=None, help="Optional: path to split JSON to report dropped protein analysis")
    args = parser.parse_args()
    
    # 1. Load sequences
    sequences = load_sequences_from_json(args.json)
    valid_proteins = set(sequences.keys())
    
    if not valid_proteins:
        print("[ERROR] No valid sequences found. Exiting.")
        return

    # 2. Parse similarity files
    semantic_pairs = {}
    raw_pairs = {}
    semantic_pairs['P'], raw_pairs['P'] = parse_similarity_file(args.sim_bp, 'BP', valid_proteins)
    semantic_pairs['C'], raw_pairs['C'] = parse_similarity_file(args.sim_cc, 'CC', valid_proteins)
    semantic_pairs['F'], raw_pairs['F'] = parse_similarity_file(args.sim_mf, 'MF', valid_proteins)
    
    # 3. Identify all usable proteins to save memory
    usable_anchors = set()
    usable_proteins = set()
    for asp in ('P', 'C', 'F'):
        for anchor, cands in semantic_pairs[asp].items():
            usable_anchors.add(anchor)
            usable_proteins.add(anchor)
            usable_proteins.update(cands['pos'])
            usable_proteins.update(cands['neg'])

    usable_proteins = list(usable_proteins)
    protein_seqs = {pid: sequences[pid] for pid in usable_proteins}

    # 4. Print stats (ported from your original script)
    longer_than_limit = sum(1 for pid in usable_proteins if len(protein_seqs[pid]) > args.max_seq_len)
    print(f"\n[INFO] Total unique proteins in dataset: {len(usable_proteins)}")
    print(f"[INFO] Long sequences are kept and will be truncated downstream "
          f"(>{args.max_seq_len} aa: {longer_than_limit})")

    # 4b. Drop analysis: proteins in split file that didn't make it into dataset
    if args.split_json:
        import json as _json
        with open(args.split_json) as _f:
            _split = _json.load(_f)
        split_proteins = set(_split["protein_to_split"].keys())
        usable_set = set(usable_proteins)
        dropped = split_proteins - usable_set

        # Per-aspect pos/neg presence built from ALL raw pairs (including non-usable anchors)
        dropped_reasons = {'no_pairs_anywhere': 0,
                           'only_pos_no_neg': 0,
                           'only_neg_no_pos': 0,
                           'pos_neg_different_aspects': 0}

        a = False
        b = False
        c = False
        for p in dropped:
            p_has_pos = {asp for asp in ('P', 'C', 'F') if raw_pairs[asp][p]['pos']}
            p_has_neg = {asp for asp in ('P', 'C', 'F') if raw_pairs[asp][p]['neg']}
            if not p_has_pos and not p_has_neg:
                dropped_reasons['no_pairs_anywhere'] += 1
            elif p_has_pos and not p_has_neg:
                if not a:
                    print("printing p:", p, {a: raw_pairs[a][p] for a in p_has_pos}, "no_neg")
                    a=True
                dropped_reasons['only_pos_no_neg'] += 1
            elif p_has_neg and not p_has_pos:
                if not b:
                    print("printing p:", p, {a: raw_pairs[a][p] for a in p_has_neg}, "no_pos")
                    b=True
                dropped_reasons['only_neg_no_pos'] += 1
            else:
                # has both but never in the same aspect
                if not c:
                    print("printing p:", p, {a: raw_pairs[a][p] for a in ('P', 'C', 'F')}, "pos_neg_diff_asp")
                    c=True
                dropped_reasons['pos_neg_different_aspects'] += 1

        print(f"\n[DROP ANALYSIS] Split proteins    : {len(split_proteins)}")
        print(f"[DROP ANALYSIS] Dataset proteins  : {len(usable_set)}")
        print(f"[DROP ANALYSIS] Dropped           : {len(dropped)}")
        print(f"[DROP ANALYSIS] Breakdown:")
        for reason, count in dropped_reasons.items():
            print(f"  {reason:35s}: {count}")
    
    # 5. Save
    dataset = {
        'usable_proteins': usable_proteins,
        'protein_seqs': protein_seqs,
        'semantic_pairs': {
            asp: {
                anchor: {
                    'pos':        data['pos'],
                    'pos_scores': data['pos_scores'],
                    'neg':        data['neg'],
                    'neg_scores': data['neg_scores'],
                }
                for anchor, data in semantic_pairs[asp].items()
            }
            for asp in ('P', 'C', 'F')
        }
    }
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"[DONE] Saved to {args.out}")

if __name__ == "__main__":
    main()