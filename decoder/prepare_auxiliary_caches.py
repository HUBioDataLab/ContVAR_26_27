#!/usr/bin/env python3
"""Precompute and cache all Encoder-2 Auxiliary Features ONCE for MF-Only splits.

Computes and stores in `data/auxiliary/` (gitignored under `data/*`):
  1. 15 Sequence-Biophysical features  (Grantham, BLOSUM62, dCharge, dHydro, dVol, dPolarity, ...)
  2.  3 ESM-2 Evolutionary features    (mean_LLR, min_LLR, wt_site_entropy via ESM-2 fp16 on GPU)
  3.  5 AlphaFold 3D Structural feats  (pLDDT, win7_pLDDT, RSA_proxy_10A, CN_14A, has_af_struct)
  -> Total: 23 protein-invariant auxiliary features aligned to all 6 MF-only splits
     (hard_{train,val,test}, easy_{train,val,test}) saved in `mf_auxiliary_bundle.npz`.

Resumable: checkpoints ESM-2 and AlphaFold progress incrementally to JSON so
interrupted runs resume cleanly without recomputing completed proteins.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm

DECODER_DIR = Path(__file__).resolve().parent
DATA_DIR = DECODER_DIR.parent / "data"
AUX_DIR = DATA_DIR / "auxiliary"

SUB_RE = re.compile(r"([A-Z])(\d+)([A-Z])")

# ---------------------------------------------------------------------------
# Biophysical Tables (exact Grantham 1974, BLOSUM62, Kyte-Doolittle, Charge, Volume)
# ---------------------------------------------------------------------------
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGE = {
    "A": 0.0, "R": 1.0, "N": 0.0, "D": -1.0, "C": 0.0, "Q": 0.0, "E": -1.0,
    "G": 0.0, "H": 0.1, "I": 0.0, "L": 0.0, "K": 1.0, "M": 0.0, "F": 0.0,
    "P": 0.0, "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4,
    "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9,
    "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}
GRANTHAM_C = {
    "A": 0.00, "R": 0.65, "N": 1.33, "D": 1.38, "C": 2.75, "Q": 0.89, "E": 0.92,
    "G": 0.74, "H": 0.58, "I": 0.00, "L": 0.00, "K": 0.33, "M": 0.00, "F": 0.00,
    "P": 0.39, "S": 1.42, "T": 0.71, "W": 0.13, "Y": 0.20, "V": 0.00,
}
GRANTHAM_P = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5, "Q": 10.5, "E": 12.3,
    "G": 9.0, "H": 10.4, "I": 5.2, "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2,
    "P": 8.0, "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}
GRANTHAM_V = {
    "A": 31, "R": 124, "N": 56, "D": 54, "C": 55, "Q": 85, "E": 83,
    "G": 3, "H": 96, "I": 111, "L": 111, "K": 119, "M": 105, "F": 132,
    "P": 32.5, "S": 32, "T": 61, "W": 170, "Y": 136, "V": 84,
}
_BLOSUM62 = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]

AUX_FEATURE_NAMES = [
    # 15 Sequence-Biophysical (indices 0..14)
    "grantham_dist", "blosum62",
    "delta_charge", "abs_delta_charge",
    "delta_hydro", "abs_delta_hydro",
    "delta_volume", "abs_delta_volume",
    "delta_polarity",
    "is_pro_gly_change", "is_aromatic_change", "is_cys_change",
    "rel_position", "is_alanine", "log_n_mutations",
    # 3 ESM-2 Evolutionary (indices 15..17)
    "esm2_llr_mean", "esm2_llr_min", "esm2_wt_entropy",
    # 5 AlphaFold Structural (indices 18..22)
    "af_plddt", "af_plddt_win7", "af_rsa_proxy_10A", "af_cn_14A", "has_af_struct",
]


def exact_grantham(a: str, b: str) -> float:
    if a not in GRANTHAM_C or b not in GRANTHAM_C:
        return 80.0
    dc = GRANTHAM_C[a] - GRANTHAM_C[b]
    dp = GRANTHAM_P[a] - GRANTHAM_P[b]
    dv = GRANTHAM_V[a] - GRANTHAM_V[b]
    return float(50.723 * math.sqrt(1.833 * dc * dc + 0.1018 * dp * dp + 0.000399 * dv * dv))


def blosum62(a: str, b: str) -> float:
    if a in AA_TO_IDX and b in AA_TO_IDX:
        return float(_BLOSUM62[AA_TO_IDX[a]][AA_TO_IDX[b]])
    return 0.0


def parse_subs(mut_str: str, snp_str: str) -> list[tuple[str, int, str]]:
    parsed = SUB_RE.findall(mut_str) or SUB_RE.findall(snp_str)
    return [(w, int(p), m) for w, p, m in parsed]


def load_sequences() -> dict[str, str]:
    raw = json.loads((DATA_DIR / "sequences" / "uniprot_dict.json").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    return {item["Entry"]: item["Sequence"] for item in raw if "Entry" in item and "Sequence" in item}


def load_all_mf_splits() -> dict[str, pd.DataFrame]:
    mf_ids = set(json.loads((DATA_DIR / "mf_go_ids.json").read_text(encoding="utf-8")))
    splits: dict[str, pd.DataFrame] = {}
    for fam in ("hard", "easy"):
        for sp in ("train", "val", "test"):
            df = pd.read_csv(DATA_DIR / "prepared" / f"{fam}_{sp}.tsv", sep="\t", low_memory=False)
            if sp != "train" and "augmented" in df.columns:
                df = df[~df["augmented"].astype(str).str.strip().str.lower().eq("true")]
            df = df[df["matched_go_id"].isin(mf_ids)].reset_index(drop=True)
            splits[f"{fam}_{sp}"] = df
    return splits


# ---------------------------------------------------------------------------
# STEP 1: 15 Sequence-Biophysical Features
# ---------------------------------------------------------------------------
def compute_biophysical_15(df: pd.DataFrame, seqs: dict[str, str]) -> np.ndarray:
    n = len(df)
    out = np.zeros((n, 15), dtype=np.float32)
    mut_col = "mutations" if "mutations" in df.columns else "snp"
    pids = df["protein_id"].astype(str).tolist()
    muts = df[mut_col].astype(str).tolist()
    snps = df["snp"].astype(str).tolist()
    is_ala = df["is_alanine"].astype(str).str.lower().eq("true").to_numpy(dtype=np.float32) if "is_alanine" in df.columns else None

    for i in range(n):
        subs = parse_subs(muts[i], snps[i])
        if not subs:
            continue
        seq_len = float(max(1, len(seqs.get(pids[i], "")) or 500))
        g_l, b_l, dc_l, dh_l, dv_l, dp_l, rp_l = [], [], [], [], [], [], []
        pg, aro, cys = 0.0, 0.0, 0.0
        for wt, pos, mut in subs:
            g_l.append(exact_grantham(wt, mut) / 215.0)
            b_l.append(blosum62(wt, mut) / 10.0)
            dc_l.append(CHARGE.get(mut, 0.0) - CHARGE.get(wt, 0.0))
            dh_l.append((HYDRO.get(mut, 0.0) - HYDRO.get(wt, 0.0)) / 9.0)
            dv_l.append((VOLUME.get(mut, 130.0) - VOLUME.get(wt, 130.0)) / 170.0)
            dp_l.append((GRANTHAM_P.get(mut, 8.0) - GRANTHAM_P.get(wt, 8.0)) / 8.0)
            if wt in ("P", "G") or mut in ("P", "G"):
                pg = 1.0
            if (wt in ("F", "W", "Y")) != (mut in ("F", "W", "Y")):
                aro = 1.0
            if wt == "C" or mut == "C":
                cys = 1.0
            rp_l.append(min(1.0, float(pos) / seq_len))

        out[i, 0] = float(np.mean(g_l))
        out[i, 1] = float(np.mean(b_l))
        out[i, 2] = float(np.mean(dc_l))
        out[i, 3] = float(np.mean(np.abs(dc_l)))
        out[i, 4] = float(np.mean(dh_l))
        out[i, 5] = float(np.mean(np.abs(dh_l)))
        out[i, 6] = float(np.mean(dv_l))
        out[i, 7] = float(np.mean(np.abs(dv_l)))
        out[i, 8] = float(np.mean(dp_l))
        out[i, 9] = pg
        out[i, 10] = aro
        out[i, 11] = cys
        out[i, 12] = float(np.mean(rp_l))
        out[i, 13] = float(is_ala[i]) if is_ala is not None else (1.0 if all(m == "A" for _, _, m in subs) else 0.0)
        out[i, 14] = float(math.log1p(len(subs)))
    return out


# ---------------------------------------------------------------------------
# STEP 2: ESM-2 Wild-Type Marginal LLR + Site Entropy (GPU fp16, Resumable)
# ---------------------------------------------------------------------------
def build_esm2_llr_cache(
    protein_to_subs: dict[str, set[tuple[str, int, str]]],
    seqs: dict[str, str],
    model_name: str,
    cache_path: Path,
) -> dict[str, list[float]]:
    """Cache mapping 'PID|WT_POS_MUT' -> [llr, wt_entropy]."""
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  [ESM-2] Loaded existing cache with {len(cache)} substitution entries.", flush=True)

    # Identify proteins that still have uncached substitutions
    todo_proteins: dict[str, list[tuple[str, int, str]]] = {}
    for pid, subs in protein_to_subs.items():
        missing = [s for s in subs if f"{pid}|{s[0]}{s[1]}{s[2]}" not in cache]
        if missing and pid in seqs:
            todo_proteins[pid] = missing

    if not todo_proteins:
        print("  [ESM-2] All substitutions already cached!", flush=True)
        return cache

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [ESM-2] Loading {model_name} on {device} (fp16) for {len(todo_proteins)} proteins...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model = model.to(device).eval()

    aa_token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in AA_ORDER}
    max_len = 1022

    def save_ckpt():
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)

    processed = 0
    with torch.no_grad():
        for pid, subs in tqdm(todo_proteins.items(), desc="ESM-2 LLR", unit="prot"):
            seq = seqs[pid]
            seq_len = len(seq)

            # Group substitutions into crops if protein > 1022 aa
            if seq_len <= max_len:
                crops = [(0, seq_len, subs)]
            else:
                # Center crop around each unique position bucket (stride 512)
                buckets: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
                for s in subs:
                    center = min(max(s[1] - 1, 0), seq_len - 1)
                    start = max(0, min(center - max_len // 2, seq_len - max_len))
                    buckets[start].append(s)
                crops = [(st, st + max_len, sub_list) for st, sub_list in buckets.items()]

            for start, end, crop_subs in crops:
                crop_seq = seq[start:end]
                inputs = tokenizer(crop_seq, return_tensors="pt", add_special_tokens=True).to(device)
                logits = model(**inputs).logits[0]  # [L+2, vocab]
                log_probs = torch.log_softmax(logits.float(), dim=-1)  # [L+2, vocab]

                for wt, pos, mut in crop_subs:
                    key = f"{pid}|{wt}{pos}{mut}"
                    rel_idx = (pos - 1) - start + 1  # +1 for <cls> token
                    if rel_idx < 1 or rel_idx >= log_probs.shape[0] - 1 or wt not in aa_token_ids or mut not in aa_token_ids:
                        cache[key] = [0.0, 0.5]
                        continue
                    lp_wt = float(log_probs[rel_idx, aa_token_ids[wt]].item())
                    lp_mut = float(log_probs[rel_idx, aa_token_ids[mut]].item())
                    llr = lp_mut - lp_wt
                    # Normalized entropy over 20 canonical AAs at this position
                    aa_lps = log_probs[rel_idx, list(aa_token_ids.values())]
                    aa_ps = torch.softmax(aa_lps, dim=-1)
                    ent = float(-(aa_ps * torch.log(aa_ps.clamp_min(1e-12))).sum().item() / math.log(20.0))
                    cache[key] = [round(llr, 4), round(ent, 4)]

            processed += 1
            if processed % 400 == 0:
                save_ckpt()

    save_ckpt()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  [ESM-2] Done! Cached {len(cache)} unique substitution LLR entries.", flush=True)
    return cache


# ---------------------------------------------------------------------------
# STEP 3: AlphaFold 3D Structural Features (pLDDT + RSA proxy, Multi-Threaded, Resumable)
# ---------------------------------------------------------------------------
def parse_alphafold_pdb_for_positions(pdb_text: str, positions: list[int]) -> dict[int, list[float]]:
    """Extract per-position [plddt/100, win7_plddt/100, rsa_proxy_10A, cn_14A/45] from AlphaFold PDB."""
    res_coords: dict[int, tuple[float, float, float]] = {}
    res_plddt: dict[int, float] = {}

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  "):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            res_seq = int(line[22:26].strip())
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            bfac = float(line[60:66])  # AlphaFold stores pLDDT (0-100) in B-factor column
            res_coords[res_seq] = (x, y, z)
            res_plddt[res_seq] = bfac
        except ValueError:
            continue

    if not res_coords:
        return {}

    sorted_res = sorted(res_coords.keys())
    coords_arr = np.array([res_coords[r] for r in sorted_res], dtype=np.float32)
    tree = cKDTree(coords_arr)

    out: dict[int, list[float]] = {}
    for pos in positions:
        if pos not in res_coords:
            continue
        pt = np.array(res_coords[pos], dtype=np.float32)
        cn10 = max(0, len(tree.query_ball_point(pt, r=10.0)) - 1)
        cn14 = max(0, len(tree.query_ball_point(pt, r=14.0)) - 1)
        rsa_proxy = max(0.0, 1.0 - min(cn10, 25) / 25.0)  # 0=deeply buried core, 1=exposed surface
        plddt = res_plddt[pos] / 100.0
        win_vals = [res_plddt[p] for p in range(pos - 3, pos + 4) if p in res_plddt]
        win_plddt = (sum(win_vals) / len(win_vals)) / 100.0 if win_vals else plddt
        out[pos] = [round(plddt, 4), round(win_plddt, 4), round(rsa_proxy, 4), round(min(cn14, 50) / 50.0, 4)]
    return out


def fetch_one_alphafold(pid: str, positions: list[int], session: requests.Session) -> tuple[str, dict[int, list[float]] | None]:
    """Fetch AlphaFold PDB (v4 direct URL, fallback to API v6/latest) and parse in memory."""
    urls = [f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_v4.pdb"]
    try:
        r = session.get(urls[0], timeout=12)
        if r.status_code == 200 and r.text.startswith(("HEADER", "TITLE", "ATOM", "MODEL")):
            return pid, parse_alphafold_pdb_for_positions(r.text, positions)
        # Fallback to AlphaFold REST API to get latest model version URL (e.g. v6)
        api_r = session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pid}", timeout=12)
        if api_r.status_code == 200:
            entries = api_r.json()
            if isinstance(entries, list) and entries and entries[0].get("pdbUrl"):
                r2 = session.get(entries[0]["pdbUrl"], timeout=15)
                if r2.status_code == 200:
                    return pid, parse_alphafold_pdb_for_positions(r2.text, positions)
    except Exception:
        pass
    return pid, None


def build_alphafold_cache(
    protein_to_positions: dict[str, set[int]],
    cache_path: Path,
    workers: int = 16,
) -> dict[str, Any]:
    """Cache mapping pid -> { 'found': bool, 'pos': { '123': [plddt, win7, rsa10, cn14] } }."""
    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"  [AlphaFold] Loaded existing cache with {len(cache)} proteins.", flush=True)

    todo = {pid: sorted(pos_set) for pid, pos_set in protein_to_positions.items() if pid not in cache}
    if not todo:
        print("  [AlphaFold] All proteins already cached!", flush=True)
        return cache

    print(f"  [AlphaFold] Fetching structures for {len(todo)} proteins ({workers} threads)...", flush=True)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2)
    session.mount("https://", adapter)

    def save_ckpt():
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)

    done = 0
    found_cnt = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one_alphafold, pid, pos_list, session): pid for pid, pos_list in todo.items()}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="AlphaFold PDB", unit="prot"):
            pid, parsed = fut.result()
            if parsed:
                cache[pid] = {"found": True, "pos": {str(k): v for k, v in parsed.items()}}
                found_cnt += 1
            else:
                cache[pid] = {"found": False, "pos": {}}
            done += 1
            if done % 250 == 0:
                save_ckpt()

    save_ckpt()
    total_found = sum(1 for v in cache.values() if v.get("found"))
    print(f"  [AlphaFold] Done! Coverage: {total_found}/{len(cache)} proteins ({total_found/max(1,len(cache))*100:.1f}%).", flush=True)
    return cache


# ---------------------------------------------------------------------------
# STEP 4: Assemble Unified Aligned 23-Feature Bundle for All 6 MF-Only Splits
# ---------------------------------------------------------------------------
def assemble_bundle(
    splits: dict[str, pd.DataFrame],
    seqs: dict[str, str],
    esm_cache: dict[str, list[float]],
    af_cache: dict[str, Any],
    bundle_path: Path,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for split_name, df in splits.items():
        n = len(df)
        bio = compute_biophysical_15(df, seqs)  # [n, 15]
        esm = np.zeros((n, 3), dtype=np.float32)
        af = np.zeros((n, 5), dtype=np.float32)

        mut_col = "mutations" if "mutations" in df.columns else "snp"
        pids = df["protein_id"].astype(str).tolist()
        muts = df[mut_col].astype(str).tolist()
        snps = df["snp"].astype(str).tolist()

        # Reasonable neutral defaults for missing AlphaFold sites (median pLDDT=0.80, RSA=0.45, CN14=0.40)
        af[:, 0] = 0.80
        af[:, 1] = 0.80
        af[:, 2] = 0.45
        af[:, 3] = 0.40
        af[:, 4] = 0.0  # has_af_struct = 0

        for i in range(n):
            pid = pids[i]
            subs = parse_subs(muts[i], snps[i])
            if not subs:
                continue

            # ESM-2 LLR & conservation
            llrs, ents = [], []
            for wt, pos, mut in subs:
                val = esm_cache.get(f"{pid}|{wt}{pos}{mut}")
                if val is not None:
                    llrs.append(val[0])
                    ents.append(val[1])
            if llrs:
                esm[i, 0] = float(np.mean(llrs)) / 10.0  # scaled roughly [-1.5, +0.5]
                esm[i, 1] = float(np.min(llrs)) / 10.0   # worst-site LLR for combinatorial
                esm[i, 2] = float(np.mean(ents))

            # AlphaFold 3D structural features
            af_entry = af_cache.get(pid)
            if af_entry and af_entry.get("found"):
                pos_dict = af_entry["pos"]
                vals = [pos_dict[str(pos)] for _, pos, _ in subs if str(pos) in pos_dict]
                if vals:
                    arr_v = np.mean(vals, axis=0)
                    af[i, 0] = float(arr_v[0])
                    af[i, 1] = float(arr_v[1])
                    af[i, 2] = float(arr_v[2])
                    af[i, 3] = float(arr_v[3])
                    af[i, 4] = 1.0  # has_af_struct = 1

        full_23 = np.concatenate([bio, esm, af], axis=1).astype(np.float32)
        arrays[split_name] = full_23
        af_cov = af[:, 4].mean() * 100
        esm_cov = (esm[:, 0] != 0.0).mean() * 100
        print(f"  [Bundle] {split_name:10s}: shape={full_23.shape} | ESM2_cov={esm_cov:.1f}% | AlphaFold_cov={af_cov:.1f}%", flush=True)

    np.savez_compressed(bundle_path, feature_names=np.array(AUX_FEATURE_NAMES), **arrays)
    print(f"\n=== AUXILIARY BUNDLE SAVED TO: {bundle_path} ===", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D",
                   help="HuggingFace ESM-2 model (default: esm2_t33_650M_UR50D; use esm2_t30_150M_UR50D for 2x speed)")
    p.add_argument("--workers", type=int, default=16, help="Concurrent threads for AlphaFold PDB downloads")
    p.add_argument("--skip-esm", action="store_true", help="Skip ESM-2 LLR computation (use existing cache if any)")
    p.add_argument("--skip-alphafold", action="store_true", help="Skip AlphaFold downloads (use existing cache if any)")
    args = p.parse_args()

    t0 = time.perf_counter()
    AUX_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78, flush=True)
    print("ContVAR ENCODER-2 AUXILIARY FEATURE CACHE BUILDER (MF-ONLY)", flush=True)
    print(f"Output Dir : {AUX_DIR}", flush=True)
    print(f"ESM-2 Model: {args.esm_model}", flush=True)
    print("=" * 78, flush=True)

    seqs = load_sequences()
    splits = load_all_mf_splits()

    # Collect unique (protein_id -> substitutions & positions) across all 6 MF-only splits
    protein_to_subs: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    protein_to_positions: dict[str, set[int]] = defaultdict(set)
    for df in splits.values():
        mut_col = "mutations" if "mutations" in df.columns else "snp"
        for pid, m_s, s_s in zip(df["protein_id"].astype(str), df[mut_col].astype(str), df["snp"].astype(str)):
            for wt, pos, mut in parse_subs(m_s, s_s):
                protein_to_subs[pid].add((wt, pos, mut))
                protein_to_positions[pid].add(pos)

    print(f"Total unique MF-only proteins: {len(protein_to_subs)} | unique substitutions: {sum(len(v) for v in protein_to_subs.values())}", flush=True)

    esm_cache_path = AUX_DIR / "esm2_llr_cache.json"
    af_cache_path = AUX_DIR / "alphafold_struct_cache.json"
    bundle_path = AUX_DIR / "mf_auxiliary_bundle.npz"

    if args.skip_esm:
        esm_cache = json.loads(esm_cache_path.read_text(encoding="utf-8")) if esm_cache_path.exists() else {}
    else:
        esm_cache = build_esm2_llr_cache(protein_to_subs, seqs, args.esm_model, esm_cache_path)

    if args.skip_alphafold:
        af_cache = json.loads(af_cache_path.read_text(encoding="utf-8")) if af_cache_path.exists() else {}
    else:
        af_cache = build_alphafold_cache(protein_to_positions, af_cache_path, workers=args.workers)

    assemble_bundle(splits, seqs, esm_cache, af_cache, bundle_path)
    print(f"Total time: {(time.perf_counter() - t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
