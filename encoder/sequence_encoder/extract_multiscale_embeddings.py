#!/usr/bin/env python3
"""
Extract multi-scale sequence embeddings (Global + Mutation-Site + Local Context Window)
from the ContVAR ESM-2 sequence encoder backbone.

Why default k=16 (window size 2k+1 = 33 residues):
  1. Matches the ContVAR graph encoder's k_idx=16 sequence-neighbor budget.
  2. In structural biology & VEP literature (PSIPRED, NetSurfP, MutPred2, DeepFRI),
     k in [10, 16] (21 to 33 residues) spans a complete secondary-structure element
     (alpha-helix / beta-hairpin + flanking loops) and local linear motifs.
  3. Because the window is stored UNPOOLED as [2k+1, 1280] centered at index k,
     any smaller window k' < 16 (e.g., k'=5, 8, 10) can be sliced at training time
     via window[k - k' : k + k' + 1] without re-running the encoder on the server.

Outputs (saved to --out_prefix):
  1. {out_prefix}_global.h5
       - "{protein_id}"       -> [1280] float32 (WT global mean-pooled)
       - "{protein_id}_{snp}" -> [1280] float32 (Variant global mean-pooled)
  2. {out_prefix}_local.h5
       - "{protein_id}_{pos}" -> [1280] float32 (WT mutation-site residue)
       - "{protein_id}_{snp}" -> [1280] float32 (Variant mutation-site residue)
  3. {out_prefix}_window.h5
       - "{protein_id}_{pos}" -> [2k+1, 1280] float16/float32 (WT unpooled local window)
       - "{protein_id}_{snp}" -> [2k+1, 1280] float16/float32 (Variant unpooled local window)
       - "mask/{key}"         -> [2k+1] bool (True for valid residues, False for N/C-terminal padding)
       - H5 root attrs: window_radius_k, window_len, center_index, embed_dim

Usage on server:
  python encoder/sequence_encoder/extract_multiscale_embeddings.py \
      --backbone /path/to/phase1_backbone_best.pt \
      --dataset_dir data/splits/multiclass \
      --sequences_json data/sequences/uniprot_dict.json \
      --out_prefix data/embeddings/phase3_seq_phase1 \
      --window_k 16 \
      --cuda 0 --resume
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from phase_2_decoder.build_embeddings import load_esm2_model, load_backbone_if_given
from phase_2_decoder.config import pick_device

SNP_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
SPLITS = ("train", "val", "test")
FAMILIES = ("easy", "hard")


def parse_snp(snp: str) -> tuple[str, int, str] | None:
    m = SNP_RE.match(str(snp))
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def load_sequences(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return {k: str(v) for k, v in raw.items()}
    return {e["Entry"]: e["Sequence"] for e in raw if e.get("Entry") and e.get("Sequence")}


def apply_snp(seq: str, pos: int, mut_aa: str) -> str:
    """Apply 1-based single amino acid substitution."""
    return seq[: pos - 1] + mut_aa + seq[pos:]


def crop_sequence_around_pos(seq: str, positions: list[int], max_len: int, center_long: bool) -> tuple[str, int]:
    """Return (cropped_seq, offset_0based) where 1-based original pos maps to pos - offset_0based."""
    L = len(seq)
    if L <= max_len or not center_long:
        return seq[:max_len], 0
    center = int(round(sum(positions) / len(positions)))
    half = max_len // 2
    start = max(0, min(center - 1 - half, L - max_len))
    return seq[start : start + max_len], start


def extract_window_from_tokens(
    token_emb_i: torch.Tensor,
    eff_len: int,
    local_pos_1based: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice [2k+1, D] centered at 1-based `local_pos_1based` from `token_emb_i` ([L_pad+2, D]).
    Note: token_emb_i has BOS at index 0, residues at 1..eff_len, and EOS at eff_len+1.
    """
    D = token_emb_i.shape[-1]
    W = 2 * k + 1
    win = np.zeros((W, D), dtype=np.float32)
    mask = np.zeros((W,), dtype=bool)

    for offset in range(-k, k + 1):
        res_idx = local_pos_1based + offset  # 1-based residue index in cropped sequence
        w_idx = offset + k                   # 0..2k index in window
        if 1 <= res_idx <= eff_len:
            win[w_idx] = token_emb_i[res_idx].numpy()
            mask[w_idx] = True

    return win, mask


def forward_multiscale_batch(
    model,
    batch_converter,
    alphabet,
    batch_items: list[tuple[str, str, list[int]]],
    device: torch.device,
    max_len: int,
    k: int,
    center_long: bool = False,
):
    """
    Run ESM-2 forward pass and return:
      - pooled_global: [B, D] float32 numpy array
      - mut_sites:     list of [n_pos_i, D] float32 numpy arrays
      - windows:       list of [n_pos_i, 2k+1, D] float32 numpy arrays
      - masks:         list of [n_pos_i, 2k+1] bool numpy arrays
    """
    assert alphabet.prepend_bos and alphabet.append_eos, (
        "ESM alphabet must have prepend_bos and append_eos enabled."
    )

    cropped_Info = [
        crop_sequence_around_pos(seq, positions, max_len, center_long)
        for _, seq, positions in batch_items
    ]
    labeled = [(key, c_seq) for (key, _, _), (c_seq, _) in zip(batch_items, cropped_Info)]

    _, _, tokens = batch_converter(labeled)
    tokens = tokens.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    token_emb = results["representations"][model.num_layers].float().cpu()  # [B, L_max+2, D]

    # Mask out BOS (0), EOS (length-1), and padding for global mean pooling
    tokens_cpu = tokens.cpu()
    padding_mask = (tokens_cpu != alphabet.padding_idx).float()
    padding_mask[:, 0] = 0.0
    lengths = (tokens_cpu != alphabet.padding_idx).sum(dim=1)
    padding_mask[torch.arange(tokens_cpu.shape[0]), lengths - 1] = 0.0
    mask_3d = padding_mask.unsqueeze(-1)
    pooled_global = ((token_emb * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp(min=1.0)).numpy()

    mut_sites, windows, masks = [], [], []
    for i, ((key, _, positions), (c_seq, offset)) in enumerate(zip(batch_items, cropped_Info)):
        eff_len = len(c_seq)
        m_list, w_list, k_list = [], [], []
        for p in positions:
            local_p = p - offset
            if not (1 <= local_p <= eff_len):
                raise ValueError(
                    f"{key}: pos={p} (local={local_p}) outside effective sequence length {eff_len}."
                )
            m_list.append(token_emb[i, local_p, :].numpy())
            w_arr, m_arr = extract_window_from_tokens(token_emb[i], eff_len, local_p, k)
            w_list.append(w_arr)
            k_list.append(m_arr)

        mut_sites.append(np.stack(m_list, axis=0))
        windows.append(np.stack(w_list, axis=0))
        masks.append(np.stack(k_list, axis=0))

    return pooled_global, mut_sites, windows, masks


def collect_jobs(dataset_dir: Path, sequences: dict[str, str], max_seq_len: int, center_long: bool):
    """Scan split TSV files and build WT and Variant forward job lists."""
    wt_pos = defaultdict(set)
    var = {}

    tsv_files = []
    for fam in FAMILIES:
        for split in SPLITS:
            p1 = dataset_dir / f"{fam}_{split}.tsv"
            p2 = dataset_dir / f"stage1_{fam}_{split}.tsv"
            if p1.exists():
                tsv_files.append(p1)
            elif p2.exists():
                tsv_files.append(p2)

    if not tsv_files:
        raise FileNotFoundError(f"No split TSV files found under {dataset_dir}")

    import csv
    for tsv_path in tsv_files:
        with open(tsv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                pid = row["protein_id"]
                snp = row["snp"]
                parsed = parse_snp(snp)
                if parsed is None or pid not in sequences:
                    continue
                wt_aa, pos, mut_aa = parsed
                seq = sequences[pid]
                if pos > len(seq) or seq[pos - 1] != wt_aa:
                    continue
                if not center_long and pos > max_seq_len:
                    continue
                wt_pos[pid].add(pos)
                var[f"{pid}_{snp}"] = (pid, pos, mut_aa)

    wt_jobs = {p: sorted(s) for p, s in wt_pos.items()}
    var_jobs = [(k, v[0], v[1], v[2]) for k, v in sorted(var.items())]
    return wt_jobs, var_jobs


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as h5:
        return set(h5.keys())


def _write(h5, key: str, arr: np.ndarray):
    if key in h5:
        del h5[key]
    h5.create_dataset(key, data=arr)


def main():
    repo_root = THIS_DIR.parents[1]
    p = argparse.ArgumentParser(
        description="Extract Global + Mutation-Site + Unpooled Local Window embeddings"
    )
    p.add_argument(
        "--out_prefix",
        default=str(repo_root / "data" / "embeddings" / "phase3_seq_phase1"),
        help="Output prefix -> writes {prefix}_global.h5, {prefix}_local.h5, {prefix}_window.h5",
    )
    p.add_argument(
        "--dataset_dir",
        default=str(repo_root / "data" / "splits" / "multiclass"),
        help="Directory containing {easy,hard}_{train,val,test}.tsv or prepared TSVs",
    )
    p.add_argument(
        "--sequences_json",
        default=str(repo_root / "data" / "sequences" / "uniprot_dict.json"),
    )
    p.add_argument(
        "--esm_model",
        default="esm2_t33_650M_UR50D",
        choices=["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D"],
    )
    p.add_argument(
        "--backbone",
        default=None,
        help="Path to fine-tuned encoder checkpoint (.pt containing esm_state_dict).",
    )
    p.add_argument(
        "--allow_raw_esm",
        action="store_true",
        help="Explicitly allow running without a fine-tuned --backbone checkpoint.",
    )
    p.add_argument(
        "--window_k",
        type=int,
        default=16,
        help="Half-window radius k around mutation site -> extracts [2k+1, D] (default: 16 -> 33 residues)",
    )
    p.add_argument(
        "--window_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for _window.h5 (float16 is ~5 GB, float32 is ~10 GB for k=16)",
    )
    p.add_argument(
        "--center_long_sequences",
        action="store_true",
        help="If set, crop sequences > max_seq_len centered on the mutation instead of dropping pos > 1022",
    )
    p.add_argument("--jobs", choices=["wt", "variant", "both"], default="both")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1022)
    p.add_argument("--cuda", default=None)
    p.add_argument("--limit", type=int, default=None, help="Limit number of forward passes for smoke test")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if not args.backbone and not args.allow_raw_esm:
        raise SystemExit(
            "[ERROR] --backbone checkpoint was not provided! "
            "Pass --backbone /path/to/phase1_backbone_best.pt (or --allow_raw_esm if intentional)."
        )
    if args.backbone and not Path(args.backbone).is_file():
        raise SystemExit(f"[ERROR] --backbone checkpoint not found: {args.backbone}")
    if not Path(args.sequences_json).is_file():
        raise SystemExit(f"[ERROR] --sequences_json not found: {args.sequences_json}")
    if not Path(args.dataset_dir).is_dir():
        raise SystemExit(f"[ERROR] --dataset_dir not found: {args.dataset_dir}")

    device = torch.device(pick_device(args.cuda))
    win_dtype = np.float16 if args.window_dtype == "float16" else np.float32
    k = args.window_k
    W = 2 * k + 1

    print("=" * 76)
    print("[CONFIG] ContVAR Sequence Encoder — Multi-Scale Embedding Extraction")
    print("=" * 76)
    print(f"[INFO] Device            : {device}")
    print(f"[INFO] Base ESM arch     : {args.esm_model} (used ONLY to initialize layer structure & alphabet)")
    if args.backbone:
        bb_path = Path(args.backbone).resolve()
        bb_mb = bb_path.stat().st_size / (1024 * 1024)
        print(f"[INFO] Fine-tuned ckpt   : {bb_path} ({bb_mb:.1f} MB)")
    else:
        print("[WARN] Fine-tuned ckpt   : NONE (running raw ESM-2 via --allow_raw_esm)")
    print(f"[INFO] Sequences JSON    : {Path(args.sequences_json).resolve()}")
    print(f"[INFO] Dataset splits dir: {Path(args.dataset_dir).resolve()}")
    print(f"[INFO] Output H5 prefix  : {Path(args.out_prefix).resolve()}_{{global,local,window}}.h5")
    print(f"[INFO] Window radius     : k={k} -> W={W} residues [p-{k} .. p+{k}] (dtype={args.window_dtype})")
    print(f"[INFO] Long seq policy   : {'center crop around mutation (--center_long_sequences)' if args.center_long_sequences else f'truncate at {args.max_seq_len}'}")
    print("=" * 76)

    sequences = load_sequences(args.sequences_json)
    wt_jobs, var_jobs = collect_jobs(
        Path(args.dataset_dir), sequences, args.max_seq_len, args.center_long_sequences
    )
    print(f"[INFO] WT forwards     : {len(wt_jobs):,} proteins ({sum(len(v) for v in wt_jobs.values()):,} sites)")
    print(f"[INFO] Variant forwards: {len(var_jobs):,}")

    global_path = Path(f"{args.out_prefix}_global.h5")
    local_path = Path(f"{args.out_prefix}_local.h5")
    window_path = Path(f"{args.out_prefix}_window.h5")
    global_path.parent.mkdir(parents=True, exist_ok=True)

    done_g = _existing_keys(global_path) if args.resume else set()
    done_l = _existing_keys(local_path) if args.resume else set()
    done_w = _existing_keys(window_path) if args.resume else set()

    tasks = []
    if args.jobs in ("wt", "both"):
        for pid, positions in wt_jobs.items():
            if args.center_long_sequences and len(sequences[pid]) > args.max_seq_len:
                # Group per position when centering long sequences
                for pos in positions:
                    lkey = f"{pid}_{pos}"
                    if pid in done_g and lkey in done_l and lkey in done_w:
                        continue
                    tasks.append((pid, sequences[pid], [pos], pid, [lkey]))
            else:
                lkeys = [f"{pid}_{pos}" for pos in positions]
                if pid in done_g and all(lk in done_l and lk in done_w for lk in lkeys):
                    continue
                tasks.append((pid, sequences[pid], positions, pid, lkeys))

    if args.jobs in ("variant", "both"):
        for key, pid, pos, mut in var_jobs:
            if key in done_g and key in done_l and key in done_w:
                continue
            tasks.append((key, apply_snp(sequences[pid], pos, mut), [pos], key, [key]))

    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("[DONE] All requested embeddings are already present in H5 files.")
        return

    tasks.sort(key=lambda t: len(t[1]))
    print(f"[INFO] Total forwards to run: {len(tasks):,}")

    model, alphabet = load_esm2_model(args.esm_model)
    first_param_before = next(model.parameters()).detach().float().norm().item()
    model = load_backbone_if_given(model, args.backbone, device).to(device).eval()
    last_layer_norm = model.layers[-1].self_attn.k_proj.weight.detach().float().norm().item()
    print(
        f"[VERIFIED] Model weights ready on {device} | "
        f"backbone={'FINE-TUNED (' + str(Path(args.backbone).name) + ')' if args.backbone else 'RAW ESM-2'} | "
        f"last_layer_k_proj_norm={last_layer_norm:.6f} (embed_norm={first_param_before:.6f})"
    )
    batch_converter = alphabet.get_batch_converter()

    mode_g = "a" if global_path.exists() and args.resume else "w"
    mode_l = "a" if local_path.exists() and args.resume else "w"
    mode_w = "a" if window_path.exists() and args.resume else "w"

    with h5py.File(global_path, mode_g) as hg, \
         h5py.File(local_path, mode_l) as hl, \
         h5py.File(window_path, mode_w) as hw:

        for h in (hg, hl, hw):
            h.attrs["esm_model"] = args.esm_model
            h.attrs["backbone_path"] = str(Path(args.backbone).resolve()) if args.backbone else "raw_esm"
            h.attrs["sequences_json"] = str(Path(args.sequences_json).resolve())
            h.attrs["dataset_dir"] = str(Path(args.dataset_dir).resolve())
            h.attrs["center_long_sequences"] = bool(args.center_long_sequences)
            h.attrs["last_layer_k_proj_norm"] = float(last_layer_norm)

        hw.attrs["window_radius_k"] = k
        hw.attrs["window_len"] = W
        hw.attrs["center_index"] = k
        hw.attrs["embed_dim"] = model.embed_dim
        mask_grp = hw.require_group("mask")

        for start in tqdm(range(0, len(tasks), args.batch_size), desc="Extracting [global|local|window]"):
            chunk = tasks[start : start + args.batch_size]
            batch_items = [(t[0], t[1], t[2]) for t in chunk]
            pooled_g, mut_sites, windows, masks = forward_multiscale_batch(
                model,
                batch_converter,
                alphabet,
                batch_items,
                device,
                args.max_seq_len,
                k=k,
                center_long=args.center_long_sequences,
            )
            for i, (_, _, _, gkey, lkeys) in enumerate(chunk):
                _write(hg, gkey, pooled_g[i])
                ms = mut_sites[i]
                ws = windows[i].astype(win_dtype)
                ks = masks[i]
                for j, lk in enumerate(lkeys):
                    _write(hl, lk, ms[j])
                    _write(hw, lk, ws[j])
                    _write(mask_grp, lk, ks[j])

    print(f"[DONE] Global embeddings : {global_path}")
    print(f"[DONE] Mut-site embeddings: {local_path}")
    print(f"[DONE] Window embeddings : {window_path} (shape=[{W}, {model.embed_dim}], center={k})")


if __name__ == "__main__":
    main()
