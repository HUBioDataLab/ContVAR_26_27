"""
Phase 3 pair embedding generation: [global | local].

Design: phase_3_variant_decoder/DESIGN.md §6.2

Reuses the infrastructure of phase_2_decoder/build_variant_embeddings.py (ESM
loading, backbone, length-sorted batching, --resume, device selection). Two
differences:

  (a) token_emb IS NOT DISCARDED. build_embeddings.mean_pool_last_layer() computes
      token_emb as [B, L, D] and throws it away after mean-pooling; the local
      embedding is already sitting there. pool_and_local() returns both the pooled
      vector and the residue vectors at the requested positions from that same
      forward pass.

  (b) A manifest instead of a FASTA directory. Sequences are built in memory from
      uniprot_dict.json; no 47k individual FASTA files are written.

Output (two H5 files per backbone):
    {prefix}_global.h5   key: "{protein}"        (WT)  / "{protein}_{snp}" (variant)
    {prefix}_local.h5    key: "{protein}_{pos}"  (WT)  / "{protein}_{snp}" (variant)

The WT side needs ONE forward per protein: every position that protein requires is
pulled from the same pass (median 2, max 114 positions per protein).

Usage:
    python -m phase_3_variant_decoder.build_pair_embeddings \\
        --out_prefix data/processed/phase3_raw_esm --cuda 0
    python -m phase_3_variant_decoder.build_pair_embeddings \\
        --out_prefix data/processed/phase3_seq_phase1 \\
        --backbone /media/.../phase1_backbone_best.pt --cuda 0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from phase_2_decoder.config import pick_device
    from phase_2_decoder.build_embeddings import load_esm2_model, load_backbone_if_given
    from phase_3_variant_decoder.config import FAMILIES, MAX_SEQ_LEN, VariantDecoderConfig
    from phase_3_variant_decoder.data.variant_dataset import load_family, load_sequences
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Run as a module from the repo root: "
        "python -m phase_3_variant_decoder.build_pair_embeddings ..."
    ) from exc


def pool_and_local(model, batch_converter, alphabet, batch_items, device, max_len):
    """Both the mean-pooled global and the requested positions' local vectors.

    Args:
        batch_items: [(key, sequence, positions)] — positions are **1-based**.

    Returns:
        pooled: [B, D] cpu tensor
        locals_: list of [n_pos_i, D] cpu tensors, one per sample

    BOS WARNING: the ESM tokenizer prepends <cls>, so the output is [B, L+2, D].
    For a 1-based pos the correct token index is `pos` ITSELF, not pos-1. Phase 0
    (extract_embeddings.py:83) slices token_emb[0, 1:-1, :] first, which is why it
    uses pos-1 there; do not mix the two conventions.
    """
    assert alphabet.prepend_bos and alphabet.append_eos, (
        "This code assumes BOS/EOS tokenisation (the pos index depends on it)."
    )

    labeled = [(k, s[:max_len]) for k, s, _ in batch_items]
    _, _, tokens = batch_converter(labeled)
    tokens = tokens.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
    token_emb = results["representations"][model.num_layers].float()  # [B, L+2, D]

    # Exactly the same masking as mean_pool_last_layer() (BOS and EOS excluded)
    padding_mask = (tokens != alphabet.padding_idx).float()
    padding_mask[:, 0] = 0
    lengths = (tokens != alphabet.padding_idx).sum(dim=1)
    padding_mask[torch.arange(tokens.shape[0], device=device), lengths - 1] = 0
    mask = padding_mask.unsqueeze(-1)
    pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    locals_ = []
    for i, (key, seq, positions) in enumerate(batch_items):
        eff_len = min(len(seq), max_len)
        for p in positions:
            if not (1 <= p <= eff_len):
                raise ValueError(
                    f"{key}: pos={p} is outside the truncated sequence (eff_len={eff_len}). "
                    "Rows with pos > max_seq_len should have been dropped BEFORE the forward pass."
                )
        idx = torch.as_tensor(list(positions), device=token_emb.device, dtype=torch.long)
        locals_.append(token_emb[i, idx, :].cpu())

    return pooled.detach().cpu(), locals_


def build_manifest(dataset_dir: str, sequences: dict[str, str], max_seq_len: int):
    """Combined job list across both families — embeddings are shared between them.

    Returns:
        wt_jobs:  {protein_id: sorted[positions]}
        var_jobs: [(key, protein_id, pos, mut_aa)]
    """
    wt_pos = defaultdict(set)
    var = {}
    for fam in FAMILIES:
        df, _ = load_family(fam, dataset_dir, sequences, max_seq_len=max_seq_len)
        for pid, snp, pos, mut in zip(df["protein_id"], df["snp"], df["pos"], df["mut_aa"]):
            wt_pos[pid].add(int(pos))
            var[f"{pid}_{snp}"] = (pid, int(pos), mut)
    wt_jobs = {p: sorted(s) for p, s in wt_pos.items()}
    var_jobs = [(k, v[0], v[1], v[2]) for k, v in sorted(var.items())]
    return wt_jobs, var_jobs


def apply_snp(seq: str, pos: int, mut_aa: str) -> str:
    """Apply the mutation at a 1-based position."""
    return seq[: pos - 1] + mut_aa + seq[pos:]


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as h5:
        return set(h5.keys())


def _write(h5, key: str, arr: np.ndarray):
    if key in h5:
        del h5[key]
    h5.create_dataset(key, data=arr)


def build(args):
    # Path checks FIRST: loading the 2.6 GB ESM model and only then reporting a
    # missing file wastes a minute. (A typical mistake is an undefined shell
    # variable turning "--backbone $CKPT/..." into "/...".)
    if args.backbone and not Path(args.backbone).is_file():
        raise SystemExit(
            f"[ERROR] --backbone not found: {args.backbone}\n"
            "        If the path starts with '/', you may have used an undefined "
            "shell variable; write the full path."
        )
    if not Path(args.sequences_json).is_file():
        raise SystemExit(f"[ERROR] --sequences_json not found: {args.sequences_json}")
    if not Path(args.dataset_dir).is_dir():
        raise SystemExit(f"[ERROR] --dataset_dir not found: {args.dataset_dir}")

    device = torch.device(pick_device(args.cuda))
    print(f"[INFO] Device: {device}")

    sequences = load_sequences(args.sequences_json)
    wt_jobs, var_jobs = build_manifest(args.dataset_dir, sequences, args.max_seq_len)
    print(f"[INFO] WT forwards     : {len(wt_jobs):,} proteins "
          f"({sum(len(v) for v in wt_jobs.values()):,} positions)")
    print(f"[INFO] Variant forwards: {len(var_jobs):,}")

    global_path = Path(f"{args.out_prefix}_global.h5")
    local_path = Path(f"{args.out_prefix}_local.h5")
    global_path.parent.mkdir(parents=True, exist_ok=True)

    done_g = _existing_keys(global_path) if args.resume else set()
    done_l = _existing_keys(local_path) if args.resume else set()
    if args.resume:
        print(f"[INFO] Resume: global={len(done_g):,} local={len(done_l):,} keys present")

    # ── job list: (key, sequence, positions, global_key, local_keys) ────────
    tasks = []
    if args.jobs in ("wt", "both"):
        for pid, positions in wt_jobs.items():
            lkeys = [f"{pid}_{p}" for p in positions]
            if pid in done_g and all(k in done_l for k in lkeys):
                continue
            tasks.append((pid, sequences[pid], positions, pid, lkeys))
    if args.jobs in ("variant", "both"):
        for key, pid, pos, mut in var_jobs:
            if key in done_g and key in done_l:
                continue
            tasks.append((key, apply_snp(sequences[pid], pos, mut), [pos], key, [key]))

    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        print("[DONE] Nothing to do, the H5 files are up to date.")
        return

    tasks.sort(key=lambda t: len(t[1]))   # shortest sequences first — memory friendly
    print(f"[INFO] Total forwards to run: {len(tasks):,}")

    model, alphabet = load_esm2_model(args.esm_model)
    model = load_backbone_if_given(model, args.backbone, device).to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    mode_g = "a" if global_path.exists() and args.resume else "w"
    mode_l = "a" if local_path.exists() and args.resume else "w"
    with h5py.File(global_path, mode_g) as hg, h5py.File(local_path, mode_l) as hl:
        for start in tqdm(range(0, len(tasks), args.batch_size), desc="pair embeddings"):
            chunk = tasks[start: start + args.batch_size]
            batch_items = [(t[0], t[1], t[2]) for t in chunk]
            pooled, locals_ = pool_and_local(
                model, batch_converter, alphabet, batch_items, device, args.max_seq_len
            )
            for i, (_, _, _, gkey, lkeys) in enumerate(chunk):
                _write(hg, gkey, pooled[i].numpy())
                loc = locals_[i].numpy()
                for j, lk in enumerate(lkeys):
                    _write(hl, lk, loc[j])

    print(f"[DONE] {global_path}")
    print(f"[DONE] {local_path}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = VariantDecoderConfig()

    p = argparse.ArgumentParser(description="Phase 3 [global | local] embedding generation")
    p.add_argument("--out_prefix", required=True,
                   help="e.g. data/processed/phase3_raw_esm -> _global.h5 and _local.h5")
    p.add_argument("--dataset_dir", default=cfg.dataset_dir)
    p.add_argument("--sequences_json", default=cfg.sequences_json)
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D",
                   choices=["esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"])
    p.add_argument("--backbone", default=None,
                   help="Fine-tuned backbone (.pt, must contain esm_state_dict)")
    p.add_argument("--jobs", choices=["wt", "variant", "both"], default="both")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--cuda", default=None)
    p.add_argument("--limit", type=int, default=None, help="first N forwards, for a smoke test")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    build(args)


if __name__ == "__main__":
    main()
