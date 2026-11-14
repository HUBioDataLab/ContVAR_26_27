"""
DESIGN §8 verification audit.

Two modes:

  --mode h5    (default) Full audit on the generated H5 files: dimensions, sequence
               construction, key alignment, fingerprint. Run once after embedding
               generation finishes.

  --mode live  BEFORE the H5 files exist, computes embeddings on the fly for a small
               sample and checks only the fingerprint. Proves the indexing is correct
               before committing to 53,347 x 3 backbone forward passes.

Expected fingerprint (cos(variant, WT), ESM-2 650M):

    residue at the mutated position   ~0.928    correct
    neighbouring residue (off by one) ~0.985    indexing is shifted
    >=15 residues away                ~0.999    badly shifted / wrong pairing
    mean-pooled global                ~0.999    global written instead of local

    python -m phase_3_variant_decoder.audit --h5_prefix data/processed/phase3
    python -m phase_3_variant_decoder.audit --mode live --n 300 --cuda 0
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

import torch

from phase_2_decoder.build_embeddings import load_backbone_if_given, load_esm2_model
from phase_2_decoder.config import pick_device
from phase_3_variant_decoder.build_pair_embeddings import apply_snp, pool_and_local
from phase_3_variant_decoder.config import (
    ESM_DIM, FAMILIES, MAX_SEQ_LEN, VariantDecoderConfig,
)
from phase_3_variant_decoder.data.variant_dataset import load_family, load_sequences

FAR_MIN_DIST = 15


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def check_sequences(df, sequences, n=20, rng=None) -> bool:
    """§8.2 — is the constructed variant sequence correct."""
    rng = rng or np.random.default_rng(0)
    rows = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 30)))
    bad = []
    for r in rows.itertuples():
        wt = sequences[r.protein_id]
        var = apply_snp(wt, r.pos, r.mut_aa)
        diffs = [i for i, (a, b) in enumerate(zip(wt, var)) if a != b]
        if len(var) != len(wt) or var[r.pos - 1] != r.mut_aa or diffs != [r.pos - 1]:
            bad.append((r.protein_id, r.snp, len(diffs)))
    ok = not bad
    print(f"  §8.2 sequence construction ({len(rows)} samples): "
          f"{'PASS' if ok else 'FAIL ' + str(bad[:3])}")
    return ok


def check_fingerprint(df, prefix, n=400, rng=None) -> bool:
    """§8.3 — the cos(variant, WT) fingerprint in the generated H5 files."""
    rng = rng or np.random.default_rng(0)
    sub = df.drop_duplicates("variant_key").sample(
        n=min(n, df.variant_key.nunique()), random_state=int(rng.integers(1 << 30)))
    at_mut, at_glob = [], []
    with h5py.File(f"{prefix}_local.h5", "r") as hl, h5py.File(f"{prefix}_global.h5", "r") as hg:
        for r in sub.itertuples():
            at_mut.append(_cos(hl[r.wt_local_key][:], hl[r.variant_key][:]))
            at_glob.append(_cos(hg[r.protein_id][:], hg[r.variant_key][:]))
    m_mut, m_glob = float(np.median(at_mut)), float(np.median(at_glob))
    ok = m_mut < 0.97 and m_glob > 0.99 and m_mut < m_glob
    print(f"  §8.3 fingerprint (n={len(sub)}): mutation={m_mut:.5f}  global={m_glob:.5f}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        print("       if mutation is ~0.97+, the local came from the wrong position "
              "or the global was written instead")
    return ok


def check_alignment(df, prefix, n=200, rng=None) -> bool:
    """§8.4 — does the pos in the WT local key match the row's pos."""
    rng = rng or np.random.default_rng(1)
    sub = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 30)))
    bad = [(r.wt_local_key, r.pos) for r in sub.itertuples()
           if r.wt_local_key != f"{r.protein_id}_{r.pos}"
           or r.variant_key != f"{r.protein_id}_{r.snp}"]
    ok = not bad
    print(f"  §8.4 key alignment ({len(sub)} samples): "
          f"{'PASS' if ok else 'FAIL ' + str(bad[:3])}")
    return ok


def check_h5_shape(prefix) -> bool:
    """Is the embedding dim consistent with the 650M model."""
    ok = True
    for kind in ("global", "local"):
        with h5py.File(f"{prefix}_{kind}.h5", "r") as h5:
            k = next(iter(h5.keys()))
            dim = int(h5[k].shape[0])
            n = len(h5.keys())
        good = dim == ESM_DIM
        ok &= good
        print(f"  H5 {kind:<6s}: {n:>7,} keys · dim {dim} "
              f"{'PASS' if good else f'FAIL (expected {ESM_DIM})'}")
    return ok


def check_fingerprint_live(df, sequences, args) -> bool:
    """Fingerprint from a live sample, before any H5 exists (was verify_local_embeddings).

    Run BEFORE generation so that 53,347 x 3 backbone forwards are not wasted.
    """
    device = torch.device(pick_device(args.cuda))
    sub = df.drop_duplicates("variant_key").copy()
    sub["seq_len"] = sub["protein_id"].map(lambda p: len(sequences[p]))
    sub = sub[sub["seq_len"] >= 2 * FAR_MIN_DIST].sort_values("seq_len").head(args.n)
    print(f"[INFO] sampled {len(sub)} variants "
          f"(length {sub.seq_len.min()}-{sub.seq_len.max()}), device {device}")

    model, alphabet = load_esm2_model(args.esm_model)
    model = load_backbone_if_given(model, args.backbone, device).to(device).eval()
    bc = alphabet.get_batch_converter()

    at_mut, at_nb, at_far, at_glob = [], [], [], []
    rows = list(sub.itertuples())
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start: start + args.batch_size]
        wt_items, var_items = [], []
        for r in chunk:
            seq = sequences[r.protein_id]
            eff = min(len(seq), MAX_SEQ_LEN)
            nb = r.pos - 1 if r.pos > 1 else r.pos + 1
            far = r.pos + FAR_MIN_DIST if r.pos + FAR_MIN_DIST <= eff else r.pos - FAR_MIN_DIST
            pos = [r.pos, nb, far]
            wt_items.append((r.protein_id, seq, pos))
            var_items.append((r.variant_key, apply_snp(seq, r.pos, r.mut_aa), pos))
        wt_g, wt_l = pool_and_local(model, bc, alphabet, wt_items, device, MAX_SEQ_LEN)
        var_g, var_l = pool_and_local(model, bc, alphabet, var_items, device, MAX_SEQ_LEN)
        for i in range(len(chunk)):
            w, v = wt_l[i].numpy(), var_l[i].numpy()
            at_mut.append(_cos(w[0], v[0]))
            at_nb.append(_cos(w[1], v[1]))
            at_far.append(_cos(w[2], v[2]))
            at_glob.append(_cos(wt_g[i].numpy(), var_g[i].numpy()))

    med = lambda x: float(np.median(x))
    print()
    print(f"  {'measurement':<34s} {'median cos':>12s}   {'expected':>10s}")
    print("  " + "-" * 60)
    for name, vals, exp in (("residue at mutated position", at_mut, "~0.928"),
                            ("neighbouring residue (pos+-1)", at_nb, "~0.985"),
                            (f">={FAR_MIN_DIST} residues away", at_far, "~0.999"),
                            ("mean-pooled global", at_glob, "~0.999")):
        print(f"  {name:<34s} {med(vals):>12.5f}   {exp:>10s}")
    ok = med(at_mut) < med(at_nb) < med(at_far)
    print()
    print("  [PASS] Separation is localised to the mutation site — indexing is correct."
          if ok else
          "  [FAIL] Ordering is not as expected: BOS shift, wrong pairing, or global "
          "written instead of local.")
    return ok


def main():
    p = argparse.ArgumentParser(description="DESIGN §8 verification audit")
    p.add_argument("--mode", choices=["h5", "live"], default="h5")
    p.add_argument("--esm_model", default="esm2_t33_650M_UR50D",
                   choices=["esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"])
    p.add_argument("--backbone", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cuda", default=None)
    p.add_argument("--h5_prefix", default="data/processed/phase3")
    p.add_argument("--embedding", default="raw_esm")
    p.add_argument("--family", choices=FAMILIES, default="hard")
    p.add_argument("--n", type=int, default=400)
    args = p.parse_args()

    prefix = f"{args.h5_prefix}_{args.embedding}"
    cfg = VariantDecoderConfig(family=args.family)
    sequences = load_sequences(cfg.sequences_json)
    df, report = load_family(args.family, cfg.dataset_dir, sequences)

    if args.mode == "live":
        print(f"[AUDIT/live] family={args.family}  ·  {len(df):,} rows  ·  "
              f"model={args.esm_model}")
        ok = check_fingerprint_live(df, sequences, args)
        raise SystemExit(0 if ok else 1)

    print(f"[AUDIT] {prefix}  ·  family={args.family}  ·  {len(df):,} rows")
    print()
    results = [
        check_h5_shape(prefix),
        check_sequences(df, sequences),
        check_alignment(df, prefix),
        check_fingerprint(df, prefix, n=args.n),
    ]
    print()
    print("RESULT:", "ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
