#!/usr/bin/env python3
"""Read a small paired batch (Global + Mutation-Site + Optional Local Window) without loading full H5s into RAM."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

DECODER_DIR = Path(__file__).resolve().parent
REPO_ROOT = DECODER_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
sys.path.insert(0, str(DECODER_DIR))


def load_batch(family='hard', split='train', n=4, window_k: int | None = None, source: str = 'trained_mean'):
    import h5py
    import numpy as np
    import pandas as pd
    from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable

    if family not in ('easy', 'hard') or split not in ('train', 'val', 'test') or n < 1:
        raise ValueError('Invalid family, split or batch size')

    source_map = {
        'trained_mean': DATA_DIR / 'embeddings' / 'trained_win7_mean',
        'trained_distance': DATA_DIR / 'embeddings' / 'trained_win7_distance',
        'baseline': DATA_DIR / 'embeddings' / 'baseline',
    }
    embed_dir = source_map.get(source, DATA_DIR / 'embeddings' / source)
    if not embed_dir.exists():
        raise FileNotFoundError(f'Embedding directory not found: {embed_dir}')

    frame = pd.read_csv(DATA_DIR / f'prepared/{family}_{split}.tsv', sep='\t').head(n)
    if frame.empty:
        raise ValueError('No prepared rows found.')
    arrays = {}
    for kind, filename, mappings in (
        ('global', 'phase3_seq_phase1_global.h5', {'WT_global': 'wt_global_key', 'VAR_global': 'var_global_key'}),
        ('mutation-site', 'phase3_seq_phase1_local.h5', {'WT_mut': 'wt_mut_key', 'VAR_mut': 'var_mut_key'}),
    ):
        with h5py.File(embed_dir / filename, 'r') as handle:
            for name, column in mappings.items():
                arrays[name] = np.stack([handle[key][:] for key in frame[column]]).astype(np.float32)
                if arrays[name].shape != (len(frame), 1280) or not np.isfinite(arrays[name]).all():
                    raise ValueError(f'Invalid {kind} batch: {name}')

    window_path = embed_dir / 'phase3_seq_phase1_window.h5'
    if window_path.exists():
        wt_col = 'wt_window_key' if 'wt_window_key' in frame.columns else 'wt_mut_key'
        var_col = 'var_window_key' if 'var_window_key' in frame.columns else 'var_mut_key'
        with h5py.File(window_path, 'r') as handle:
            sample_shape = handle[frame[wt_col].iloc[0]].shape
            if len(sample_shape) == 2 and sample_shape[0] == 7:
                # Fixed 7-residue window (trained context)
                for name, column in (('WT_window', wt_col), ('VAR_window', var_col)):
                    arrays[name] = np.stack([handle[key][:] for key in frame[column]]).astype(np.float32)
                if 'mask' in handle:
                    arrays['WINDOW_mask'] = np.stack([handle['mask'][key][:] for key in frame[wt_col]]).astype(bool)
            else:
                saved_k = int(handle.attrs.get('window_radius_k', 16))
                center = int(handle.attrs.get('center_index', saved_k))
                use_k = saved_k if window_k is None else min(window_k, saved_k)
                sl = slice(center - use_k, center + use_k + 1)
                for name, column in (('WT_window', wt_col), ('VAR_window', var_col)):
                    arrays[name] = np.stack([handle[key][sl] for key in frame[column]]).astype(np.float32)
                if 'mask' in handle:
                    arrays['WINDOW_mask'] = np.stack([handle['mask'][key][sl] for key in frame[wt_col]]).astype(bool)
            arrays['DELTA_window'] = arrays['VAR_window'] - arrays['WT_window']

    go_path = DATA_DIR / 'embeddings/go_terms_embeddings_anc2vec.npz'
    table = GOEmbeddingTable(go_path)
    arrays['GO'] = table.lookup(frame['matched_go_id'].tolist()).detach().cpu().numpy()
    arrays['DELTA_mut'] = arrays['VAR_mut'] - arrays['WT_mut']
    arrays['DELTA_global'] = arrays['VAR_global'] - arrays['WT_global']
    arrays['labels'] = frame['label_idx'].to_numpy(dtype=np.int64)
    return frame, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('easy', 'hard'), default='hard')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='train')
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--window_k', type=int, default=None, help='Optional sub-window radius k <= saved_k')
    parser.add_argument('--source', choices=('trained_mean', 'trained_distance', 'baseline'), default='trained_mean',
                        help='Feature source directory under data/embeddings')
    args = parser.parse_args()
    frame, arrays = load_batch(args.family, args.split, args.n, window_k=args.window_k, source=args.source)
    print(f'=== Loaded from source: {args.source} ===')
    print(frame[['sample_id', 'protein_id', 'snp', 'matched_go_id', 'label', 'label_idx']].to_string(index=False))
    for name, array in arrays.items():
        print(f'{name}: shape={array.shape}, dtype={array.dtype}')
    print('Labels: 0=LOF, 1=GOF, 2=NEUTRAL. Verification successful.')


if __name__ == '__main__':
    main()
