"""Read-only sequence/embedding lineage audit. Runs directly, without torch/GPU.

Writes reports only; never repairs IDs, fetches sequences, or changes H5 files.
Alternative sources are evidence candidates, not replacements for the primary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

try:
    import h5py
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Use the contvar environment with numpy, pandas, h5py.")

SNP = re.compile(r"^([A-Z])(\d+)([A-Z])$")
ACCESSION = re.compile(r'(?<![A-Z0-9])(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?![A-Z0-9])')
SOURCE_EXTENSIONS = ('.fasta', '.fa', '.faa', '.dat')
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'node_modules', 'envs', 'wandb'}
ROLES = ('wt_global', 'var_global', 'wt_site', 'var_site')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sequence_hash(seq):
    return hashlib.sha256(seq.encode('utf-8')).hexdigest()


def open_text(path):
    return gzip.open(path, 'rt', encoding='utf-8') if str(path).lower().endswith('.gz') else Path(path).open(encoding='utf-8')


def field(record, names):
    for name in names:
        if name in record:
            return record[name]
    return None


def sequence_records(path):
    """Yield (source ID, sequence, secondary IDs) without accepting alias matches.

    JSON list Entry/Sequence, ID->sequence, and UniProt REST results are supported.
    Table headers are required. FASTA source IDs retain their complete first token.
    """
    path = Path(path)
    suffix = path.name.lower().removesuffix('.gz')
    if suffix.endswith('.json'):
        with open_text(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and isinstance(raw.get('results'), list):
            raw = raw['results']
        if isinstance(raw, dict):
            for pid, value in raw.items():
                if isinstance(value, dict):
                    seq = field(value, ('Sequence', 'sequence', 'seq'))
                    if isinstance(seq, dict):
                        seq = seq.get('value')
                else:
                    seq = value
                if seq is not None and not isinstance(seq, str):
                    raise ValueError(f'Unexpected JSON sequence value for {pid!r}')
                yield str(pid), seq or '', []
        elif isinstance(raw, list):
            for record in raw:
                if not isinstance(record, dict):
                    raise ValueError('JSON list must contain objects')
                pid = field(record, ('Entry', 'protein_id', 'primaryAccession', 'accession', 'id'))
                seq = field(record, ('Sequence', 'sequence', 'seq'))
                if isinstance(seq, dict):
                    seq = seq.get('value')
                if pid is None:
                    raise ValueError('JSON record has no recognized protein-ID field')
                if seq is not None and not isinstance(seq, str):
                    raise ValueError(f'Unexpected JSON sequence value for {pid!r}')
                yield str(pid), seq or '', record.get('secondaryAccessions', [])
        else:
            raise ValueError('Unsupported JSON sequence schema')
    elif suffix.endswith(('.tsv', '.csv', '.tab')):
        with open_text(path) as f:
            reader = csv.DictReader(f, delimiter=',' if suffix.endswith('.csv') else '\t')
            names = reader.fieldnames or []
            pid_key = next((x for x in ('Entry', 'protein_id', 'accession', 'id', 'File_Name') if x in names), None)
            seq_key = next((x for x in ('Sequence', 'sequence', 'seq') if x in names), None)
            if not pid_key or not seq_key:
                raise ValueError('No recognized ID/sequence columns; not a sequence table')
            for row in reader:
                yield row[pid_key], row[seq_key] or '', []
    elif suffix.endswith(('.fasta', '.fa', '.faa')):
        with open_text(path) as f:
            header, chunks = None, []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if header is not None:
                        yield header, ''.join(chunks), []
                    header, chunks = line[1:].split()[0], []
                elif header is None:
                    raise ValueError('Sequence before first FASTA header')
                else:
                    chunks.append(line)
            if header is not None:
                yield header, ''.join(chunks), []
    elif suffix.endswith('.dat'):
        with open_text(path) as f:
            accessions, chunks, in_sequence = [], [], False
            for line in f:
                if line.startswith('AC   '):
                    accessions.extend(x.strip() for x in line[5:].split(';') if x.strip())
                elif line.startswith('SQ   '):
                    in_sequence = True
                elif line.startswith('//'):
                    if accessions:
                        yield accessions[0], ''.join(chunks), accessions[1:]
                    accessions, chunks, in_sequence = [], [], False
                elif in_sequence:
                    chunks.append(''.join(line.split()))
    else:
        raise ValueError('Unsupported sequence format')


def id_candidates(pid):
    stripped = pid.strip()
    token = stripped.split()[0] if stripped else ''
    values = [(stripped, 'whitespace_trim'), (token, 'first_header_token')]
    parts = token.split('|')
    if len(parts) >= 3 and parts[0] in ('sp', 'tr'):
        values.append((parts[1], 'uniprot_fasta_header'))
    for match in ACCESSION.finditer(token.upper()):
        values.append((match.group(), 'accession_in_source_identifier'))
    for value, why in list(values):
        if re.search(r'-\d+$', value):
            values.append((re.sub(r'-\d+$', '', value), why + '_isoform_base'))
    return {(value, why) for value, why in values if value and value != pid}


def read_primary(path):
    """Preserve exact IDs and sequence text; do not silently fix the training input."""
    sequences, duplicates, conflicts, empties = {}, Counter(), set(), set()
    for pid, seq, _ in sequence_records(path):
        if not seq:
            empties.add(pid)
            continue
        if pid in sequences:
            duplicates[pid] += 1
            if sequences[pid] != seq:
                conflicts.add(pid)
        sequences[pid] = seq
    if not sequences:
        raise ValueError('Primary source contains no nonempty sequences')
    return sequences, {'nonempty_unique_ids': len(sequences), 'empty_ids': sorted(empties),
                       'duplicate_ids': dict(duplicates), 'conflicting_ids': sorted(conflicts)}


def find_layout(root, family):
    for folder in (root / 'multiclass', root, root / 'cascade', root / 'parallel'):
        for prefix in ('', 'stage1_'):
            paths = [folder / f'{prefix}{family}_{split}.tsv' for split in ('train', 'val', 'test')]
            if all(p.is_file() for p in paths):
                return paths
    raise ValueError(f'No complete train/val/test files for {family} under {root}')


def bool_column(series, name):
    text = series.astype(str).str.strip().str.lower()
    if not text.isin(('true', 'false', '1', '0')).all():
        raise ValueError(f'Invalid boolean values in {name}')
    return text.isin(('true', '1'))


def load_rows(root, families):
    frames, inventory = [], []
    for family in families:
        for split, path in zip(('train', 'val', 'test'), find_layout(root, family)):
            frame = pd.read_csv(path, sep='\t', keep_default_na=False)
            required = {'protein_id', 'snp', 'matched_go_id', 'label'}
            if not required.issubset(frame.columns):
                raise ValueError(f'{path}: missing columns {sorted(required - set(frame.columns))}')
            if not frame.label.isin(('LOF', 'GOF', 'NEUTRAL')).all():
                raise ValueError(f'{path}: unknown label')
            for name in ('augmented', 'is_combinatorial'):
                frame[name] = bool_column(frame[name], name) if name in frame else False
            frame['family'], frame['split'], frame['source_row'] = family, split, range(len(frame))
            frames.append(frame)
            inventory.append({'path': str(path.resolve()), 'sha256': digest(path), 'rows': len(frame)})
    return pd.concat(frames, ignore_index=True), inventory


def sequence_status(pid, snp, sequences, max_len):
    match = SNP.fullmatch(snp)
    if match is None:
        return 'malformed_mutation'
    if pid not in sequences:
        return 'missing_wt_sequence'
    pos = int(match[2])
    if pos < 1:
        return 'position_below_1'
    if pos > max_len:
        return 'position_above_max_seq_len'
    if pos > len(sequences[pid]):
        return 'position_outside_sequence'
    if sequences[pid][pos - 1] != match[1]:
        return 'wt_amino_acid_mismatch'
    return 'eligible'


def scan_roots(roots, out_dir, max_depth, max_files):
    files, warnings, seen = [], [], set()
    def onerror(exc):
        warnings.append(str(exc))
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            warnings.append(f'Search root not found: {root}')
            continue
        for current, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
            here = Path(current)
            depth = len(here.relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (here/d).is_symlink()
                             and (here/d).resolve() != out_dir and depth < max_depth)
            for name in sorted(names):
                path = here/name
                suffix = name.lower().removesuffix('.gz')
                candidate = suffix.endswith(SOURCE_EXTENSIONS) or suffix.endswith(('.h5', '.hdf5')) or (
                    suffix.endswith(('.json', '.tsv', '.csv')) and any(x in suffix for x in ('uniprot', 'sequence', 'protein')))
                if candidate and path.is_file() and path.resolve() not in seen:
                    seen.add(path.resolve())
                    files.append(path.resolve())
                    if len(files) >= max_files:
                        warnings.append(f'Discovery stopped at --max_candidate_files={max_files}; search is incomplete')
                        return files, warnings
    return files, warnings


def inspect_h5(path, required, dimension, keys_only):
    """required maps key -> affected rows; missing files are reportable, not empty successes."""
    path = Path(path).expanduser()
    states = {key: 'missing_file' for key in required}
    meta = {'path': str(path.resolve()), 'required_keys': len(required), 'validation': 'keys_only' if keys_only else 'numeric'}
    if not path.is_file():
        meta['error'] = 'file_not_found'
        return states, meta
    meta.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    try:
        with h5py.File(path, 'r') as h:
            meta.update(root_keys=len(h), attributes={str(k): str(v)[:500] for k, v in h.attrs.items()})
            for n, key in enumerate(sorted(required), 1):
                if key not in h:
                    state = 'missing_key'
                else:
                    try:
                        value = h[key]
                        if not isinstance(value, h5py.Dataset):
                            state = 'not_a_dataset'
                        elif value.shape != (dimension,):
                            state = 'wrong_shape'
                        elif value.dtype.kind != 'f':
                            state = 'not_float'
                        elif keys_only:
                            state = 'key_present_unchecked_values'
                        else:
                            vector = value[:]
                            state = 'nonfinite' if not np.isfinite(vector).all() else 'zero_vector' if not np.any(vector != 0) else 'valid'
                    except (OSError, ValueError, RuntimeError, TypeError) as exc:
                        state = 'unreadable_vector'
                states[key] = state
                if n % 10000 == 0:
                    print(f'[H5] {path.name}: {n:,}/{len(required):,}', flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        meta['error'] = str(exc)
        # A failed file inspection does not establish coverage for remaining keys.
        states = {k: ('unreadable_file' if v == 'missing_file' else v) for k, v in states.items()}
    meta['statuses'] = dict(Counter(states.values()))
    meta['changed_during_audit'] = (path.stat().st_size != meta['bytes'] or path.stat().st_mtime_ns != meta['mtime_ns'])
    return states, meta


def audit_sources(paths, wanted, sequences, primary_path):
    candidates, inventory = [], []
    base_to_wanted = defaultdict(set)
    for pid in wanted:
        base_to_wanted[re.sub(r'-\d+$', '', pid)].add(pid)
    for path in paths:
        info = {'path': str(path), 'records': 0, 'matched_candidates': 0, 'status': 'ok'}
        # Keep evidence already emitted if a source is partially unreadable; flag the source.
        try:
            for source_id, seq, secondary in sequence_records(path):
                info['records'] += 1
                if not seq:
                    continue
                matches = {(source_id, 'exact_id')} if source_id in wanted else set()
                for alias, why in id_candidates(source_id):
                    if alias in wanted:
                        matches.add((alias, why))
                for alias in secondary or []:
                    if alias in wanted:
                        matches.add((alias, 'uniprot_secondary_accession'))
                source_base = re.sub(r'-\d+$', '', source_id)
                for requested in base_to_wanted.get(source_base, ()):
                    if requested != source_id:
                        matches.add((requested, 'isoform_related_candidate'))
                for requested, why in matches:
                    normalized = ''.join(seq.split()).upper()
                    is_primary_exact = path.resolve() == primary_path.resolve() and why == 'exact_id'
                    if is_primary_exact and sequences.get(requested) == seq and normalized == seq:
                        continue
                    candidates.append({'protein_id': requested, 'source_path': str(path), 'source_id': source_id,
                        'match_type': why, 'length': len(seq), 'sequence_sha256': sequence_hash(seq),
                        'sequence_role_hint': 'variant_candidate' if re.search(r'(?:^|[_|])[A-Z]\d+[A-Z](?:[_|.]|$)',source_id)
                            else 'wt_candidate' if re.search(r'(?:^|[_|])wt(?:[_|.]|$)',source_id,re.I) else 'unknown',
                        'normalization_changes_sequence': normalized != seq,
                        'same_as_primary_sequence': requested in sequences and sequences[requested] == seq,
                        'has_primary_sequence': requested in sequences,
                        'is_primary_source': path.resolve() == primary_path.resolve(),
                        'candidate_only_not_applied': True})
                    info['matched_candidates'] += 1
        except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
            info.update(status='unreadable_or_unsupported', error=str(exc))
        inventory.append(info)
        print(f"[SOURCE] {path.name}: {info['status']}; records={info['records']:,}; candidates={info['matched_candidates']:,}", flush=True)
    return candidates, inventory


def join_unique(values):
    return '|'.join(sorted(set(str(x) for x in values)))


def aggregate_rows(df, keys):
    return df.groupby(keys, dropna=False, sort=True).agg(
        dataset_rows=('snp', 'size'), families=('family', join_unique), splits=('split', join_unique),
        mutations=('snp', join_unique), labels=('label', join_unique), augmented_rows=('augmented', 'sum'),
        protocol_rows=('protocol_included', 'sum'), eligible_rows=('pipeline_eligible', 'sum'),
        effective_rows=('effective', 'sum')).reset_index()


def run(args):
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    families = ('easy', 'hard') if args.family == 'both' else (args.family,)
    primary = Path(args.sequences_json).expanduser().resolve()
    if not primary.is_file():
        raise ValueError(f'Primary sequence file not found: {primary}')
    df, inputs = load_rows(Path(args.dataset_dir).expanduser().resolve(), families)
    output_names = ('summary.json', 'REPORT_TR.md', 'sequence_candidates.tsv', 'sequence_source_inventory.tsv',
        'embedding_gaps.tsv', 'missing_wt_sequences.tsv', 'sequence_filter_failures.tsv',
        'variants_requiring_embeddings.tsv', 'row_diagnosis.tsv', 'alternative_h5_matches.tsv',
        'alternative_h5_inventory.tsv', 'log_evidence.tsv', 'coverage_summary.tsv',
        'artifact_inventory.json.gz', 'inventory_comparison.tsv')
    protected = {primary, Path(args.global_h5).expanduser().resolve(), Path(args.local_h5).expanduser().resolve()}
    protected.update(Path(x['path']) for x in inputs)
    protected.update(Path(x).expanduser().resolve() for x in args.sequence_source + args.log_file)
    if args.compare_inventory:
        protected.add(Path(args.compare_inventory).expanduser().resolve())
    if any((out/name).resolve() in protected for name in output_names):
        raise ValueError('A report path overlaps an input. Choose a separate --out_dir.')
    sequences, sequence_meta = read_primary(primary)
    sequence_meta.update(path=str(primary), sha256=digest(primary), bytes=primary.stat().st_size)
    print(f'[DATA] {len(df):,} rows; {df.protein_id.nunique():,} proteins', flush=True)
    parsed = df.snp.str.extract(r'^([A-Z])(\d+)([A-Z])$')
    df['position'] = pd.to_numeric(parsed[1], errors='coerce').astype('Int64')
    df['sequence_status'] = [sequence_status(p, s, sequences, args.max_seq_len) for p, s in zip(df.protein_id, df.snp)]
    df['sequence_length'] = df.protein_id.map(lambda p: len(sequences[p]) if p in sequences else None).astype('Int64')
    df['primary_sequence_sha256'] = df.protein_id.map({p: sequence_hash(sequences[p]) for p in set(df.protein_id) & sequences.keys()})
    df['primary_sequence_conflict'] = df.protein_id.isin(sequence_meta['conflicting_ids'])
    df['pipeline_eligible'] = df.sequence_status.eq('eligible')
    df['protocol_included'] = df.split.eq('train') | ~df.augmented
    df['effective'] = df.pipeline_eligible & df.protocol_included
    df['variant_key'] = df.protein_id + '_' + df.snp
    df['wt_global_key'] = df.protein_id
    df['var_global_key'] = df.variant_key
    df['wt_site_key'] = df.protein_id + '_' + df.position.astype(str)
    df['var_site_key'] = df.variant_key
    if args.reference_dataset_dir:
        old, reference_inputs = load_rows(Path(args.reference_dataset_dir).expanduser().resolve(), families)
        old_variants = set(old.protein_id + '_' + old.snp)
        df['variant_new_vs_reference'] = ~df.variant_key.isin(old_variants)
    else:
        reference_inputs = []
        df['variant_new_vs_reference'] = 'not_checked'

    # All raw required keys are inspected; eligible and excluded rows remain separate.
    required_g = set(df.wt_global_key) | set(df.var_global_key)
    required_l = set(df.loc[df.position.notna(), 'wt_site_key']) | set(df.var_site_key)
    global_states, global_meta = inspect_h5(args.global_h5, required_g, args.expected_dim, args.keys_only)
    local_states, local_meta = inspect_h5(args.local_h5, required_l, args.expected_dim, args.keys_only)
    accepted = {'valid', 'key_present_unchecked_values'}
    for role in ROLES:
        states = global_states if role.endswith('global') else local_states
        df[f'{role}_status'] = df[f'{role}_key'].map(states).fillna('invalid_mutation_key')
    df['embedding_inputs_present'] = df[[f'{r}_status' for r in ROLES]].isin(accepted).all(axis=1)
    df['diagnosis'] = df.sequence_status.where(~df.pipeline_eligible, 'eligible_but_embedding_input_missing_or_invalid')
    df.loc[df.pipeline_eligible & df.embedding_inputs_present, 'diagnosis'] = 'eligible_inputs_present'

    inventory = {'format':'contvar_sequence_lineage_inventory_v1', 'numeric_validation':not args.keys_only,
        'expected_dim':args.expected_dim, 'primary_sequences_sha256':sequence_meta['sha256'],
        'sequence_ids_present':sorted(set(df.protein_id) & sequences.keys()),
        'sequence_ids_requested':sorted(set(df.protein_id)),
        'global_states':global_states, 'site_states':local_states,
        'artifacts':{'global':global_meta,'site':local_meta}}
    with gzip.open(out/'artifact_inventory.json.gz','wt',encoding='utf-8') as f:
        json.dump(inventory,f,default=int)
    comparison, comparison_meta = [], {'provided':False}
    if args.compare_inventory:
        with open_text(Path(args.compare_inventory).expanduser()) as f:
            previous = json.load(f)
        if previous.get('format') != inventory['format'] or previous.get('expected_dim') != args.expected_dim:
            raise ValueError('Reference inventory format/dimension does not match this audit')
        for role,states in (('global',global_states),('site',local_states)):
            earlier = previous[f'{role}_states']
            for key in sorted(states.keys() & earlier.keys()):
                current_state, earlier_state = states[key],earlier[key]
                if current_state == earlier_state:
                    continue
                status = ('present_here_missing_or_invalid_in_reference' if current_state in accepted and earlier_state not in accepted
                    else 'missing_or_invalid_here_present_in_reference' if earlier_state in accepted and current_state not in accepted
                    else 'validation_status_differs')
                comparison.append({'role':role,'key':key,'reference_status':earlier_state,'current_status':current_state,'comparison':status})
        previous_ids = set(previous['sequence_ids_present'])
        previous_requested = set(previous['sequence_ids_requested'])
        for pid in sorted(set(df.protein_id) & previous_requested):
            old_present,new_present = pid in previous_ids,pid in sequences
            if old_present != new_present:
                comparison.append({'role':'sequence','key':pid,'reference_status':'present' if old_present else 'missing',
                    'current_status':'present' if new_present else 'missing','comparison':
                    'present_here_missing_or_invalid_in_reference' if new_present else 'missing_or_invalid_here_present_in_reference'})
        comparison_meta = {'provided':True,'path':str(Path(args.compare_inventory).expanduser().resolve()),
            'same_primary_sequence_file_sha256':previous['primary_sequences_sha256']==sequence_meta['sha256'],
            'reference_numeric_validation':previous['numeric_validation'],
            'differences':len(comparison),'difference_counts':dict(Counter((x['role']+':'+x['comparison']) for x in comparison))}
    pd.DataFrame(comparison,columns=['role','key','reference_status','current_status','comparison']).to_csv(out/'inventory_comparison.tsv',sep='\t',index=False)

    discovered, discovery_warnings = scan_roots(args.search_root, out, args.max_depth, args.max_candidate_files)
    sequence_paths = {primary} | {Path(p).expanduser().resolve() for p in args.sequence_source}
    excluded_h5 = {Path(args.global_h5).expanduser().resolve(), Path(args.local_h5).expanduser().resolve()}
    alternative_h5 = []
    sequence_auto = []
    for path in discovered:
        if path.suffix.lower() in ('.h5', '.hdf5'):
            if path not in excluded_h5:
                alternative_h5.append(path)
        else:
            sequence_auto.append(path)
    if len(sequence_auto) > args.max_sequence_files:
        discovery_warnings.append(f'Sequence candidate limit {args.max_sequence_files} reached; increase it for exhaustive discovery')
    sequence_paths.update(sequence_auto[:args.max_sequence_files])
    candidates, source_inventory = audit_sources(sorted(sequence_paths), set(df.protein_id), sequences, primary)
    candidate_cols = ['protein_id', 'source_path', 'source_id', 'match_type', 'length', 'sequence_sha256', 'sequence_role_hint',
                      'normalization_changes_sequence', 'same_as_primary_sequence', 'has_primary_sequence',
                      'is_primary_source', 'candidate_only_not_applied']
    candidate_df = pd.DataFrame(candidates, columns=candidate_cols).drop_duplicates()
    candidate_df.to_csv(out/'sequence_candidates.tsv', sep='\t', index=False)
    pd.DataFrame(source_inventory).to_csv(out/'sequence_source_inventory.tsv', sep='\t', index=False)
    candidate_proteins = set(candidate_df.protein_id)
    df['alternative_sequence_candidate_found'] = df.protein_id.isin(candidate_proteins)

    gap_rows = []
    for role in ROLES:
        g = df[df.effective & ~df[f'{role}_status'].isin(accepted)]
        for key, group in g.groupby(f'{role}_key', sort=True):
            gap_rows.append({'role': role, 'key': key, 'protein_id': group.protein_id.iloc[0],
                'status': group[f'{role}_status'].iloc[0], 'dataset_rows': len(group),
                'families': join_unique(group.family), 'splits': join_unique(group.split),
                'mutations': join_unique(group.snp), 'variant_new_vs_reference': join_unique(group.variant_new_vs_reference)})
    gaps = pd.DataFrame(gap_rows, columns=['role', 'key', 'protein_id', 'status', 'dataset_rows', 'families', 'splits', 'mutations', 'variant_new_vs_reference'])
    gaps.to_csv(out/'embedding_gaps.tsv', sep='\t', index=False)
    missing_seq = aggregate_rows(df[df.sequence_status.eq('missing_wt_sequence')], ['protein_id'])
    missing_seq['alternative_candidate_found'] = missing_seq.protein_id.isin(candidate_proteins)
    missing_seq['primary_empty_record_seen'] = missing_seq.protein_id.isin(sequence_meta['empty_ids'])
    missing_seq.to_csv(out/'missing_wt_sequences.tsv', sep='\t', index=False)
    df[~df.pipeline_eligible].to_csv(out/'sequence_filter_failures.tsv', sep='\t', index=False)
    need_variant = df.effective & ~(df.var_global_status.isin(accepted) & df.var_site_status.isin(accepted))
    aggregate_rows(df[need_variant], ['protein_id', 'snp', 'variant_key', 'position', 'primary_sequence_sha256',
        'var_global_status', 'var_site_status']).to_csv(out/'variants_requiring_embeddings.tsv', sep='\t', index=False)
    df.to_csv(out/'row_diagnosis.tsv', sep='\t', index=False)

    alt_matches, alt_inventory = [], []
    if len(alternative_h5) > args.max_h5_files:
        discovery_warnings.append(f'Alternative H5 limit {args.max_h5_files} reached; discovery is incomplete')
    for path in alternative_h5[:args.max_h5_files]:
        info = {'path': str(path), 'status': 'ok', 'matching_gap_keys': 0}
        try:
            with h5py.File(path, 'r') as h:
                role_hint = 'global' if 'global' in path.name.lower() else 'site' if any(x in path.name.lower() for x in ('local', 'site')) else 'unknown'
                for gap in gap_rows:
                    key = gap['key']
                    if key in h:
                        value = h[key]
                        shape = list(value.shape) if isinstance(value, h5py.Dataset) else None
                        alt_matches.append({'requested_role': gap['role'], 'key': key, 'candidate_h5': str(path),
                            'filename_role_hint': role_hint, 'shape': str(shape), 'same_checkpoint_verified': False,
                            'usable_replacement_verified': False})
                        info['matching_gap_keys'] += 1
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            info.update(status='unreadable', error=str(exc))
        alt_inventory.append(info)
        print(f"[ALT H5] {path.name}: {info['status']}; gap-key matches={info['matching_gap_keys']:,}", flush=True)
    pd.DataFrame(alt_matches, columns=['requested_role','key','candidate_h5','filename_role_hint','shape','same_checkpoint_verified','usable_replacement_verified']).to_csv(out/'alternative_h5_matches.tsv', sep='\t', index=False)
    pd.DataFrame(alt_inventory, columns=['path','status','matching_gap_keys','error']).to_csv(out/'alternative_h5_inventory.tsv', sep='\t', index=False)

    log_evidence = []
    pattern = re.compile(r'out.of.memory|traceback|killed|--limit|--jobs|--resume|--dataset_dir|--sequences_json|--backbone|--out_prefix|--max_seq_len|WT forwards|Variant forwards|Total forwards|Nothing to do|protein has no sequence|WT residue mismatch|pos >', re.I)
    for filename in args.log_file:
        path = Path(filename).expanduser()
        try:
            with open_text(path) as f:
                for line_number, line in enumerate(f, 1):
                    if pattern.search(line):
                        # Keep diagnostics short, and avoid credential-like lines.
                        if re.search(r'api.?key|password|token\s*[=:]', line, re.I):
                            continue
                        log_evidence.append({'path':str(path.resolve()), 'line':line_number, 'text':line.strip()[:600]})
        except (OSError, UnicodeError) as exc:
            discovery_warnings.append(f'Log unreadable: {path}: {exc}')
    pd.DataFrame(log_evidence, columns=['path','line','text']).to_csv(out/'log_evidence.tsv', sep='\t', index=False)

    summaries = []
    for (family, split), group in df.groupby(['family','split'], sort=False):
        used = group[group.effective]
        row = {'family':family, 'split':split, 'raw_rows':len(group), 'augmented_rows':int(group.augmented.sum()),
            'protocol_rows':int(group.protocol_included.sum()), 'pipeline_rows':int(group.pipeline_eligible.sum()),
            'effective_rows':len(used), 'ready_rows':int(used.embedding_inputs_present.sum()),
            'missing_or_invalid_embedding_rows':int((~used.embedding_inputs_present).sum()),
            'sequence_filter_counts':group.sequence_status.value_counts().to_dict()}
        summaries.append(row)
        print(f"[{family.upper()}/{split}] raw={len(group):,}; eligible+protocol={len(used):,}; ready={row['ready_rows']:,}; embedding gaps={row['missing_or_invalid_embedding_rows']:,}", flush=True)
    sequence_issue_proteins = set(missing_seq.protein_id)
    effective = df[df.effective]
    unique = {'all_proteins':df.protein_id.nunique(), 'all_variants':df.variant_key.nunique(),
        'missing_wt_sequences':len(missing_seq),
        'missing_wt_sequences_with_candidates':len(sequence_issue_proteins & candidate_proteins),
        'eligible_missing_wt_global':effective.loc[~effective.wt_global_status.isin(accepted),'protein_id'].nunique(),
        'eligible_missing_wt_site':effective.loc[~effective.wt_site_status.isin(accepted),'wt_site_key'].nunique(),
        'eligible_missing_variants':df.loc[need_variant,'variant_key'].nunique(),
        'conflicting_primary_sequence_ids':len(sequence_meta['conflicting_ids'])}
    source_changes = []
    for item in inputs + reference_inputs:
        if digest(item['path']) != item['sha256']:
            source_changes.append(item['path'])
    if digest(primary) != sequence_meta['sha256']:
        source_changes.append(str(primary))
    for name,meta in (('global',global_meta),('mutation_site',local_meta)):
        if meta.get('changed_during_audit'):
            source_changes.append(meta['path'])
    conflict_in_use = bool(effective.primary_sequence_conflict.any())
    ok = bool(len(effective)) and bool(effective.embedding_inputs_present.all()) and not source_changes and not conflict_in_use
    summary = {'audit_format':'contvar_sequence_lineage_v1', 'utc_time':datetime.now(timezone.utc).isoformat(),
        'ready_for_current_eligible_protocol_rows':ok, 'numeric_validation_performed':not args.keys_only,
        'mf_filter_applied':False, 'max_seq_len':args.max_seq_len, 'expected_dim':args.expected_dim,
        'inputs':inputs, 'reference_inputs':reference_inputs, 'primary_sequences':sequence_meta,
        'artifacts':{'global':global_meta,'mutation_site':local_meta}, 'unique_counts':unique,
        'per_split':summaries, 'discovery':{'roots':args.search_root,'files_found':len(discovered),'warnings':discovery_warnings},
        'reference_inventory_comparison':comparison_meta,
        'source_files_changed_during_audit':source_changes,
        'interpretation':[
            'A missing primary sequence is excluded by the embedding manifest builder before model inference.',
            'max_seq_len limits mutation positions; long proteins with positions inside the prefix remain eligible.',
            'An eligible missing vector proves an artifact gap, not whether generation was never run, interrupted, or copied incompletely.',
            'Other-source sequences, secondary accessions, and isoforms are candidates only; no ID or sequence was replaced.',
            'Alternative H5 key matches do not establish a matching checkpoint, role, or numerical correctness.',
            'No GO aspect filtering, combinatorial exclusion, contradictory-label removal, training, network fetch, or source mutation was performed.',
            'Combinatorial labels and conflicting primary sequences require separate scientific/provenance review.',
            'If no roots, source files, reference dataset, or logs are supplied, their absence cannot be investigated.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=int),encoding='utf-8')
    pd.DataFrame(summaries).drop(columns='sequence_filter_counts').to_csv(out/'coverage_summary.tsv',sep='\t',index=False)
    report = ['# Sekans ve embedding kaynak kontrolü', '',
        'Bu rapor kaynakları değiştirmeden oluşturuldu. MF filtresi henüz uygulanmadı.', '',
        '| Kontrol | Tekil sayı |','|---|---:|',
        *[f'| {key} | {value:,} |' for key,value in unique.items()], '',
        '## Sonuçları nasıl okuyacağız?', '',
        '- `missing_wt_sequences.tsv`: ana sekans sözlüğünde bulunmayan proteinler. Başka kaynak adayları ayrıca işaretlidir.',
        '- `sequence_candidates.tsv`: exact ID, FASTA header, secondary accession, dosya adındaki accession veya isoform ilişkisiyle bulunan adaylar. Mutant olabilecek kaynaklar sequence_role_hint ile işaretlenir; WT olarak otomatik kabul edilmez.',
        '- `sequence_filter_failures.tsv`: kaynakta sekans yok, pozisyon sınır dışında veya WT aminoasidi eşleşmiyor gibi kesin filtre nedenleri.',
        '- `embedding_gaps.tsv`: sekansı ve pozisyonu uygun örneklerde eksik/bozuk vektörler. WT ve mutant ayrıdır.',
        '- `variants_requiring_embeddings.tsv`: mevcut sekansla yeniden çıkarılabilecek mutant embedding adayları.',
        '- `alternative_h5_matches.tsv`: diğer H5 dosyalarındaki anahtar eşleşmeleri. Aynı checkpoint olduğu doğrulanmadan kullanılamaz.',
        '- `row_diagnosis.tsv`: tüm satırlarda kaynak, filtre, protokol ve dört embedding durumu.',
        '- `log_evidence.tsv`: verilen loglardaki çıkarım komutu, limit veya hata kanıtları.', '',
        '- `inventory_comparison.tsv`: referans envanterle anahtar/sekans farkları. Server’da bulunan fakat localde olmayanları ayırt eder; tek başına kopyalama hatasının nedenini kanıtlamaz.', '',
        '**Eksik vektörün nedeni log/provenance kanıtı olmadan kesin olarak kesinti veya kopyalama hatası diye adlandırılmaz.**', '',
        'Global çıktı mevcut çıkarımda ilk max_seq_len residue üzerinden alınır. H5 adındaki local, mutation-site vektörüdür.', '',
        '## Arama sınırları', '', *(f'- {x}' for x in discovery_warnings or ['Keşif limitine ulaşıldığına dair uyarı yok. Arama yalnız verilen köklerle sınırlıdır.'])]
    (out/'REPORT_TR.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('[UNIQUE] '+json.dumps(unique,default=int), flush=True)
    print(f"[RESULT] {'READY' if ok else 'GAPS FOUND'}; no data or embeddings changed.")
    print(f'[SAVED] {out}')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_dir',required=True,help='V4 root, multiclass/, parallel/, or cascade/; each row set is loaded once')
    p.add_argument('--sequences_json',required=True,help='Primary sequence input used by the embedding builder')
    p.add_argument('--global_h5',required=True)
    p.add_argument('--local_h5',required=True,help='Mutation-site H5, not a window cache')
    p.add_argument('--out_dir',required=True)
    p.add_argument('--family',choices=('easy','hard','both'),default='both')
    p.add_argument('--reference_dataset_dir',help='Old splits used to build embeddings; identifies newly added variants')
    p.add_argument('--compare_inventory',help='artifact_inventory.json.gz from the local audit; compare server/local input availability')
    p.add_argument('--sequence_source',action='append',default=[],help='Additional explicit JSON/FASTA/TSV/CSV/UniProt DAT source; repeatable, .gz supported')
    p.add_argument('--search_root',action='append',default=[],help='Bounded read-only discovery under these roots; repeatable')
    p.add_argument('--log_file',action='append',default=[],help='Embedding-generation log to search for command/interruption evidence; repeatable')
    p.add_argument('--max_depth',type=int,default=5)
    p.add_argument('--max_candidate_files',type=int,default=2000)
    p.add_argument('--max_sequence_files',type=int,default=60)
    p.add_argument('--max_h5_files',type=int,default=30)
    p.add_argument('--max_seq_len',type=int,default=1022)
    p.add_argument('--expected_dim',type=int,default=1280)
    p.add_argument('--keys_only',action='store_true',help='Faster coverage only; skips finite/nonzero vector checks')
    args = p.parse_args()
    for name in ('dataset_dir','sequences_json','global_h5','local_h5','out_dir'):
        if not getattr(args,name).strip():
            p.error(f'--{name} must be nonempty; check exported variables')
    for name in ('max_depth','max_candidate_files','max_sequence_files','max_h5_files','max_seq_len','expected_dim'):
        if getattr(args,name) < (0 if name == 'max_depth' else 1):
            p.error(f'--{name} is outside the supported range')
    try:
        raise SystemExit(run(args))
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        p.exit(2, f'[ERROR] {exc}\n')


if __name__ == '__main__':
    main()
