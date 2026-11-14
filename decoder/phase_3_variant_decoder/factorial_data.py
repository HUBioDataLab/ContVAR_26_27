"""Matched V4 data selection and fail-closed frozen-input preflight for all arms."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from phase_3_variant_decoder.audit_input_embeddings import audit_h5,sha256
from phase_3_variant_decoder.config import VariantDecoderConfig
from phase_3_variant_decoder.data.variant_dataset import load_family,load_sequences,build_go_vocab,attach_indices
from phase_3_variant_decoder.data.go_embeddings import GOEmbeddingTable
from phase_3_variant_decoder.data.pair_dataset import build_pair_tensors
from phase_3_variant_decoder.input_controls import fit_normalization


def source_root(path,family):
    path=Path(path).expanduser().resolve()
    for root in (path/'multiclass',path,path/'parallel',path/'cascade'):
        if all((root/f'{family}_{s}.tsv').is_file() for s in ('train','val','test')) or all(
            (root/f'stage1_{family}_{s}.tsv').is_file() and (root/f'stage2_{family}_{s}.tsv').is_file()
            for s in ('train','val','test')):
            return root
    raise ValueError(f'No complete split family under {path}: {family}')


def ontology_terms(path):
    """Namespace lookup includes alt IDs; no relabeling of the original GO lookup key."""
    terms={};block={}
    def finish():
        if 'id' not in block:return
        value=(block.get('namespace'),block.get('obsolete',False))
        terms[block['id']]=value
        for alias in block.get('alt',[]):terms[alias]=value
    with Path(path).open(encoding='utf-8') as f:
        for raw in f:
            line=raw.strip()
            if line.startswith('['):
                finish();block={};continue
            if line.startswith('id: GO:'):block['id']=line[4:]
            elif line.startswith('namespace: '):block['namespace']=line[11:]
            elif line.startswith('alt_id: '):block.setdefault('alt',[]).append(line[8:])
            elif line=='is_obsolete: true':block['obsolete']=True
    finish()
    if not terms:raise ValueError('No GO terms in --go_obo')
    return terms


def split_views(tensors):
    """Own index storage; pandas Copy-on-Write arrays can be non-writable."""
    views={}
    for split in ('train','val','test'):
        part=tensors.df[tensors.df.split==split]
        views[split]={'n':len(part),**{key:torch.tensor(part[key].to_numpy(copy=True),dtype=torch.long,
            device=tensors.wt_feat.device) for key in ('wt_row','var_row','go_idx','label_idx')}}
    return views


def prepare_inputs(args,family,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    data_root=source_root(args.dataset_dir,family)
    cfg=VariantDecoderConfig(family,args.embedding,args.input_mode)
    for key in ('sequences_json','global_h5','local_h5','go_embeddings'):
        value=getattr(args,key)
        if value:setattr(cfg,key,str(Path(value).expanduser().resolve()))
    cfg.dataset_dir=str(data_root);cfg.max_seq_len=args.max_seq_len
    cfg.drop_contradictory=args.drop_contradictory
    sequences=load_sequences(cfg.sequences_json)
    frame,filters=load_family(family,str(data_root),sequences,max_seq_len=args.max_seq_len,
                              drop_contradictory=args.drop_contradictory)
    if not frame.label.isin(('LOF','GOF','NEUTRAL')).all():raise ValueError('Unexpected target labels')
    report={'family':family,'source_root':str(data_root),'raw_rows':filters.raw_rows,
            'sequence_filter_steps':filters.steps,'pipeline_rows':len(frame),'steps':[],
            'aspect':args.aspect,'missing_policy':'explicit_subset' if args.allow_missing_embeddings else 'error',
            'go_obo_sha256':sha256(args.go_obo) if args.go_obo else None}
    def save():
        (root/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def keep(mask,name):
        nonlocal frame
        report['steps'].append({'filter':name,'removed':int((~mask).sum())})
        frame=frame.loc[mask].copy()
    for col in ('augmented','is_combinatorial'):
        if col not in frame:frame[col]=False
        text=frame[col].astype(str).str.lower()
        if not text.isin(('true','false','0','1')).all():raise ValueError(f'Invalid boolean field {col}')
        frame[col]=text.isin(('true','1'))
    # Audit original raw identities, not only the successfully encoded subset.
    raw=[]
    for split in ('train','val','test'):
        path=data_root/f'{family}_{split}.tsv'
        if not path.exists():path=data_root/f'stage1_{family}_{split}.tsv'
        part=pd.read_csv(path,sep='\t',keep_default_na=False)
        part['split']=split;raw.append(part)
        if 'target_lof' in part or 'target_gof' in part:
            if not {'target_lof','target_gof'}<=set(part):raise ValueError('Incomplete parallel targets')
            if not (part.target_lof.eq(part.label.eq('LOF').astype(int)).all() and part.target_gof.eq(part.label.eq('GOF').astype(int)).all()):
                raise ValueError('Parallel targets must be LOF-vs-rest / GOF-vs-rest')
    raw=pd.concat(raw,ignore_index=True)
    for a,b in (('train','val'),('train','test'),('val','test')):
        left,right=raw[raw.split==a],raw[raw.split==b]
        if family=='hard' and set(left.protein_id)&set(right.protein_id):
            raise ValueError(f'HARD proteins overlap between {a}/{b}')
        l=set(zip(left.protein_id,left.snp));r=set(zip(right.protein_id,right.snp))
        if l&r:raise ValueError(f'Variants overlap between {a}/{b}')
    keep(frame.split.eq('train')|~frame.augmented,'exclude augmentation from validation/test')
    keep(frame.pos.ge(1),'position must be 1-based')
    if args.exclude_combinatorial:keep(~frame.is_combinatorial,'exclude combinatorial descriptions')
    if args.aspect=='MF':
        if not args.go_obo:
            save();raise ValueError('MF-only experiments require --go_obo. Use --aspect all explicitly for development.')
        terms=ontology_terms(args.go_obo)
        unknown=~frame.matched_go_id.isin(terms)
        report['unknown_go_rows']=int(unknown.sum());report['unknown_go_ids']=sorted(frame.loc[unknown,'matched_go_id'].unique())
        if unknown.any() and args.unknown_go=='error':
            save();raise ValueError('GO IDs missing from ontology; see data_audit.json. Check snapshot or explicitly use --unknown_go exclude.')
        keep(frame.matched_go_id.map(lambda g: terms.get(g,(None,False))==('molecular_function',False)),
             'retain nonobsolete Molecular Function terms')
    if not len(frame):save();raise ValueError('No rows after data selection')
    report['before_embedding_check']={s:int(frame.split.eq(s).sum()) for s in ('train','val','test')}
    frame['wt_global_key']=frame.protein_id
    frame['wt_site_key']=frame.wt_local_key
    status_columns=[]
    for mode,path,columns in (('global',cfg.global_h5,['wt_global_key','variant_key']),
                              ('local',cfg.local_h5,['wt_site_key','variant_key'])):
        if args.input_mode!='both' and args.input_mode!=mode:continue
        required=[k for column in columns for k in frame[column]]
        states,meta=audit_h5(path,required)
        report[f'{mode}_h5']=meta
        for column in columns:
            frame[f'{mode}_{column}_status']=frame[column].map(states)
            status_columns.append(f'{mode}_{column}_status')
    table=GOEmbeddingTable(cfg.go_embeddings)
    frame['go_status']=frame.matched_go_id.map(lambda g:'missing_key' if g not in table.go_to_row
        else 'zero_vector' if not table.vectors[table.go_to_row[g]].any() else 'valid')
    status_columns.append('go_status')
    covered=frame[status_columns].eq('valid').all(axis=1)
    gaps=frame.loc[~covered]
    gaps.to_csv(root/'missing_inputs.tsv',sep='\t',index=False)
    report['rows_with_missing_or_invalid_input']=len(gaps)
    report['missing_by_split']={s:int(gaps.split.eq(s).sum()) for s in ('train','val','test')}
    report['go_artifact_sha256']=table.artifact_sha256
    save()
    if len(gaps) and not args.allow_missing_embeddings:
        raise ValueError(f'{family}: {len(gaps):,} rows have missing/invalid inputs. See {root}/missing_inputs.tsv. '
                         'Complete embeddings or explicitly use --allow_missing_embeddings for a covered-subset experiment.')
    keep(covered,'explicitly selected input-complete subset' if len(gaps) else 'input coverage')
    frame=attach_indices(frame.reset_index(drop=True),build_go_vocab(frame))
    for split in ('train','val','test'):
        part=frame[frame.split==split]
        if not len(part):save();raise ValueError(f'Empty {split} partition')
    if set(frame.loc[frame.split=='train','label'])!={'LOF','GOF','NEUTRAL'}:
        save();raise ValueError('Training requires all three classes')
    report['split_rows']={s:int(frame.split.eq(s).sum()) for s in ('train','val','test')}
    report['label_counts']={s:frame.loc[frame.split==s,'label'].value_counts().to_dict() for s in report['split_rows']}
    report['retained_combinatorial_rows']=int(frame.is_combinatorial.sum())
    identity=['split','protein_id','snp','matched_go_id','label','augmented']
    report['filtered_rows_sha256']=hashlib.sha256(frame[identity].to_csv(index=False).encode()).hexdigest()
    report['sources']={str(p):sha256(p) for p in sorted(data_root.glob(f'*{family}_*.tsv'))}
    report['sequences_sha256']=sha256(cfg.sequences_json)
    frame[identity].to_csv(root/'selected_rows.tsv',sep='\t',index=False)
    # No assembled NPY cache: inspect actual H5 vectors and avoid stale-cache reuse.
    tensors=build_pair_tensors(frame,cfg,cache_dir=None)
    tensors.attach_go_embeddings(cfg.go_embeddings)
    report['feature_tensors_sha256']={name:hashlib.sha256(memoryview(
        np.ascontiguousarray(value.detach().cpu().numpy())).cast('B')).hexdigest()
        for name,value in (('WT',tensors.wt_feat),('VAR',tensors.var_feat),('GO',tensors.go_features))}
    views=split_views(tensors)
    normalization=fit_normalization(tensors,views['train'])  # CPU float64; safe before MPS/CUDA transfer.
    save()
    return tensors,normalization,report
