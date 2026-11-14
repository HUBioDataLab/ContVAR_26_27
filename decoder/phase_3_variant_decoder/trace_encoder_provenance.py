"""Trace encoder sequence sources using commands, configs, logs and raw ID tables.

Read-only; no GPU, torch, pickle loading, training, downloads or ID replacement.
Code defaults/config declarations are explicitly distinguished from run evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

import pandas as pd

try:
    from .audit_sequence_lineage import open_text, sequence_records, read_primary, digest, sequence_hash, id_candidates
except ImportError:
    from audit_sequence_lineage import open_text, sequence_records, read_primary, digest, sequence_hash, id_candidates

KEYS = ('sequences_json','sequences_path','uniprot','goa','dataset_dir','go_dataset','dataset',
        'triplet_json','reference_csv','proteins_csv','backbone','checkpoint','global_h5','local_h5',
        'out_prefix','data_root','graph_root','esm2_embedding_path','go_tsv_dir','split_json','split_file')
KEY_PATTERN = '|'.join(sorted(KEYS,key=len,reverse=True))
ARG_PATTERN = re.compile(r'--('+KEY_PATTERN+r'|json|output)\s*(?:=|\s)\s*("[^"]+"|\x27[^\x27]+\x27|[^\s\\,\]}]+)')
RELEVANT = re.compile(r'uniprot|map_sequences|build_go_dataset|build_pair_embeddings|build_variant_embeddings|phase1_dms|train_go|sequences_json|triplet_data|target_seq|semantic_similarity|phase3_seq_phase1|Loading sequences|Loading GOA',re.I)
SKIP_DIRS = {'.git','.venv','__pycache__','node_modules','envs','media','sequence_lineage'}
SKIP_NAMES = {'audit_sequence_lineage.py','trace_encoder_provenance.py','test_sequence_lineage.py','test_encoder_provenance.py'}


def text_type(path):
    if path.name in ('.bash_history','.zsh_history'):
        return 'shell_history_not_execution_proof'
    if path.suffix.lower() in ('.log','.out','.err') or path.name=='output.log':
        return 'run_log'
    if path.name=='wandb-metadata.json':
        return 'recorded_run_metadata'
    if path.suffix.lower() in ('.py','.sh','.ipynb'):
        return 'source_code_not_execution_proof'
    return 'config_or_record_not_execution_proof'


def discover(roots,out,max_depth,max_files):
    """Do not spend the discovery budget on thousands of per-variant FASTAs."""
    metadata,sequence,goa,warnings = set(),set(),set(),[]
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.is_dir():
            warnings.append(f'Missing root: {root}')
            continue
        root_count=0
        def err(exc):
            warnings.append(str(exc))
        for current,dirs,names in os.walk(root,followlinks=False,onerror=err):
            here=Path(current);depth=len(here.relative_to(root).parts)
            dirs[:]=sorted(d for d in dirs if d not in SKIP_DIRS and not (here/d).is_symlink()
                          and (here/d).resolve()!=out and depth<max_depth)
            for name in sorted(names):
                path=here/name; low=name.lower(); stem=low.removesuffix('.gz')
                if name in SKIP_NAMES or not path.is_file():
                    continue
                is_goa=stem.endswith('.gaf') or ('goa' in low and stem.endswith(('.tsv','.txt')))
                is_sequence=(stem.endswith(('.json','.tsv','.csv','.fasta','.fa','.faa','.dat'))
                    and any(x in low for x in ('uniprot','swissprot','swiss_prot','reviewed','sequences','proteins 2')))
                is_meta=path.suffix.lower() in ('.py','.sh','.log','.out','.err','.yaml','.yml','.toml','.ipynb') or (
                    path.suffix.lower()=='.json' and any(x in low for x in ('config','metadata','summary','args','params','provenance')))
                if not (is_goa or is_sequence or is_meta):
                    continue
                root_count+=1
                if root_count>max_files:
                    warnings.append(f'Per-root candidate limit {max_files} reached: {root}; other roots are still scanned')
                    break
                path=path.resolve()
                if is_goa: goa.add(path)
                elif is_sequence: sequence.add(path)
                if is_meta: metadata.add(path)
            if root_count>max_files:
                break
    return sorted(metadata),sorted(sequence),sorted(goa),warnings


def clean_value(value):
    return str(value).strip().strip('"\x27').rstrip(',)')


def inspect_evidence(path,max_bytes):
    rows=[];kind=text_type(path)
    def emit(line,key,value,context,origin=None):
        value=clean_value(value)
        if not value or re.search(r'api.?key|password|authorization|bearer\s',str(context),re.I):
            return
        rows.append({'source_file':str(path),'line_or_json_pointer':str(line),'evidence_kind':origin or kind,
                     'parameter':key,'value':value,'context':str(context).strip()[:700],
                     'proves_target_h5_provenance':False})
    if path.stat().st_size>max_bytes:
        return [],{'path':str(path),'status':'skipped_size_limit','bytes':path.stat().st_size}
    try:
        text=path.read_text(encoding='utf-8',errors='replace')
        if path.suffix=='.json':
            raw=json.loads(text)
            def walk(obj,pointer=''):
                if isinstance(obj,dict):
                    for k,v in obj.items():
                        ptr=pointer+'/'+str(k)
                        if k=='args' and isinstance(v,list):
                            command=shlex.join(str(x) for x in v)
                            for m in ARG_PATTERN.finditer(command):
                                emit(ptr,m[1],m[2],command)
                        elif k in ('program','root','cwd','codePath') and isinstance(v,str):
                            emit(ptr,str(k),v,f'{ptr}={v}')
                        elif isinstance(v,(str,int,float)) and any(re.search(r'(^|[./_])'+re.escape(k2)+r'($|[./_])',ptr,re.I) for k2 in KEYS):
                            emit(ptr,str(k),v,f'{ptr}={v}')
                        else:
                            walk(v,ptr)
                elif isinstance(obj,list):
                    for i,v in enumerate(obj): walk(v,pointer+'/'+str(i))
            walk(raw)
        if path.suffix=='.ipynb':
            raw=json.loads(text)
            text='\n'.join(''.join(cell.get('source',[])) for cell in raw.get('cells',[]) if cell.get('cell_type')=='code')
        yaml_parent=None
        for n,line in enumerate(text.splitlines(),1):
            if re.search(r'^\s*('+KEY_PATTERN+r')\s*:\s*$',line):
                yaml_parent=line.strip().split(':')[0]
            value=re.match(r'^\s*value:\s*(.+)$',line)
            if value and yaml_parent:
                emit(n,yaml_parent,value[1],line)
            if not (RELEVANT.search(line) or re.search(r'how\s*=\s*[\x27"]inner[\x27"]',line)):
                continue
            for m in ARG_PATTERN.finditer(line): emit(n,m[1],m[2],line)
            # Map-source logs emitted by the existing preprocessing code.
            m=re.search(r'(Loading UniProt data from|Loading GOA data from|Loading sequences from)\s*:\s*(.+)',line,re.I)
            if m and kind in ('run_log','recorded_run_metadata'):
                emit(n,m[1],m[2],line,'runtime_loading_message')
            if any(x in line for x in ('how=\'inner\'','how="inner"','map_sequences.py','uniprot_dict.json','target_seq')):
                emit(n,'recipe_or_reference',line.strip(),line)
        return rows,{'path':str(path),'status':'ok','bytes':path.stat().st_size,'evidence_rows':len(rows)}
    except (OSError,ValueError,UnicodeError) as exc:
        return rows,{'path':str(path),'status':'unreadable','error':str(exc)}


def source_ids(path,wanted,is_goa):
    """Return membership, plus candidate sequences; metadata-only TSVs stay metadata."""
    ids=set();candidates=[]; info={'path':str(path),'status':'ok','rows':0,'kind':'goa' if is_goa else 'unknown'}
    try:
        low=path.name.lower().removesuffix('.gz')
        if is_goa:
            with open_text(path) as f:
                for line in f:
                    if not line.strip() or line.startswith('!') or line.startswith('#'):
                        continue
                    cols=line.rstrip('\n').split('\t')
                    if len(cols)<2: continue
                    if cols[1] in ('Entry','DB_Object_ID','protein_id'): continue
                    if cols[0] not in ('UniProtKB','UniProt'):
                        # Do not interpret a different table's second column as an accession.
                        continue
                    info['rows']+=1;ids.add(cols[1])
            if not ids:
                raise ValueError('No recognizable UniProtKB GOA/GAF records; membership is unknown')
        elif low.endswith(('.tsv','.csv','.tab')):
            with open_text(path) as f:
                reader=csv.DictReader(f,delimiter=',' if low.endswith('.csv') else '\t')
                headers=reader.fieldnames or []
                id_col=next((x for x in ('Entry','protein_id','accession') if x in headers),None)
                seq_col=next((x for x in ('Sequence','sequence','seq') if x in headers),None)
                info['columns']=headers
                if not id_col:
                    if 'File_Name' in headers and seq_col:
                        info['kind']='dms_filename_sequence_table_wt_rows_only'
                        for row in reader:
                            info['rows']+=1
                            name=row['File_Name']
                            if not name.upper().endswith('_WT'):
                                continue
                            for pid,_ in id_candidates(name):
                                if pid not in wanted: continue
                                ids.add(pid)
                                seq=row[seq_col]
                                if seq:
                                    candidates.append({'protein_id':pid,'source_path':str(path),'source_id':name,'length':len(seq),
                                        'sequence_sha256':sequence_hash(seq),'sequence':seq,'accepted_for_encoding':False})
                        info.update(unique_ids=len(ids),requested_ids_present=len(ids&wanted))
                        return ids,candidates,info
                    raise ValueError('No direct accession column; unsupported sequence/metadata table')
                info['kind']='sequence_table' if seq_col else 'uniprot_metadata_only_no_sequence_column'
                for row in reader:
                    pid=row[id_col];info['rows']+=1;ids.add(pid)
                    if pid in wanted and seq_col and row[seq_col]:
                        seq=row[seq_col]
                        candidates.append({'protein_id':pid,'source_path':str(path),'source_id':pid,'length':len(seq),
                            'sequence_sha256':sequence_hash(seq),'sequence':seq,'accepted_for_encoding':False})
        else:
            info['kind']='sequence_export'
            for pid,seq,_ in sequence_records(path):
                info['rows']+=1
                parts=pid.split('|')
                canonical=parts[1] if len(parts)>=3 and parts[0] in ('sp','tr') else pid
                ids.add(canonical)
                if canonical in wanted and seq:
                    candidates.append({'protein_id':canonical,'source_path':str(path),'source_id':pid,'length':len(seq),
                        'sequence_sha256':sequence_hash(seq),'sequence':seq,'accepted_for_encoding':False})
        info.update(unique_ids=len(ids),requested_ids_present=len(ids&wanted))
    except (OSError,ValueError,TypeError,KeyError,csv.Error) as exc:
        info.update(status='unreadable_or_unsupported',error=str(exc))
        # Partial parsing cannot establish absence.
    return ids,candidates,info


def run(args):
    out=Path(args.out_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    primary=Path(args.primary_sequences).expanduser().resolve()
    primary_sha=digest(primary)
    sequences,primary_meta=read_primary(primary)
    missing_path=Path(args.missing_ids).expanduser().resolve()
    missing_sha=digest(missing_path)
    missing=pd.read_csv(missing_path,sep='\t',keep_default_na=False)
    if 'protein_id' not in missing: raise ValueError('--missing_ids requires a protein_id column')
    wanted=set(missing.protein_id)
    meta,seq_files,goa_files,warnings=discover(args.search_root,out,args.max_depth,args.max_files_per_root)
    meta=sorted(set(meta)|{Path(p).expanduser().resolve() for p in args.config_file+args.log_file+args.history_file})
    seq_files=sorted(set(seq_files)|{Path(p).expanduser().resolve() for p in args.uniprot_source})
    goa_files=sorted(set(goa_files)|{Path(p).expanduser().resolve() for p in args.goa_source})
    evidence,meta_inventory=[],[]
    for p in meta:
        if p.is_file():
            rows,info=inspect_evidence(p,args.max_text_bytes);evidence.extend(rows);meta_inventory.append(info)
        else: warnings.append(f'Explicit evidence file missing: {p}')
    for e in evidence:
        value=e['value'];existing=[]
        if '\n' not in value and len(value)<4096 and not re.search(r'[${}]',value):
            path=Path(value).expanduser()
            options=[path] if path.is_absolute() else [Path(e['source_file']).parent/path,*[Path(root).expanduser()/path for root in args.search_root]]
            for candidate in options:
                try:
                    if candidate.is_file(): existing.append(str(candidate.resolve()))
                except OSError: pass
        e['existing_path_candidates']='|'.join(sorted(set(existing)))
    # Evidence may name an existing raw input outside the discovery roots.
    for e in evidence:
        value=e['value']
        for filename in e['existing_path_candidates'].split('|'):
            if not filename: continue
            low=Path(filename).name.lower()
            if e['parameter'] in ('uniprot','sequences_json','sequences_path') and low.endswith(('.tsv','.csv','.json','.fasta','.gz')):
                seq_files.append(Path(filename))
            elif e['parameter'] in ('goa','goa_tsv'):
                goa_files.append(Path(filename))
    explicit_seq={Path(p).expanduser().resolve() for p in args.uniprot_source}|{primary}
    explicit_goa={Path(p).expanduser().resolve() for p in args.goa_source}
    def cap(paths,explicit,label):
        automatic=sorted(set(paths)-explicit)
        if len(automatic)>args.max_source_files:
            warnings.append(f'{label} source limit {args.max_source_files} reached; explicit files are still included')
        return sorted(explicit|set(automatic[:args.max_source_files]))
    seq_files=cap(seq_files,explicit_seq,'sequence');goa_files=cap(goa_files,explicit_goa,'GOA')
    output_names=('encoder_evidence.tsv','evidence_file_inventory.tsv','raw_source_inventory.tsv',
        'source_membership.tsv','protein_trace.tsv','source_sequence_candidates.tsv','summary.json','REPORT_TR.md')
    protected={primary,missing_path,*meta,*seq_files,*goa_files,*[Path(p).expanduser().resolve() for p in args.h5]}
    if any((out/name).resolve() in protected for name in output_names):
        raise ValueError('Report path overlaps an input; choose a separate --out_dir')
    inventory,membership,candidates=[],[],[]
    seq_present=defaultdict(list);meta_present=defaultdict(list);goa_present=defaultdict(list)
    successful_seq,successful_goa=[],[]
    for is_goa,paths in ((False,seq_files),(True,goa_files)):
        for path in paths:
            ids,found,info=source_ids(path,wanted,is_goa);inventory.append(info);candidates.extend(found)
            if info['status']=='ok':
                (successful_goa if is_goa else successful_seq).append(str(path))
            for pid in sorted(wanted):
                state='present' if pid in ids else 'absent' if info['status']=='ok' else 'unknown'
                membership.append({'protein_id':pid,'source_path':str(path),'source_kind':info['kind'],'membership':state})
                if state=='present':
                    target=goa_present if is_goa else meta_present if info['kind']=='uniprot_metadata_only_no_sequence_column' else seq_present
                    target[pid].append(str(path))
            print(f"[SOURCE] {path.name}: {info['status']}; {info['kind']}; missing-ID matches={len(ids&wanted):,}",flush=True)
    traces=[]
    for pid in sorted(wanted):
        in_primary=pid in sequences
        hypothesis=('not_missing_in_this_primary' if in_primary else 'sequence_found_in_other_source' if seq_present[pid]
            else 'metadata_ID_found_but_sequence_not_recovered' if meta_present[pid] else 'not_found_in_scanned_sources')
        traces.append({'protein_id':pid,'in_primary_sequence_file':in_primary,'sequence_source_count':len(seq_present[pid]),
            'metadata_only_source_count':len(meta_present[pid]),'goa_source_count':len(goa_present[pid]),
            'sequence_sources':'|'.join(seq_present[pid]),'metadata_sources':'|'.join(meta_present[pid]),
            'goa_sources':'|'.join(goa_present[pid]),'status':hypothesis,
            'inner_join_exclusion_possible':bool(seq_present[pid]) and bool(successful_goa) and not bool(goa_present[pid]),
            'historical_input_pair_confirmed':False})
    h5_info=[]
    for name in args.h5:
        path=Path(name).expanduser().resolve();item={'path':str(path),'exists':path.is_file()}
        if path.is_file():
            item.update(bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            try:
                import h5py
                with h5py.File(path,'r') as h:
                    item.update(root_keys=len(h),attributes={str(k):str(v)[:2000] for k,v in h.attrs.items()})
            except (OSError,ImportError,ValueError) as exc: item['error']=str(exc)
        h5_info.append(item)
    if digest(primary)!=primary_sha or digest(missing_path)!=missing_sha:
        raise ValueError('Primary sequences or missing-ID list changed during the audit; rerun with stable inputs')
    evidence_columns=['source_file','line_or_json_pointer','evidence_kind','parameter','value','existing_path_candidates','context','proves_target_h5_provenance']
    pd.DataFrame(evidence,columns=evidence_columns).drop_duplicates().to_csv(out/'encoder_evidence.tsv',sep='\t',index=False)
    pd.DataFrame(meta_inventory).to_csv(out/'evidence_file_inventory.tsv',sep='\t',index=False)
    pd.DataFrame(inventory).to_csv(out/'raw_source_inventory.tsv',sep='\t',index=False)
    pd.DataFrame(membership).to_csv(out/'source_membership.tsv',sep='\t',index=False)
    pd.DataFrame(traces).to_csv(out/'protein_trace.tsv',sep='\t',index=False)
    pd.DataFrame(candidates,columns=['protein_id','source_path','source_id','length','sequence_sha256','sequence','accepted_for_encoding']).drop_duplicates().to_csv(out/'source_sequence_candidates.tsv',sep='\t',index=False)
    counts={'requested_proteins':len(wanted),'present_in_current_primary':sum(p in sequences for p in wanted),
        'found_in_other_sequence_sources':sum(bool(seq_present[p]) for p in wanted if p not in sequences),
        'found_only_as_metadata_id':sum(bool(meta_present[p]) and not bool(seq_present[p]) for p in wanted if p not in sequences),
        'possible_GOA_inner_join_exclusions':sum(t['inner_join_exclusion_possible'] for t in traces),
        'evidence_records':len(evidence)}
    summary={'format':'contvar_encoder_provenance_trace_v1','counts':counts,
        'primary_sequence_file':{'path':str(primary),'sha256':primary_sha,'unique_nonempty_ids':len(sequences)},
        'missing_ID_file':{'path':str(missing_path),'sha256':missing_sha},'h5_metadata':h5_info,
        'evidence_kind_counts':dict(Counter(e['evidence_kind'] for e in evidence)),
        'source_status_counts':dict(Counter(i['status'] for i in inventory)),'warnings':warnings,
        'limits':{'max_depth':args.max_depth,'max_files_per_root':args.max_files_per_root,
                  'max_source_files':args.max_source_files,'max_text_bytes':args.max_text_bytes},
        'interpretation':['A code/config path is not proof that a historical run used it.',
            'Source loading logs and recorded run arguments are stronger evidence; linking them to the target H5 still needs run/output correlation.',
            'map_sequences.py inner-joins GOA Entry IDs with an Entry/Sequence table; which historical files were used is not assumed.',
            'UniProt metadata tables without Sequence cannot regenerate amino-acid sequences.',
            'GOA absence is only relative to inspected snapshots; it does not establish absence from all GO annotations.',
            'No checkpoint/pickle tensors were loaded, no model ran, no source file was changed, no candidate sequence was accepted automatically.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# Encoder kaynak izleme raporu','',
        'Kod varsayımları, config tanımları ve gerçek çalışma kayıtları ayrı tutulur. Bu kontrol geçmişte kaydedilmemiş bilgileri kesinmiş gibi tamamlamaz.','',
        '| Kontrol | Sayı |','|---|---:|',*[f'| {k} | {v:,} |' for k,v in counts.items()],'',
        '## Önce bakılacak dosyalar','',
        '- `encoder_evidence.tsv`: kaynak dosya/satır, parametre ve kanıt türü. `runtime_loading_message` ve `recorded_run_metadata` öncelikli incelenir.',
        '- `protein_trace.tsv`: eksik protein ham sekans kaynağında mı, yalnız metadata tablosunda mı, GOA listesinde mi?',
        '- `source_sequence_candidates.tsv`: başka kaynakta bulunan aday sekanslar; henüz encoding için onaylanmış değiller.',
        '- `raw_source_inventory.tsv`: hangi tabloda Sequence kolonu var, hangisi sadece ID/metadata?',
        '- `source_membership.tsv`: her protein için kaynak bazında present/absent/unknown.', '',
        '## Arama sınırları','',*[f'- {x}' for x in warnings or ['Belirtilen köklerde dosya sayısı limitine ulaşıldığına dair uyarı yok.']], '',
        'Aday kaynaklar geçmişte kullanılan dosyalarla aynı sürüm olmayabilir. Inner join açıklaması olasıdır; geçmiş komut ve girdi çifti doğrulanmadan kesin neden sayılmaz.']
    (out/'REPORT_TR.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('[SUMMARY] '+json.dumps(counts),flush=True)
    print(f'[SAVED] {out}')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary_sequences',required=True)
    p.add_argument('--missing_ids',required=True,help='TSV with protein_id; e.g. missing_wt_sequences.tsv')
    p.add_argument('--search_root',action='append',default=[])
    p.add_argument('--config_file',action='append',default=[])
    p.add_argument('--log_file',action='append',default=[])
    p.add_argument('--history_file',action='append',default=[],help='Only encoder-related evidence is emitted; history is not proof of completion')
    p.add_argument('--uniprot_source',action='append',default=[])
    p.add_argument('--goa_source',action='append',default=[])
    p.add_argument('--h5',action='append',default=[])
    p.add_argument('--out_dir',required=True)
    p.add_argument('--max_depth',type=int,default=8)
    p.add_argument('--max_files_per_root',type=int,default=6000)
    p.add_argument('--max_source_files',type=int,default=80)
    p.add_argument('--max_text_bytes',type=int,default=16*1024*1024)
    args=p.parse_args()
    for name in ('primary_sequences','missing_ids','out_dir'):
        if not getattr(args,name).strip(): p.error(f'--{name} must be nonempty')
    for name in ('max_depth','max_files_per_root','max_source_files','max_text_bytes'):
        if getattr(args,name)<(0 if name=='max_depth' else 1): p.error(f'Invalid --{name}')
    try: raise SystemExit(run(args))
    except (OSError,ValueError,KeyError) as exc: p.exit(2,f'[ERROR] {exc}\n')


if __name__=='__main__':
    main()
