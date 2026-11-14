"""Train four independent decoder configurations on identical frozen V4 inputs.

python -m phase_3_variant_decoder.compare_four_decoders --help
Parallel = two independent OvR networks, not effect/direction cascade routing.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from phase_2_decoder.config import pick_device
from phase_3_variant_decoder.config import LABELS,WANDB_PROJECT,WANDB_ENTITY
from phase_3_variant_decoder.data.pair_dataset import iterate_batches
from phase_3_variant_decoder.factorial_data import prepare_inputs,split_views
from phase_3_variant_decoder.factorial_models import ARMS,FORMAT,DECISION,PairDecisionModel,binary_targets,parallel_decisions,load_model
from phase_3_variant_decoder.input_controls import classification_metrics


def resolve_device(requested='auto'):
    """Auto: Apple MPS, then NVIDIA CUDA, then CPU; explicit requests fail clearly."""
    preferred=str(requested or 'auto').strip().lower()
    if preferred=='cuda':preferred='cuda:0'
    if preferred.startswith('cuda:'):
        index=preferred.split(':',1)[1]
        if not index.isdigit():raise ValueError(f'Invalid CUDA device: {requested}')
    return torch.device(pick_device(None if preferred=='auto' else preferred))


def gather(tensors,view,idx):
    idx=idx.to(tensors.wt_feat.device)
    return tensors.gather_with_go(view['wt_row'][idx],view['var_row'][idx],view['go_idx'][idx])


def binary_metrics(y,pred):
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    cm=np.bincount(y*2+pred,minlength=4).reshape(2,2)
    tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*tp,den,out=np.zeros(2,dtype=float),where=den>0)
    return {'macro_f1':float(f.mean()),'accuracy':float(tp.sum()/cm.sum()),'confusion':cm.tolist()}


def training_weights(labels,target_class,policy):
    local_labels=labels.detach().cpu()
    y=binary_targets(local_labels,target_class).long() if target_class is not None else local_labels
    counts=torch.bincount(y,minlength=2 if target_class is not None else 3).float()
    if not (counts>0).all():raise ValueError('Each training target class must have examples')
    weights=len(y)/(len(counts)*counts) if policy=='balanced' else torch.ones_like(counts)
    return counts,weights.to(labels.device)


def loss_rows(logits,labels,target_class,weights):
    if target_class is None:
        return F.cross_entropy(logits,labels,reduction='none')*weights[labels]
    y=binary_targets(labels,target_class)
    return F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights[y.long()]


@torch.no_grad()
def evaluate_network(model,tensors,view,target_class,weights,batch_size):
    model.eval();outputs=[];loss=0.
    for idx in iterate_batches(view['n'],batch_size,False):
        logits=model(*gather(tensors,view,idx))
        y=view['label_idx'][idx.to(logits.device)]
        values=loss_rows(logits,y,target_class,weights)
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite output/loss')
        outputs.append(logits.cpu());loss+=float(values.sum())
    logits=torch.cat(outputs)
    labels=view['label_idx'].cpu()
    metric=(classification_metrics(labels.numpy(),logits.argmax(1).numpy()) if target_class is None
        else binary_metrics(binary_targets(labels,target_class).numpy(),(logits>0).numpy()))
    return {**metric,'loss':loss/view['n']},logits


def fit_network(args,tensors,views,normalization,processing,target_class,seed,destination,provenance,on_epoch=None):
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=PairDecisionModel(processing,tensors.input_dim,tensors.go_features.shape[1],3 if target_class is None else 1,
        args.projection_dim,args.hidden_dims,args.dropout,normalization).to(tensors.wt_feat.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    counts,weights=training_weights(views['train']['label_idx'],target_class,args.class_weighting)
    generator=torch.Generator().manual_seed(seed)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/'decoder.pt';history=[];best=-1.;stale=0
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;start=time.perf_counter()
        for idx in iterate_batches(views['train']['n'],args.batch_size,True,generator):
            logits=model(*gather(tensors,views['train'],idx))
            labels=views['train']['label_idx'][idx.to(logits.device)]
            loss=loss_rows(logits,labels,target_class,weights).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(idx)
        train,_=evaluate_network(model,tensors,views['train'],target_class,weights,args.batch_size)
        val,_=evaluate_network(model,tensors,views['val'],target_class,weights,args.batch_size)
        row={'epoch':epoch,'train_loss':total/views['train']['n'],'train_eval_loss':train['loss'],
             'val_loss':val['loss'],'train_macro_f1':train['macro_f1'],'val_macro_f1':val['macro_f1'],
             'f1_gap':train['macro_f1']-val['macro_f1'],'epoch_seconds':time.perf_counter()-start}
        history.append(row)
        if val['macro_f1']>best:
            best=val['macro_f1'];stale=0
            torch.save({'format':FORMAT,'labels':list(LABELS),'model_spec':model.spec,
                'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'target_class':target_class,'seed':seed,'selection':{'epoch':epoch,'val_macro_f1':best,
                    'metric':'three-class macro-F1' if target_class is None else 'binary macro-F1'},
                'loss':'CrossEntropy' if target_class is None else 'BCEWithLogits',
                'class_weighting':args.class_weighting,'training_class_counts':counts.cpu(),
                'training_class_weights':weights.cpu(),'encoder_frozen':True,'test_used_for_selection':False,
                'input_provenance':provenance,'go_vocab':dict(tensors.df[['matched_go_id','go_idx']].drop_duplicates().itertuples(index=False,name=None)),
                'go_vectors':tensors.go_features.detach().cpu()},path)
        else:stale+=1
        if on_epoch:on_epoch(row)
        if not args.quiet:
            print(f"[TRAIN] {processing}/{('three_class' if target_class is None else LABELS[target_class])} s{seed} "
                  f"epoch={epoch} train_F1={train['macro_f1']:.4f} val_F1={val['macro_f1']:.4f}",flush=True)
        if stale>=args.early_stop_patience:break
    del model,optimizer
    model,checkpoint=load_model(path,tensors.wt_feat.device)
    (destination/'history.json').write_text(json.dumps(history,indent=2))
    return model,checkpoint,history,path


@torch.no_grad()
def evaluate_arm(models,tensors,view,strategy,batch_size):
    for model in models:model.eval()
    outputs=[]
    for idx in iterate_batches(view['n'],batch_size,False):
        inputs=gather(tensors,view,idx)
        if strategy=='parallel':logits=torch.stack([m(*inputs) for m in models],dim=1)
        else:logits=models[0](*inputs)
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite final decoder logits')
        outputs.append(logits.cpu())
    logits=torch.cat(outputs)
    if strategy=='parallel':pred,scores=parallel_decisions(logits)
    else:pred,scores=logits.argmax(1),logits.softmax(1)
    y=view['label_idx'].cpu().numpy();metric=classification_metrics(y,pred.numpy())
    metric.update({f'{name.lower()}_support':int((y==i).sum()) for i,name in enumerate(LABELS)})
    if strategy=='parallel':
        metric['both_positive_rate']=float((logits>0).all(1).float().mean())
        metric['both_nonpositive_rate']=float((logits<=0).all(1).float().mean())
    return metric,logits.numpy(),scores.numpy(),pred.numpy()


def predict_saved(checkpoints,wt,var,go,device='cpu'):
    """Saved independent LOF/GOF checkpoints are supplied in that order."""
    models=[];metadata=[]
    for path in checkpoints:
        model,ckpt=load_model(path,device);models.append(model);metadata.append(ckpt)
    if len(models)==2:
        if [x['target_class'] for x in metadata]!=[0,1]:raise ValueError('Expected LOF then GOF checkpoint')
        if metadata[0]['input_provenance']!=metadata[1]['input_provenance']:raise ValueError('Checkpoint input provenance differs')
        if metadata[0]['model_spec']!=metadata[1]['model_spec']:raise ValueError('Independent heads use different model configurations')
        if metadata[0]['go_vocab']!=metadata[1]['go_vocab'] or not torch.equal(metadata[0]['go_vectors'],metadata[1]['go_vectors']):
            raise ValueError('Independent heads use different GO vectors')
    elif len(models)!=1 or metadata[0]['target_class'] is not None:
        raise ValueError('Expected one three-class or two independent binary checkpoints')
    with torch.no_grad():
        inputs=(wt.to(device),var.to(device),go.to(device))
        if len(models)==2:
            logits=torch.stack([m(*inputs) for m in models],dim=1)
            pred,scores=parallel_decisions(logits)
        else:
            logits=models[0](*inputs);pred,scores=logits.argmax(1),logits.softmax(1)
    return {'prediction':pred.cpu(),'class_decision_scores':scores.cpu(),'raw_logits':logits.cpu()}


def run(args):
    from phase_3_variant_decoder.factorial_reports import make_reports,publish,start_wandb,epoch_logger
    device=resolve_device(args.device)
    print(f'[DEVICE] requested={args.device}; selected={device}',flush=True)
    if args.cpu_threads:torch.set_num_threads(args.cpu_threads)
    families=('easy','hard') if args.family=='both' else (args.family,)
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='four_decoders_',dir=Path(args.out_dir).resolve()))
    manifest=root/'four_decoder_comparison.json'
    report={'format':FORMAT,'config':vars(args),'runtime':{'device':str(device),'torch_version':str(torch.__version__)},
        'results':[],'data':{},'artifacts':{},'wandb_url':None,
        'protocol':{'test_used_for_selection':False,'encoder_frozen':True,'parallel_decision':DECISION,
          'parallel_models':'Two independent networks, optimizers, and validation-selected checkpoints',
          'scores':'Softmax class decision scores are uncalibrated. Parallel scores are derived from [LOF logit, GOF logit, 0], not three-class likelihood training.',
          'comparison_limit':'Score difference uses a scalar bottleneck; this is not subtraction-only or parameter-count-matched ablation.'}}
    def save():manifest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    prepared={}
    # Fail preflight before starting any training or W&B run.
    for family in families:
        tensors,norm,audit=prepare_inputs(args,family,root/family/'data')
        prepared[family]=(tensors,norm);report['data'][family]=audit
        print(f"[DATA] {family.upper()} {audit['split_rows']}; explicit subset={args.allow_missing_embeddings}; aspect={args.aspect}",flush=True)
    save()
    if args.preflight_only:
        report['preflight_only']=True;save();print(f'[PREFLIGHT] {manifest}');return report
    run_obj,wandb=start_wandb(args,report) if not args.no_wandb else (None,None)
    if run_obj is not None:report['wandb_url']=run_obj.url
    try:
        for family in families:
            tensors,norm=prepared.pop(family);tensors.to(device)
            views=split_views(tensors)
            audit=report['data'][family]
            provenance={k:audit[k] for k in ('filtered_rows_sha256','go_artifact_sha256','sequences_sha256','sources','feature_tensors_sha256')}
            provenance.update(input_mode=args.input_mode,embedding=args.embedding,
                global_h5=audit.get('global_h5'),local_h5=audit.get('local_h5'))
            for seed in args.seeds:
                # Every arm sees the exact same tensor/row mapping.
                for arm_key,spec in ARMS.items():
                    destination=root/family/f's{seed}'/arm_key
                    models=[];checkpoints=[];histories={};paths=[]
                    targets=(0,1) if spec['strategy']=='parallel' else (None,)
                    for target in targets:
                        name='three_class' if target is None else LABELS[target].lower()
                        callback=epoch_logger(run_obj,spec['section'],family,seed,name) if run_obj is not None else None
                        model,ckpt,history,path=fit_network(args,tensors,views,norm,spec['processing'],target,
                            seed+(target or 0),destination/name,provenance,callback)
                        models.append(model);checkpoints.append(ckpt);histories[name]=history;paths.append(path)
                    result={'family':family,'seed':seed,'arm':arm_key,**spec,
                        'parameter_count':sum(sum(p.numel() for p in m.parameters()) for m in models),
                        'independent_network_count':len(models),'checkpoints':[str(p.relative_to(root)) for p in paths],
                        'selection':{('three_class' if c['target_class'] is None else LABELS[c['target_class']].lower()):c['selection'] for c in checkpoints},
                        'histories':histories,'metrics':{},'filtered_rows_sha256':audit['filtered_rows_sha256']}
                    for split in ('train','val','test'):
                        metric,logits,scores,pred=evaluate_arm(models,tensors,views[split],spec['strategy'],args.batch_size)
                        result['metrics'][split]=metric
                        frame=tensors.df[tensors.df.split==split][['protein_id','snp','matched_go_id','label','augmented']].copy()
                        frame['prediction']=np.asarray(LABELS)[pred]
                        for i,label in enumerate(LABELS):frame[f'class_score_{label.lower()}']=scores[:,i]
                        if spec['strategy']=='parallel':
                            frame['lof_logit'],frame['gof_logit']=logits[:,0],logits[:,1]
                        frame.to_csv(destination/f'{split}_predictions.tsv',sep='\t',index=False)
                    report['results'].append(result);save()
                    print(f"[TEST] {family.upper()} / {spec['title']} / s{seed}: macro-F1={result['metrics']['test']['macro_f1']:.4f}",flush=True)
                    del models,checkpoints;gc.collect()
            del tensors,views;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        report['artifacts']=make_reports(report,root);save()
        if run_obj is not None:
            publish(run_obj,wandb,report,root,args.create_workspace,args.upload_checkpoints);save()
    except Exception:
        save()
        if run_obj is not None:run_obj.finish(exit_code=1)
        raise
    if run_obj is not None:run_obj.finish()
    print(f'[DONE] {manifest}')
    if report['wandb_url']:print(f"[WANDB] {report['wandb_url']}")
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset_dir');p.add_argument('--sequences_json');p.add_argument('--go_embeddings')
    p.add_argument('--global_h5');p.add_argument('--local_h5')
    p.add_argument('--family',choices=('easy','hard','both'),default='both')
    p.add_argument('--embedding',choices=('raw_esm','seq_phase0','seq_phase1'),default='seq_phase1')
    p.add_argument('--input_mode',choices=('both','global','local'),default='both')
    p.add_argument('--aspect',choices=('MF','all'),default='MF');p.add_argument('--go_obo')
    p.add_argument('--unknown_go',choices=('error','exclude'),default='error')
    p.add_argument('--exclude_combinatorial',action='store_true');p.add_argument('--drop_contradictory',action='store_true')
    p.add_argument('--allow_missing_embeddings',action='store_true')
    p.add_argument('--max_seq_len',type=int,default=1022)
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--early_stop_patience',type=int,default=15)
    p.add_argument('--batch_size',type=int,default=256);p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight_decay',type=float,default=.0001)
    p.add_argument('--dropout',type=float,default=.2);p.add_argument('--projection_dim',type=int,default=128)
    p.add_argument('--hidden_dims',type=int,nargs='+',default=[128,64])
    p.add_argument('--class_weighting',choices=('none','balanced'),default='none')
    p.add_argument('--device',default='auto',help='auto (MPS/CUDA/CPU), cpu, mps, cuda, or cuda:N')
    p.add_argument('--cuda',dest='device',help='Legacy NVIDIA device index, e.g. --cuda 6')
    p.add_argument('--cpu_threads',type=int);p.add_argument('--quiet',action='store_true')
    p.add_argument('--preflight_only',action='store_true')
    p.add_argument('--technical_smoke',action='store_true',help='Label reports as technical smoke tests, not benchmark results')
    p.add_argument('--out_dir',default='runs/four_decoders')
    p.add_argument('--no_wandb',action='store_true');p.add_argument('--wandb_mode',choices=('online','offline','disabled'),default='online')
    p.add_argument('--wandb_project',default=WANDB_PROJECT);p.add_argument('--wandb_entity',default=WANDB_ENTITY)
    p.add_argument('--wandb_run_name');p.add_argument('--create_workspace',action='store_true')
    p.add_argument('--upload_checkpoints',action='store_true');p.add_argument('--upload_existing')
    args=p.parse_args(argv)
    if args.create_workspace and (args.no_wandb or args.wandb_mode!='online'):
        p.error('--create_workspace requires online W&B')
    if args.upload_existing and args.no_wandb:p.error('--upload_existing requires W&B')
    if args.upload_existing:
        from phase_3_variant_decoder.factorial_reports import upload_saved
        return upload_saved(args)
    for name in ('dataset_dir','sequences_json','go_embeddings'):
        if not getattr(args,name) or not getattr(args,name).strip():p.error(f'Provide nonempty --{name}')
    for name in ('global_h5','local_h5'):
        needed=args.input_mode=='both' or args.input_mode==('global' if name=='global_h5' else 'local')
        if needed and (not getattr(args,name) or not getattr(args,name).strip()):p.error(f'Provide nonempty --{name}')
    for name in ('epochs','early_stop_patience','batch_size','projection_dim','max_seq_len'):
        if getattr(args,name)<1:p.error(f'--{name} must be positive')
    if len(set(args.seeds))!=len(args.seeds):p.error('Seeds must be unique')
    if not 0<=args.dropout<1 or not np.isfinite(args.lr) or args.lr<=0 or not np.isfinite(args.weight_decay) or args.weight_decay<0:
        p.error('Invalid optimizer/dropout setting')
    if args.cpu_threads is not None and args.cpu_threads<1:p.error('--cpu_threads must be positive')
    if not args.hidden_dims or any(x<1 for x in args.hidden_dims):p.error('Invalid --hidden_dims')
    return run(args)


if __name__=='__main__':main()
