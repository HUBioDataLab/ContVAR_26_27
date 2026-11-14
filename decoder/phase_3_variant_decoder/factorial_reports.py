"""Separate, ordered W&B sections and local reports for the four decoder arms."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from phase_3_variant_decoder.factorial_models import ARMS,FORMAT,DECISION
from phase_3_variant_decoder.outcome_tables import _table

OVERVIEW='00_Four_Decoder_Comparison'
METRICS=('macro_f1','accuracy','lof_f1','gof_f1','neutral_f1','gof_recall')


def score_rows(report):
    rows=[]
    for result in report['results']:
        for split,metric in result['metrics'].items():
            rows.append({'family':result['family'].upper(),'seed':result['seed'],'arm':result['arm'],
                'model':result['title'],'split':split,'parameters':result['parameter_count'],
                'independent_networks':result['independent_network_count'],
                **{k:v for k,v in metric.items() if k!='confusion'}})
    return rows


def aggregates(scores):
    result=[]
    for (family,arm,split),g in pd.DataFrame(scores).groupby(['family','arm','split'],sort=False):
        row={'family':family,'arm':arm,'model':ARMS[arm]['title'],'split':split,'seeds':len(g),
             'n':int(g.n.iloc[0]),'parameters':int(g.parameters.iloc[0])}
        for metric in METRICS:
            row[metric+'_mean']=float(g[metric].mean())
            row[metric+'_seed_sd']=float(g[metric].std(ddof=1)) if len(g)>1 else None
        result.append(row)
    return result


def make_reports(report,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);scores=score_rows(report);means=aggregates(scores)
    marker='TECHNICAL SMOKE TEST — ' if report['config'].get('technical_smoke') else ''
    pd.DataFrame(scores).to_csv(root/'scores.csv',index=False)
    pd.DataFrame(means).to_csv(root/'seed_summary.csv',index=False)
    artifacts={OVERVIEW:{}}
    def register(section,key,path):
        artifacts.setdefault(section,{})[key]=str(Path(path).relative_to(root))
    path=root/'overview.png'
    visible=[r for r in means if r['split'] in ('val','test')]
    _table(['Family','Model','Partition','Samples','Seeds','Macro-F1','Accuracy','GOF recall'],
        [[r['family'],r['model'].replace(' + ','\n'),r['split'].upper(),f"{r['n']:,}",r['seeds'],
          f"{r['macro_f1_mean']:.4f}",f"{r['accuracy_mean']:.4f}",f"{r['gof_recall_mean']:.4f}"] for r in visible],
        path,marker+'Four decoder experiments — matched frozen inputs',
        'Her family içinde dört model aynı satırlarda eğitilir ve ölçülür. Çoklu seed varsa değerler seed ortalamasıdır. Checkpoint seçimi yalnız validation ile yapılır. Paralel çıktılar sabit OvR kuralıyla birleştirilir.',
        widths=[.07,.29,.10,.10,.06,.13,.12,.13])
    register(OVERVIEW,'01_Results',path)
    for family in sorted({r['family'] for r in means}):
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        labels=[]
        for i,(arm,spec) in enumerate(ARMS.items()):
            labels.append(spec['title'].replace(' + ','\n'))
            selected=next(r for r in means if r['family']==family and r['arm']==arm and r['split']=='test')
            for j,metric in enumerate(('macro_f1','gof_recall')):
                bar=axes[j].bar(i,selected[metric+'_mean'],yerr=selected[metric+'_seed_sd'] or 0.,capsize=4)
                axes[j].bar_label(bar,fmt='%.3f',padding=4)
        for ax,title in zip(axes,('Test macro-F1','Test GOF recall')):
            ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1.12),ylabel='Score (0–1)',title=title)
            ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
        fig.suptitle(marker+f'{family} — four decoder configurations')
        fig.text(.5,-.04,'Aynı örnekler kullanılır. Hata çubukları varsa seed SD; tek seed için belirsizlik tahmini değildir.',ha='center',fontsize=8)
        path=root/f'{family.lower()}_comparison.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
        register(OVERVIEW,f'02_{family}_Comparison',path)
    for result in report['results']:
        section=result['section'];prefix=f"{result['family']}_s{result['seed']}"
        path=root/f'{prefix}_{result["arm"]}_counts.png'
        metric=result['metrics']['test'];cm=np.array(metric['confusion'])
        _table(['True class','Predicted LOF','Predicted GOF','Predicted NEUTRAL','Support'],
            [[label,*map(int,cm[i]),int(cm[i].sum())] for i,label in enumerate(('LOF','GOF','NEUTRAL'))],
            path,marker+f"{result['family'].upper()} / test — {result['title']} — seed {result['seed']}",
            'Her hücre örnek sayısıdır. Satır gerçek etiketi, sütun tahmini gösterir. Yüzde veya satır-normalizasyon kullanılmaz.',
            widths=[.20,.19,.19,.24,.18])
        register(section,prefix+'_Prediction_Counts',path)
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for head,history in result['histories'].items():
            epochs=[r['epoch'] for r in history]
            for ax,train_key,val_key in ((axes[0],'train_macro_f1','val_macro_f1'),(axes[1],'train_eval_loss','val_loss')):
                ax.plot(epochs,[r[train_key] for r in history],label=head+' / train',linestyle='--')
                ax.plot(epochs,[r[val_key] for r in history],label=head+' / validation')
        axes[0].set(title='Binary macro-F1 (each head vs rest)' if result['strategy']=='parallel' else 'Three-class macro-F1',ylabel='Macro-F1',ylim=(0,1.05))
        axes[1].set(title='Training and validation objective',ylabel='BCE loss' if result['strategy']=='parallel' else 'Cross-entropy loss')
        for ax in axes:ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(marker+f"{result['family'].upper()} — {result['title']} — seed {result['seed']}")
        caption='Paralel yapıda eğriler her bağımsız binary modelin macro-F1 değeridir; üç sınıflı F1 final sonuç tablosundadır.' if result['strategy']=='parallel' else 'Tek modelin üç sınıflı macro-F1 değeridir.'
        fig.text(.5,-.04,caption,ha='center',fontsize=8)
        path=root/f'{prefix}_{result["arm"]}_training.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
        register(section,prefix+'_Training',path)
    text=['# '+marker+'Four decoder experiments','',
        '| Configuration | Processing | Outputs |','|---|---|---|',
        *[f"| {s['title']} | {s['processing']} | {s['strategy']} |" for s in ARMS.values()], '',
        'All configurations use identical selected rows and fixed encoder vectors within a family.',
        'Parallel targets: LOF=(1,0), GOF=(0,1), NEUTRAL=(0,0). Both independent networks receive every example.',
        DECISION,
        'Separate scoring uses sigmoid(score_WT)-sigmoid(score_VAR), a learned positive scale initialized at 10, and a linear output classifier.',
        'WT/VAR weights and dropout masks are shared within a scorer. The two binary networks share no trainable parameters.',
        'Unweighted BCE/CE is the default; optional balanced weighting is recorded. No threshold fitting or cascade routing.',
        'Each independent binary checkpoint is selected by its own validation binary macro-F1. Three-class checkpoints use validation three-class macro-F1.',
        'The final common metric is three-class macro-F1. Loss values across objectives are not directly comparable.',
        'Score-difference designs have a scalar bottleneck and different parameter counts; this is not a subtraction-only ablation.',
        'GO-only controls and old cascade experiments are not run or modified by this entry point.', '',
        '## Data coverage','',json.dumps(report['data'],indent=2), '',
        '## Results','',pd.DataFrame(scores).to_csv(index=False)]
    (root/'comparison_report.md').write_text('\n'.join(text),encoding='utf-8')
    return artifacts


def start_wandb(args,report):
    import wandb
    config={**vars(args),'experiment':FORMAT,'test_used_for_selection':False,
            'data_coverage':{f:{k:a[k] for k in ('split_rows','filtered_rows_sha256','aspect','missing_policy')} for f,a in report['data'].items()}}
    run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,
        name=args.wandb_run_name or f'phase3-four-decoders-{args.family}',mode=args.wandb_mode,
        group='decoder-2x2',job_type='decoder-four-experiments',tags=['phase3','decoder-2x2',args.aspect],config=config)
    run.config.update({'four_decoder_comparison_id':run.id})
    return run,wandb


def epoch_logger(run,section,family,seed,head):
    prefix=f'{section}/{family}/s{seed}/{head}/Training'
    run.define_metric(prefix+'/epoch')
    run.define_metric(prefix+'/*',step_metric=prefix+'/epoch')
    def log(row):run.log({prefix+'/'+key:value for key,value in row.items()})
    return log


def create_workspace(run,artifacts):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    sections=[]
    for section in (OVERVIEW,*[s['section'] for s in ARMS.values()]):
        panels=[wr.MediaBrowser(title=key.replace('_',' '),media_keys=[f'{section}/{key}']) for key in sorted(artifacts.get(section,{}))]
        sections.append(ws.Section(name=section,panels=panels,is_open=True))
    return ws.Workspace(entity=run.entity,project=run.project,name=f'Four decoders - {run.id}',sections=sections,
        runset_settings=ws.RunsetSettings(filters=f"Config('four_decoder_comparison_id') == '{run.id}'"),
        auto_generate_panels=False).save().url


def publish(run,wandb,report,root,workspace=False,checkpoints=False):
    root=Path(root);rows=score_rows(report);means=aggregates(rows)
    payload={f'{OVERVIEW}/00_Scores':wandb.Table(dataframe=pd.DataFrame(rows)),
             f'{OVERVIEW}/00_Seed_Summary':wandb.Table(dataframe=pd.DataFrame(means)),
             f'{OVERVIEW}/00_Protocol':wandb.Table(columns=['Item','Definition'],data=[
                 ['Parallel',DECISION],['Input processing','Joint Concat or separate shared-weight sigmoid scores followed by signed difference'],
                 ['Selection','Validation only; independent binary head checkpoint selection'],
                 ['Scores','Uncalibrated class decision scores; parallel softmax is not three-class likelihood training'],
                 ['Class weighting',report['config']['class_weighting']]])}
    for section,images in report['artifacts'].items():
        for key,relative in images.items():payload[f'{section}/{key}']=wandb.Image(str(root/relative))
    for arm,spec in ARMS.items():
        payload[spec['section']+'/00_Scores']=wandb.Table(dataframe=pd.DataFrame([r for r in rows if r['arm']==arm]))
    run.log(payload)
    for result in report['results']:
        prefix=f"{result['section']}/{result['family']}/s{result['seed']}"
        run.summary[prefix+'/parameter_count']=result['parameter_count']
        for split,metric in result['metrics'].items():
            for key,value in metric.items():
                if isinstance(value,(float,int)):run.summary[f'{prefix}/{split}/{key}']=value
    for row in means:
        for key,value in row.items():
            if isinstance(value,(float,int)):
                run.summary[f"{OVERVIEW}/{row['family']}/{row['arm']}/{row['split']}/{key}"]=value
    if workspace:
        try:report['workspace_url']=create_workspace(run,report['artifacts'])
        except Exception as exc:
            report['workspace_error']=str(exc)
            print(f'[WARN] Ordered W&B workspace could not be saved: {exc}; results remain logged.',flush=True)
    # Persist final workspace info before logging the result artifact.
    manifest=root/'four_decoder_comparison.json'
    manifest.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    artifact=wandb.Artifact(f'four-decoders-{run.id}',type='decoder-comparison')
    for name in ('four_decoder_comparison.json','scores.csv','seed_summary.csv','comparison_report.md'):
        artifact.add_file(str(root/name),name=name)
    for images in report['artifacts'].values():
        for relative in images.values():artifact.add_file(str(root/relative),name=relative)
    if checkpoints:
        for result in report['results']:
            for relative in result['checkpoints']:
                path=(root/relative).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('Missing/external checkpoint artifact')
                artifact.add_file(str(path),name=relative)
    run.log_artifact(artifact)


def upload_saved(args):
    from types import SimpleNamespace
    source=Path(args.upload_existing).expanduser().resolve();root=source.parent
    report=json.loads(source.read_text())
    if report.get('format')!=FORMAT or not report.get('artifacts'):raise ValueError('Expected a completed four_decoder_comparison.json')
    for images in report['artifacts'].values():
        for relative in images.values():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError(f'Missing/external report image: {relative}')
    config={**report['config'],'wandb_project':args.wandb_project,'wandb_entity':args.wandb_entity,
            'wandb_run_name':args.wandb_run_name or report['config'].get('wandb_run_name'),'wandb_mode':args.wandb_mode}
    run,wandb=start_wandb(SimpleNamespace(**config),report)
    report['wandb_url']=run.url
    try:
        # Re-upload also restores all training histories, without retraining.
        for result in report['results']:
            for head,history in result['histories'].items():
                callback=epoch_logger(run,result['section'],result['family'],result['seed'],head)
                for row in history:callback(row)
        publish(run,wandb,report,root,args.create_workspace,args.upload_checkpoints)
    except Exception:
        run.finish(exit_code=1);raise
    run.finish();print('[WANDB]',run.url)
    return run.url
