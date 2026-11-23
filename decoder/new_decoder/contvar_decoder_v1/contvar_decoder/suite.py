"""Full four-model comparison on easy/hard with validation-only lambda selection.

Fit 6 distinct models per family: four baselines and lambda=.1 joint controls.
Score-difference lambda controls are mathematically redundant and not retrained.
All training completes before any test metrics are evaluated.
"""
import argparse
from copy import copy
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, PACKAGE, WORKSPACE
from .model import load_decoder
from .prepare import DEFAULT_OUT, sha256_file
from .train import evaluate, load_datasets, train, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,default=PACKAGE)
    p.add_argument("--window-pool",choices=("none","mean"),default="none")
    p.add_argument("--window-cache",type=Path)
    p.add_argument("--data-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--out",type=Path,default=WORKSPACE/"results/full_seed42")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--attention-heads",type=int,default=4)
    p.add_argument("--dropout",type=float,default=.1)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.out.exists():
        p.error("Choose a new --out directory")
    if min(args.epochs,args.patience,args.batch_size,args.threads)<1:
        p.error("Training limits must be positive")
    if args.smoke:
        args.epochs=1
    args.balance=False
    args.exclude_augmented_train=False
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    started=time.monotonic()
    manifest=json.loads((args.data_dir/"manifest.json").read_text())
    # Pin source files before the first optimizer step; refuse modified tables.
    for family in ("easy","hard"):
        for split in ("train","val","test"):
            for head in ("parallel","multiclass"):
                path=args.data_dir/head/f"{family}_{split}.tsv"
                if sha256_file(path)!=manifest["prepared"][f"{family}/{split}"]["sha256"][head]:
                    raise ValueError(f"Prepared table changed: {path}")
    plan=dict(smoke=args.smoke,seed=args.seed,max_epochs=args.epochs,patience=args.patience,
              source_layouts={"parallel":"parallel ZIP", "multiclass":"multiclass ZIP"},
              evaluation_augmented_excluded=True, multiple_label_policy="retain source labels and report groups",
              balance_classes=False,lambda_candidates=[0.,.1],
              model_selection="best validation macro-F1; ties prefer lambda=0",
              test_access="only after all 12 distinct model training runs complete",
              prepared_manifest_sha256=sha256_file(args.data_dir/"manifest.json"),
              source_code_sha256={p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")})
    write_json(args.out/"plan.json",plan)
    fitted={}
    for family in ("easy","hard"):
        store=FeatureStore(args.package,window_pool=args.window_pool,window_cache=args.window_cache)
        try:
            args.family=family
            ds={head:load_datasets(args,head,store) for head in ("parallel","multiclass")}
            store.preload([d for sets in ds.values() for d in sets.values()])
            fitted[family]={}
            for head in ("parallel","multiclass"):
                for mode in ("score_difference","joint"):
                    for lam in ([0.,.1] if mode=="joint" else [0.]):
                        cfg=copy(args);cfg.head=head;cfg.pair_mode=mode;cfg.lambda_reverse=lam
                        name=f"{head}_{mode}_lambda{lam:g}"
                        dest=args.out/family/name
                        result=train(cfg,head,mode,ds[head],torch.device(args.device),dest)
                        fitted[family][name]=result
                        write_json(args.out/"training_progress.json",fitted)
        finally:
            store.close()
    selection={}
    for family,models in fitted.items():
        selection[family]={}
        for head in ("parallel","multiclass"):
            for mode in ("score_difference","joint"):
                names=[f"{head}_{mode}_lambda{lam:g}" for lam in ([0.,.1] if mode=="joint" else [0.])]
                selection[family][f"{head}_{mode}"]=max(names,key=lambda n:models[n]["val"]["macro_f1"])
    write_json(args.out/"selection_before_test.json",selection)
    # The test dataset is loaded for the first time after selection is frozen.
    for family,models in fitted.items():
        args.family=family
        store=FeatureStore(args.package,window_pool=args.window_pool,window_cache=args.window_cache)
        try:
            ds={h:load_datasets(args,h,store,("test",))["test"] for h in ("parallel","multiclass")}
            store.preload(ds.values())
            for name,result in models.items():
                dest=args.out/family/name
                model,checkpoint=load_decoder(dest/"best.pt",args.device)
                if checkpoint["go_sha256"]!=store.go.artifact_sha256:
                    raise ValueError("GO input differs from training")
                test,predictions=evaluate(model,DataLoader(ds[result["head"]],batch_size=args.batch_size),args.device)
                result["test"]=test
                predictions.to_csv(dest/"test_predictions.tsv",sep="\t",index=False)
                write_json(dest/"results.json",result)
                print(f"TEST {family}/{name}: n={test['n']} macro-F1={test['macro_f1']:.4f} reverse={test['reverse_mse']:.6g}",flush=True)
        finally:
            store.close()
    report=dict(plan=plan,selection_before_test=selection,models=fitted,
                distinct_training_runs=sum(len(m) for m in fitted.values()),
                total_seconds=time.monotonic()-started,
                score_difference_lambda_control="No retraining: reversal penalty is structurally zero; baseline is the control",
                test_used_for_selection=False)
    write_json(args.out/"summary.json",report)
    print(f"Completed suite in {report['total_seconds']:.1f}s",flush=True)


if __name__=="__main__":
    main()
