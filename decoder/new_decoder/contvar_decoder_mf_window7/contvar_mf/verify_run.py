"""Independently check the completed 48-run result package and checkpoint readback."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader

from .data import FeatureStore, PairDataset, model_inputs
from .model import load_decoder
from .suite import verify_preparation
from .train import seed_everything
from .util import LABELS, WORKSPACE, sha256_file, utc_now, write_json


def main():
    root=WORKSPACE / "results/seed42"
    summary=json.loads((root / "summary.json").read_text(encoding="utf-8"))
    prepared=WORKSPACE / "prepared"
    manifest=verify_preparation(prepared)
    lock=json.loads((root / "selection_before_test.json").read_text(encoding="utf-8"))
    assert len(summary["models"])==summary["distinct_training_runs"]==48
    assert summary["plan"]["settings"]["seed"]==42
    assert not summary["historical_results_used"] and not summary["test_used_for_selection"]
    assert sha256_file(WORKSPACE / "offline_run.py")==summary["plan"]["provenance"]["offline_runner_sha256"]
    for name,digest in summary["plan"]["provenance"]["source_code_sha256"].items():
        assert sha256_file(WORKSPACE / "contvar_mf" / name)==digest
        assert sha256_file(root / "code_snapshot" / name)==digest
    checks=[]
    for name,result in summary["models"].items():
        c=result["condition"]
        folder=root / name
        config=json.loads((folder / "config.json").read_text(encoding="utf-8"))
        assert config["encoder_frozen"] and config["anc2vec_frozen"]
        assert config["classification_only"] and not config["reverse_penalty"]
        assert config["settings"]["seed"]==42 and config["settings"]["patience"]==5
        history=json.loads((folder / "history.json").read_text(encoding="utf-8"))
        best=max(history,key=lambda r:r["validation"]["macro_f1"])
        assert best["epoch"]==result["best_epoch"] and len(history)==result["epochs_run"]<=20
        assert np.isclose(best["validation"]["macro_f1"],result["validation"]["macro_f1"])
        assert sha256_file(folder / "best.pt")==lock["checkpoints"][name]["checkpoint_sha256"]
        expected=pd.read_csv(prepared / c["cohort"] / f"{c['family']}_test.tsv",sep="\t",keep_default_na=False,low_memory=False)
        pred=pd.read_csv(folder / "test_predictions.tsv",sep="\t",keep_default_na=False,low_memory=False)
        assert pred.sample_id.tolist()==expected.sample_id.tolist()
        assert pred.label.tolist()==expected.label.tolist()
        scores=pred[[f"score_{label.lower()}" for label in LABELS]].to_numpy()
        assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
        if c["head"]=="multiclass":
            np.testing.assert_allclose(scores.sum(1),1,atol=1e-6)
            predicted=scores.argmax(1)
        else:
            np.testing.assert_allclose(scores[:,2],1-scores[:,:2].max(1),atol=1e-6)
            predicted=np.where(scores[:,:2].max(1)>=.5,scores[:,:2].argmax(1),2)
        assert pred.prediction.tolist()==[LABELS[i] for i in predicted]
        common=pred.common_test.astype(str).str.lower().eq("true").to_numpy()
        for label,mask,m in [("native",np.ones(len(pred),dtype=bool),result["test"]),
                             ("common",common,result["test"]["common_test"])]:
            y=pred.label_idx.to_numpy()[mask];p=predicted[mask]
            assert len(y)==m["n"]
            assert np.isclose(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0),m["macro_f1"],atol=1e-7)
            assert np.isclose(accuracy_score(y,p),m["accuracy"],atol=1e-7)
            assert np.isclose(matthews_corrcoef(y,p),m["mcc"],atol=1e-7)
            assert confusion_matrix(y,p,labels=[0,1,2]).tolist()==m["confusion_matrix"]
        assert int(common.sum())==manifest["common_test"][c["family"]]["rows"]
        assert (folder / "test_predictions.tsv").stat().st_mtime >= (root / "selection_before_test.json").stat().st_mtime
        checks.append(name)
    # Independent frequency baselines, using Counters rather than the fitting
    # implementation's pandas groupby/unstack path.
    stored_baselines=json.loads((root / "baselines.json").read_text(encoding="utf-8"))
    baseline_checks=0
    for key,rules in stored_baselines.items():
        cohort,family=key.split("/")
        train=pd.read_csv(prepared / cohort / f"{family}_train.tsv",sep="\t",keep_default_na=False,low_memory=False)
        test=pd.read_csv(prepared / cohort / f"{family}_test.tsv",sep="\t",keep_default_na=False,low_memory=False)
        overall=Counter(train.label)
        by_go,by_protein_go=defaultdict(Counter),defaultdict(Counter)
        for protein,go,label in train[["protein_id","matched_go_id","label"]].itertuples(index=False,name=None):
            by_go[go][label]+=1;by_protein_go[(protein,go)][label]+=1
        common=test.common_test.astype(str).str.lower().eq("true").to_numpy()
        for rule,result in rules.items():
            predictions=[]
            for protein,go in test[["protein_id","matched_go_id"]].itertuples(index=False,name=None):
                counts=overall
                if rule!="majority":counts=by_go.get(go,overall)
                if rule=="protein_go_majority":counts=by_protein_go.get((protein,go),counts)
                predictions.append(max(range(3),key=lambda i:counts[LABELS[i]]))
            predictions=np.asarray(predictions)
            for domain,mask in [("native",np.ones(len(test),dtype=bool)),("common",common)]:
                y=test.label_idx.to_numpy()[mask];p=predictions[mask]
                assert np.isclose(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0),result[domain]["macro_f1"])
                assert confusion_matrix(y,p,labels=[0,1,2]).tolist()==result[domain]["confusion_matrix"]
                baseline_checks+=1
    # Check actual persisted weights against exported scores on a first batch,
    # for each validation-selected model. This adds no training run.
    torch.set_num_threads(4)
    seed_everything(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    store=FeatureStore(prepared)
    readbacks=[]
    for name in summary["best_by_validation"].values():
        c=summary["models"][name]["condition"]
        dataset=PairDataset(prepared / c["cohort"] / f"{c['family']}_test.tsv",store,c["window_mode"])
        batch=next(iter(DataLoader(dataset,batch_size=128)))
        model,_=load_decoder(root / name / "best.pt",device)
        with torch.no_grad(): output=model(**model_inputs(batch,device))
        saved=pd.read_csv(root / name / "test_predictions.tsv",sep="\t",keep_default_na=False,low_memory=False).iloc[:len(batch["label"])]
        np.testing.assert_allclose(output["scores"].cpu().numpy(),saved[[f"score_{s.lower()}" for s in LABELS]].to_numpy(),atol=2e-6,rtol=2e-6)
        assert saved.prediction.tolist()==[LABELS[i] for i in output["prediction"].cpu().tolist()]
        readbacks.append(name)
    historical=json.loads((WORKSPACE / "verification/historical_files_before.json").read_text(encoding="utf-8"))
    changed=[name for name,value in historical.items() if sha256_file(WORKSPACE.parent / name)!=value["sha256"]]
    assert not changed, changed
    report=json.loads((WORKSPACE / "reports/report_provenance.json").read_text(encoding="utf-8"))
    assert report["summary_sha256"]==sha256_file(root / "summary.json") and not report["historical_results_used"]
    assert report["report_code_sha256"]==sha256_file(WORKSPACE / "contvar_mf/report.py")
    tables={"tum_deneyler.csv":48,"window_karsilastirmalari.csv":32,
            "ortak_test_kapsam_farklari.csv":24,"baselines.csv":12,"veri_ozeti.csv":12,"validation_secimleri.csv":4}
    for filename,expected_rows in tables.items():
        table=pd.read_csv(WORKSPACE / "reports" / filename)
        assert len(table)==expected_rows,(filename,len(table))
    exported=pd.read_csv(WORKSPACE / "reports/tum_deneyler.csv")
    assert exported.name.nunique()==48
    for row in exported.itertuples():
        expected=summary["models"][row.name]
        assert np.isclose(row.test_macro_f1,expected["test"]["macro_f1"])
        assert np.isclose(row.common_macro_f1,expected["test"]["common_test"]["macro_f1"])
    result=dict(passed=True,checked_models=len(checks),independent_prediction_metrics=True,
                independent_baseline_metrics=baseline_checks,
                exported_tables_checked=tables,
                checkpoint_first_batch_readbacks=readbacks,historical_files_checked=len(historical),
                historical_files_changed=changed,extra_training_runs=0,checked_at=utc_now())
    write_json(WORKSPACE / "verification/final_qa.json",result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":
    main()
