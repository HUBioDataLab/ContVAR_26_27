"""Create shareable result tables from an already completed full suite."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import WORKSPACE
from .train import write_json

NAMES={"parallel_score_difference":"Paralel + skor farkı", "parallel_joint":"Paralel + joint",
       "multiclass_score_difference":"Üç sınıflı + skor farkı", "multiclass_joint":"Üç sınıflı + joint"}


def baseline_tables(root):
    result={}
    for family in ("easy","hard"):
        folder=root/"prepared_splits/multiclass"
        train=pd.read_csv(folder/f"{family}_train.tsv",sep="\t")
        majority=int(train.label_idx.value_counts().index[0])
        counts=pd.crosstab(train.matched_go_id,train.label_idx).reindex(columns=[0,1,2],fill_value=0)
        go_rule=counts.idxmax(axis=1).to_dict()
        grouped=train.groupby(["protein_id","matched_go_id","label_idx"]).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
        pair_rule=grouped.idxmax(axis=1).to_dict()
        result[family]={}
        for split in ("val","test"):
            frame=pd.read_csv(folder/f"{family}_{split}.tsv",sep="\t")
            y=frame.label_idx.to_numpy()
            go=np.array([go_rule.get(g,majority) for g in frame.matched_go_id])
            pg=np.array([pair_rule.get((p,g),go_rule.get(g,majority)) for p,g in zip(frame.protein_id,frame.matched_go_id)])
            result[family][split]={name:float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0))
                                   for name,pred in (("majority_class",np.full(len(y),majority)),("go_only_train_majority",go),("protein_go_train_majority_with_go_fallback",pg))}
    return result


def report(suite,root=WORKSPACE):
    root=Path(root);suite=Path(suite)
    full=json.loads((suite/"summary.json").read_text())
    if full["plan"]["smoke"]:
        raise ValueError("Full result report must not be built from a smoke run")
    manifest=json.loads((root/"prepared_splits/manifest.json").read_text())
    docs=root/"docs";docs.mkdir(exist_ok=True)
    baselines=baseline_tables(root)
    write_json(suite/"training_only_reference_rules.json",baselines)
    selected=[];all_rows=[]
    for family,models in full["models"].items():
        for name,res in models.items():
            base=f"{res['head']}_{res['pair_mode']}"
            v,t=res["val"],res["test"]
            row=dict(family=family,model=base,lambda_reverse=res["lambda_reverse"],
                     selected_by_validation=full["selection_before_test"][family][base]==name,
                     best_epoch=res["best_epoch"],epochs_run=res["epochs_run"],
                     n_train=res["n_train"],n_val=v["n"],n_test=t["n"],
                     val_macro_f1=v["macro_f1"],test_macro_f1=t["macro_f1"],test_accuracy=t["accuracy"],
                     test_lof_f1=t["lof"]["f1"],test_gof_f1=t["gof"]["f1"],test_neutral_f1=t["neutral"]["f1"],
                     test_macro_auprc=t["macro_auprc"],test_reverse_mse=t["reverse_mse"],
                     test_reverse_mae=t["reverse_mae"],test_class_swap_mse=t["class_swap_mse"],
                     parameters=res["parameter_count"],training_seconds=res["seconds"],
                     checkpoint=f"results/{suite.name}/{family}/{name}/best.pt")
            all_rows.append(row)
            if row["selected_by_validation"]:
                selected.append(row)
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (docs/"tum_deneyler.csv").write_text(out.getvalue())
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    (docs/"secili_modeller.csv").write_text(out.getvalue())
    text=f'''# Tam veri deney sonuçları

Bu rapor `results/{suite.name}/summary.json` dosyasından otomatik üretildi. Seed {full['plan']['seed']}; en fazla {full['plan']['max_epochs']} epoch; early-stopping patience {full['plan']['patience']}. Eğitimler küçük örneklemle sınırlı değildir: mevcut frozen girdileri geçerli olan bütün uygun training satırları kullanılmıştır. Test sonucu checkpoint veya lambda seçmek için kullanılmamıştır.

**Kapsam:** yalnız tek mutasyon; değerlendirmede augmentation yok; kaynak etiketleri ve partition atamaları korunur. Çoklu etiketli tek-mutasyon grupları kullanıcı tercihiyle korunur ve ayrıca raporlanır. Tam kaynak ZIP'te embeddingi olmayan satırlar bu deneyin kapsamına alınamaz.

## Dört kombinasyonun final karşılaştırması

Her ailede joint modelin lambda değeri validation macro-F1 ile seçildi. Aşağıdaki satırlar `selection_before_test.json` ile test hesaplarından önce sabitlenmiş adaylardır. Macro-F1 yüksek, reverse MSE düşük olmalıdır; fark modellerinin yapısal sıfır reverse hatası sınıflandırma üstünlüğü değildir.

| Aile | Mimari | Seçilen lambda | Epoch | Val Macro-F1 | Test Macro-F1 | Test LOF F1 | Test GOF F1 | Test NEUTRAL F1 | Test reverse MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in selected:
        text+=f"| {r['family']} | {NAMES[r['model']]} | {r['lambda_reverse']:g} | {r['best_epoch']} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | {r['test_lof_f1']:.4f} | {r['test_gof_f1']:.4f} | {r['test_neutral_f1']:.4f} | {r['test_reverse_mse']:.4g} |\n"
    text+='\nValidation ile genel model seçimi:\n\n'
    for family in ("easy","hard"):
        best=max([r for r in selected if r["family"]==family],key=lambda r:r["val_macro_f1"])
        text+=f"- **{family}: {NAMES[best['model']]}**, lambda={best['lambda_reverse']:g}; validation Macro-F1={best['val_macro_f1']:.4f}, bu seçimin test Macro-F1'i={best['test_macro_f1']:.4f}. [Checkpoint](../{best['checkpoint']}).\n"
    text+='''
## Reverse loss açık/kapalı kontrolü

Bu tablo kontrolün her iki adayını da gösterir. Testte daha iyi olan lambda sonradan seçilmedi. Fark modellerinde ceza yapısal olarak sıfır olduğundan aynı eğitimin lambda=0.1 kopyası çalıştırılmadı.

| Aile | Joint output | Lambda | Val Macro-F1 | Test Macro-F1 | Test reverse MSE |
|---|---|---:|---:|---:|---:|
'''
    for r in all_rows:
        if r["model"].endswith("_joint"):
            text+=f"| {r['family']} | {NAMES[r['model']]} | {r['lambda_reverse']:g} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | {r['test_reverse_mse']:.4g} |\n"
    text+='''
## Train'den türetilen basit referanslar

Bu kurallar yalnız train etiketleriyle kurulmuştur. GO-only kontrolü, skorun ne kadarının mutasyon dışındaki dağılımlardan açıklanabileceğini görmeye yardım eder. Protein–GO kuralı tanımadığı protein/GO için GO kuralına, o da genel train çoğunluğuna döner. Bu kurallar testte fit edilmez.

| Aile | Çoğunluk sınıfı test F1 | GO-only test F1 | Protein–GO + fallback test F1 |
|---|---:|---:|---:|
'''
    for f in ("easy","hard"):
        b=baselines[f]["test"]
        text+=f"| {f} | {b['majority_class']:.4f} | {b['go_only_train_majority']:.4f} | {b['protein_go_train_majority_with_go_fallback']:.4f} |\n"
    text+='''
**Yorum:** bu koşuda joint modeller dört karşılaştırmada da skor farkı modellerinden daha yüksek test Macro-F1 verdi. Lambda=0.1 reverse hatasını belirgin düşürdü, fakat dört joint kontrolünde de test Macro-F1'i düşürdü; validation seçimi de hepsinde lambda=0 oldu. Bu, tek seed ve denenmiş tek pozitif lambda için gözlemdir.

Easy'de train'den türetilen protein–GO kuralı (0.7031), bütün yeni modellerin test skorundan yüksektir. Hard'da GO-only referans 0.5536, validation ile seçilen model 0.5661'dir. Dolayısıyla bu sonuçlar mimarinin mutasyon bilgisini etkili kullandığını veya annotation önbilgisini aştığını tek başına göstermiyor. Joint üstünlüğünün hangi kısmının protein/GO dağılımından geldiği ayrıca incelenmelidir. Testte en yüksek gözlenen yeni-model skoru easy'de üç sınıflı joint 0.6871; hard'da üç sınıflı joint 0.5665'tir. Validation ile seçilen modelin farklı olması raporda korunur.
'''
    text+='''
## Veri ve çalışma kapsamı

| Aile | Train | Validation | Test |
|---|---:|---:|---:|
'''
    for f in ("easy","hard"):
        ns=[manifest["prepared"][f"{f}/{s}"]["rows"] for s in ("train","val","test")]
        text+=f"| {f} | {ns[0]} | {ns[1]} | {ns[2]} |\n"
    text+=f"\nToplam {full['distinct_training_runs']} ayrı eğitim ve {full['distinct_training_runs']} tam test değerlendirmesi; toplam suite süresi {full['total_seconds']/60:.1f} dakika. Her koşulun epoch sayısı, parametreleri ve süresi CSV'de bulunur.\n"
    text+='''
## Dosyalar ve yorum sınırı

- [Seçilmiş sekiz model CSV](secili_modeller.csv), [bütün on iki koşul CSV](tum_deneyler.csv).
- Her model klasöründe `config.json`, `history.json`, `best.pt`, `results.json`, `val_predictions.tsv`, `test_predictions.tsv` bulunur.
- `results.json` içinde görülmüş/görülmemiş GO ve çoklu etiketli girdi grupları için alt sonuçlar vardır. Confusion matrix sırası LOF, GOF, NEUTRAL'dır; satırlar gerçek, sütunlar tahmin etiketidir.
- `results/package_smoke` yalnız yazılım kontrolüdür; bu raporun skorlarına karıştırılmaz.

Sonuçlar tek seed ve mevcut cache sürümü içindir; tekrarlı seed belirsizliği ölçülmedi. Cache'in checkpoint kimliği H5 içeriğinden biyolojik olarak yeniden kanıtlanmadı; artifact hash'leri korundu. GO-specific etiketler kaynak bağlamına bağlıdır. Model skorları deneysel aktivite veya kalibre klinik olasılık değildir. Bu sınırlar farklı kaynak/ölçek/encoder katkısının gösterildiği anlamına gelmez.
'''
    (docs/"SONUCLAR.md").write_text(text)
    write_json(suite/"report_selection.json",dict(selected=selected,training_only_references=baselines))
    print(f"Wrote {len(selected)} selected and {len(all_rows)} total experiment rows.")
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,default=WORKSPACE/"results/full_seed42")
    p.add_argument("--root",type=Path,default=WORKSPACE)
    a=p.parse_args();report(a.suite,a.root)


if __name__=="__main__":
    main()
