"""Export only NEW 48-run results: tables, plots, diagrams, HTML and PDF."""
import argparse
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score

from .util import LABELS, WORKSPACE, run_conditions, sha256_file, utc_now, write_json
from .suite import baseline_predictions

WINDOWS = ["none", "mean", "pair_mean", "esim"]
WN = dict(none="Penceresiz", mean="Mean-7", pair_mean="B mean-7", esim="B ESIM-7")
CN = dict(old="Eski", full="Geniş")


def diagram(mode, diff=False):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.03))
    ax.axis("off")
    nodes = {}
    def box(k,x,y,text,color="#e2edf8",w=.24,h=.10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",facecolor=color,edgecolor="#26334d"))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
        nodes[k]=(x,y,w,h)
    def arrow(a,b,color="#526178"):
        x,y,w,h=nodes[a];xx,yy,ww,hh=nodes[b]
        if a=="q":
            if b=="fusion":
                points=[(x+w,y+h/2),(.67,y+h/2),(.67,yy+hh+.035),(xx+ww/2,yy+hh)]
            else:
                points=[(x,y+h/2),(.325,y+h/2),(.325,yy+hh/2),(xx,yy+hh/2)]
            ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=color,lw=1.1,ls="--")
            start,end=points[-2],points[-1]
        elif a=="sw" and b=="d":
            points=[(x+w,y+h/2),(.985,y+h/2),(.985,yy+hh/2),(xx+ww,yy+hh/2)]
            ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=color,lw=1.4)
            start,end=points[-2],points[-1]
        elif abs(x-xx)>.08:
            start=(x+w,y+h/2) if xx>x else (x,y+h/2)
            end=(xx,yy+hh/2) if xx>x else (xx+ww,yy+hh/2)
        else:
            start=(x+w/2,y) if yy<y else (x+w/2,y+h)
            end=(xx+ww/2,yy+hh) if yy<y else (xx+ww/2,yy)
        ax.annotate("",xy=end,xytext=start,
                    arrowprops=dict(arrowstyle="->",color=color,lw=1.4,shrinkA=2,shrinkB=2))
    ax.set_title("Saf skor farki" if diff else "Joint — "+WN[mode],loc="left",fontsize=19,fontweight="bold")
    extra="\n+ kendi mean-7 vektoru" if mode=="mean" else ""
    box("wt",.01,.73,"WT global + mutasyon noktasi"+extra,w=.29)
    box("var",.01,.49,"VAR global + mutasyon noktasi"+extra,"#e3f2e8",w=.29)
    box("wr",.37,.73,"Adapter + GO reader\npaylasilan agirliklar")
    box("vr",.37,.49,"Adapter + GO reader\npaylasilan agirliklar","#e3f2e8")
    box("q",.37,.91,"Anc2Vec [200] → query [128]","#eae0f3",h=.055)
    arrow("wt","wr");arrow("var","vr");arrow("q","wr","#8c65a6");arrow("q","vr","#8c65a6")
    if diff:
        box("sw",.72,.73,"S(WT, GO)")
        box("sv",.72,.49,"S(VAR, GO)","#e3f2e8")
        box("d",.72,.25,"S(VAR,GO) − S(WT,GO)","#fff0d3")
        box("out",.72,.04,"LOF / GOF / NEUTRAL","#f5e0e0")
        for a,b in [("wr","sw"),("vr","sv"),("sw","d"),("sv","d"),("d","out")]:arrow(a,b)
        ax.text(.02,.14,"Her iki tarafta ayni scorer.\nWT skoru VAR girdisini kullanmaz.\nParallel: iki binary cikti; multiclass: uc sinif.",fontsize=12)
    else:
        local_text="\n+ ortak local" if mode in ("pair_mean","esim") else ""
        box("fusion",.72,.59,"Fusion [128]\nWT + VAR + GO"+local_text,"#fff0d3",h=.15)
        box("out",.72,.32,"Parallel: 2 binary head\nveya\nMulticlass: 3 sinif","#f5e0e0",h=.15)
        for a,b in [("wr","fusion"),("vr","fusion"),("q","fusion"),("fusion","out")]:arrow(a,b)
        if mode in ["pair_mean","esim"]:
            box("raw",.01,.20,"WT + VAR window\nmerkezden 7 × 1280",w=.29)
            text="Ayri masked mean\n→ karsilastirma: 1 token" if mode=="pair_mean" else "Hizali konumlarda MLP\n→ karsilastirma: 7 token"
            box("pair",.37,.20,text,"#fff0d3")
            box("local",.37,.01,"GO local reader\n→ ortak ozet [128]","#eae0f3")
            for a,b in [("raw","pair"),("pair","local"),("q","local"),("local","fusion")]:arrow(a,b)
            ax.text(.01,.045,"[WT, VAR, VAR−WT, WT⊙VAR]",fontsize=10)
        else:
            ax.text(.02,.17,"Her tarafin ozeti 128 boyut.\nEncoder ve Anc2Vec sabit.\nDecoder sifirdan ogrenilir.",fontsize=12)
    fig.tight_layout()
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=WORKSPACE/"results/seed42")
    p.add_argument("--out",type=Path,default=WORKSPACE/"reports")
    a=p.parse_args()
    source=a.results/"summary.json"
    if not source.exists():
        raise RuntimeError("NEW 48-run summary missing: no placeholder results will be generated")
    summary=json.loads(source.read_text(encoding="utf-8"))
    if summary["distinct_training_runs"]!=48 or len(summary["models"])!=48 or summary["historical_results_used"]:
        raise ValueError("Expected exactly 48 NEW completed runs")
    data=json.loads(Path(summary["preparation_manifest"]).read_text(encoding="utf-8"))
    baselines=json.loads((a.results/"baselines.json").read_text(encoding="utf-8"))
    a.out.mkdir(parents=True,exist_ok=True)
    figures=a.out/"figures";figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    rows=[]
    for c in run_conditions():
        r=summary["models"][c["name"]]
        row={**c,"seed":summary["plan"]["settings"]["seed"],
             "window_size":0 if c["window_mode"]=="none" else 7,
             "source_cohort":data["source_cohorts"][c["cohort"]],
             "prepared_train":f"prepared/{c['cohort']}/{c['family']}_train.tsv",
             "prepared_val":f"prepared/{c['cohort']}/{c['family']}_val.tsv",
             "prepared_test":f"prepared/{c['cohort']}/{c['family']}_test.tsv",
             "best_epoch":r["best_epoch"],"epochs_run":r["epochs_run"],"parameters":r["parameter_count"],
             "training_seconds":r["seconds"],"n_train":r["training_rows"],"n_val":r["validation_rows"],
             "val_macro_f1":r["validation"]["macro_f1"]}
        for prefix,m in [("test",r["test"]),("common",r["test"]["common_test"])]:
            for k in ["n","macro_f1","mcc","accuracy","balanced_accuracy","macro_auprc","classification_loss"]:row[f"{prefix}_{k}"]=m[k]
            row[f"{prefix}_both_heads_positive_rate"]=m.get("both_heads_positive_rate")
            for label in LABELS:
                for k in ["precision","recall","f1","support","auprc","auroc"]:row[f"{prefix}_{label.lower()}_{k}"]=m[label.lower()][k]
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(a.out/"tum_deneyler.csv",index=False)
    scope=pd.read_csv(a.results/"common_scope_comparisons.csv")
    scope.to_csv(a.out/"ortak_test_kapsam_farklari.csv",index=False)
    lookup={(r["cohort"],r["family"],r["window_mode"],r["head"],r["pair_mode"]):r for r in rows}
    comparisons=[]
    for cohort in ["old","full"]:
        for family in ["easy","hard"]:
            for head,pair in [("parallel","score_difference"),("multiclass","score_difference"),("parallel","joint"),("multiclass","joint")]:
                pairs=[("none","mean")]+([("mean","pair_mean"),("pair_mean","esim")] if pair=="joint" else [])
                for before,after in pairs:
                    x,y=lookup[(cohort,family,before,head,pair)],lookup[(cohort,family,after,head,pair)]
                    comparison=dict(cohort=cohort,family=family,head=head,pair_mode=pair,before=before,after=after,
                        before_test_macro_f1=x["test_macro_f1"],after_test_macro_f1=y["test_macro_f1"],
                        delta_test_macro_f1=y["test_macro_f1"]-x["test_macro_f1"],
                        delta_common_macro_f1=y["common_macro_f1"]-x["common_macro_f1"],
                        delta_test_mcc=y["test_mcc"]-x["test_mcc"])
                    for label in LABELS:
                        for domain in ["test","common"]:
                            metric=f"{domain}_{label.lower()}_f1"
                            comparison[f"delta_{metric}"]=y[metric]-x[metric]
                    comparisons.append(comparison)
    window=pd.DataFrame(comparisons);window.to_csv(a.out/"window_karsilastirmalari.csv",index=False)
    audit=[];filter_audit=[]
    for key,r in data["tables"].items():
        if "final" in r:
            c,f,s=key.split("/")
            audit.append(dict(cohort=c,family=f,split=s,source_rows=r["source_rows"],**r["final"],
                              technical_excluded_rows=r["technical_excluded_rows"]))
            for reason,count in r["exclusions"].items():
                filter_audit.append(dict(cohort=c,family=f,split=s,reason=reason,rows=count))
            for reason,count in r["technical_exclusions"].items():
                filter_audit.append(dict(cohort=c,family=f,split=s,reason=reason,rows=count))
    audit=pd.json_normalize(audit);audit.to_csv(a.out/"veri_ozeti.csv",index=False)
    pd.DataFrame(filter_audit).to_csv(a.out/"filtre_dokumu.csv",index=False)
    baseline_rows=[]
    validation_baselines={}
    prepared_path=Path(summary["preparation_manifest"]).parent
    for key,rules in baselines.items():
        c,f=key.split("/")
        train=pd.read_csv(prepared_path/c/f"{f}_train.tsv",sep="\t",keep_default_na=False,low_memory=False)
        val=pd.read_csv(prepared_path/c/f"{f}_val.tsv",sep="\t",keep_default_na=False,low_memory=False)
        validation_baselines[key]={rule:float(f1_score(val.label_idx,scores.argmax(1),labels=[0,1,2],average="macro",zero_division=0))
                                   for rule,scores in baseline_predictions(train,val).items()}
        for rule,m in rules.items():
            row=dict(cohort=c,family=f,baseline=rule,native_macro_f1=m["native"]["macro_f1"],
                                      common_macro_f1=m["common"]["macro_f1"],native_mcc=m["native"]["mcc"],
                                      val_macro_f1=validation_baselines[key][rule])
            for label in LABELS:
                for domain in ["native","common"]:
                    row[f"{domain}_{label.lower()}_f1"]=m[domain][label.lower()]["f1"]
            baseline_rows.append(row)
    bf=pd.DataFrame(baseline_rows);bf.to_csv(a.out/"baselines.csv",index=False)
    chosen=[]
    for key,name in summary["best_by_validation"].items():
        result=summary["models"][name];c=result["condition"]
        native=result["test"];common=native["common_test"]
        reference=baselines[key]["protein_go_majority"]
        chosen.append(dict(group=key,name=name,window=WN[c["window_mode"]],head=c["head"],pair_mode=c["pair_mode"],
            val_macro_f1=result["validation"]["macro_f1"],test_macro_f1=native["macro_f1"],test_mcc=native["mcc"],
            baseline_test_macro_f1=reference["native"]["macro_f1"],
            delta_test=native["macro_f1"]-reference["native"]["macro_f1"],
            common_macro_f1=common["macro_f1"],baseline_common_macro_f1=reference["common"]["macro_f1"],
            delta_common=common["macro_f1"]-reference["common"]["macro_f1"]))
    chosen=pd.DataFrame(chosen);chosen.to_csv(a.out/"validation_secimleri.csv",index=False)
    chosen_view=pd.DataFrame({"Grup":chosen.group,
        "Model":[f"{r.window} / {'P' if r.head=='parallel' else 'M'}-{'J' if r.pair_mode=='joint' else 'SD'}" for r in chosen.itertuples()],
        "Val F1":chosen.val_macro_f1.round(4),"Test F1":chosen.test_macro_f1.round(4),
        "Ref. F1":chosen.baseline_test_macro_f1.round(4),"Fark":chosen.delta_test.round(4),
        "Ortak F1":chosen.common_macro_f1.round(4),"Ortak ref.":chosen.baseline_common_macro_f1.round(4),
        "Ortak fark":chosen.delta_common.round(4)})
    notes=[]
    for before,after in [("none","mean"),("mean","pair_mean"),("pair_mean","esim")]:
        part=window.loc[(window.before==before)&(window.after==after)]
        notes.append(f"{WN[before]} → {WN[after]}: {int((part.delta_test_macro_f1>0).sum())}/{len(part)} kosulda artış; ortalama Δ Macro-F1={part.delta_test_macro_f1.mean():+.4f}.")
    notes.append(f"Ortak testte geniş kapsam: {int((scope.delta_macro_f1>0).sum())}/{len(scope)} eşleştirmede artış; ortalama Δ Macro-F1={scope.delta_macro_f1.mean():+.4f}.")
    notes+=["B mean–ESIM farkı, hizalı karşılaştırmanın ve tokenları GO reader'a kadar korumanın birlikte etkisidir. İki B modelinin parametre sayısı eşittir.",
            "Eski/geniş kayıt ve validation düzenleri farklıdır. Ortak test eğitim düzenini karşılaştırır; yalnız protein sayısının nedensel katkısını ayırmaz.",
            "Tek seed 42 kullanıldı. Bu farklar betimleyicidir; tekrarlı seed belirsizliği veya istatistiksel üstünlük kanıtı değildir.",
            f"Ortak hard testte yalnız {data['common_test']['hard']['labels']['GOF']} GOF kaydı vardır; GOF skorundaki küçük örneklem duyarlılığı dikkate alınmalıdır.",
            "Kaynak etiket çelişkileri ve tekrarlar korunmuştur. Teknik elemeler veri manifestinde kaydedilmiştir. Eski eğitimlerin sayısal sonuçları kullanılmamıştır."]
    saved=[]
    with PdfPages(a.out/"ContVAR_MF_Window7_Rapor.pdf",metadata={"Title":"ContVAR MF Window7 - 48 new experiments","Author":"ContVAR"}) as pdf:
        def save(fig,name):
            fig.text(.99,-.012,f"ContVAR | MF / window-7 / seed 42 | {len(saved)+1}",
                     ha="right",va="top",fontsize=7,color="#526178")
            fig.savefig(figures/f"{name}.png",dpi=160,bbox_inches="tight")
            fig.savefig(figures/f"{name}.svg",bbox_inches="tight")
            pdf.savefig(fig,bbox_inches="tight");plt.close(fig);saved.append(name)
        def table_page(title, table, name, note=""):
            fig, ax=plt.subplots(figsize=(14,8))
            ax.axis("off")
            ax.set_title(title,loc="left",fontsize=18,fontweight="bold",pad=22)
            artist=ax.table(cellText=table.values,colLabels=table.columns,loc="center",cellLoc="center")
            artist.auto_set_font_size(False);artist.set_fontsize(9);artist.scale(1,1.8)
            for (r,c),cell in artist.get_celld().items():
                if r==0:cell.set_facecolor("#20364f");cell.set_text_props(color="white",weight="bold")
                elif r%2==0:cell.set_facecolor("#eef2f7")
            if note:fig.text(.07,.07,note,fontsize=10,va="bottom")
            fig.tight_layout();save(fig,name)
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.94,"ContVAR — MF / window-7 / seed 42",fontsize=20,fontweight="bold")
        paragraphs=[f"48 yeni egitim. Toplam egitim suresi: {summary['total_training_seconds']/60:.1f} dakika.",
            "Yalnız is_combinatorial=False ve molecular_function; augmentation yalnız train. Encoder/Anc2Vec sabit, decoder sıfırdan. Reverse cezası yok.",
            "Checkpointler validation Macro-F1 ile seçildi. Native ve ortak testler bütün eğitimlerden sonra hesaplandı.",
            f"GO ontology: {data['ontology']['version']}.",
            "2 kapsam x 2 split x 12 model = 48. Her grupta penceresiz 4, mean-7 4, B mean-7 2, B ESIM-7 2 model var.",
            "Window: mutasyon dahil toplam 7 amino asit (yaricap 3). Ham 33 konumun 13:20 dilimi; padding maskeleri korunur.",
            "Native test: her kapsamin kendi testi. Ortak test: iki testin ayni kayit/etiket/tekrar kesişimi; her iki train/val'den ayridir."]
        y=.86
        for paragraph in paragraphs:
            lines=textwrap.wrap(paragraph,105,break_long_words=False)
            fig.text(.06,y,"\n".join(lines),fontsize=11,va="top");y-=.03*len(lines)+.025
        save(fig,"00_protokol")
        settings=summary["plan"]["settings"]
        model_protocol=pd.DataFrame([
            ["Penceresiz","Global + mutasyon noktasi","Yok","P/M x J/SD",4],
            ["Mean-7","Global + nokta + ayri window mean","Yok","P/M x J/SD",4],
            ["B mean-7","Global + mutasyon noktasi","Window mean -> 1 karsilastirma tokeni","P/M x J",2],
            ["B ESIM-7","Global + mutasyon noktasi","Hizali 7 karsilastirma tokeni","P/M x J",2],
        ],columns=["Kosul","WT/VAR yollari","GO ile okunan ortak kol","Cikti/karar","Adet"])
        table_page("Her kapsam / split icin 12 model",model_protocol,"deney_matrisi",
                   "P: parallel, M: multiclass; J: joint, SD: score difference. B kosullari joint-only; reverse ceza yok.")
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.94,"Egitim ve referans protokolu",fontsize=19,fontweight="bold")
        details=[
            f"AdamW: lr={settings['lr']}, weight decay={settings['weight_decay']}, batch={settings['batch_size']}, "
            f"en fazla {settings['epochs']} epoch, patience={settings['patience']}.",
            f"Decoder width={settings['width']}, attention heads={settings['attention_heads']}, dropout={settings['dropout']}, "
            f"gradient clip={settings['gradient_clip']}. Sinif agirligi yok. Her modelin kendi sifirdan ogrenilen agirliklari var.",
            "Parallel: LOF ve GOF icin iki sigmoid head, esik 0.5. Ikisi de negatifse NEUTRAL; ikisi pozitifse buyuk olan secilir. "
            "Bu skorlar uc sinif icin normalize olasiliklar degildir. Multiclass: uc sinif softmax ve argmax.",
            "Referanslar yeniden, yalniz ilgili train etiketlerinden hesaplanir: majority tum train'in sik sinifi; "
            "go_majority ayni GO'nun sik sinifi; protein_go_majority ayni protein-GO'nun sik sinifi.",
            "Bilinmeyen protein-GO icin GO sayimlarina, bilinmeyen GO icin genel train sayimlarina donulur. "
            "Referans skorlari train sinif oranlaridir. Bunlar ek neural egitim degildir.",
            "Hard test proteinleri train'de olmadigi icin protein_go_majority, go_majority ile ayni sonucu verir. "
            "Referanslarda da test etiketleri kullanilmaz.",
            "B ESIM, local window icin hizali karsilastirma uyarlamasidir: ortak 1280->128 adapter, "
            "[WT,VAR,VAR-WT,WT*VAR] MLP'si ve yedi tokeni okuyan GO reader. Soft-alignment veya BiLSTM eklenmedi.",
        ]
        y=.86
        for paragraph in details:
            lines=textwrap.wrap(paragraph,108,break_long_words=False)
            fig.text(.06,y,"\n".join(lines),fontsize=11,va="top");y-=.03*len(lines)+.025
        save(fig,"egitim_referans_protokolu")
        overview=audit[["cohort","family","split","rows","proteins","go_terms","labels.LOF","labels.GOF","labels.NEUTRAL","technical_excluded_rows"]].copy()
        overview.columns=["Kapsam","Split","Bölüm","N","Protein","GO","LOF","GOF","NEUTRAL","Teknik eleme"]
        overview["Kapsam"]=overview["Kapsam"].map(CN)
        table_page("MF veri kapsamı ve sınıf destekleri",overview,"veri_tablosu",
                   f"Ortak test: easy N={data['common_test']['easy']['rows']} (GOF={data['common_test']['easy']['labels']['GOF']}); "
                   f"hard N={data['common_test']['hard']['rows']} (GOF={data['common_test']['hard']['labels']['GOF']}).")
        table_page("Validation secimi ve yeni protein-GO referansi",chosen_view,"validation_secimleri",
                   f"4 validation seciminin {int((chosen.delta_test>0).sum())}/4'u native testte referansi asti. "
                   "Fark = model - referans. Secim test skoru ile yapilmadi.\n"
                   "Referans: train protein-GO sinif sayimlari, GO/genel fallback. P: parallel, M: multiclass; J: joint, SD: skor farki.")
        for cohort in ["old","full"]:
            for family in ["easy","hard"]:
                part=frame.loc[(frame.cohort==cohort)&(frame.family==family)].sort_values("number")
                table=pd.DataFrame({"Model":[("P" if r.head=="parallel" else "M")+"-"+("SD" if r.pair_mode=="score_difference" else "J") for r in part.itertuples()],
                    "Window":[WN[w] for w in part.window_mode],
                    "Val F1":part.val_macro_f1.round(4).to_numpy(),"Test F1":part.test_macro_f1.round(4).to_numpy(),
                    "Ortak F1":part.common_macro_f1.round(4).to_numpy(),"Test MCC":part.test_mcc.round(4).to_numpy(),
                    "LOF F1":part.test_lof_f1.round(4).to_numpy(),"GOF F1":part.test_gof_f1.round(4).to_numpy(),
                    "NEUTRAL F1":part.test_neutral_f1.round(4).to_numpy()})
                table_page(f"{CN[cohort]} / {family} — 12 yeni koşul",table,f"tablo_{cohort}_{family}",
                           "P: parallel, M: multiclass; SD: skor farkı, J: joint. Val/test/ortak F1 sütunları Macro-F1'dir.")
        for mode in WINDOWS:save(diagram(mode),f"mimari_{mode}")
        save(diagram("mean",True),"mimari_score_difference")
        fig,axes=plt.subplots(2,2,figsize=(16,10),sharey=True)
        for ax,(c,f) in zip(axes.flat,[(c,f) for c in ["old","full"] for f in ["easy","hard"]]):
            part=frame.loc[(frame.cohort==c)&(frame.family==f)].sort_values("number")
            names=[f"{WN[r.window_mode]}\n{r.head[0].upper()}-{'SD' if r.pair_mode=='score_difference' else 'J'}" for r in part.itertuples()]
            ax.bar(np.arange(len(part)),part.test_macro_f1,color=["#4e79a7" if h=="parallel" else "#d78d43" for h in part["head"]])
            for r,col in zip(bf.loc[(bf.cohort==c)&(bf.family==f)].itertuples(),["gray","#8b62a5","#39815c"]):
                ax.axhline(r.native_macro_f1,color=col,ls="--",lw=1,label=r.baseline)
            ax.set_xticks(np.arange(len(part)),names,rotation=65,ha="right",fontsize=8)
            ax.set(title=f"{CN[c]} / {f}",ylabel="Test Macro-F1",ylim=(0,1));ax.legend(fontsize=7)
        fig.tight_layout();save(fig,"01_tum_modeller")
        fig,axes=plt.subplots(2,2,figsize=(13,9),sharey=True)
        for ax,(c,f) in zip(axes.flat,[(c,f) for c in ["old","full"] for f in ["easy","hard"]]):
            for head,color in [("parallel","#4e79a7"),("multiclass","#d78d43")]:
                vals=[lookup[(c,f,w,head,"joint")]["test_macro_f1"] for w in WINDOWS]
                ax.plot(range(4),vals,"o-",label=head,color=color)
                for x,v in enumerate(vals):
                    other_head="multiclass" if head=="parallel" else "parallel"
                    other=lookup[(c,f,WINDOWS[x],other_head,"joint")]["test_macro_f1"]
                    above=v>other or (v==other and head=="parallel")
                    ax.annotate(f"{v:.3f}",(x,v),xytext=(0,8 if above else -13),
                                textcoords="offset points",ha="center",fontsize=8)
            ax.set_xticks(range(4),[WN[w] for w in WINDOWS],rotation=15)
            for r,col in zip(bf.loc[(bf.cohort==c)&(bf.family==f)].itertuples(),["gray","#8b62a5","#39815c"]):
                ax.axhline(r.native_macro_f1,color=col,ls="--",lw=.9,label=r.baseline)
            ax.set(title=f"{CN[c]} / {f}",ylabel="Test Macro-F1",ylim=(0,1));ax.legend()
        fig.tight_layout();save(fig,"02_window_joint")
        for cohort in ["old","full"]:
            fig,axes=plt.subplots(2,2,figsize=(14,10),sharey=True)
            for ax,(family,head) in zip(axes.flat,[(f,h) for f in ["easy","hard"] for h in ["parallel","multiclass"]]):
                for label,color in zip(LABELS,["#4e79a7","#d78d43","#39815c"]):
                    vals=[lookup[(cohort,family,w,head,"joint")][f"test_{label.lower()}_f1"] for w in WINDOWS]
                    ax.plot(range(4),vals,"o-",color=color,label=label)
                    ref=baselines[f"{cohort}/{family}"]["protein_go_majority"]["native"][label.lower()]["f1"]
                    ax.axhline(ref,color=color,ls=":",lw=1,label=f"{label} ref.")
                ax.set_xticks(range(4),[WN[w] for w in WINDOWS],rotation=15)
                ax.set(title=f"{CN[cohort]} / {family} / {head} joint",ylabel="Test sinif F1",ylim=(0,1))
                ax.legend(fontsize=8,ncol=2)
            fig.tight_layout();save(fig,f"02_siniflar_{cohort}")
        fig,ax=plt.subplots(figsize=(11,8))
        names=scope[["window_mode","head","pair_mode"]].drop_duplicates()
        vals=np.array([[scope.loc[(scope.family==f)&(scope.window_mode==r.window_mode)&(scope["head"]==r.head)&(scope.pair_mode==r.pair_mode),"delta_macro_f1"].iloc[0]
                        for f in ["easy","hard"]] for r in names.itertuples()])
        heatmap_labels=[f"{WN[r.window_mode]} / {r.head} / {r.pair_mode}" for r in names.itertuples()]
        for rule in ["majority","go_majority","protein_go_majority"]:
            changes=[baselines[f"full/{family}"][rule]["common"]["macro_f1"]-
                     baselines[f"old/{family}"][rule]["common"]["macro_f1"] for family in ["easy","hard"]]
            vals=np.vstack([vals,changes]);heatmap_labels.append("Baseline: "+rule)
        limit=max(.01,float(np.abs(vals).max()))
        im=ax.imshow(vals,cmap="RdBu",vmin=-limit,vmax=limit,aspect="auto")
        for i in range(len(vals)):
            for j in range(2):ax.text(j,i,f"{vals[i,j]:+.4f}",ha="center",va="center",color="white" if abs(vals[i,j])>limit*.6 else "black")
        ax.set_xticks([0,1],[f"Easy (N={data['common_test']['easy']['rows']})",f"Hard (N={data['common_test']['hard']['rows']})"])
        ax.set_yticks(range(len(heatmap_labels)),heatmap_labels)
        ax.axhline(len(names)-.5,color="#526178",lw=1)
        ax.set_title("Aynı ortak testte geniş − eski Macro-F1");fig.colorbar(im,ax=ax);fig.tight_layout();save(fig,"03_ortak_test_kapsam")
        for kind in ["native","common"]:
            fig,axes=plt.subplots(2,2,figsize=(11,9))
            for ax,(key,name) in zip(axes.flat,summary["best_by_validation"].items()):
                r=summary["models"][name];m=r["test"] if kind=="native" else r["test"]["common_test"]
                cm=np.array(m["confusion_matrix"]);ax.imshow(cm,cmap="Blues")
                for i in range(3):
                    for j in range(3):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()*.5 else "black")
                ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS)
                baseline=baselines[key]["protein_go_majority"]["native" if kind=="native" else "common"]["macro_f1"]
                ax.set(title=f"{key}\nvalidation seçimi: {r['condition']['window_mode']}\nF1={m['macro_f1']:.4f}, ref={baseline:.4f}, n={m['n']}",xlabel="Tahmin",ylabel="Gerçek")
            fig.tight_layout();save(fig,"04_confusion_"+kind)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(key,name) in zip(axes.flat,summary["best_by_validation"].items()):
            history=json.loads((a.results/name/"history.json").read_text(encoding="utf-8"))
            ax.plot([r["epoch"] for r in history],[r["validation"]["macro_f1"] for r in history],"o-",label="model")
            ax.axvline(summary["models"][name]["best_epoch"],ls="--",color="gray")
            for rule,col in zip(["majority","go_majority","protein_go_majority"],["gray","#8b62a5","#39815c"]):
                ax.axhline(validation_baselines[key][rule],color=col,ls=":",lw=1,label=rule)
            ax.set(title=key,xlabel="Epoch",ylabel="Validation Macro-F1",ylim=(0,1))
            ax.legend(fontsize=7)
        fig.tight_layout();save(fig,"05_ogrenme_egrileri")
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for ax,(key,name) in zip(axes.flat,summary["best_by_validation"].items()):
            pred=pd.read_csv(a.results/name/"test_predictions.tsv",sep="\t",keep_default_na=False,low_memory=False)
            c,f=key.split("/")
            train=pd.read_csv(prepared_path/c/f"{f}_train.tsv",sep="\t",keep_default_na=False,low_memory=False)
            reference=baseline_predictions(train,pred)["protein_go_majority"]
            for i,(label,color) in enumerate(zip(LABELS,["#4e79a7","#d78d43","#39815c"])):
                y=pred.label_idx.to_numpy()==i
                if y.any():
                    precision,recall,_=precision_recall_curve(y,pred[f"score_{label.lower()}"])
                    ap=summary["models"][name]["test"][label.lower()]["auprc"]
                    ax.plot(recall,precision,color=color,label=f"{label} AP={ap:.3f}")
                    bp,br,_=precision_recall_curve(y,reference[:,i])
                    bap=average_precision_score(y,reference[:,i])
                    ax.plot(br,bp,"--",color=color,lw=1,label=f"{label} ref={bap:.3f}")
            ax.set(title=key+" (kesikli: protein–GO ref.)",xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
        fig.tight_layout();save(fig,"06_pr_egrileri")
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.94,"Yeni sonuçların yorumu",fontsize=19,fontweight="bold")
        y=.86
        for paragraph in notes:
            lines=textwrap.wrap(paragraph,110,break_long_words=False)
            fig.text(.06,y,"\n".join(lines),fontsize=11,va="top")
            y-=.03*len(lines)+.03
        save(fig,"07_yorum")
    show=["number","cohort","family","window_mode","head","pair_mode","val_macro_f1","test_macro_f1","common_macro_f1","test_mcc"]
    best=frame.loc[frame["name"].isin(summary["best_by_validation"].values())]
    best_table=best[show].copy()
    for col in ["val_macro_f1","test_macro_f1","common_macro_f1","test_mcc"]:
        best_table[col]=best_table[col].map(lambda x:f"{x:.4f}")
    def markdown_table(table):
        lines=["| "+" | ".join(map(str,table.columns))+" |", "| "+" | ".join(["---"]*len(table.columns))+" |"]
        lines.extend("| "+" | ".join(map(str,row))+" |" for row in table.itertuples(index=False,name=None))
        return "\n".join(lines)
    md=["# Yeni ContVAR MF / window-7 sonuçları","","Yalnızca yeni 48 eğitim; seed 42. Eski sonuçlar kullanılmadı.",""]+["- "+n for n in notes]+["","## Dosyalar","",
       "- tum_deneyler.csv: 48 koşulun tüm metrikleri","- window_karsilastirmalari.csv",
       "- ortak_test_kapsam_farklari.csv","- veri_ozeti.csv / filtre_dokumu.csv / baselines.csv","- validation_secimleri.csv: test ile seçilmeyen 4 model ve referans farkları",
       "- ContVAR_MF_Window7_Rapor.pdf ve figures/ PNG/SVG çizimleri","",
       "## Validation ile seçilen modeller ve referans","",markdown_table(chosen_view),"",
       "Bu dört seçimden "+str(int((chosen.delta_test>0).sum()))+" tanesi native testte protein-GO referansını geçti. Easy'de aynı protein için GO etiketi sıklığı güçlü bir referans oluşturuyor; hard'da proteinler train'de bulunmadığından bu yöntem GO sıklığına dönüyor.","",
       "Macro-F1, LOF/GOF/NEUTRAL F1 değerlerinin eşit ağırlıklı ortalamasıdır. MCC, tüm karışıklık matrisini kullanan tamamlayıcı ölçüdür.","",
       "Önce aynı kapsam/split içinde window koşullarını karşılaştırın. Eski/geniş farkını ortak test sütunlarından okuyun; native test örnekleri farklıdır.","",
       "## Yeni hesaplanan referanslar","",markdown_table(bf.round(4)),"",
       "Referansların tamamı yalnız train etiketlerinden hesaplanmış sayım yöntemleridir. Ek neural eğitim yapılmadı."]
    (a.out/"SONUCLAR.md").write_text("\n".join(md),encoding="utf-8")
    body="<h1>ContVAR — Yeni MF / window-7 sonuçları</h1><p>Yalnızca yeni 48 eğitim. Seed 42.</p><ul>"
    body+="".join("<li>"+html.escape(n)+"</li>" for n in notes)+"</ul>"
    body+="<h2>Validation ile seçilen modeller ve referans</h2><div class='tablewrap'>"+chosen_view.to_html(index=False)+"</div>"
    body+="<h2>Veri</h2><div class='tablewrap'>"+audit.to_html(index=False)+"</div>"
    for name in saved:body+=f'<h2>{html.escape(name)}</h2><img src="figures/{name}.png" alt="{html.escape(name)}">'
    body+="<h2>Bütün koşullar</h2><div class='tablewrap'>"+frame[show].round(4).to_html(index=False)+"</div>"
    (a.out/"SONUCLAR.html").write_text('<!doctype html><html lang="tr"><meta charset="utf-8"><title>ContVAR MF Window7</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #ccc}.tablewrap{overflow:auto}img{width:100%}h2{margin-top:40px}</style>'+body+"</html>",encoding="utf-8")
    write_json(a.out/"report_provenance.json",dict(created=utc_now(),summary_sha256=sha256_file(source),
        report_code_sha256=sha256_file(__file__),historical_results_used=False,training_runs=48,figures=saved))
    print(f"REPORT_COMPLETE {a.out.resolve()}")


if __name__=="__main__":
    main()
