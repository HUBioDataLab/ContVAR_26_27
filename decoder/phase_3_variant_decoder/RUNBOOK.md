# Phase 3 — çalıştırma sırası

## Ayrı deney — üç sınıflı Concat ve kayıtlı cascade

WT/VAR/GO'yu birlikte okuyup LOF/GOF/NEUTRAL üreten Concat'ın mevcut checkpoint'ini
eski cascade ile aynı örneklerde karşılaştırma komutu:
[Concat–cascade ve mutasyon duyarlılığı](docs/CONCAT_VS_CASCADE_TR.md).
Compare kaldırılmıştır. `--direct_manifest` mevcut Concat ağırlıklarını kullanır;
eğitim başlatmaz. Genel sonuçlara EASY challenge ve HARD protein–GO içi ranking
eklenir. W&B bölümü `00_Concat_vs_Cascade`; yeni saved view'da ilk sıradadır.

## Grafikler ve W&B sunumu

Encoder etiketleri, metrik tanımları ve kayıtlı sonuçları eğitim yapmadan yeniden sunma komutu için [W&B reporting guide](docs/WANDB_REPORTING.md) belgesine bakın. `seq_phase1` grafiklerde **Phase 0 (GO) -> Phase 1 (DMS)** olarak gösterilir. `Local` adı korunur; mevcut kodda tek mutasyon pozisyonunun embedding’idir.

Tasarım: [DESIGN.md](DESIGN.md) · Tüm komutlar **repo kökünden** çalıştırılır.

**Güncel mimari: [iki aşamalı WT/varyant skor cascade'i](docs/CASCADE.md).**
Cascade'in varsayılan loss'u **hinge**'dir. Hinge ve squared hinge'i aynı veri ve
ayarlarla, ayrı W&B bölümlerinde çalıştırmak için
[loss karşılaştırması ve server komutu](docs/CASCADE_HINGE_EXPERIMENT.md).

`train` komutu artık varsayılan olarak GO girdili cascade çalıştırır. Aşağıdaki
eski sweep ve sonuç karşılaştırma akışı legacy hinge deneyi içindir; yeni mimarinin
eğitim/değerlendirme komutları CASCADE.md içindedir.

GO terimlerini Anc2vec girdisine çevirmek ve kapsam analizi üretmek için:
[Anc2vec edinme, analiz ve decoder bağlantısı](docs/ANC2VEC.md).

Ana veri kaynağı `splits_cascade/` dizinidir. Sunucuda yolu
`PHASE3_DATASET_DIR=/media/ubuntu/8TB/$USER/contvar_phase3/datasets/splits_cascade`
ile sabitleyin. Ağırlıklı/ağırlıksız cascade karşılaştırmasının güncel tek-komut
akışı ve üretilen grafikler [CASCADE.md](docs/CASCADE.md) içinde belgelenir.

```bash
cd /home/mfurkan/contvar/CONTVAR-Sekans-model-
conda activate contvar
```

---

## A. Veri hazırlama kontrolü  (~30 sn, GPU gerekmez)

```bash
python -m phase_3_variant_decoder.data.variant_dataset
```

**Beklenen** (`splits_cascade`, çelişkili kayıt filtresi kapalı):

```
easy   ham 66.368  →  KALAN 57.310 (%86.4)   train 46.953 · val 5.819 · test 4.538
hard   ham 68.006  →  KALAN 58.787 (%86.4)   train 47.098 · val 5.825 · test 5.864
       sekansı yok −6.092/−6.183 · pos>1022 −2.941/−3.011 · WT uyuşmuyor −24
```

---

## B. Lokal embedding doğrulaması  (~3 dk, 1 GPU)

**Tam sample kapsamı için:**
[Encoder/GO embedding kapsam kontrolü](docs/INPUT_EMBEDDING_COVERAGE_TR.md)
ile mevcut H5'leri GPU kullanmadan doğrudan denetleyin. Bu kontrol ham satırları,
eğitim filtrelerinden kalanları ve WT/VAR global/local + GO girdileri tamam olanları
ayrı raporlar. Aşağıdaki cosine fingerprint bir tanı kontrolüdür; tek başına
checkpoint veya residue doğruluğunun kanıtı değildir.

**Aşama C'den önce örnek kontrol yapılmalı.** Bu fingerprint pozisyon seçimine
ilişkin tanısal ipucu sağlar. Kesin eşleşme kontrolü için bilinen checkpoint'ten
örnek vektörler yeniden üretilip karşılaştırılmalıdır.

```bash
python -m phase_3_variant_decoder.audit --mode live --n 300 --cuda 3
```

**İlk raw ESM-2 kontrolünde gözlenen örnek değerler (checkpoint'e bağlıdır):**

```
mutasyon pozisyonundaki rezidü     0.929      ← kritik sayı
komşu rezidü (pos±1)               0.985
≥15 rezidü uzakta                  0.999
mean-pooled global                 0.999
[PASS]
```

`[FAIL]` veya yüksek local cosine görülürse BOS/pozisyon kayması, yanlış eşleştirme
ve checkpoint etkisini inceleyin. Bu eşikler tek başına hata tanısı koymaz.

---

## C. Embedding üretimi  (backbone başına 53.347 forward)

Üç komut **birbirinden bağımsız, paralel koşulabilir** — ayrı GPU ver.
Hepsi `--resume` destekliyor; kesilirse aynı komutu tekrar çalıştır.

```bash
# C1 — raw ESM-2 (backbone yok)          GPU 3
nohup python -m phase_3_variant_decoder.build_pair_embeddings \
  --out_prefix data/processed/phase3_raw_esm \
  --esm_model esm2_t33_650M_UR50D \
  --cuda 3 --batch_size 8 --resume \
  > logs/phase3_emb_raw.log 2>&1 &

# C2 — phase 0 backbone                  GPU 4
nohup python -m phase_3_variant_decoder.build_pair_embeddings \
  --out_prefix data/processed/phase3_seq_phase0 \
  --esm_model esm2_t33_650M_UR50D \
  --backbone /media/ubuntu/8TB/mfurkan/contvar/checkpoints/run_phase_0_epoch_40_continue_full_model_best.pt \
  --cuda 4 --batch_size 8 --resume \
  > logs/phase3_emb_p0.log 2>&1 &

# C3 — phase 1 backbone                  GPU 1
nohup python -m phase_3_variant_decoder.build_pair_embeddings \
  --out_prefix data/processed/phase3_seq_phase1 \
  --esm_model esm2_t33_650M_UR50D \
  --backbone /media/ubuntu/8TB/mfurkan/contvar/checkpoints/phase1_backbone_best.pt \
  --cuda 1 --batch_size 8 --resume \
  > logs/phase3_emb_p1.log 2>&1 &
```

Çıktı, prefix başına iki dosya (toplam ~0,69 GB):

```
data/processed/phase3_{raw_esm,seq_phase0,seq_phase1}_global.h5    10.688 + 42.659 anahtar
data/processed/phase3_{raw_esm,seq_phase0,seq_phase1}_local.h5     37.788 + 42.659 anahtar
```

**Model:** `esm2_t33_650M_UR50D` (1280-d) — phase 0/1/2 ile aynı. Komutlarda açıkça
yazılı; varsayılan da budur ama devredilebilirlik için elle belirtiliyor. `--backbone`
verilen koşularda da taban model aynı, üzerine `esm_state_dict` yükleniyor.

**Bellek yetmezse** `--batch_size 4` (phase 2'nin varsayılanı). `concat_*` tipleri bu
dosyalardan türetilir, **ayrıca üretim gerekmez.**

### C sonrası kontrol

```bash
python - <<'EOF'
import h5py
for p in ["raw_esm", "seq_phase0", "seq_phase1"]:
    for kind, exp in (("global", 53347), ("local", 80447)):
        f = f"data/processed/phase3_{p}_{kind}.h5"
        with h5py.File(f, "r") as h5:
            n = len(h5.keys())
        print(f"{f:<52s} {n:>7,} / {exp:,}  {'OK' if n == exp else 'EKSİK'}")
EOF
```

---

## D. Hiperparametre araması — 20 sweep (2 aile × 5 embedding × 2 input_mode)

Sabit hiperparametreyle koşmak yerine **WandB Sweeps / `method: bayes`** ile aranıyor.
Her `(aile, embedding)` kendi optimal ayarını bulur.

### D1 — Cache'i kur (20 kombinasyon, her biri bir kez)

Sweep agent her trial'ı **ayrı process** olarak başlatır; H5'ten 80 bin dataset okumak
**~45 s** sürüyor ve her trial'da tekrarlanırdı. Cache bunu **~2,5 s**'ye indiriyor.

```bash
CD=data/processed/phase3_cache
for FAM in hard easy; do
  for MODE in both global; do
    for E in raw_esm seq_phase0 seq_phase1; do
      python -m phase_3_variant_decoder.data.pair_dataset --build_cache \
        --family $FAM --embedding $E --input_mode $MODE \
        --global_h5 data/processed/phase3_${E}_global.h5 \
        --local_h5  data/processed/phase3_${E}_local.h5 --cache_dir $CD
    done
    for P in phase0 phase1; do
      python -m phase_3_variant_decoder.data.pair_dataset --build_cache \
        --family $FAM --embedding concat_$P --input_mode $MODE \
        --global_h5 data/processed/phase3_seq_${P}_global.h5 \
        --local_h5  data/processed/phase3_seq_${P}_local.h5 \
        --global_h5_secondary data/processed/phase3_raw_esm_global.h5 \
        --local_h5_secondary  data/processed/phase3_raw_esm_local.h5 --cache_dir $CD
    done
  done
done
```

Toplam **17 GB**. İş bitince silinebilir — türetilmiş veridir, H5'lerden yeniden kurulur.

### D2–D4 — Sweep'ler: tek orkestratör

Elle 20 kez `wandb sweep` + `wandb agent` yazmak yerine
[run_sweeps.py](run_sweeps.py). Oluşturulan sweep'ler `runs/sweeps.json`'a yazılır;
sonraki komutlar oradan okur, ID taşımak gerekmez.

```bash
# 1) YAML'ları üret  (lite: 20 trial · 4 parametre · full: 80 trial · 8 parametre)
python -m phase_3_variant_decoder.make_sweeps --preset lite

# 2) 20 sweep oluştur — GPU kullanmaz, birkaç saniye
python -m phase_3_variant_decoder.run_sweeps create --preset lite

# 3) durumu gör
python -m phase_3_variant_decoder.run_sweeps status

# 4) boş GPU'lara dağıt — önce ne olacağına bak
python -m phase_3_variant_decoder.run_sweeps launch --gpus 3 6 --dry_run
python -m phase_3_variant_decoder.run_sweeps launch --gpus 3 6

# 5) biten sweep'lerin en iyisini kalıcılaştır
python -m phase_3_variant_decoder.run_sweeps collect --cuda 3
```

`launch` her GPU'ya bir kabuk betiği yazıp arka planda başlatır; o GPU'daki sweep'ler
**sırayla** koşar. Yani 20 sweep'i 2 GPU ile bitirmek için tek komut yeter, tmux'ta
20 pane açmak gerekmez.

```
20 sweep → 2 GPU
  GPU 3  (10 sweep)     GPU 6  (10 sweep)
      hard/raw_esm/both     hard/seq_phase0/both
      ...                   ...
izlemek için:  tail -f logs/sweeps/agent_gpu*.log
```

`launch` varsayılan olarak **yalnızca tamamlanmamış** sweep'leri dağıtır — kesilip
tekrar çalıştırılırsa biten sweep'leri atlar.

| preset | trial | aranan parametre | sabit |
|---|---|---|---|
| **lite** | 20 | `margin` `w_gof` `w_neutral` `lr` | wd=1e-4 · dropout=0,3 · bs=256 · hidden=512-1024-512 |
| **full** | 80 | yukarıdakiler + `weight_decay` `dropout` `batch_size` `hidden_dims` | — |

`lite`de sabitlenenler mimari/optimizasyon ayarları; veri değiştiğinde optimal değerleri
fazla kaymaz. Aranmaya devam edenler ölçülen asıl sorunun eksenleri: sınıf dengesizliği
(`gof_recall` 0,20 · `neutral_recall` 0,11) ve bandın yanlış yere oturması
(`w/2` = 0,038, `m_n` = 0,1'in altında).

**20 sweep = 2 aile × 5 embedding tipi × 2 input_mode.** Ablasyon için ayrıca bir şey
yapmana gerek yok, `create` ikisini de kuruyor.

### D5 — `input_mode` ablasyonu: phase 3'ün merkezî iddiası

`both` = `[global | lokal]` · `global` = lokal olmadan. Aradaki fark, lokal embedding'in
**gerçek** katkısı. D2'deki döngü ikisini de üretiyor; ayrıca bir şey yapmana gerek yok.

Sabit hiperparametreyle hızlı bir ön bakış istersen:

```bash
for MODE in both global; do
  python -m phase_3_variant_decoder.train \
    --architecture legacy \
    --family hard --embedding raw_esm --input_mode $MODE \
    --global_h5 data/processed/phase3_raw_esm_global.h5 \
    --local_h5  data/processed/phase3_raw_esm_local.h5 \
    --cache_dir data/processed/phase3_cache --cuda 3
done
```

> `global` modu lokal H5'e **hiç dokunmuyor** — dosya silinse bile çalışır. Ablasyonun
> dürüstlüğü kod düzeyinde garanti (doğrulandı).

### Her koşu otomatik kaydediliyor

- **`runs/results.csv`** — koşu başına bir satır, 120 kolon, append-only. Sweep yarıda
  kesilse bile o ana kadarki her şey diskte.
- **wandb** — her koşu ayrı run; `fam:` `emb:` `mode:` `src:` tag'leriyle filtrelenebilir.

### İzlenecek iki şey (DESIGN §8.6)

```
[012] loss=0.0412  val_macroF1=0.4821 (c=+0.031 w=0.180)  val_neutral_recall=0.312
                                                            ↑ ~0'a düşerse dejenere çözüm
```

---

## E. Analiz raporu

```bash
# nihai rapor — 10 bölüm
python -m phase_3_variant_decoder.report

# satır-bazlı skor dökümü (hata analizi + delta sinyali teşhisi)
python -m phase_3_variant_decoder.evaluate \
  --checkpoint phase_3_variant_decoder/runs/decoder_hard_raw_esm_both.pt \
  --global_h5 data/processed/phase3_raw_esm_global.h5 \
  --local_h5  data/processed/phase3_raw_esm_local.h5 \
  --dump_scores phase_3_variant_decoder/runs/scores_hard_raw_esm_both.csv --cuda 3

# rapora delta teşhisini de ekle
python -m phase_3_variant_decoder.report \
  --scores phase_3_variant_decoder/runs/scores_hard_raw_esm_both.csv

# results.csv silindiyse wandb'den üret
python -m phase_3_variant_decoder.report --from_wandb
```

Rapor bölümleri:

| # | bölüm | ne söyler |
|---|---|---|
| 1 | Leaderboard | en iyi koşular, hiperparametreleriyle |
| 2 | Grid | embedding tipi × aile × input_mode, hücre başına en iyi |
| 3 | **ABLASYON: input_mode** | **`both − global` = lokal embedding'in gerçek katkısı** |
| 4 | easy vs hard | genelleme uçurumu = protein ezberi payı |
| 5 | Teşhis kırılımları | görülmemiş GO terimi (hard'da ~%16) · alanin kestirmesi |
| 6 | Sınıf dengesi | NEUTRAL recall = dejenere çözüm göstergesi |
| 7 | Marjin sağlığı | `w/2` gerçekten `m_n` ile `m` arasında mı |
| 8 | Hiperparametre duyarlılığı | Spearman ρ + en iyi %10 aralığı (≥12 koşu gerekir) |
| 9 | Phase 2 karşılaştırması | `lof_gof_auroc` vs taban çizgisi 0,54 |
| 10 | Delta sinyali | sınıf başına `d` dağılımı, "ölü sinyal" kontrolü (`--scores` ile) |

### Nasıl okunur

| bakılacak | anlamı |
|---|---|
| **Bölüm 3, `fark` sütunu** | Lokal embedding olmadan ne kaybediyoruz. ~0 çıkarsa phase 3'ün merkezî varsayımı çürümüş demektir — raporun bunu açıkça yazması gerekir. |
| `test_lof_gof_auroc` | **Phase 2 taban çizgisi 0,54.** Eşikten bağımsız, asıl gösterge. |
| `macro_f1` tek başına | Yanıltıcı: %83,9 LOF olduğu için hiçbir şey öğrenmeyen model bile ~0,33 alır. |
| `easy` vs `hard` | Ne kadarı gerçek varyant sinyali, ne kadarı protein ezberi. |
| `seen` vs `seen_not` | Train'de görülmemiş GO terimli satırlar (hard test'in %16,5'i) metriği ne kadar sulandırıyor. |
| `ala` vs `ala_not` | "Mutant alanin ise LOF" kestirmesi. Asıl performans `ala_not`'ta. |
| `neutral_recall` | Dejenere çözüm dedektörü. |

---

## Ertelenen

`--stage1 goa` bayrağı var ama **uygulanmadı**; kullanılırsa hata verir.
Gerekçe ve tasarım: [docs/GOA_INTEGRATION_DEFERRED.md](docs/GOA_INTEGRATION_DEFERRED.md).
Sonuçlar zayıf çıkarsa ilk bakılacak yer orası.


## Separate input-control experiment

`python -m phase_3_variant_decoder.compare_input_controls` trains GO-only MLP,
WT + GO MLP, Mutation + GO MLP and Pair-MLP on the same frozen embeddings, with a
train-only GO lookup reference. It uses its own three-class classification
objective and W&B group; cascade and hinge experiments remain separate.
See [input-control protocol and server commands](docs/INPUT_CONTROL_EXPERIMENT.md).
`--family both` produces separate EASY/HARD runs; `--upload_existing` uploads a
saved `input_comparison.json` without retraining.

## Mutation sensitivity: frozen-checkpoint evaluation

`python -m phase_3_variant_decoder.evaluate_mutation_sensitivity` adds an EASY
train-majority challenge and within-protein–GO variant ranking (HARD primary).
It loads existing cascade checkpoints without retraining or selecting thresholds
on test. Results use the dedicated `00_Mutation_Sensitivity` W&B section;
`--create_workspace` creates a new saved view with this section at the top.
See [definitions, data counts, and server commands](docs/MUTATION_SENSITIVITY_EVALUATION_TR.md).

Protocol v2 freezes the EASY challenge using raw-train majority labels before
input filtering. HARD reports covered original row pairs as primary and clean
distinct-variant pairs separately. Raw manifests retain every excluded row/pair
and its reason. On the audited local data, original counts are 570 EASY rows and
165 HARD groups / 4,572 pairs; covered counts are 499 and 144 / 3,799 respectively.
The former 498 EASY count also reflected a train-majority change, not just
missing model inputs. See [full reconciliation](../docs/MUTATION_COHORT_RECONCILIATION_TR_2026_10_27.md).

Use `python -m phase_3_variant_decoder.audit_mutation_cohorts --help` for a
CPU-only data audit without checkpoints. Re-run evaluation to obtain v2 model
metrics; uploading a v1 JSON does not recompute its cohorts or results.
