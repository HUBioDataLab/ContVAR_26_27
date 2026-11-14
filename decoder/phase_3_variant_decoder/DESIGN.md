# Phase 3 — Varyant GO-Değişimi Decoder'ı

> **2026-09-25 mimari güncellemesi:** Güncel uygulama
> [GO koşullu, iki aşamalı WT/varyant skor cascade'i](docs/CASCADE.md).
> Aşağıdaki belge önceki `--architecture legacy` deneyinin tasarım kaydıdır;
> cascade'in loss, çıktı ve seçim kurallarını tanımlamaz.

> **Durum:** tasarım onaylandı, uygulama bekliyor.
> Bu doküman tek başına yeterlidir — uygulayan kişinin başka bir kaynağa ihtiyacı yoktur.
> Eski tasarım (`phase_3_variant_finetune/`) geçersizdi ve **silindi**; kararlarının
> akıbeti [SUPERSEDED.md](SUPERSEDED.md), ölçümleri
> [docs/GOA_INTEGRATION_DEFERRED.md](docs/GOA_INTEGRATION_DEFERRED.md) §6'da.

---

## 1. Amaç ve bağlam

Bir varyantın (tek amino asit değişimi) hangi GO fonksiyonunu **kaybettirdiğini (LOF)**,
**kazandırdığını (GOF)** veya **değiştirmediğini (NEUTRAL)** tahmin eden bir decoder eğitmek.

**Neden yeni bir faz:** phase 2 decoder'ı varyant ile WT'yi ayırt etmiyor. Ölçüldü:

| ölçüm | sonuç |
|---|---|
| `\|s_var − s_wt\|` medyanı (aynı GO indeksinde) | 0,000168 (`s_wt` medyanı 0,848 iken) |
| `s_var − s_wt` ile LOF/GOF ayırma AUROC | **0,54** |
| Phase 2'nin en iyi F decoder'ında `bm_test_auroc` | **0,4533** (rastgeleden kötü) |

Sebebi yapısal: phase 2 eğitiminde varyant embedding'lerinden **hiç gradyan akmadı** —
`variant_h5` yalnızca epoch sonunda gözlemsel metrik için kullanılıyordu. Decoder'ın varyantı
temsil etmek için hiçbir sebebi olmadı.

**Bu fazın iki kökten farkı:**

1. **Lokal embedding.** Mean-pooled protein uzayında WT ile varyant ayırt edilemiyor
   (`cos = 0,99985`). Mutasyon pozisyonundaki rezidüde ise `cos = 0,92835` — **~190 kat**
   daha güçlü ayrışma, ve mutasyon bölgesine lokalize. Decoder artık
   `[global | lokal]` çifti alıyor.
2. **Çift-bazlı, üç sınıflı, sıralama tabanlı loss.** WT ve varyant aynı decoder'dan geçiyor,
   karar ikisinin **farkı** üzerinden veriliyor.

---

## 2. Veri seti

> **Güncel cascade veri kaynağı:** `splits_cascade/` altındaki
> `stage{1,2}_{easy,hard}_{train,val,test}.tsv` dosyalarıdır. Stage 1 dosyaları
> aşağıda belgelenen satırların `FUNCTIONAL/NEUTRAL` hedefli ana görünümünü,
> Stage 2 dosyaları bunların LOF/GOF alt kümesini taşır. Loader iki görünümün
> birebir tutarlılığını doğrular. Aşağıdaki `novel_llm_dataset` sayıları eski
> üç sınıflı tasarımın tarihsel analizidir; güncel sayılar `docs/CASCADE.md` ve
> çalışma zamanı filtre raporunda tutulur.

`data/processed/novel_llm_dataset/` — `{easy,hard}_{train,val,test}.tsv`, 12 kolon:

```
protein_id  feature_type  ftId  snp  atomic_description  label
matched_go_id  matched_go_name  cosine_similarity  match_source  notes  is_combinatorial
```

`label ∈ {LOF, GOF, NEUTRAL}` · `match_source` %100 `novel_prediction_llm`.

### 2.1 İki aile: easy ve hard

| | train | val | test | LOF / GOF / NEUTRAL (train) | protein örtüşmesi |
|---|---|---|---|---|---|
| **easy** | 46.150 | 5.768 | 4.553 | 38.618 / 4.038 / 3.494 | train∩val 2.665 · train∩test 1.990 |
| **hard** | 46.072 | 5.799 | 5.783 | 38.695 / 3.967 / 3.410 | **0 · 0** |

**Tek fark split kuralı.** Easy satır-bazlı rastgele bölünmüş; hard **protein-disjoint** —
train/val/test arasında tek bir ortak protein yok.

Bu fark tesadüfi değil, deneyin kendisi. Veri protein başına kümeleniyor: bir protein train'de
30 varyantıyla geçiyorsa ve hepsi aynı GO terimi için LOF'sa, easy'nin test'i o proteinin 31.
varyantını sorduğunda model **mutasyona hiç bakmadan** doğru cevap verebilir. Veri %83,9 LOF
olduğu için bu kestirme çok kârlı. Hard'da yapısal olarak imkânsız.

**İkisi ayrı deney olarak koşulur ve yan yana raporlanır.** Aradaki fark, modelin ne kadarının
gerçek varyant sinyali, ne kadarının protein ezberi olduğunu okur.

### 2.2 Ölçülmüş özellikler

- **SNP notasyonu %100 temiz `X123Y`** (57.654/57.654). Indel, çoklu-alternatif, notasyon
  temizliği gerekmiyor.
- **Aspect:** train satırlarının %58,6 F · %35,2 P · %0,4 C · **%5,8 hiçbir aspect'te yok**
  (GO'dan kaldırılmış / obsolete terimler).
- **Alanin yanlılığı: satırların %41,9'u alanine mutasyonu.** `MUTAGEN`'de %49,3, `VARIANT`'ta
  %2,6 — klasik alanin taraması. Alanine çevrilen WT rezidüleri: S %12,3 · R %10,6 · D %10,2 ·
  K %9,5 · C %6,9 (polar/yüklü, işlevsel yan zincirler).
  **Etiket dağılımı ayrışıyor:** alanin satırları %87,1 LOF / %4,8 GOF · alanin-olmayanlar
  %81,3 LOF / **%11,7 GOF**. Yani GOF sinyalinin çoğu alanin-olmayan yarıda.
  Split'ler arası oran stabil (train %42,4 · val %40,4 · test %39,7), dağılım kayması yok.
- **`is_combinatorial` ~%14:** fenotip çoklu mutanta ait ("...when associated with A-210") ama
  biz tek SNP uyguluyoruz. **Tutuluyor, özel işlem yok** (karar verildi).
  Açık çelişki ölçüldü ve ihmal edilebilir: aynı `(protein, snp, GO)` için zıt etiket sadece
  57 grup / 125 satır (%0,22), bunların 27'si combinatorial. Asıl maliyet ölçülemeyen
  atıf hatası; bilinen çekince olarak raporda kalır.
- **GO terimi seyrekliği:** terim başına medyan **3 satır**, terimlerin **%31,4'ü tek satırlık**,
  %69,9'u ≤5 satırlık. Hard'da val/test satırlarının **%16,9 / %15,5'i** train'de hiç geçmeyen
  bir terim taşıyor. Bu, bu tasarımın bilinen en büyük zayıflığı — bkz. §7.

### 2.3 Filtre zinciri (ölçülmüş, uygulama bunu yeniden üretmeli)

Sekans kaynağı: `/media/ubuntu/8TB/mfurkan/contvar_data/raw/uniprot_dict.json`
(166.613 kayıt, `[{"Entry": ..., "Sequence": ...}, ...]` biçiminde).

| adım | easy | hard |
|---|---|---|
| ham satır | 56.471 | 57.654 |
| 1) proteinin sekansı yok (1.319 protein) | −5.244 | −5.309 |
| 2) `pos > 1022` | −2.487 | −2.531 |
| 3) pozisyon sekans uzunluğunun dışında | −1 | −1 |
| 4) `seq[pos−1] != wt_aa` | −22 | −22 |
| **kalan** | **48.717 (%86,3)** | **49.791 (%86,4)** |
| | train 39.752 · val 4.962 · test 4.003 | train 40.004 · val 4.952 · test 4.835 |

**`pos > 1022` neden zorunlu:** ESM-2 girdiyi 1022 rezidüde kırpıyor; kırpmanın dışındaki bir
pozisyonun rezidü embedding'i hiç üretilmiyor, yani lokal yarı tanımsız. (Eski tasarımda bu
tartışmalı bir karardı; lokal embedding'e geçilince mimarinin zorunlu sonucu oldu.)

**Opsiyonel filtre (bayrakla, varsayılan kapalı):** §2.2'deki 125 çelişkili satır. Aynı girdiye
zıt hedef veren tek satır grubu, elemek neredeyse bedava. İncelendiğinde LLM etiketleme
artefaktı çıktı — iki farklı spesifik aktivite aynı üst terime eşlenmiş
(`Loss of C-glucosyltransferase` LOF + `Acquisition of O-glucosyltransferase` GOF, ikisi de
`UDP-glucosyltransferase activity`), ya da aynı olgu bir kez kayıp bir kez kazanç yazılmış.

---

## 3. Mimari

### 3.1 Girdi — `[global | lokal]`

```
WT      : [ global_wt(protein)       | residue_wt (pos) ]   → logits_wt  → s_wt
varyant : [ global_var(protein, snp) | residue_var(pos) ]   → logits_var → s_var
```

- **Lokal = mutasyon pozisyonundaki tek rezidünün son-katman embedding'i** (1280-d).
  Tanım, phase 0'daki `X_local` ile birebir aynı:
  [`extract_embeddings.py:83`](../phase_0_go_pretrained/extract_embeddings.py#L83)
  `residue_emb = token_emb[0, 1:-1, :]` → `residue_emb[pos-1]`.
- **WT'nin lokali de aynı pozisyondan alınır** — iki forward pozisyon-eşleşmelidir.
  Sonuç: WT lokali protein başına sabit bir vektör değil; `{protein}_{pos}` anahtarıyla saklanır.
- Aynı decoder her iki tarafı da geçirir (paylaşılan ağırlıklar). Model hangi tarafın varyant
  olduğunu **bilmez** — ayrım yalnızca embedding'lerin kendisinden gelir.

**Girdi boyutu:** `raw_esm` / `seq_phase0` / `seq_phase1` → **2560**;
`concat_phase0` / `concat_phase1` → **5120**
(`[primary_global | raw_global | primary_local | raw_local]`).

### 3.2 Decoder

[`FFNDecoder`](../phase_2_decoder/models/ffn_decoder.py) **olduğu gibi yeniden kullanılır** —
`input_dim` parametreli, 2560/5120 kabul ediyor. Sıfırdan init edilir; phase 2 checkpoint'i
**yüklenmez**.

```
Linear(input_dim → 512) → BN → ReLU → Dropout(0.30)
Linear(512 → 1024)      → BN → ReLU → Dropout(0.30)
Linear(1024 → 512)      → BN → ReLU → Dropout(0.20)
Linear(512 → N_GO)                                   # sigmoid UYGULANMAZ
```

### 3.3 Vocab ve çıkış katmanı

- **Tek çok-aspect decoder.** F+P+C tek çıkış katmanında. Aspect'e bölmek C'yi (train 190 satır)
  kullanılamaz kılar ve deney sayısını üçe katlar; paylaşılan gövde aspect'ler arası transfer
  sağlar.
- **Vocab = o ailenin `train ∪ val ∪ test` terimleri.** Hiçbir satır düşmez.

> ⚠️ **Kayıtlı çekince.** Hard val/test satırlarının %16,9 / %15,5'i train'de hiç geçmeyen bir
> terim taşıyor → o satırlar **hiç eğitilmemiş, rastgele-init** nöronda skorlanacak. Bu bilinçli
> bir karardır; görünür kılmak için metrikler **iki kez** raporlanır (§5.2).

---

## 4. Loss — Margin Ranking (hinge)

```
d = s_wt − s_var          s_* = sigmoid(logits[go_idx]),   d ∈ [−1, 1]

LOF      y=+1  →  w_LOF     · max(0,  m − d)        sıfır olur:  d ≥ +m
GOF      y=−1  →  w_GOF     · max(0,  m + d)        sıfır olur:  d ≤ −m
NEUTRAL        →  w_NEUTRAL · max(0, |d| − m_n)     sıfır olur:  |d| ≤ m_n

L_batch = (1/N) · Σ_i  L_i
```

**Yalnızca satırın kendi `go_idx`'i denetlenir**; diğer tüm çıkışlar maskelenir.

**Parametreler:** `w_*` varsayılan **1.0** (CLI'dan verilebilir) · `m` ve `m_n` CLI parametresi,
başlangıç önerisi **`m = 0,3` · `m_n = 0,1`**.

### 4.1 Neden BCE değil

Veri %100 LLM etiketi, insan doğrulaması yok. BCE hedefi 0/1'e çivilediği için modeli emin
olamayacağı bir kesinliğe zorluyor ve gürültülü etikette ağır ceza veriyor. Hinge marjin
sağlandığı anda **gradyanı tam olarak sıfırlıyor** — kolay örnekler susuyor, model zor olanlara
odaklanıyor. %83,9 LOF'ta bunun pratik etkisi büyük.

Ayrıca **tutarlılık kazancı:** §5.1'deki çıkarım kuralı zaten `d` ekseninde eşikliyor.
BCE dolaylı olarak `d`'yi optimize ediyordu; hinge doğrudan onu optimize ediyor.

### 4.2 Uygulama kuralları

- **Hinge sigmoid olasılık uzayında hesaplanır**, logit uzayında değil. `d ∈ [−1,1]` olduğu için
  marjinin doğal bir ölçeği olur ve §5.1'deki eşiklerle doğrudan kıyaslanabilir.
- **Zorunlu kısıt: `m_n < m`.** Aksi halde NEUTRAL ile LOF/GOF hedefleri çelişir; kod bunu
  başlangıçta doğrulamalı ve ihlalde hata vermeli.
- **NEUTRAL iki taraflıdır** (`|d|`). Tek taraflı `max(0, m_n − d)` yazılırsa LOF formülünün
  birebir aynısı olur ve NEUTRAL satırları LOF gibi eğitilir. **Bu koda girmemeli.**

### 4.3 Bilinçli kabul edilen kayıp: mutlak seviye çapası yok

Saf hinge yalnızca *farkı* kısıtlıyor; tüm skorlar herhangi bir sabite kayabilir. Delta ile
çıkarım yaptığımız için bu tur sorun değil, ama `s_wt` artık "bu protein bu fonksiyona sahip"
anlamına **gelmiyor**. Bu, §7'deki GOA entegrasyonunu doğrudan etkiler.

---

## 5. Değerlendirme

### 5.1 Çıkarım kuralı

```
d = s_wt − s_var          (§4 ile aynı işaret yönü)

d > c + w/2       → LOF        (WT yüksek, varyant kaybetmiş)
|d − c| ≤ w/2     → NEUTRAL
d < c − w/2       → GOF        (varyant yüksek)
```

Üç sınıfı tek boyutlu bir eksende ayırmak **iki kesim noktası** gerektirir.
`(merkez c, genişlik w)` parametrelemesi kullanılır — `(τ_LOF, τ_GOF)` ile matematiksel olarak
aynı iki serbest parametre, ama "band ne kadar kaymış" doğrudan okunur. Bandın sıfırda
merkezlenmemesi **beklenir**: veri %83,9 LOF olduğu için gövde varyant skorunu genel olarak
aşağı itme baskısı görüyor.

> **Eşikler val'de aranır, test'e değiştirilmeden uygulanır.**
> Phase 2'nin [`benchmark_eval.py:66-72`](../phase_2_decoder/benchmark_eval.py#L66-L72)
> hatası tekrarlanmamalı — orada eşik, değerlendirilen split'in **kendi** accuracy'sinde
> aranıyordu, dolayısıyla raporlanan test F1/accuracy'si iyimserdi.

### 5.2 Metrikler

**Seçim metriği: makro-F1**, val'de aranan eşiklerle, 3 sınıf üzerinden.

Her epoch loglanır (seçime **girmez**): sınıf bazında precision/recall, karışıklık matrisi,
`d` üzerinden one-vs-rest AUPRC/AUROC, ve aşağıdaki kırılımlar.
Test metrikleri de her epoch hesaplanıp loglanır ama **seçimde kullanılmaz** (val eğrisinin
test eğrisini takip edip etmediğini sonradan okumak için).

**İki alt küme kırılımı — ikisi de bedava, ikisi de teşhis amaçlı:**

1. **Vocab kırılımı:** tüm satırlar vs. train'de görülmüş terimli satırlar.
   Aradaki fark, §3.3'teki eğitilmemiş nöron çekincesinin metriği ne kadar sulandırdığını okur.
2. **Alanin kırılımı:** `mut == A` vs `mut != A`.
   *Risk:* lokal embedding mutant rezidünün kimliğini doğrudan taşıyor, dolayısıyla model
   "mutant alanin ise LOF" kestirmesini bedavaya öğrenebilir — satırların %42'sinde %87 isabet.
   Bu bir sızıntı değil, gerçek biyolojik sinyal; ama **alanin-olmayan alt kümedeki performans
   asıl gösterge** olarak ayrıca okunur (GOF'un çoğu orada).

### 5.3 Deney gridi

**2 aile (easy / hard) × 5 embedding tipi = 10 koşu.**

Embedding tipleri phase 2'nin adlandırmasını korur
([`config.py:12`](../phase_2_decoder/config.py#L12)):
`raw_esm` · `seq_phase0` · `seq_phase1` · `concat_phase0` · `concat_phase1`.
Concat tipleri mevcut H5'lerden türetilir, **ek ESM forward'ı gerektirmez**.

---

## 6. Uygulama adımları

Dizin: `phase_3_variant_decoder/`

### 6.1 `data/variant_dataset.py`
TSV parse → `(protein_id, snp, pos, wt_aa, mut_aa, go_id, go_idx, label, is_combinatorial, split)`.
§2.3 filtreleri, **her adımın düşürdüğü satır sayısını basan özet rapor**. Vocab kurulumu (§3.3).

### 6.2 `build_pair_embeddings.py`

[`build_variant_embeddings.py`](../phase_2_decoder/build_variant_embeddings.py) üzerine kurulur.
Etrafındaki altyapı (uzunluğa göre batch'leme, `--resume`, `--backbone` ile phase_0/phase_1
yükleme, `pick_device`) **olduğu gibi yeniden kullanılır**. İki değişiklik:

**(a) `token_emb` atılmayacak.**
[`build_embeddings.py:133-142`](../phase_2_decoder/build_embeddings.py#L133-L142) `token_emb`'i
`[B, L, D]` hesaplayıp mean-pool sonrası atıyor — lokal embedding zaten orada, kimse almıyor.
`mean_pool_last_layer()`'ın yanına `pool_and_local()` yazılır: aynı forward'dan hem `pooled`
hem istenen pozisyonların rezidü vektörleri döner.

> ⚠️ **BOS kayması.** ESM tokenizer'ı sekansın başına `<cls>`, sonuna `<eos>` ekler; çıktı
> `[B, L+2, D]`. 1-tabanlı `pos` için doğru indeks **`token_emb[i, pos]`** — `pos-1` değil.
> (Phase 0 önce `token_emb[0, 1:-1, :]` diye dilimlediği için `pos-1` kullanıyor; iki
> konvansiyonu karıştırmayın. Biz dilimlemeden indeksliyoruz → `pos`.)
>
> **Bu bir doğrulama konusu değil, kod konusu:** doğru indeksi yaz ve yanına
> `assert alphabet.prepend_bos and alphabet.append_eos` koy. ESM-2 için ikisi de True,
> ama sessizce doğru kalmasına güvenmek yerine açıkça iddia edilsin.

**(b) FASTA dizini yerine manifest.** Mevcut kod varyant başına ayrı FASTA dosyası okuyor
(`fasta_dir.glob("*.fasta")`, key = dosya adı). 42.659 varyant için 42.659 dosya yazmak
gereksiz — yerine `(key, protein_id, pos, wt_aa, mut_aa)` manifest'i alınıp sekans
`uniprot_dict.json`'dan bellekte kurulur (`seq[pos−1] == wt_aa` doğrulamasıyla).
Bu, ayrı bir sekans-üretim adımını tamamen ortadan kaldırır.

**Forward sayısı (§2.3 filtreleri uygulandıktan sonra, easy+hard birleşik):**

```
varyant forward :  42.659
WT forward      :  10.688      ← 37.788 (protein, pozisyon) çifti var, ama protein başına
                                  medyan 2 (maks 114) pozisyon TEK forward'dan çekiliyor
──────────────────────────────
backbone başına :  53.347 forward
```

`pos > 1022` filtresi forward'dan **önce** uygulanmalı; `seq[:max_len]` kırpması sessizce
yanlış rezidü verir. `pool_and_local()` bunu ayrıca assert'le kontrol eder.

**Çıktı** (backbone başına iki H5):

| dosya | anahtar | kayıt | fp32 |
|---|---|---|---|
| `*_global.h5` | `{protein}` ve `{protein}_{snp}` | 10.688 + 42.659 | 0,273 GB |
| `*_local.h5` | `{protein}_{pos}` ve `{protein}_{snp}` | 37.788 + 42.659 | 0,412 GB |

Üç backbone (raw / phase_0 / phase_1) toplam ~2,1 GB; concat tipleri bunlardan türetilir.

### 6.3 `data/pair_dataset.py`
`VariantPairDataset` → `(x_wt[D], x_var[D], go_idx, label, weight)`; `D` = 2560 veya 5120,
embedding tipinden türetilir.
Phase 2'nin `GOAnnotationDataset`'i (protein başına tek örnek) bu iş için **uygun değil**.

### 6.4 `losses.py`
§4'teki hinge. `m_n < m` doğrulaması burada.

### 6.5 `train.py`
- §6.4 loss'u, §5.2 seçim ve loglama, WandB (`contvar_gridsearch` projesi, ayrı tag).
- CLI deseni phase 2'nin `train.py` / `benchmark_eval.py` ile aynı:
  `--family {easy,hard} --embedding {raw_esm,seq_phase0,seq_phase1,concat_phase0,concat_phase1}
   --global_h5 --local_h5 [--global_h5_secondary --local_h5_secondary]
   --margin --w_lof --w_gof --w_neutral --cuda`
  (revizyon: `margin_neutral`/`margin_ratio` kaldırıldı — NEUTRAL artık LOF/GOF ile
  aynı tek `margin`'e göre değerlendiriliyor, kullanıcı talebiyle.)
- **`--stage1 {none,goa}` bayrağı şimdiden konur**, varsayılan `none`. §7 açıldığında kod yolu
  hazır olsun ve ablasyon tek argümanla açılıp kapansın.

### 6.6 `evaluate.py`
§5.1 eşik araması (val) → test uygulaması. §5.2 metrik seti ve iki kırılım.
Aileler arası karşılaştırma tablosu (easy vs hard).

### 6.7 Eski dokümanların tasfiyesi — TAMAMLANDI
`phase_3_variant_finetune/` silindi. Kararlarının akıbeti [SUPERSEDED.md](SUPERSEDED.md)'de
tablo halinde, ölçümleri [docs/GOA_INTEGRATION_DEFERRED.md](docs/GOA_INTEGRATION_DEFERRED.md)
§6'da duruyor.

---

## 7. Ertelenen iş: GOA entegrasyonu

**Uygulanmayacak, ama sonuçlar zayıf çıkarsa ilk bakılacak yer.**
Detay: [docs/GOA_INTEGRATION_DEFERRED.md](docs/GOA_INTEGRATION_DEFERRED.md)

Çözdüğü problem §2.2'deki seyreklik: ~6.000 çıkışlı bir katman, satır başına tek indeks
denetleniyor, terim başına medyan 3 satır. GOA'nın tam da oraya denk geldiği ölçüldü:
≤5 satırlık 4.031 "ince" terimin %87,3'ü GOA'da ve terim başına medyan 18 GOA proteini var
(~6 kat); hard'ın "görülmemiş terim" satırlarının %85'i GOA'da mevcut.

---

## 8. Doğrulama

1. **Veri hazırlama:** filtre raporundaki sayılar §2.3 tablosuyla tutmalı
   (sekansı yok → 5.244/5.309 · `pos>1022` → 2.487/2.531 · kalan → 48.717/49.791).
2. **Sekans üretimi:** manifest'ten kurulan rastgele 20 varyant sekansında `seq[pos−1] == mut_aa`
   ve WT'den **tam 1 pozisyon** farkı doğrulanmalı.
3. **Boru hattı parmak izi — en kritik tek doğrulama.** Yeni H5'te mutasyon pozisyonundaki
   `cos(var, wt)` medyanı **~0,928** çıkmalı. Referans değerler:

   | ne | beklenen cos | bu çıkarsa |
   |---|---|---|
   | mutasyon pozisyonundaki rezidü | **0,928** | ✅ doğru |
   | komşu rezidü (bir kayma) | ~0,98–0,99 | indeksleme kaymış |
   | ≥15 rezidü uzaktaki pozisyon | 0,99975 | ciddi kayma / yanlış eşleşme |
   | mean-pooled global | 0,99985 | lokal yerine global yazılmış |

   Bu kontrol tek bir hatayı değil, **assert'le kapatılamayan bir hata sınıfını** hedefliyor:
   (a) iki H5 arasında anahtar hizalaması — WT lokali `{protein}_{pos}`, varyant lokali
   `{protein}_{snp}`; yanlış kaydı çekmek geçerli bir vektör döndürür,
   (b) `x_wt` ↔ `x_var` eşleştirmesinin gerçekten aynı `(protein, pos)`'a ait olması,
   (c) backbone / H5 karışması (`--backbone seq_phase1` ile koşup raw yoluna yazmak, ya da
   `seq_phase0` global'ini `raw_esm` lokaliyle eşlemek).
   Üçü de sessizdir ve üçünü de bu tek ölçüm ele verir. Maliyeti birkaç saniye; yakalamazsa
   bedeli 53.347 × 3 backbone ESM forward'ı artı 10 eğitim koşusu.

   **Ölçüldü (ESM-2 650M, n=300):** mutasyon pozisyonu **0,92912** · komşu 0,98522 ·
   ≥15 uzak 0,99856 · global 0,99929 → `[PASS]`. Araç:
   `python -m phase_3_variant_decoder.verify_local_embeddings --n 300 --cuda 0`

   *(BOS kayması buraya dahil değil — o §6.2'de assert ile kapatılıyor.)*
4. **Çift hizalaması:** rastgele satırlarda WT lokalinin anahtarındaki `pos` ile satırın `pos`'u
   eşleşmeli; varyant lokali de aynı pozisyondan gelmeli.

   *Not — izoform riski zaten kapandı:* UniProt pozisyonları kanonik izoforma göre verilir ve
   `uniprot_dict.json` başka bir izoform taşısaydı tüm pozisyonlar kayardı. §2.3'teki filtre 4
   (`seq[pos−1] == wt_aa`) bunu test etti: 49.813 satırda **22 uyuşmazlık (%0,04)**. Sekanslar
   yanlış olsaydı amino asit kompozisyonu gereği ~%90 uyuşmazlık beklenirdi. Ayrı bir kontrol
   gerekmiyor.
5. **Öğrenme kanıtı:** `d` ile LOF/GOF ayırma AUROC'u, phase 2 taban çizgisi **0,54**'ten
   anlamlı yükselmeli. Yükselmiyorsa sırasıyla: sınıf ağırlıkları (§4), GOA entegrasyonu (§7).
6. **Dejenere çözüm kontrolü.** Eğitim loss'u düşerken val makro-F1 yerinde sayıyorsa ve
   **NEUTRAL recall'ı ~0** ise model "her varyantı aşağı it" numarasına yerleşmiştir.
   NEUTRAL per-class recall'ı bu teşhisin doğrudan göstergesidir, her epoch izlenmeli.
   *Risk yapısal:* "her varyantı `m` kadar aşağı it" hamlesi LOF hinge'ini satırların %83,9'unda
   sıfırlar; buna direnen tek şey GOF (%8,6) + NEUTRAL (%7,4) = %16. `w_GOF` / `w_NEUTRAL`
   gerekirse burada devreye girer.

   *Ve bu numara embedding uzayında GERÇEKTEN erişilebilir — ölçüldü:*

   ```
   ‖ortalama delta‖ / ortalama ‖delta‖  =  0,3925     (rastgele olsa ~0,028 = 1/√1280)
   delta·u > 0 olan çift oranı          =  %86,8      (u = ortalama delta yönü, n=6.655)
   aynı protein, farklı snp delta'ları  :  medyan cos +0,386
   FARKLI proteinlerin delta'ları       :  medyan cos +0,114   ← sıfır değil, pozitif
   ```

   Yani WT→varyant farkları **ortak bir `u` yönü** paylaşıyor. `u` boyunca monoton azalan
   tek bir fonksiyon, çiftlerin %86,8'inde `s_var < s_wt` üretir ve veri %83,9 LOF olduğu
   için satırların ~%78'ini **hiçbir varyant-spesifik bilgi öğrenmeden** doğru yapar.
   Model bunu "varyant mı" diye bakarak değil, sadece yön duyarlılığıyla başarır —
   mutlak konuma hiç bakmaz. NEUTRAL ve GOF terimleri bu çözümü bozan tek kuvvet.
7. **Marjin sağlanmış mı (§4 ↔ §5.1 bağı).** Val'de bulunan bandın yarı genişliği `w/2`,
   **`m_n` ile `m` arasına** düşmeli. Ayrıca `|d|` dağılımı sınıf bazında raporlanmalı —
   LOF/GOF'ta kütlenin `m`'in ötesinde, NEUTRAL'da `m_n`'in içinde toplanması beklenir.
   Hinge sıfırlanmış ama metrik kötüyse sorun eşiklerde değil **temsildedir**.
8. **Vocab çekincesi (§3.3):** tüm-satır metriği ile "train'de görülmüş terim" alt kümesi
   arasındaki fark raporlanmalı; hard'da uçurum varsa eğitilmemiş nöronlar metriği
   sulandırıyordur.
9. **CLI:** aynı script 5 embedding tipi × 2 aile ile hatasız koşmalı; concat modunda
   `--global_h5_secondary` / `--local_h5_secondary` zorunluluğu doğrulanmalı.

---

## 9. Kaynak dosya konumları (bu makinede)

```
Veri seti         splits_cascade/stage{1,2}_{easy,hard}_{train,val,test}.tsv
WT sekansları     /media/ubuntu/8TB/mfurkan/contvar_data/raw/uniprot_dict.json
GOA              data/raw/goa_2025-12-04_swissprot_noiea.tsv        (§7 için)
GO ontolojisi     data/raw/go-basic.obo                              (§7 için)
Backbone'lar      /media/ubuntu/8TB/mfurkan/contvar/checkpoints/
                    run_phase_0_epoch_40_continue_full_model_best.pt   (phase 0)
                    phase1_backbone_best.pt                            (phase 1)
```
