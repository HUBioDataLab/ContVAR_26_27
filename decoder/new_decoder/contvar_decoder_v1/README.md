# ContVAR Decoder v1 — gönderim paketi

> **17 Kasım 2026 mean-window çalışma kopyası:** Güncel kurulum ve deney komutları için [BASLA.md](BASLA.md) dosyasını kullanın. Yeni ana komut `python -m contvar_decoder.window_suite`'tir. Aşağıdaki açıklamalar ve `metadata/file_manifest.json`, orijinal devir sürümünün tarihsel kayıtlarıdır. `results/full_seed42` değiştirilmeden korunmuştur.

Bu paket iki hazır biyolojik sinyalle çalışan GO-conditioned decoder'ı, doğru hedef biçimleriyle hazırlanmış splitleri, gerekli frozen embeddingleri, testleri ve tam veri deneylerini bir araya getirir. Encoder yeniden eğitilmez. Başlangıç tarihi: 9 Kasım 2026.

Önce [mimari açıklamasını](docs/MIMARI.md), deney tamamlandıktan sonra [tam sonuç raporunu](docs/SONUCLAR.md) okuyun. Görsel: [mimari.svg](docs/mimari.svg). Paket içindeki `results/full_seed42/summary.json` tam sayısal kayıttır; `results/package_smoke/` yalnız pipeline kontrolüdür.

## İçerik

```text
contvar_decoder_v1/
  contvar_decoder/       # Model, native GO reader, hazırlama, eğitim, inference, suite
  tests/                # Model + veri kapsamı + hedef dönüşümü kontrolleri
  docs/                 # Katmanlar, veri politikası, sonuçlar ve mimari görseli
  source_splits/        # Orijinal multiclass/parallel/cascade ZIP ve kaynak raporları
  prepared_splits/
    multiclass/         # Üç sınıflı kaynak splitlerden hazırlanmış tablolar
    parallel/           # Paralel head kaynak splitlerden hazırlanmış tablolar
    manifest.json       # Hash'ler, satır sayıları, kaynak eşdeğerliği kontrolleri
  encoder_inputs/data/
    embeddings/        # WT/VAR global ve mutation-site H5 dosyaları
    go/                # 200-D Anc2vec NPZ
    sequences/         # Hazırlama için WT sekansları
  results/full_seed42/  # Checkpointler, epoch geçmişleri ve örnek düzeyinde tahminler
  metadata/            # Köken, test doğrulaması ve dosya manifesti
  requirements.txt
  requirements-lock.txt
  pyproject.toml
```

Tam paket bütün veriyi içerir. Kod/sonuç paketi daha küçüktür; aynı embedding girdileri ayrıca gerekir. Böyle bir kurulumda komutlara `--package /path/to/gonderilecek_decoder_baslangic_v1` verilebilir. Yolun altında `data/embeddings` ve `data/go` bulunmalıdır.

## Kurulum ve ilk kontrol

Python 3.10+ kullanın. Komutlar bu README'nin bulunduğu klasörden çalıştırılır. Paket, kurulu bir `.venv` veya kişisel makineye ait Python yolu içermez.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest tests -q
python -m contvar_decoder.verify --full
```

Kayıtlı sürümler `environment.json` ve `requirements-lock.txt` içindedir. Farklı işletim sisteminde birebir sürümler bulunmuyorsa `requirements.txt` işlevsel alt sınırları içerir; farklı ortamla elde edilen sonuçların birebir aynı olması garanti edilmez. Mevcut tam deney CPU, PyTorch 2.15.0 ve seed 42 ile yürütülür; CUDA/MPS isteğe bağlıdır.

## Doğru veri nasıl seçiliyor?

| Model | Kullanılan kaynak | Hazır girdi klasörü | Öğrenme hedefi |
|---|---|---|---|
| Paralel + skor farkı | `parallel-*.zip` | `prepared_splits/parallel` | `target_lof`, `target_gof` |
| Paralel + ortak girdi | `parallel-*.zip` | `prepared_splits/parallel` | `target_lof`, `target_gof` |
| Üç sınıflı + skor farkı | `multiclass-*.zip` | `prepared_splits/multiclass` | `target` = LOF/GOF/NEUTRAL |
| Üç sınıflı + ortak girdi | `multiclass-*.zip` | `prepared_splits/multiclass` | Aynı üç sınıflı hedef |

İki formatın kaynak satırları eşdeğer olsa da gerçek hedef sütunları kendi kaynak arşivinden okunur ve doğrulanır. Cascade stage1/stage2 ayrı bir referans yapıdır; bu dört modelin verisi olarak üst üste eklenmez.

Easy ve hard ayrı modellerdir. Kaynak train/val/test atamaları korunur. Single SNP (`X123Y`) ve `is_combinatorial=False` zorunludur. `head_split_report.txt` gereğince augmentation train'de tutulur, validation ve testte çıkarılır. Kullanıcı kararıyla aynı girdiye birden fazla etiket taşıyan **tek mutasyonlu** satırlar korunur ve ayrı raporlanır. Etiket sırası: **LOF=0, GOF=1, NEUTRAL=2**.

| Aile | Train | Validation | Test |
|---|---:|---:|---:|
| Easy | 41.720 | 4.677 | 4.515 |
| Hard | 41.872 | 4.973 | 4.915 |

Bunlar gerekli mevcut embeddingleri geçerli olan satırlardır; kaynak ZIP'lerin tamamı değildir. Eksik girdilere sıfır/rastgele vektör verilmez. Dışlanan örnekler her formatın `*_excluded.tsv` dosyalarında nedenleriyle bulunur. [Veri politikası](docs/VERI_VE_DENEY_PROTOKOLU.md) kapsamı ayrıntılandırır.

## Mevcut checkpoint ile tahmin

`docs/SONUCLAR.md` hangi modelin validation ile seçildiğini belirtir. Aşağıdaki örnek mevcut bir modelin validation tablosuna tekrar tahmin üretir:

```bash
python -m contvar_decoder.predict \
  results/full_seed42/hard/multiclass_joint_lambda0/best.pt \
  --family hard --split val --out results/new_validation_readback
```

Inference modeli eğitmez. GO hash'i, etiket sırası, tek mutasyon kapsamı ve easy/hard ailesi kontrol edilir. Çıktı: `metrics.json` ve `predictions.tsv`. Bu komut varsayılan olarak cached örnekler içindir; yeni ham protein sekansından ESM çıkarımı yapmaz.

## Tam deneyleri yeniden çalıştırma

Paket hazır split içerir. Kaynaktan yeniden hazırlamak isterseniz mevcut klasörün üstüne yazmak yerine yeni çıktı yolu seçin:

```bash
python -m contvar_decoder.prepare_all --out prepared_splits_rebuilt
python -m contvar_decoder.suite \
  --data-dir prepared_splits_rebuilt --epochs 20 --patience 5 --seed 42 \
  --out results/new_full_seed42
```

Mevcut hazır verilerle doğrudan:

```bash
python -m contvar_decoder.suite --epochs 20 --patience 5 --seed 42 \
  --out results/new_full_seed42
```

Suite dört mimariyi easy ve hard üzerinde eğitir. Joint modellerde lambda=0 ve 0.1 ayrı koşullardır; skor farkı modellerinde reverse loss yapısal olarak sıfır olduğu için ikinci kez aynı eğitim yapılmaz. Toplam **12 ayrı eğitim** vardır. Önce bütün eğitimler tamamlanır, her modelin checkpoint'i validation macro-F1 ile seçilir, joint modellerin lambda seçimi yine yalnız validation ile sabitlenir. Sonra test değerlendirmeleri yapılır. Seçim dosyası `selection_before_test.json` test hesaplarından önce yazılır.

Tek bir mimariyi çalıştırmak için:

```bash
python -m contvar_decoder.train --family hard --head parallel \
  --pair-mode score_difference --epochs 20 --out results/new_parallel_difference

python -m contvar_decoder.train --family hard --head multiclass \
  --pair-mode joint --lambda-reverse 0.1 --epochs 20 --out results/new_joint_reverse
```

`--evaluate-test` tekli eğitim komutunda final test değerlendirmesini açar. Hiperparametreleri test skorlarıyla seçmeyin. Sınıf ağırlıkları bu teslimin ana deneyinde kapalıdır; `--balance` ayrı bir deneydir. Tekrarlı seed çalışmaları `--seed 43`, `--seed 44` ile yeni klasörlere yapılabilir; bu teslimin ana sonuçları tek seed'e aittir.

## Yapılanlar ve kapsam sınırı

Hazır olan: mutation-site/global adapter'ları, paylaşılan attention reader, joint fusion veya ayrı skor çıkarma, paralel veya üç sınıflı head, reverse metriği/opsiyonel loss, doğru split kontrolü, checkpoint ve raporlar.

Local sequence window, 3B pocket, ayrı comparison reader/LOCAL_PAIR ve pretrained STAR-GO ağırlık transferi bu sürümde yoktur. Gelecekte pooled local/pocket sinyalleri `scale_dims` ve availability maskesiyle eklenebilir; residue pencereleri için ayrıca uygun readout gerekir. Cache üstünden ESM/LoRA eğitimi yapılmaz.
