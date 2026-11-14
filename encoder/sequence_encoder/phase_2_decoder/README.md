# Sekans_decoder — Phase 3: GO Function Decoder (Sekans Tabanlı)

ESM-2 protein embedding'lerinden GO fonksiyon anotasyonlarını tahmin eden decoder modülü.

## Pipeline Konumu

```
Phase 1 (GO Pretrain)  →  Phase 2 (DMS Fine-tune)  →  Phase 3 (Decoder) ← Biz buradayız
server_go_pretrained/     src/scripts_yeni/            Sekans_decoder/
```

## Modül Yapısı

```
Sekans_decoder/
├── config.py              # Tüm hiperparametreler ve dosya yolları
├── data/
│   ├── parse_goa.py       # GO anotasyon parser
│   └── dataset.py         # H5 embedding + multi-hot label dataset
├── models/
│   └── ffn_decoder.py     # FFN decoder (512→1024→512→N_GO)
├── evaluate.py            # mAP, F1, MCC, Precision, Recall metrikleri
├── train.py               # Eğitim loop'u
├── inference.py           # Tek protein tahmin + ContVAR_fitness
├── predict_variants.py    # Toplu varyant tahmini (CSV çıktı)
├── test_eval.py           # Test seti değerlendirme
└── README.md              # Bu dosya
```

## Gerekli Dosyalar

| Dosya | Açıklama |
|-------|----------|
| `data/raw/goa_2025-12-04_swissprot_noiea.tsv` | GO anotasyonları (SwissProt, GAF 2.2, IEA/ND hariç) |
| `data/raw/go-basic.obo` | GO ontoloji hiyerarşisi (ancestor propagation için) |
| `data/processed/esm2_t33_650M_UR50D_protein_embedding.h5` | ESM-2 650M mean-pooled embedding'ler |
| `phase_0_go_pretrained/protein_uniref50.tsv` | Protein → UniRef50 cluster eşlemesi |
| `phase0_protein_split_removed_graphless.json` | Protein → train/val/test split ataması |
| `embeddings_variable.h5` | Varyant embedding'leri (predict_variants için) |

`go-basic.obo` indirmek için:
```bash
wget https://purl.obolibrary.org/obo/go/go-basic.obo -P data/raw/
pip install obonet
```

## Kullanım

### 1. Eğitim

```bash
# Molecular Function (MF) decoder eğitimi
python -m Sekans_decoder.train --aspect F

# Biological Process (BP) decoder eğitimi
python -m Sekans_decoder.train --aspect P

# Cellular Component (CC) decoder eğitimi
python -m Sekans_decoder.train --aspect C
```

### 2. Test Değerlendirme

```bash
python -m Sekans_decoder.test_eval --aspect F
```

### 3. Varyant Tahmini

```bash
python -m Sekans_decoder.predict_variants \
    --var_h5 embeddings_variable.h5 \
    --out predictions_f.csv \
    --aspect F
```

## Çıktılar

- `go_vocab_{f|p|c}.json` — GO term → index eşlemesi
- `decoder_best_{f|p|c}.pt` — En iyi model checkpoint'u
- `predictions_{f|p|c}.csv` — Varyant tahmin sonuçları

## Mimari

```
ESM-2 Mean-Pool (1280)
        ↓
  Linear(1280→512) → BN → ReLU → Dropout(0.30)
  Linear(512→1024) → BN → ReLU → Dropout(0.30)
  Linear(1024→512) → BN → ReLU → Dropout(0.20)
  Linear(512→N_GO) → raw logits
        ↓
   BCEWithLogitsLoss (eğitim)
   sigmoid → threshold (inference)
```

## GO Annotation Preprocessing

`data/parse_goa.py` iki aşamalı ön-işleme uygular:

### 1. Qualifier Filtresi (GAF 2.2)

GAF 2.2 formatında her satır, protein ile GO terimi arasındaki ilişki türünü belirtir. Aspect başına yalnızca anlamlı ilişkiler tutulur:

| Aspect | Geçerli qualifier'lar |
|--------|----------------------|
| F (MF) | `enables`, `contributes_to` |
| P (BP) | `involved_in`, `acts_upstream_of`, `acts_upstream_of_or_within`, ve pozitif/negatif etki varyantları |
| C (CC) | `is_active_in`, `part_of`, `located_in`, `colocalizes_with` |

### 2. Ancestor Propagation (True Path Rule)

GO'nun temel kuralı: bir protein T terimi ile anotlanmışsa, T'nin hiyerarşideki tüm ancestor terimleri için de annotated sayılır. `go-basic.obo` dosyası `obonet` ile yüklenir; her doğrudan anotasyon için `is_a` ve `part_of` kenarları üzerinden tüm ancestor'lar label setine eklenir.

- Ortalama genişleme: ~2.5× (örn. TP53 MF: 34 → 82 term)
- Bu olmadan model parent term'leri tahmin ettiğinde yanlış cezalandırılır

### Test Monitoring

Her epoch sonunda test seti de değerlendirilir (gözlem amaçlı). Model seçimi (checkpoint kaydetme, early stopping) yalnızca validation mAP üzerinden yapılır.

## Notlar

- `encoder_output_dim` H5 dosyasından otomatik çıkarılır; elle ayarlamaya gerek yok.
- WandB logging opsiyoneldir. `WANDB_API_KEY` env var veya `config.wandb_api_key` ile aktifleşir.
- Test metrikleri WandB'de `test/mAP`, `test/F1_macro` vb. olarak val ile aynı run'da loglanır.
