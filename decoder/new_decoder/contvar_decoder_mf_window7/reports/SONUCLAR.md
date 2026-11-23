# Yeni ContVAR MF / window-7 sonuçları

Yalnızca yeni 48 eğitim; seed 42. Eski sonuçlar kullanılmadı.

- Penceresiz → Mean-7: 9/16 kosulda artış; ortalama Δ Macro-F1=+0.0064.
- Mean-7 → B mean-7: 5/8 kosulda artış; ortalama Δ Macro-F1=+0.0015.
- B mean-7 → B ESIM-7: 3/8 kosulda artış; ortalama Δ Macro-F1=-0.0054.
- Ortak testte geniş kapsam: 17/24 eşleştirmede artış; ortalama Δ Macro-F1=+0.0130.
- B mean–ESIM farkı, hizalı karşılaştırmanın ve tokenları GO reader'a kadar korumanın birlikte etkisidir. İki B modelinin parametre sayısı eşittir.
- Eski/geniş kayıt ve validation düzenleri farklıdır. Ortak test eğitim düzenini karşılaştırır; yalnız protein sayısının nedensel katkısını ayırmaz.
- Tek seed 42 kullanıldı. Bu farklar betimleyicidir; tekrarlı seed belirsizliği veya istatistiksel üstünlük kanıtı değildir.
- Ortak hard testte yalnız 17 GOF kaydı vardır; GOF skorundaki küçük örneklem duyarlılığı dikkate alınmalıdır.
- Kaynak etiket çelişkileri ve tekrarlar korunmuştur. Teknik elemeler veri manifestinde kaydedilmiştir. Eski eğitimlerin sayısal sonuçları kullanılmamıştır.

## Dosyalar

- tum_deneyler.csv: 48 koşulun tüm metrikleri
- window_karsilastirmalari.csv
- ortak_test_kapsam_farklari.csv
- veri_ozeti.csv / filtre_dokumu.csv / baselines.csv
- validation_secimleri.csv: test ile seçilmeyen 4 model ve referans farkları
- ContVAR_MF_Window7_Rapor.pdf ve figures/ PNG/SVG çizimleri

## Validation ile seçilen modeller ve referans

| Grup | Model | Val F1 | Test F1 | Ref. F1 | Fark | Ortak F1 | Ortak ref. | Ortak fark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old/easy | B mean-7 / M-J | 0.6917 | 0.5753 | 0.6111 | -0.0359 | 0.6106 | 0.6124 | -0.0019 |
| old/hard | Penceresiz / M-J | 0.5359 | 0.489 | 0.4671 | 0.0218 | 0.5449 | 0.4609 | 0.084 |
| full/easy | Penceresiz / M-J | 0.6571 | 0.5805 | 0.5823 | -0.0018 | 0.5982 | 0.6054 | -0.0072 |
| full/hard | Mean-7 / M-J | 0.5325 | 0.5 | 0.4632 | 0.0368 | 0.5257 | 0.4942 | 0.0315 |

Bu dört seçimden 2 tanesi native testte protein-GO referansını geçti. Easy'de aynı protein için GO etiketi sıklığı güçlü bir referans oluşturuyor; hard'da proteinler train'de bulunmadığından bu yöntem GO sıklığına dönüyor.

Macro-F1, LOF/GOF/NEUTRAL F1 değerlerinin eşit ağırlıklı ortalamasıdır. MCC, tüm karışıklık matrisini kullanan tamamlayıcı ölçüdür.

Önce aynı kapsam/split içinde window koşullarını karşılaştırın. Eski/geniş farkını ortak test sütunlarından okuyun; native test örnekleri farklıdır.

## Yeni hesaplanan referanslar

| cohort | family | baseline | native_macro_f1 | common_macro_f1 | native_mcc | val_macro_f1 | native_lof_f1 | common_lof_f1 | native_gof_f1 | common_gof_f1 | native_neutral_f1 | common_neutral_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old | easy | majority | 0.2912 | 0.2883 | 0.0 | 0.2843 | 0.8735 | 0.865 | 0.0 | 0.0 | 0.0 | 0.0 |
| old | easy | go_majority | 0.4346 | 0.4302 | 0.2096 | 0.6229 | 0.8745 | 0.8672 | 0.2197 | 0.2258 | 0.2097 | 0.1975 |
| old | easy | protein_go_majority | 0.6111 | 0.6124 | 0.4893 | 0.7351 | 0.9023 | 0.8955 | 0.356 | 0.3636 | 0.5752 | 0.5782 |
| old | hard | majority | 0.2836 | 0.2772 | 0.0 | 0.282 | 0.8508 | 0.8316 | 0.0 | 0.0 | 0.0 | 0.0 |
| old | hard | go_majority | 0.4671 | 0.4609 | 0.2578 | 0.4994 | 0.8484 | 0.8101 | 0.1958 | 0.1053 | 0.3571 | 0.4674 |
| old | hard | protein_go_majority | 0.4671 | 0.4609 | 0.2578 | 0.4994 | 0.8484 | 0.8101 | 0.1958 | 0.1053 | 0.3571 | 0.4674 |
| full | easy | majority | 0.2912 | 0.2883 | 0.0 | 0.2807 | 0.8737 | 0.865 | 0.0 | 0.0 | 0.0 | 0.0 |
| full | easy | go_majority | 0.4346 | 0.4493 | 0.1818 | 0.5747 | 0.8626 | 0.8546 | 0.1887 | 0.2105 | 0.2525 | 0.2828 |
| full | easy | protein_go_majority | 0.5823 | 0.6054 | 0.4256 | 0.7192 | 0.8879 | 0.8875 | 0.3333 | 0.3492 | 0.5257 | 0.5795 |
| full | hard | majority | 0.2862 | 0.2772 | 0.0 | 0.2837 | 0.8586 | 0.8316 | 0.0 | 0.0 | 0.0 | 0.0 |
| full | hard | go_majority | 0.4632 | 0.4942 | 0.2344 | 0.4668 | 0.838 | 0.8027 | 0.176 | 0.2105 | 0.3757 | 0.4694 |
| full | hard | protein_go_majority | 0.4632 | 0.4942 | 0.2344 | 0.4668 | 0.838 | 0.8027 | 0.176 | 0.2105 | 0.3757 | 0.4694 |

Referansların tamamı yalnız train etiketlerinden hesaplanmış sayım yöntemleridir. Ek neural eğitim yapılmadı.