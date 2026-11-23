# 48 yeni deneyin yorumu

Bu sonuçlar yalnızca yeni MF / window-7 / seed-42 deneylerinden geliyor. Önceki eğitimlerin skorları karşılaştırmalara katılmadı. Toplam 48 eğitim tamamlandı; kaydedilen eğitim sürelerinin toplamı 54,2 dakika.

## Validation ile seçilen modeller

Her kapsam/split içinde 12 koşul arasından validation Macro-F1'i en yüksek model seçildi. Aşağıdaki dört model test skoruna bakılarak seçilmedi. Dördü de multiclass + joint.

| Kapsam / split | Seçilen window koşulu | Validation Macro-F1 | Kendi testinde Macro-F1 | Yeni protein-GO baseline | Model - baseline |
|---|---|---:|---:|---:|---:|
| Eski / easy | B mean-7 | 0,6917 | 0,5753 | 0,6111 | -0,0359 |
| Eski / hard | Penceresiz | 0,5359 | 0,4890 | 0,4671 | +0,0218 |
| Geniş / easy | Penceresiz | 0,6571 | 0,5805 | 0,5823 | -0,0018 |
| Geniş / hard | Mean-7 | 0,5325 | 0,5000 | 0,4632 | +0,0368 |

Easy'de train ve testte aynı proteinler bulunabiliyor; farklı varyantlar ayrılıyor. Protein-GO etiketi sıklığı bu nedenle güçlü bir referans. Seçilen iki easy model bu referansı aşamadı. Hard'da test proteinleri train'de yok; protein-GO sayım referansı GO sayımına dönüyor. Seçilen iki hard model bu referansı geçti. Bu baseline bir embedding modeli değildir: yalnızca ilgili train etiketlerinin sayımlarını kullanır.

Validation'da en yüksek skoru alan koşul testte her zaman en yüksek skoru vermedi. Testte daha yüksek çıkan başka bir koşulu sonradan seçilmiş model gibi sunmadık; 48 koşulun test skorlarının tamamı tabloda duruyor.

## Window ne kazandırdı?

| Aynı kapsam/split/head/karar yapısında karşılaştırma | Test Macro-F1 artan koşul | Ortalama fark |
|---|---:|---:|
| Penceresiz -> standart mean-7 | 9 / 16 | +0,0064 |
| Standart mean-7 -> B mean-7 | 5 / 8 | +0,0015 |
| B mean-7 -> B ESIM-7 | 3 / 8 | -0,0054 |

Bu çalışmada window eklemek her koşulu iyileştirmedi. Ortak local kolun B mean kontrolündeki ortalama katkısı küçük. ESIM de B mean'e göre genel bir üstünlük göstermedi; ortalama Macro-F1 biraz düştü.

ESIM'in B mean karşısındaki sekiz sonucu:

| Kapsam / split | Head | B mean-7 | B ESIM-7 | Fark |
|---|---|---:|---:|---:|
| Eski / easy | Parallel | 0,5897 | 0,5956 | +0,0059 |
| Eski / easy | Multiclass | 0,5753 | 0,5681 | -0,0072 |
| Eski / hard | Parallel | 0,4739 | 0,4795 | +0,0056 |
| Eski / hard | Multiclass | 0,4967 | 0,4769 | -0,0198 |
| Geniş / easy | Parallel | 0,5783 | 0,5652 | -0,0130 |
| Geniş / easy | Multiclass | 0,5882 | 0,5607 | -0,0274 |
| Geniş / hard | Parallel | 0,4888 | 0,4857 | -0,0031 |
| Geniş / hard | Multiclass | 0,4764 | 0,4923 | +0,0159 |

İki B modelinin parametre sayısı eşit. B mean iki pencereyi önce ayrı ortalayıp bir karşılaştırma tokenı üretir. B ESIM hizalı yedi konumu karşılaştırıp yedi tokenı GO reader'a taşır. Dolayısıyla bu karşılaştırma, özetleme sırası ile tokenları GO okumasına kadar korumanın birlikte etkisini ölçer. Kullanılan ESIM, bu projeye özel hizalı yerel karşılaştırma uyarlamasıdır; soft-alignment/BiLSTM içeren tam ESIM kurulmadı.

## Geniş kapsam yararlı oldu mu?

Kapsamların kendi test örnekleri farklı olduğu için yukarıdaki native test skorlarını birbirinden çıkararak kapsam etkisi söylemiyoruz. Aynı model konfigürasyonlarını iki testin kesişiminde eşleştiriyoruz.

| Ortak test | Kayıt | Geniş kapsamın arttığı eşleştirme | Ortalama geniş - eski Macro-F1 |
|---|---:|---:|---:|
| Easy | 1.030 | 7 / 12 | +0,0112 |
| Hard | 399 | 10 / 12 | +0,0147 |
| Toplam eşleştirme | İki ayrı test | 17 / 24 | +0,0130 |

Geniş kapsam bu karşılaştırmalarda ortalama olarak daha iyi; katkı her konfigürasyonda aynı değil. Train içerikleri ve validation atamaları da değiştiği için bunu iki eğitim düzeninin sonucu olarak yorumluyoruz. Yalnızca protein sayısının etkisini ayırmış olmuyoruz. Ortak testin her iki tarafın train ve validation bölümlerinden ayrıldığı kontrol edildi.

## En zor sınıf ve yorumun sınırı

Validation ile seçilen dört modelde LOF F1 yaklaşık 0,83-0,89 iken GOF F1 0,19-0,35. Modellerin en zayıf tarafı bu dört seçimde GOF. Yüksek LOF başarısı, nadir GOF sınıfının da aynı düzeyde çözüldüğü anlamına gelmiyor; Macro-F1 bu ayrımı görmemizi sağlıyor.

Yalnızca seed 42 kullanıldı. Küçük farkları istatistiksel üstünlük olarak yorumlamıyoruz. Ortak hard testte sadece 17 GOF kaydı var; bu alt grupta birkaç tahmin skoru belirgin değiştirebilir.

## Veri ve eski sonuçlar

- Eski kapsam: `prepared_old_subset_without_1499_proteins`; geniş kapsam: `prepared_full_all_proteins`.
- Yalnız `is_combinatorial=False` ve MF seçildi. Augmentation train'de korundu, validation/testten çıkarıldı. Etiket çelişkileri ve tekrarlar korundu.
- Eski kapsamda embedding'i bulunamayan aynı 9 benzersiz kayıt, easy ve hard train tablolarından dokuzar satır olarak çıkarıldı. Validation/testte teknik eleme yok. Diğer koşulların tamamı aynı hazırlanmış kayıt listelerini kullandı.
- Window, ham 33 konumdan `[13:20]` kesilerek toplam 7 konuma indirildi. Maskeler kullanıldı. Encoder ve Anc2Vec yeniden eğitilmedi; decoder ağırlıkları sıfırdan öğrenildi. Reverse cezası kullanılmadı.
- Eski sonuçlar arşiv olarak duruyor. Kontrol edilen 326 eski dosyanın SHA-256 kimliği değişmedi. Yeni raporlarda yalnızca bu 48 eğitim ve yeniden hesaplanan baseline'lar kullanıldı.

Tam sayısal kaynaklar: `tum_deneyler.csv`, `validation_secimleri.csv`, `window_karsilastirmalari.csv`, `ortak_test_kapsam_farklari.csv`, `baselines.csv`, `veri_ozeti.csv`, `filtre_dokumu.csv`. Tahminler ve checkpoint'ler `results/seed42/` altında. Son bağımsız doğrulama `verification/final_qa.json` dosyasında.
