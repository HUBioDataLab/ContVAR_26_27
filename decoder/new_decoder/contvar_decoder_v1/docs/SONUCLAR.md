# Tam veri deney sonuçları

Bu rapor `results/full_seed42/summary.json` dosyasından otomatik üretildi. Seed 42; en fazla 20 epoch; early-stopping patience 5. Eğitimler küçük örneklemle sınırlı değildir: mevcut frozen girdileri geçerli olan bütün uygun training satırları kullanılmıştır. Test sonucu checkpoint veya lambda seçmek için kullanılmamıştır.

**Kapsam:** yalnız tek mutasyon; değerlendirmede augmentation yok; kaynak etiketleri ve partition atamaları korunur. Çoklu etiketli tek-mutasyon grupları kullanıcı tercihiyle korunur ve ayrıca raporlanır. Tam kaynak ZIP'te embeddingi olmayan satırlar bu deneyin kapsamına alınamaz.

## Dört kombinasyonun final karşılaştırması

Her ailede joint modelin lambda değeri validation macro-F1 ile seçildi. Aşağıdaki satırlar `selection_before_test.json` ile test hesaplarından önce sabitlenmiş adaylardır. Macro-F1 yüksek, reverse MSE düşük olmalıdır; fark modellerinin yapısal sıfır reverse hatası sınıflandırma üstünlüğü değildir.

| Aile | Mimari | Seçilen lambda | Epoch | Val Macro-F1 | Test Macro-F1 | Test LOF F1 | Test GOF F1 | Test NEUTRAL F1 | Test reverse MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | Paralel + skor farkı | 0 | 15 | 0.5836 | 0.5699 | 0.8333 | 0.4193 | 0.4573 | 1.197e-12 |
| easy | Paralel + joint | 0 | 6 | 0.6790 | 0.6755 | 0.8643 | 0.6033 | 0.5590 | 233.7 |
| easy | Üç sınıflı + skor farkı | 0 | 14 | 0.5997 | 0.6017 | 0.8352 | 0.4449 | 0.5251 | 0 |
| easy | Üç sınıflı + joint | 0 | 7 | 0.6785 | 0.6871 | 0.8619 | 0.6162 | 0.5832 | 94.79 |
| hard | Paralel + skor farkı | 0 | 8 | 0.5014 | 0.5041 | 0.7882 | 0.3245 | 0.3998 | 5.228e-13 |
| hard | Paralel + joint | 0 | 12 | 0.5873 | 0.5661 | 0.8245 | 0.4535 | 0.4202 | 485.5 |
| hard | Üç sınıflı + skor farkı | 0 | 12 | 0.5082 | 0.4947 | 0.7757 | 0.2832 | 0.4252 | 0 |
| hard | Üç sınıflı + joint | 0 | 5 | 0.5758 | 0.5665 | 0.8294 | 0.4303 | 0.4397 | 64.89 |

Validation ile genel model seçimi:

- **easy: Paralel + joint**, lambda=0; validation Macro-F1=0.6790, bu seçimin test Macro-F1'i=0.6755. [Checkpoint](../results/full_seed42/easy/parallel_joint_lambda0/best.pt).
- **hard: Paralel + joint**, lambda=0; validation Macro-F1=0.5873, bu seçimin test Macro-F1'i=0.5661. [Checkpoint](../results/full_seed42/hard/parallel_joint_lambda0/best.pt).

## Reverse loss açık/kapalı kontrolü

Bu tablo kontrolün her iki adayını da gösterir. Testte daha iyi olan lambda sonradan seçilmedi. Fark modellerinde ceza yapısal olarak sıfır olduğundan aynı eğitimin lambda=0.1 kopyası çalıştırılmadı.

| Aile | Joint output | Lambda | Val Macro-F1 | Test Macro-F1 | Test reverse MSE |
|---|---|---:|---:|---:|---:|
| easy | Paralel + joint | 0 | 0.6790 | 0.6755 | 233.7 |
| easy | Paralel + joint | 0.1 | 0.6402 | 0.6279 | 0.02999 |
| easy | Üç sınıflı + joint | 0 | 0.6785 | 0.6871 | 94.79 |
| easy | Üç sınıflı + joint | 0.1 | 0.6323 | 0.6318 | 0.03541 |
| hard | Paralel + joint | 0 | 0.5873 | 0.5661 | 485.5 |
| hard | Paralel + joint | 0.1 | 0.5512 | 0.5113 | 0.1004 |
| hard | Üç sınıflı + joint | 0 | 0.5758 | 0.5665 | 64.89 |
| hard | Üç sınıflı + joint | 0.1 | 0.5687 | 0.5582 | 0.07844 |

## Train'den türetilen basit referanslar

Bu kurallar yalnız train etiketleriyle kurulmuştur. GO-only kontrolü, skorun ne kadarının mutasyon dışındaki dağılımlardan açıklanabileceğini görmeye yardım eder. Protein–GO kuralı tanımadığı protein/GO için GO kuralına, o da genel train çoğunluğuna döner. Bu kurallar testte fit edilmez.

| Aile | Çoğunluk sınıfı test F1 | GO-only test F1 | Protein–GO + fallback test F1 |
|---|---:|---:|---:|
| easy | 0.2772 | 0.6191 | 0.7031 |
| hard | 0.2789 | 0.5536 | 0.5536 |

**Yorum:** bu koşuda joint modeller dört karşılaştırmada da skor farkı modellerinden daha yüksek test Macro-F1 verdi. Lambda=0.1 reverse hatasını belirgin düşürdü, fakat dört joint kontrolünde de test Macro-F1'i düşürdü; validation seçimi de hepsinde lambda=0 oldu. Bu, tek seed ve denenmiş tek pozitif lambda için gözlemdir.

Easy'de train'den türetilen protein–GO kuralı (0.7031), bütün yeni modellerin test skorundan yüksektir. Hard'da GO-only referans 0.5536, validation ile seçilen model 0.5661'dir. Dolayısıyla bu sonuçlar mimarinin mutasyon bilgisini etkili kullandığını veya annotation önbilgisini aştığını tek başına göstermiyor. Joint üstünlüğünün hangi kısmının protein/GO dağılımından geldiği ayrıca incelenmelidir. Testte en yüksek gözlenen yeni-model skoru easy'de üç sınıflı joint 0.6871; hard'da üç sınıflı joint 0.5665'tir. Validation ile seçilen modelin farklı olması raporda korunur.

## Veri ve çalışma kapsamı

| Aile | Train | Validation | Test |
|---|---:|---:|---:|
| easy | 41720 | 4677 | 4515 |
| hard | 41872 | 4973 | 4915 |

Toplam 12 ayrı eğitim ve 12 tam test değerlendirmesi; toplam suite süresi 32.8 dakika. Her koşulun epoch sayısı, parametreleri ve süresi CSV'de bulunur.

## Dosyalar ve yorum sınırı

- [Seçilmiş sekiz model CSV](secili_modeller.csv), [bütün on iki koşul CSV](tum_deneyler.csv).
- Her model klasöründe `config.json`, `history.json`, `best.pt`, `results.json`, `val_predictions.tsv`, `test_predictions.tsv` bulunur.
- `results.json` içinde görülmüş/görülmemiş GO ve çoklu etiketli girdi grupları için alt sonuçlar vardır. Confusion matrix sırası LOF, GOF, NEUTRAL'dır; satırlar gerçek, sütunlar tahmin etiketidir.
- `results/package_smoke` yalnız yazılım kontrolüdür; bu raporun skorlarına karıştırılmaz.

Sonuçlar tek seed ve mevcut cache sürümü içindir; tekrarlı seed belirsizliği ölçülmedi. Cache'in checkpoint kimliği H5 içeriğinden biyolojik olarak yeniden kanıtlanmadı; artifact hash'leri korundu. GO-specific etiketler kaynak bağlamına bağlıdır. Model skorları deneysel aktivite veya kalibre klinik olasılık değildir. Bu sınırlar farklı kaynak/ölçek/encoder katkısının gösterildiği anlamına gelmez.
