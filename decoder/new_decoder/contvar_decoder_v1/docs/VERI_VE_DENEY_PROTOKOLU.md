# Veri ve değerlendirme protokolü

Kaynaklar `source_splits/` içindeki orijinal ZIP'lerdir. `head_split_report.txt`, multiclass ve parallel veri tanımlarını ve **evaluation'da augmented=True satırların çıkarılmasını** belirtir. Üç formatın satır eşdeğerliği ve native hedefleri `prepare_all` ile doğrulanır; kaynak ZIP'ler değiştirilmez.

## Filtre sırası ve kapsam

1. Validation/test için augmented=True kayıtları çıkarılır. Train augmentation'ı korunur.
2. is_combinatorial=True kanıtlar çıkarılır. Tekli SNP biçiminde yazılması yeterli değildir; etiket birleşik mutasyon bağlamını anlatabilir.
3. Tek bir X123Y SNP, WT sekansı, pozisyon 1–1022 ve sekans uzunluğu, WT aminoasit eşleşmesi ve gerçek aminoasit değişimi aranır.
4. WT/VAR global ve mutation-site cache anahtarları, her vektörün 1280 boyutu ve finite değerleri doğrulanır.
5. GO ID'nin sabit 200-D artifact içinde bulunması gerekir. GO ID lookup yapılır; dataset go_idx'si artifact satırı kabul edilmez.

Her dışlama `*_excluded.tsv` içinde kaynak sample_id, aile/split ve neden ile saklanır. Kaynak partition atamaları korunur; easy ve hard birleştirilmez. Geçerli cohort iki head formatında aynıdır.

Ham satırdan geçerli satıra düşüşü ve dışlama nedenlerini [veri_kapsami.csv](veri_kapsami.csv) özetler.

| Aile | Train | Validation | Test | Train augmentation |
|---|---:|---:|---:|---:|
| Easy | 41.720 | 4.677 | 4.515 | 1.731 |
| Hard | 41.872 | 4.973 | 4.915 | 1.408 |

Ham satır sayıları daha büyüktür. Tam deney, mevcut frozen cache'in desteklediği uygun bütün satırlarla yapılır. Eksik embeddingler, bilinmeyen checkpoint'ten yeniden üretilmiş gibi gösterilmez. Baştan sona encoder+decoder eğitimi bu paketin görevi değildir.

Easy'de protein paylaşımı vardır; aynı protein+mutasyon train/val/test arasında tekrarlanmıyor. Hard protein ID düzeyinde ayrıktır. Bu kontroller homoloji düzeyinde ayrıklığı kanıtlamaz. Asıl split stratejisi verilen kaynak atamalarıdır; yeni rastgele split oluşturulmaz.

## Çoklu etiket grupları

Kullanıcının açık tercihiyle tek mutasyon koşulunu sağlayan `(protein_id,snp,GO)` gruplarında birden fazla kaynak etiket varsa satırlar korunur. Bu, biyolojik etiket hatası hükmü değildir: farklı alt aktiviteler aynı GO terimine eşlenmiş olabilir. Modelin girişinde bu deneysel bağlam bulunmaz.

Gruplar easy'de 86, hard'da 94 anahtar içerir; mevcut cohort ve split başına sayımlar audit dosyalarındadır. `multiple_labels_for_input` yalnız değerlendirme metadata'sıdır; model girdisine verilmez. Bu alt grubun sınıf sonuçları mümkün olan partitionlarda ayrı hesaplanır. Majority-vote ile etiket seçilmez, iki aile üst üste sayılmaz.

## Dört kombinasyon ve loss kontrolü

İki bağımsız faktör: **joint / ayrı scorer skor farkı** ve **parallel / multiclass output**. Explicit VAR−WT embedding ekleme bu deneyin tanımı değildir. Her ailede dört baseline eğitimine ek olarak yalnız joint modellerde lambda=0.1 kontrolü eğitilir. Toplam 6×2=12 ayrı eğitim. Scalar skor farkı modelindeki reverse cezası zaten sıfır olduğu için aynı eğitim lambda ile tekrar çalıştırılmaz.

Bütün modeller aynı kullanılabilir cohort, aynı source partition atamaları, seed, batch, optimizer ve epoch/early-stop bütçesini kullanır. Head hedefleri kendi kaynak arşivinden gelir. Sınıf ağırlıkları ana deneyde kapalıdır. Üç sınıflı CE ve iki binary BCE farklı amaçlardır; sayısal loss değerleri birbirine eşdeğer skala gibi kıyaslanmaz.

## Teste erişim ve seçim

`suite.py`, 12 eğitimin tamamında sadece train ve validation DataLoader'larını oluşturur. Her checkpoint validation macro-F1 ile seçilir. Joint modellerin lambda seçimi de validation ile yapılır; eşitlikte lambda=0 tercih edilir. `selection_before_test.json` kaydı bundan sonra yazılır. Test örnekleri daha sonra yüklenip tüm kayıtlı adaylar için metrikler hesaplanır. Ana dört-model karşılaştırması validation ile sabitlenen adayları kullanır; testte en yüksek skoru veren lambda'yı sonradan seçmez.

Birincil ölçüt üç sınıfın Macro-F1 ortalamasıdır. Ayrıca accuracy, sınıf precision/recall/F1, AUPRC/AUROC, confusion matrix, MCC, reverse MSE/MAE ve skor büyüklüğü raporlanır. GO'nun train'de görülmesine ve çoklu etiketli girdi grubuna göre alt sonuçlar da kaydedilir. Desteklenmeyen AUROC hücreleri null'dır. Paralel neutral skoru yardımcı karar skorudur; üç kalibre olasılık gibi yorumlanmaz.

Bu teslimin tam sonuçları tek seed'e aittir. Deney tekrarları ve güven aralıkları ayrı çalışmadır. Önceki 128/64 örnekli smoke skorları ana sonuçlara karıştırılmaz. Tam model seçimi ve test değerlendirmesi, bu paket içindeki current kaynak ve hash'lere dayanır.
