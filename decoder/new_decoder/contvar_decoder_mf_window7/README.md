# ContVAR — MF / window-7 deneyleri

Bu klasör, 18 Kasım 2026'da kullanıcı tarafından onaylanan yeni 48 eğitim için ayrı çalışma sürümüdür.

Mevcut durum: 48 bilimsel eğitim, 48 native test ve eşleştirilmiş ortak test değerlendirmeleri tamamlandı. Bağımsız doğrulama geçti. Önce reports/YORUM.md dosyasını, ardından 24 sayfalık reports/ContVAR_MF_Window7_Rapor.pdf raporunu okuyabilirsiniz. Ayrıntılı durum DURUM.md dosyasında.

Temel sonuç: ESIM, B mean kontrolünü 3/8 koşulda geçti; ortalama test Macro-F1 farkı -0,0054. Geniş kapsam aynı ortak testte 17/24 eşleştirmede artış verdi; ortalama fark +0,0130. Tek seed kullanıldığı için bu sonuçlar betimleyicidir. Eski sonuçlar arşiv olarak korundu ve yeni raporlarda kullanılmadı.

## Sabit protokol

- Veri kökü: ../New_Decoder_Data_with_local_Context.
- Eski: prepared_old_subset_without_1499_proteins.
- Geniş: prepared_full_all_proteins.
- Her kaynağın kendi easy/hard train/validation/test atamaları korunur.
- Yalnız is_combinatorial=False ve resmi GO namespace'i molecular_function.
- Var olan augmentation train'de kalır, validation/testten çıkarılır. Yeni augmentation üretilmez.
- Çelişkili etiketler ve tekrarlar korunur.
- Birkaç gerçekten kullanılamayan girdi nedenleri kaydedilerek elenebilir; bütün model koşulları aynı kayıtlarda çalışır. Yaygın hatalar sistematik sorun olarak durdurulur.
- Ham window dosyası phase3_seq_phase1_window-001.h5: [33,1280], merkez 16. Kesim [13:20], toplam 7. Maskeler korunur. Eski mean-33 cache kullanılmaz.
- Encoder/Anc2Vec sabit, decoder sıfırdan. Seed 42. Reverse loss ve ek seed yok.

## 48 koşul

Her kapsam × easy/hard için:

| Window | Modeller | Adet |
|---|---|---:|
| none | parallel/multiclass × joint/score_difference | 4 |
| mean | parallel/multiclass × joint/score_difference | 4 |
| pair_mean | parallel/multiclass joint | 2 |
| esim | parallel/multiclass joint | 2 |

Toplam 2 kapsam × 2 split × 12 = 48. Sıra: eski penceresiz 8, eski window 16, geniş penceresiz 8, geniş window 16.

B mean: önce iki pencere ayrı ortalanır, ortak adapter ve [WT,VAR,VAR−WT,WT⊙VAR] MLP'siyle bir token oluşturulur; GO reader okur.
B ESIM: ortak adapter ve aynı MLP her hizalı konuma uygulanır; yedi token doğrudan GO reader'a gider. Sonradan mean yok. İki B yönteminde WT/VAR reader'larına ayrı mean-window girdisi verilmez. Bu, tam recurrent/soft-alignment ESIM yerine hizalı yerel karşılaştırma uyarlamasıdır.

## Eğitim

AdamW, lr 3e-4, weight decay 1e-4, batch 128, width 128, attention heads 4, dropout .1, gradient clip 5, en fazla 20 epoch, patience 5. Sınıf ağırlığı kapalı. Checkpoint validation Macro-F1 ile seçilir.

Bütün 48 fit tamamlanıp seçim dosyası kilitlenmeden final test hesaplanmaz. Her model kendi testinde ve iki kapsamın test kesişiminde ölçülür. Ortak testte easy için protein–mutasyon, hard için protein train/validation örtüşmesi kontrol edilir. Kayıt/etiket ve tekrar adetleri eşleştirilir.

## Çalıştırma

Bu klasörü PowerShell'de açın. Mevcut ../contvar_decoder_v1/.venv kullanılır.

    .\run_all.ps1

Aşamalar ayrı da çalışabilir:

    .\run_all.ps1 -Stage prepare
    .\run_all.ps1 -Stage check
    .\run_all.ps1 -Stage fit
    .\run_all.ps1 -Stage evaluate
    .\run_all.ps1 -Stage report
    .\run_all.ps1 -Stage verify
    .\run_all.ps1 -Stage package

All aşaması seçimleri kilitleyip testleri, raporu, son doğrulamayı ve kod/sonuç paketini üretir. Tamamlanmış koşular aynı planla devam ettirilir; farklı kod/veri konfigürasyonunun eski checkpoint'i otomatik kabul edilmez.

requirements.txt, bu çalışmada kullanılan sürümleri kaydeder. Büyük ham embedding dosyaları ve 2 GB'lık hazırlanmış cache sonuç ZIP'ine eklenmez; tekrar çalıştırma için kaynak veri klasörü gerekir.

Python modülleri offline_run.py üzerinden, -I -S ile çalışır. Başlangıç eklentileri/.pth dosyaları yüklenmez; Python socket bağlantı/DNS/bind/sendto girişimleri engellenir. Öğrenme sırasında internet gerekmez. Çalıştırıcının --self-test seçeneği vardır.

## Çıktılar

- prepared/manifest.json: filtre sayıları, teknik elemeler, kaynak kimlikleri, namespace sürümü, ortak test ve sınıf dağılımları.
- prepared/features_window7.h5: seçilen örnekler için yeni kırpılmış girdiler; kaynak H5 dosyalarını değiştirmez.
- results/seed42/: 48 ayrı checkpoint, ayarlar, epoch geçmişi, val/test tahminleri, metrikler.
- results/seed42/selection_before_test.json: testten önce kilitlenen seçimler.
- reports/SONUCLAR.html, SONUCLAR.md, ContVAR_MF_Window7_Rapor.pdf.
- reports/*.csv: bütün koşullar, window farkları, ortak test kapsam farkları, baselines, veri özeti.
- reports/figures/: yeni mimari şemaları ve PNG/SVG grafikler.

Rapor modülü 48 yeni koşunun tamamlanmış summary dosyası olmadan çalışmaz. Tarihsel full_seed42 ve window_mean_seed42 sonuçlarını rapor girdisi olarak kullanmaz.

## Namespace kaynağı

Kullanılan resmi GO üretim JSON dosyası ontology/go-basic.json içinde sabittir. İndirilen yayın sürümü 2026-07-26; URL ve dosya kimliği yanındaki go-basic.source.json içindedir. Veri paketindeki 7.799 GO kimliğinin tamamı resmi namespace bilgisiyle eşleşti. İlk denemede indirilen editör kaynağı ontology/go.obo MF seçimi için kullanılmaz.

## Korunan tarihsel çalışma

../contvar_decoder_v1 kaynak kodu, prepared tabloları ve sonuçları değiştirilmez. verification/historical_files_before.json başlamadan önceki 326 dosyanın kimliğini içerir. Bunlar yeni sonuç tablolarının veri kaynağı değildir.
