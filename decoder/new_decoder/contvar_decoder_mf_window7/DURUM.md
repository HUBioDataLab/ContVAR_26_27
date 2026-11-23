# Çalışma durumu — 18 Kasım 2026

Kullanıcının onayladığı 48 yeni eğitim ve test değerlendirmeleri tamamlandı. Seed 42, MF, window size 7; reverse cezası yok. Bağımsız son doğrulama geçti. Sonuç yorumu reports/YORUM.md içinde; 24 sayfalık rapor reports/ContVAR_MF_Window7_Rapor.pdf dosyasında.

## Tamamlananlar

- Windows izin yolu hatası çözüldü. Yeni PowerShell süreçlerinde POWERSHELL_TELEMETRY_OPTOUT=true devralındığı doğrulandı.
- Resmi GO üretim yayını 2026-07-26 ile namespace eşleştirmesi tamamlandı.
- MF veri hazırlığı ve window-7 cache tamamlandı: prepared/manifest.json.
- 24 birim/protokol testi geçti: verification/tests.xml.
- Gerçek veride 12 konfigürasyonun GPU ileri/gradient kontrolleri geçti: verification/prepared_and_gradient_checks.json.
- 48 koşullu suite tamamlandı. Checkpoint seçimi yalnız validation Macro-F1; bütün fitlerden sonra native ve ortak testler hesaplandı.
- Tahminlerden bağımsız F1/MCC/karışıklık matrisi kontrolleri, 24 baseline değerlendirmesi ve 4 seçilen checkpoint'in GPU geri okuması geçti: verification/final_qa.json.
- 48 koşullu sonuç tablosu, 32 window karşılaştırması, 24 eşleştirilmiş kapsam karşılaştırması ve yeni baseline'lar hazır.

## Nihai veri sayıları

| Kapsam/split | Train | Val | Test |
|---|---:|---:|---:|
| Eski/easy | 23776 | 2653 | 2974 |
| Eski/hard | 23921 | 3003 | 3065 |
| Geniş/easy | 31576 | 3250 | 3330 |
| Geniş/hard | 31461 | 3560 | 3552 |

Eski eğitim tablolarında aynı 9 benzersiz protein–mutasyon–GO kaydının girdileri yoktu; easy ve hard train'den dokuzar satır nedenleriyle çıkarıldı. Diğer bölümlerde teknik eleme yok. Etiket çelişkileri ve tekrarlar korundu. Ortak MF test: easy 1030, hard 399 satır.

## Deney kayıtları

- logs/suite.log: epoch ve koşul ilerlemesi.
- results/seed42/training_progress.json: tamamlanan fit sayısı ve validation sonuçları.
- results/seed42/summary.json: 48 tamamlanmış modelin validation, native test ve ortak test sonuçları.
- results/seed42/selection_before_test.json: testten önce kilitlenen checkpoint seçimleri.

## Yeniden çalıştırma

Kesinti olursa bu klasörde .\run_all.ps1 çalıştırılabilir. Aynı planla tamamlanmış koşular tekrar eğitilmez; yarım koşu last.pt ile devam eder. Veri/kod kimliği değişirse otomatik reuse reddedilir.

Tamamlanan raporlama ve paketleme aşamaları:

1. offline_run.py --log logs/report.log contvar_mf.report
2. offline_run.py --log logs/final_qa.log contvar_mf.verify_run
3. Rapor ve grafiklerin görsel kontrolü; gerekli yazım/düzen düzeltmeleri.
4. package_run.py ile yalnız yeni kod/sonuç paketinin hazırlanması.
5. DURUM.md ve verification/current_status.json son hâle güncellenmesi.

Eski sonuçlar yeni raporlarda kullanılmadı. Eski kaynak/sonuçların 326 dosyası başlangıçta ve son kontrolde değişmeden doğrulandı. Eğitim çekirdeği model.py/data.py/train.py/suite.py/util.py/go_embeddings.py ve offline_run.py plan tarafından SHA-256 ile sabitlendi. Eğitim sürelerinin toplamı 54,2 dakika. Sonuç paketi deliverables/ContVAR_MF_Window7_Code_Results.zip altında; ham embedding'ler ve büyük feature cache ZIP'e dahil değil.
