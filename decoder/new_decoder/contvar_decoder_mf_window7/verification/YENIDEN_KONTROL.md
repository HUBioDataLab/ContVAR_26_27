# Eğitim, inference ve sonuç bütünlüğü yeniden kontrolü

Kontrol zamanı: 18 Kasım 2026 06:12:51 (Türkiye saati).

**Sonuç: Kaydedilmiş 48 eğitimin, test/inference çıktılarının ve sonuç hazırlığının tamamlandığı doğrulandı. Eski decoder ağırlıklarıyla veya sonuçlarıyla karışma bulunmadı.**

Bu kontrol sırasında yeni eğitim başlatılmadı. Var olan 48 checkpoint yeniden GPU’ya yüklenerek tüm validation ve test örneklerinde ileri hesaplama yapıldı.

| Denetim | Sonuç |
|---|---|
| Eğitim planı, epoch geçmişi ve loglar | 48/48 koşul; hepsi max-epoch veya patience kuralına uygun bitmiş |
| En iyi checkpoint seçimi | 48/48, validation Macro-F1 ve en erken eşitlik kuralıyla uyumlu |
| Checkpoint içindeki metadata ve parametre şekilleri | MF/window-7 biçimi, seed 42, doğru head/pair/window, reverse cezası kapalı |
| Gerçek ağırlık değerleri | 43.251.584 parametre değeri sonlu; 48 ağırlık seti birbirinden farklı |
| Eski checkpoint karşılaştırması | Klasörde bulunan 52 eski checkpoint’in hiçbirinin tensor içeriği yeni 48 setle aynı değil |
| Yeni GPU inference | 48 model × validation/test = 96 tam değerlendirme, 304.644 tahmin |
| Sınıf tahminleri | Yeniden üretilen 304.644 tahminin tamamı kayıtlı sınıfla aynı |
| Skor farkı | En büyük mutlak fark 2.980194e-8; float32/CSV sayısal hassasiyeti içinde |
| Loss farkı | En büyük mutlak fark 9.252930e-7; kabul edilen sayısal hassasiyet içinde |
| Test ve validation metrikleri | F1, MCC, accuracy, balanced accuracy, sınıf precision/recall/F1, AP ve AUROC bağımsız yeniden hesaplanarak eşleştirildi |
| Ortak test dosyaları | 48/48 dosya, native tahminlerin doğru ortak alt kümesiyle eşleşiyor |
| Veri seçimi | 12 hazırlanmış tablo kaynak satırlara kadar izlendi; MF, combinatorial ve evaluation augmentation kuralları doğrulandı |
| Namespace | 7.799 GO kimliği resmi ontology namespace/alias bilgisiyle yeniden doğrulandı |
| Split ve ortak test sızıntısı | Easy varyant, hard protein ayrımı ve her iki kapsamın train/validation ortak test dışılığı geçti |
| Baseline | Train sayımlarından 36 validation/native/common değerlendirmesi yeniden hesaplandı |
| CSV raporları | 48 model, 32 window farkı, 24 kapsam farkı, 12 baseline, 4 validation seçimi ve 12 veri özeti tutarlı |
| PDF ve grafikler | 24 sayfa, 24 PNG, 24 SVG, HTML bağlantıları ve son görsel kontrolün PDF kimliği eşleşiyor |
| Kaynak ve embedding kimlikleri | 34 kod/manifest/ontology/kaynak tablo/embedding/cache kimliği doğrulandı |
| Eski arşiv | Başlangıçta kaydedilen 326 eski dosya değişmemiş |

## Eski ağırlıklarla karışma nasıl kontrol edildi?

Checkpoint dosyaları yalnızca isimleriyle karşılaştırılmadı. `.pt` arşivleri veri olarak okundu; metadata, model konfigürasyonu, epoch, veri/kod kimlikleri ve gerçek tensor depolama blokları denetlendi. Yeni 48 checkpoint’in ağırlık içerikleri, 44 eski eğitim checkpoint’i ve 8 eski smoke kontrolüyle karşılaştırıldı. Eşleşme yok. Yeni checkpoint kimlikleri testten önce kilitlenen kimliklerle de aynı.

Kaydedilmiş eğitim kodu modelleri sıfırdan oluşturuyor; devam yüklemesi yalnızca aynı yeni koşulun kendi klasöründeki `last.pt` için var. Yeni yükleyici `contvar_mf_window7_v1` biçimini istiyor. ESM/Anc2Vec embedding’lerinin sabit tutulması planın parçası; burada karşılaştırılan eğitilmiş ağırlıklar decoder’a ait.

## Sonuç hazırlığı ve ZIP

CSV sayıları ve bütün sayısal alanları, tahminlerden bağımsız hesaplanan metriklerle kontrol edildi. Grafik ve PDF’nin kaynak summary dosyası, rapor kodu ve gözden geçirilmiş PDF kimliği de doğrulandı. Kontrol sonuçları daha önceki bulguları değiştirmedi: standart mean 9/16, B mean 5/8, ESIM 3/8 karşılaştırmada artış; geniş kapsam aynı ortak testte 17/24 eşleştirmede artış.

ZIP’teki 532 kayıt mevcut dosyalarla karşılaştırıldı. Model, veri, kod ve rapor dosyalarında fark yok. Tek açıklanabilir fark `logs/package.log`: ZIP kapandıktan sonra paketlemenin bittiğini söyleyen mesaj dosyaya ekleniyor; ZIP bu son mesajdan önceki logu içeriyor. Bu fark ağırlıklara, tahminlere veya rapora etki etmiyor. ZIP’in SHA-256 kimliği de ilk tamamlandığı andaki kimlikle aynı:

`6b5e37c413d77de7066676e7a2565469757414f657bf34dc7e088634531c9ec9`

## Çalıştırma ortamı

PowerShell’in MSIX başlatma hatası nedeniyle önce komut katmanı çalışmadı. Güvenilen yerel Python doğrudan, shell kullanmadan, gizli pencere ve process-local ortam değişkenleriyle çalıştırıldı. Python `-I -S` ve mevcut offline ağ koruması kullanıldı. İlk PyTorch kullanıcı/cache ortam hatası giderildikten sonra tam inference kontrolü exit code 0 ile tamamlandı. Sistem ortam değişkenleri değiştirilmedi.

## Kanıt dosyaları

- `recheck_after_disconnect.json`: bu kontrolün ayrıntılı dökümü ve kimlikler.
- `full_forward_recheck.json`: 48 modelin iki bölüm için yeniden inference kanıtı.
- `full_forward_recheck.py`: ağırlık güncellemeden tam inference karşılaştırmasını yapan kod.
- `audit_helpers.mjs` ve `recheck_after_disconnect.mjs`: bağımsız dosya/tensor/CSV/metrik kontrolleri.
- `../logs/full_forward_recheck.log`: 48/48 yeniden inference tamamlanma kaydı.

Orijinal checkpoint’ler, prediction dosyaları, bilimsel raporlar ve ZIP bu kontrolde değiştirilmedi.