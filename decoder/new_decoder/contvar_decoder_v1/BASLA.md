# ContVAR mean window çalışması

VS Code'da üst klasördeki `ContVAR.code-workspace` dosyasını veya doğrudan bu klasörü açın. Python interpreter olarak `.venv\Scripts\python.exe` seçin. Codex'e bu çalışma klasörünü verin; kapsam `AGENTS.md` içinde kayıtlıdır.

İlk tam deney tamamlandı: `results/window_mean_seed42/RAPOR.md` dosyasını okuyarak başlayın. 4 mimari × 2 örnek kapsamı × 2 window koşulu × 2 split = 32 eğitim ve test değerlendirmesi vardır. Mevcut sonuçları incelemek için yeniden eğitim başlatmanız gerekmez.

26 test geçti; tamamlanan 32 koşunun checkpoint, tahmin, satır/etiket eşleşmesi ve seçim kayıtları doğrulandı. Tarihsel sonuçların 78 dosyası değişmedi. Mean window, 16 eşleştirilmiş koşulun 8'inde test Macro-F1'i artırdı, 8'inde düşürdü; bu tek seed deneyi genel bir üstünlük göstermiyor. Ayrıntılar ve baseline karşılaştırmaları rapordadır.

Bu bilgisayarda VS Code'un Python uzantısı kurulu. Codex uzantısı listede görünmedi. VS Code içinde de Codex kullanmak isterseniz [OpenAI'nin resmi IDE kurulum sayfasını](https://developers.openai.com/codex/ide/) kullanın; uzantıyı açıp giriş yaptıktan sonra bu projeyi seçin. Codex paneli görünmüyorsa komut paletindeki `Codex: Open Codex Sidebar` komutu kullanılabilir.

Codex'te devam etmek için başlangıç mesajı:

> Bu ContVAR mean-window çalışma kopyasında devam edeceğiz. Önce AGENTS.md ve BASLA.md dosyalarını, verification/ kayıtlarını ve results/window_mean_seed42/RAPOR.md varsa onu oku. Window, WT ve VAR için ayrı maskeli ortalama ile eklenir. Encoder/GO sabittir; comparison reader/ESIM kapsam dışındadır. Eski çıktıları değiştirme. İlk olarak mevcut deneyin durumunu özetle; eğitim zaten çalışıyorsa ikinci bir kopya başlatma. Yeni çalışma istendiğinde mevcut .venv/Scripts/python.exe ortamını kullan.

## Hedef

Mevcut mutation-site ve global girdilerine maskeli ortalaması alınmış 33 aminoasitlik window eklemek. WT/VAR aynı adapter ve GOReader ağırlıklarını paylaşır. Ayrı karşılaştırma reader'ı, ESIM, pocket veya encoder eğitimi bu çalışmada yoktur.

`PairedGODecoder` zaten ek ölçekleri destekler; eğitimde `scale_dims` üç ölçekle kurulur. İki çıktılı/üç sınıflı head ve joint/skor farkı seçenekleri korunur. `--window-pool none` eski iki ölçekli ağ; `--window-pool mean --window-cache cache/window_mean.h5` üç ölçekli ağdır.

## Veri kaynakları

- Üst klasördeki `New_Decoder_Data_with_local_Context`: yeni frozen global/local/window/GO dosyaları ve native splitler.
- Üst klasördeki `New_Decoder Data`: tarihsel checkpoint doğrulaması için eski global/local girdileri.
- `prepared_splits`: devirde doğrulanmış tarihsel örnek kapsamı; değiştirilmez.
- `prepared_window/old`: aynı canonical örnekler, yeni girdilerle hazırlanır.
- `prepared_window/full`: geniş cache'in desteklediği uygun native örnekler.
- Eski paketlerin `prepared` tabloları kaynak partition'larla uyuşmadığından doğrudan eğitim girdisi yapılmaz.
- Gayem'in ayrıca kullandığı kod/koşu kaydı bu klasörde ayrı bir depo olarak yoktur; bu uygulama devir ZIP'indeki canonical v1 kodunu temel alır.

## Kurulum (yalnız ortam yeniden kurulacaksa)

PowerShell terminalinde, bu klasörden:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt matplotlib
```

Bu bilgisayarda kullanılan gerçek sürümler `requirements-windows-lock.txt` ve `verification/environment.json` içine kaydedilir. Orijinal `requirements-lock.txt` geçmiş macOS ortamının kaydıdır. Aktivasyon zorunlu değildir; tam Python yolu kullanılır. Farklı Python/GPU ortamlarında resmi PyTorch kurulum seçeneğini kullanın.

Aynı Windows paket sürümlerini kurmak için `python -m pip install -r requirements-windows-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128` kullanılabilir.

## Komutlar

```powershell
# Birim/entegrasyon testleri
.\run_window.ps1 -Action test

# Bir kez: tüm window'ları maske ve merkez vektörü ile doğrular, float32 mean cache'i çıkarır;
# native splitlerden old/full cohort'larını hazırlar. Mevcut çıktının üstüne yazmaz.
.\run_window.ps1 -Action prepare

# Pipeline denemesi: küçük veri, bir epoch; bilimsel sonuç değildir
.\run_window.ps1 -Action smoke -Out results/window_smoke_check

# Tam karşılaştırma: old/full × easy/hard × none/mean × 4 model = 32 eğitim
.\run_window.ps1 -Action train -Out results/window_mean_seed42 -Device cuda -Seed 42

# Yalnız mevcut sonuçlardan raporu/grafikleri yeniden oluşturur
.\run_window.ps1 -Action report -Out results/window_mean_seed42
```

PowerShell script çalıştırma politikası izin vermiyorsa Python komutlarını doğrudan kullanın; sistem politikasını değiştirmek gerekmez:

```powershell
.\.venv\Scripts\python.exe -m contvar_decoder.window_prepare
.\.venv\Scripts\python.exe -m contvar_decoder.window_suite --device cuda --seed 42 --out results/window_mean_seed42
```

`--resume`, aynı plan/kod/veriyle tamamlanmış model koşularını yeniden kullanır. Yarım bir model eğitiminin optimizer durumundan devam etmez. Çalışan süreci yeniden başlatmadan önce logu kontrol edin; mevcut çıktı klasörlerini silmeyin.

## Deney kuralları

- Eski ve geniş örnek kapsamlarının her birinde window kapalı/açık aynı örnekler, aynı yeni global/mutation-site vektörleri ve aynı GO dosyasıyla çalışır.
- Toplantıdaki old+window, full-window ve full+window gruplarına ek olarak old-window kontrolü vardır. Böylece 16 mimari/veri/window koşulu, easy/hard ayrı eğitimleriyle 32 koşu eder.
- Seed 42; epoch üst sınırı 20, patience 5, batch 128, AdamW 3e-4; sınıf ağırlıkları kapalı, reverse lambda=0. Reverse metrikleri hesaplanır. Bunlar tarihsel ayarlarla kontrollü ilk karşılaştırmadır.
- Train augmentation korunur; validation/test augmentation ve birleşik mutasyon kanıtları dışlanır. Kaynak etiketler/partition'lar korunur. Çoklu etiketler raporlanır.
- Her checkpoint yalnız validation Macro-F1 ile seçilir. Test değerlendirmesi bütün eğitimlerden ve `selection_before_test.json` kaydından sonra yapılır.
- Geniş veri modelleri ayrıca aynı eski test alt kümesinde değerlendirilir. Baseline kuralları yalnız ilgili train verisinden hesaplanır.
- Tek seed sonuçları istatistiksel üstünlük kanıtlamaz. Kaynak test partition'ları geçmişte görülmüştür; sonuçlar kontrollü takip deneyi olarak raporlanır.

## Sonuçları nereden okuyacağım?

Tamamlanan deney klasöründe `RAPOR.md`, `tum_sonuclar.csv`, `window_katkisi.csv`, `baseline_sonuclari.csv`, `ortak_test_veri_katkisi.csv` ve `grafikler/` bulunur. Her koşulun klasöründe `best.pt`, `config.json`, `history.json`, `results.json`, `val_predictions.tsv`, `test_predictions.tsv` vardır.

Mean checkpoint ile inference örneği:

```powershell
.\.venv\Scripts\python.exe -m contvar_decoder.predict results/window_mean_seed42/old/hard/mean/multiclass_joint/best.pt --package ../New_Decoder_Data_with_local_Context --data-dir prepared_window/old --window-cache cache/window_mean.h5 --family hard --split val --device cuda --out results/mean_readback_new
```

Eski checkpointlerde `--window-cache` gerekmez. Tarihsel doğrulamada eski girdiler kullanılır; yeni embeddinglerle eski skorları yeniden üretme iddiası yapılmaz.

Mean cache'in HDF5 dosyası oluşturma zamanlarını da saklar. Sayısal olarak aynı ortalamalar yeniden üretilse bile dosya hash'i değişebilir. Mevcut checkpointlerle birebir artifact doğrulaması için `cache/window_mean.h5` dosyasını koruyun; `ContVAR_Mean_Window_Cache.zip` yedeği bu dosyayı aynı byte'larla geri yükler. Kaynak büyük embeddingler yedek kod/sonuç paketinde çoğaltılmaz.

Tam deney tamamlandıktan sonra çıktı tutarlılığını denetlemek için:

```powershell
.\.venv\Scripts\python.exe verify_window_run.py results/window_mean_seed42
```
