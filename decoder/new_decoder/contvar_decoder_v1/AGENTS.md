# ContVAR mean-window çalışma kapsamı

- Başlangıç: `BASLA.md`. Devir belgesi üst klasördeki `CODEX_DEVIR_14Kasim2026.md`.
- Yalnız mean window entegrasyonu: WT ve VAR ayrı ayrı maskeli ortalanır; encoder ve GO vektörleri sabittir. ESIM/LOCAL_PAIR/comparison reader ekleme.
- Dört model korunur: parallel/multiclass × joint/score_difference. Skor farkı, paylaşılan scorer'ın VAR ve WT ham skorlarının farkıdır.
- Kaynak split/etiketler korunur. Tek SNP ve açık is_combinatorial=False zorunlu. Augmentation yalnız train'de; çoklu kaynak etiketleri tutulup raporlanır.
- `prepared_window/old`, canonical devir cohort'udur; Gayem'in tam koşusuyla aynı olduğu varsayılmaz. Eski paketlerin prepared tablolarını native kaynak splitlerle karıştırma.
- Window kapalı/açık koşulları aynı örnekler ve yeni global/mutation-site girdileriyle çalışır. Full-trained modeller ayrıca ortak eski test alt kümesinde değerlendirilir.
- Önce testler ve smoke; sonra yeni çıktı klasöründe tam deney. Test metrikleri checkpoint/hiperparametre seçmek için kullanılmaz. Bir seed istatistiksel üstünlük kanıtı değildir.
- Eski `results/full_seed42` ve kaynak ZIP/veri dosyalarını değiştirme. `metadata/file_manifest.json` tarihsel teslim manifestidir; değiştirilmiş çalışma kopyasının manifesti değildir.
- Bu bilgisayarda `.venv/Scripts/python.exe` kullanılır. Test: `.venv/Scripts/python.exe -m pytest tests -q`.
- Kullanıcı yalnız açıklama istediğinde veya plan seçerken eğitim başlatma. Somut kod/eğitim görevi verildiğinde bu kapsamda ilerle.
