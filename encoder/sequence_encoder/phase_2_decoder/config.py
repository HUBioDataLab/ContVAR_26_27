"""
Sekans_decoder konfigürasyonu.

Tüm hiperparametreler ve dosya yolları burada merkezi olarak tanımlanır.
ESM-2 650M mean-pooled çıktı boyutu 1280'dir.
"""

from pathlib import Path
import torch


EMBEDDING_TYPES = ("raw_esm", "seq_phase0", "seq_phase1", "concat_phase0", "concat_phase1")


def pick_device(preferred: str | None = None) -> str:
    """Sekans_decoder için ortak cihaz seçimi: preferred > MPS > CUDA > CPU."""
    if preferred is not None:
        preferred = str(preferred).strip()
        if preferred.lower() == "cpu":
            return "cpu"
        if preferred.lower() == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MPS istendi fakat bu sistemde mevcut degil.")
        if preferred.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA istendi fakat bu sistemde mevcut degil.")
            idx = int(preferred.split(":", 1)[1])
            if idx >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA:{idx} requested but only {torch.cuda.device_count()} visible")
            return preferred
        if preferred.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA istendi fakat bu sistemde mevcut degil.")
            idx = int(preferred)
            if idx >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA:{idx} requested but only {torch.cuda.device_count()} visible")
            return f"cuda:{idx}"
        raise ValueError(f"Invalid device selection: {preferred}")

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DecoderConfig:

    def __init__(self, aspect: str = "F", embedding_type: str = "raw_esm"):
        """
        Args:
            aspect: GO ontoloji namespace'i
                    "F" = Molecular Function (MF)
                    "P" = Biological Process (BP)
                    "C" = Cellular Component (CC)
            embedding_type: Kullanılacak protein embedding kaynağı.
                    "raw_esm"       = ham ESM-2 embedding
                    "seq_phase0"    = Phase0 fine-tuned embedding
                    "seq_phase1"    = Phase1 fine-tuned embedding
                    "concat_phase0" = seq_phase0 + raw_esm birleşimi
                    "concat_phase1" = seq_phase1 + raw_esm birleşimi
        """
        self.aspect = aspect
        asp = aspect.lower()
        repo_root = Path(__file__).resolve().parents[1]
        self.embedding_type = embedding_type
        self.use_qualifier_filter = True  # False -> parse_goa_tsv qualifier filtresini atlar
        self._return_metrics = False       # True -> train() dict metrik döndürür (grid_search icin)

        # ── Encoder ──────────────────────────────────────────────────
        # ESM-2 650M mean-pooled embedding boyutu
        # İleride Phase 1/2 fine-tuned embedding kullanılacaksa
        # bu değer değiştirilmeli
        self.encoder_output_dim = 1280

        # ── Decoder mimarisi ─────────────────────────────────────────
        self.hidden_dims = [512, 1024, 512]
        self.dropout = 0.3

        # ── GO sınıflandırma ─────────────────────────────────────────
        self.min_go_freq = 10          # Bu frekanstan az görülen GO terimleri atılır
        self.null_function_weight = 3.0  # NULL_FUNCTION sınıfı için ekstra ağırlık
        self.pos_weight_clamp = 20.0   # pos_weight üst sınırı
        self.use_go_propagation = True  # GO ancestor propagation (annotation + eval)

        # ── Eğitim ───────────────────────────────────────────────────
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 32
        self.epochs = 80
        self.early_stop_patience = 15
        self.seed = 42
        self.eval_threshold = 0.5      # Inference/eval sigmoid eşik değeri
        self.cuda = None

        # ── Dosya yolları (aspect-spesifik) ──────────────────────────
        # Repo köküne göre mutlak path kullanıyoruz ki çalışma dizinine bağımlı olmasın.
        # embeddings_h5 için burada beklenen dosya protein-level embedding H5'tir.
        self.goa_tsv = str(repo_root / "data" / "raw" / "goa_2025-12-04_swissprot_noiea.tsv")
        self.benchmark_tsv = str(repo_root / "data" / "raw" / "variant_specific_go_benchmark.tsv")
        self.go_obo_path = str(repo_root / "data" / "raw" / "go-basic.obo")
        self.raw_esm_h5 = str(repo_root / "data" / "processed" / "esm2_t33_650M_UR50D_protein_embedding.h5")
        self.seq_phase0_h5 = None
        self.seq_phase1_h5 = None
        self.embeddings_h5 = self.raw_esm_h5           # embedding_type'a göre resolve_embeddings_h5() ile güncellenir
        self.embeddings_h5_secondary = None            # concat_phase0/concat_phase1 icin ikinci kaynak (raw_esm)
        self.variant_h5 = None  # keyed by "{protein_id}_{snp}"; None → benchmark atlanır
        self.variant_h5_secondary = None  # concat_phase0/concat_phase1 icin ikinci varyant embedding H5'i
        self.go_vocab_json = str(repo_root / "phase_2_decoder" / f"go_vocab_{asp}.json")
        self.decoder_checkpoint = str(repo_root / "phase_2_decoder" / f"decoder_best_{asp}.pt")
        self.uniref_tsv = str(repo_root / "phase_0_go_pretrained" / "protein_uniref50.tsv")
        self.split_json = str(repo_root / "phase0_protein_split_removed_graphless.json")

        # ── Weights & Biases ─────────────────────────────────────────
        self.wandb_project = "ContVAR-Sekans"
        self.wandb_entity = None       # None = kişisel workspace
        self.wandb_run_name = f"sekans-decoder-{asp}"
        self.wandb_api_key = None      # Env var WANDB_API_KEY tercih edilir
        self.wandb_enabled = True      # False → WandB tamamen devre dışı
        self.wandb_tags: list[str] = []       # ör. grid_search.py: ["stage_final", "final_run"]
        self.wandb_extra_config: dict = {}    # ör. grid_search.py: {"stage": "final", "is_final": True}

        # embedding_type=="raw_esm" disinda kaynak path'i henuz set edilmemis olabilir
        # (cagiran taraf once DecoderConfig(...) olusturup sonra seq_phase0_h5/seq_phase1_h5
        # atar, en sonda resolve_embeddings_h5()'i tekrar cagirir) — burada sessizce gec.
        try:
            self.resolve_embeddings_h5()
        except ValueError:
            pass

    def resolve_embeddings_h5(self):
        """
        embedding_type + raw_esm_h5/seq_phase0_h5/seq_phase1_h5'e göre embeddings_h5
        (ve concat modlarında embeddings_h5_secondary) alanlarını günceller.

        embedding_type veya *_h5 alanlarından biri __init__ sonrasında değiştirilirse
        (örn. grid_search.py, run_*_decoder.py CLI override'ları) bu metod tekrar
        çağrılmalıdır.
        """
        if self.embedding_type not in EMBEDDING_TYPES:
            raise ValueError(
                f"Bilinmeyen embedding_type: {self.embedding_type!r} (seçenekler: {EMBEDDING_TYPES})"
            )
        source_by_type = {
            "raw_esm": (self.raw_esm_h5, None),
            "seq_phase0": (self.seq_phase0_h5, None),
            "seq_phase1": (self.seq_phase1_h5, None),
            "concat_phase0": (self.seq_phase0_h5, self.raw_esm_h5),
            "concat_phase1": (self.seq_phase1_h5, self.raw_esm_h5),
        }
        primary, secondary = source_by_type[self.embedding_type]
        if primary is None:
            raise ValueError(
                f"embedding_type={self.embedding_type!r} için gerekli H5 yolu ayarlanmamış "
                "(seq_phase0_h5 / seq_phase1_h5 kontrol et)."
            )
        self.embeddings_h5 = primary
        self.embeddings_h5_secondary = secondary

    def validate_paths(self):
        required = {
            "goa_tsv": self.goa_tsv,
            "embeddings_h5": self.embeddings_h5,
            "split_json": self.split_json,
        }
        split_path = Path(self.split_json)
        if split_path.exists():
            import json
            with open(split_path) as f:
                split_data = json.load(f)
            if "protein_to_split" not in split_data:
                required["uniref_tsv"] = self.uniref_tsv
        missing = {name: path for name, path in required.items() if not Path(path).exists()}
        if missing:
            pretty = "\n".join(f"  - {name}: {path}" for name, path in missing.items())
            raise FileNotFoundError(
                "Sekans_decoder için gerekli dosyalar eksik:\n"
                f"{pretty}\n"
                "Not: once protein-level embedding H5 uretmek icin "
                "`python -m Sekans_decoder.build_embeddings` calistir."
            )
