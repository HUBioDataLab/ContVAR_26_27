# go_pretrain/go_model.py
# ESM2 backbone + 3 GO Projection Head (MF, BP, CC) model tanımı
#
# Stage 1 (GO Pretraining):
#   Backbone (ESM2, trainable) → 3 ayrı projection head → GO triplet loss
#
# Stage 2 (DMS Training):
#   Backbone'un fine-tuned ağırlıkları alınır, GO head'leri atılır
#   Yeni global + local head eklenir

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """GO ontolojisi için projeksiyon kafası."""
    
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class GOPretrainModel(nn.Module):
    """
    ESM2 backbone + 3 GO projection head.
    
    3 head backbone'dan gelen pooled embedding'i alır ve
    ontoloji-spesifik uzaylara projekte eder.
    
    Forward çıktıları:
        h:    backbone embedding (ESM2 mean-pooled) — Stage 2'de DMS'te kullanılacak
        z_mf: MF projeksiyon çıktısı
        z_bp: BP projeksiyon çıktısı
        z_cc: CC projeksiyon çıktısı
    """
    
    def __init__(self, esm_model, esm_alphabet, out_dim=256, freeze_layers=0):
        """
        Args:
            esm_model:     ESM2 model objesi (esm.pretrained'den yüklenen)
            esm_alphabet:  ESM2 alphabet objesi (tokenizer)
            out_dim:       Projeksiyon çıktı boyutu
            freeze_layers: İlk N transformer katmanını dondur (0=hepsini eğit)
        """
        super().__init__()
        
        self.esm_model = esm_model
        self.alphabet = esm_alphabet
        
        # ESM2 embedding boyutunu al
        # ESM2 modelinin son katmanının çıktı boyutu
        self.esm_dim = esm_model.embed_dim
        
        # Opsiyonel: İlk N katmanı dondur
        if freeze_layers > 0:
            for name, param in esm_model.named_parameters():
                # "layers.0.", "layers.1.", ... şeklinde isimlendiriliyor
                if "layers." in name:
                    layer_idx = int(name.split("layers.")[1].split(".")[0])
                    if layer_idx < freeze_layers:
                        param.requires_grad = False
            print(f"[INFO] Froze first {freeze_layers} ESM2 layers")
        
        # 3 ontolojiye özel projeksiyon kafaları
        self.head_mf = ProjectionHead(self.esm_dim, out_dim)
        self.head_bp = ProjectionHead(self.esm_dim, out_dim)
        self.head_cc = ProjectionHead(self.esm_dim, out_dim)
        
        print(f"[INFO] GOPretrainModel: esm_dim={self.esm_dim}, out_dim={out_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
    
    def _pool(self, token_embeddings, attention_mask=None):
        """
        Token embedding'lerini mean-pool yaparak protein-seviye embedding'e dönüştür.
        ESM2'nin özel token'larını (BOS, EOS, padding) hariç tutar.
        
        Args:
            token_embeddings: (B, L, D) — ESM2 çıktısı
            attention_mask:   (B, L) — padding=0, valid=1 (opsiyonel)
        
        Returns:
            pooled: (B, D) — protein seviye embedding
        """
        if attention_mask is not None:
            # Padding ve özel tokenları (BOS=idx 0, EOS=son valid idx) hariç tut
            # Basit yaklaşım: mask'i genişlet ve çarp
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            # BOS (ilk) ve EOS (son) token'ı atla
            pooled = token_embeddings[:, 1:-1, :].mean(dim=1)
        
        return pooled
    
    def encode(self, tokens, repr_layers=None):
        """
        Token'ları ESM2'den geçirip pooled embedding döndür.
        
        Args:
            tokens: (B, L) — tokenize edilmiş sekanslar
            repr_layers: Hangi katmanların representation'larını al (None=son katman)
        
        Returns:
            h: (B, D) — pooled backbone embedding
        """
        if repr_layers is None:
            repr_layers = [self.esm_model.num_layers]
        
        results = self.esm_model(tokens, repr_layers=repr_layers, return_contacts=False)
        
        # Son katmanın representation'ını al
        token_emb = results["representations"][self.esm_model.num_layers]  # (B, L, D)
        
        # Padding mask: padding token = alphabet.padding_idx
        padding_mask = (tokens != self.alphabet.padding_idx).float()
        
        # BOS ve EOS token'larını da hariç tut
        # BOS = cls_idx (index 0 in sequence), EOS = eos_idx
        # Genelde: padding_mask zaten padding'i 0 yapıyor
        # BOS ve EOS'u da 0 yapalım
        padding_mask[:, 0] = 0  # BOS
        # EOS: her sekans için son non-padding token
        for i in range(tokens.shape[0]):
            # Son valid pozisyonu bul
            valid_positions = (tokens[i] != self.alphabet.padding_idx).nonzero(as_tuple=True)[0]
            if len(valid_positions) > 0:
                last_valid = valid_positions[-1].item()
                padding_mask[i, last_valid] = 0  # EOS
        
        h = self._pool(token_emb, attention_mask=padding_mask)
        return h
    
    def forward(self, tokens):
        """
        Tam forward pass: ESM2 → pool → 3 head.
        
        Args:
            tokens: (B, L) — tokenize edilmiş protein sekansları
        
        Returns:
            h:    (B, esm_dim) — backbone embedding (pool'lanmış)
            z_mf: (B, out_dim) — MF head çıktısı
            z_bp: (B, out_dim) — BP head çıktısı
            z_cc: (B, out_dim) — CC head çıktısı
        """
        h = self.encode(tokens)
        
        z_mf = self.head_mf(h)
        z_bp = self.head_bp(h)
        z_cc = self.head_cc(h)
        
        return h, z_mf, z_bp, z_cc
    
    def save_full(self, path):
        """
        Tüm model ağırlıklarını kaydet (backbone + 3 head).
        """
        torch.save({
            'esm_state_dict': self.esm_model.state_dict(),
            'head_mf_state_dict': self.head_mf.state_dict(),
            'head_bp_state_dict': self.head_bp.state_dict(),
            'head_cc_state_dict': self.head_cc.state_dict(),
            'esm_dim': self.esm_dim,
            'out_dim': self.head_mf.net[0].out_features,
        }, path)
        print(f"[SAVE] Full model saved → {path}")


# =============================================
# Helper: ESM2 Model Yükleme
# =============================================

def load_esm2_model(model_name="esm2_t6_8M_UR50D"):
    """
    ESM2 modelini yükle. Farklı esm paket versiyonlarını destekler.

    Seçenekler:
        esm2_t6_8M_UR50D    —  8M param, hızlı, düşük bellek  (geliştirme/test)
        esm2_t12_35M_UR50D  —  35M param, orta
        esm2_t30_150M_UR50D — 150M param, büyük
        esm2_t33_650M_UR50D — 650M param, en büyük (en iyi sonuç ama çok bellek)

    Returns:
        model, alphabet
    """
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "ESM paketi bulunamadi. Kurulum icin `pip install fair-esm` calistirin."
        ) from exc

    supported_models = (
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
    )
    errors = []

    # --- Yöntem 1: Eski API — esm.pretrained.<model_name>() ---
    pretrained_mod = getattr(esm, "pretrained", None)
    if pretrained_mod is not None:
        loader = getattr(pretrained_mod, model_name, None)
        if loader is not None:
            try:
                model, alphabet = loader()
                print(f"[INFO] Loaded {model_name} via esm.pretrained.{model_name}()")
                print(f"       embed_dim={model.embed_dim}, layers={model.num_layers}")
                return model, alphabet
            except Exception as exc:
                errors.append(f"esm.pretrained.{model_name}(): {exc}")

    # --- Yöntem 2: Hub API — esm.pretrained.load_model_and_alphabet_hub() ---
    if pretrained_mod is not None:
        hub_loader = getattr(pretrained_mod, "load_model_and_alphabet_hub", None)
        if hub_loader is not None:
            try:
                model, alphabet = hub_loader(model_name)
                print(f"[INFO] Loaded {model_name} via load_model_and_alphabet_hub()")
                print(f"       embed_dim={model.embed_dim}, layers={model.num_layers}")
                return model, alphabet
            except Exception as exc:
                errors.append(f"load_model_and_alphabet_hub({model_name}): {exc}")

    # --- Yöntem 3: torch.hub.load() ---
    try:
        model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
        print(f"[INFO] Loaded {model_name} via torch.hub.load()")
        print(f"       embed_dim={model.embed_dim}, layers={model.num_layers}")
        return model, alphabet
    except Exception as exc:
        errors.append(f"torch.hub.load(..., {model_name}): {exc}")

    raise RuntimeError(
        "ESM2 modeli yuklenemedi: "
        f"'{model_name}'. Denenen yollar: {' | '.join(errors) if errors else 'yok'}. "
        "Kurulumu ve model adini kontrol edin. "
        f"Desteklenen modeller: {', '.join(supported_models)}"
    )


def tokenize_sequences(sequences, alphabet, max_len=1022):
    """
    Protein sekanslarını ESM2 token'larına dönüştür.
    
    Args:
        sequences: list of (label, sequence) tuples
        alphabet:  ESM2 alphabet
        max_len:   Maksimum sekans uzunluğu
    
    Returns:
        batch_tokens: (B, L) padded token tensor
    """
    batch_converter = alphabet.get_batch_converter()
    
    # Sekansları max_len'e kırp
    truncated = [(label, seq[:max_len]) for label, seq in sequences]
    
    batch_labels, batch_strs, batch_tokens = batch_converter(truncated)
    return batch_tokens
