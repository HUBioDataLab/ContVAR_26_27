"""
Varyant fonksiyon değişim tahmini (tek protein).

WT proteininin GO anotasyonları veritabanından biliniyor (GOA).
Decoder sadece varyant embedding'i üzerinde çalışır.

    wt_label  = 1.0  (GOA'da anotasyonlu)  veya  0.0  (yok)
    var_score = sigmoid(decoder(var_emb))  — her GO terimi için [0,1]

    wt_label=1 ve var_score < (1 - threshold)  →  LOSS   (fonksiyon kaybı)
    wt_label=0 ve var_score > threshold         →  GAIN   (yeni fonksiyon kazanımı)
    diğer                                       →  NEUTRAL

ContVAR_fitness (tek skaler):
    Sadece WT'de anotasyonlu GO terimleri üzerinden hesaplanır.
    mean(var_score[wt_annotated]) / 1.0
    1.0 = tüm WT fonksiyonları korunmuş | 0.0 = tam fonksiyon kaybı
"""

import json
import h5py
import torch
try:
    from .models.ffn_decoder import FFNDecoder
    from .config import pick_device
except ImportError:
    from models.ffn_decoder import FFNDecoder
    from config import pick_device

DEVICE = torch.device(pick_device())


def load_decoder(checkpoint: str, go_vocab_json: str,
                 encoder_output_dim: int = 1280,
                 hidden_dims: list[int] = None,
                 dropout: float = 0.3) -> tuple:
    """Kaydedilmiş decoder ve vocabulary'yi yükler."""
    with open(go_vocab_json) as f:
        go_vocab = json.load(f)

    n_classes = len(go_vocab)
    checkpoint_obj = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        encoder_output_dim = checkpoint_obj.get("input_dim", encoder_output_dim)
        hidden_dims = checkpoint_obj.get("hidden_dims", hidden_dims)
        dropout = checkpoint_obj.get("dropout", dropout)
        n_classes = checkpoint_obj.get("n_classes", n_classes)
        state_dict = checkpoint_obj["model_state_dict"]
    else:
        state_dict = checkpoint_obj

    decoder = FFNDecoder(encoder_output_dim, n_classes, hidden_dims, dropout)
    decoder.load_state_dict(state_dict)
    decoder.eval().to(DEVICE)

    return decoder, go_vocab


def load_embedding(h5_path: str, protein_id: str,
                   secondary_h5_path: str | None = None) -> torch.Tensor:
    """
    H5 dosyasından tek protein/varyant embedding'i yükler → [D].

    secondary_h5_path verilirse (concat_phase0/concat_phase1 checkpoint'leri için),
    aynı protein_id/varyant key'i o dosyadan da yüklenip [primary | secondary]
    sırasıyla concat edilir — dataset.py'deki GOAnnotationDataset ile aynı sıra.
    """
    with h5py.File(h5_path, "r") as h5f:
        emb = torch.FloatTensor(h5f[protein_id][:])
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
    if secondary_h5_path:
        with h5py.File(secondary_h5_path, "r") as h5f_sec:
            emb_sec = torch.FloatTensor(h5f_sec[protein_id][:])
            if emb_sec.dim() == 2:
                emb_sec = emb_sec.mean(dim=0)
        emb = torch.cat([emb, emb_sec], dim=-1)
    return emb


def predict_function_change(wt_go_terms: set, var_emb: torch.Tensor,
                            decoder: torch.nn.Module, go_vocab: dict,
                            threshold: float = 0.1) -> dict:
    """
    WT ile varyant arasındaki fonksiyon değişimlerini tahmin eder.

    Args:
        wt_go_terms: WT proteininin bilinen GO terimleri (veritabanından)
        var_emb:     Varyant encoder embedding [D]
        decoder:     Eğitilmiş FFNDecoder
        go_vocab:    GO term → index eşlemesi
        threshold:   LOSS/GAIN karar eşiği

    Returns:
        {
            "GO:0003700": {"wt": 1, "var": 0.12, "status": "LOSS"},
            "GO:0009999": {"wt": 0, "var": 0.73, "status": "GAIN"},
            ...
            "ContVAR_fitness": 0.81
        }
    """
    decoder.eval()
    with torch.no_grad():
        var_probs = torch.sigmoid(
            decoder(var_emb.unsqueeze(0).to(DEVICE))
        ).squeeze().cpu()

    results = {}
    annotated_var_scores = []

    for go_term, idx in go_vocab.items():
        if go_term == "NULL_FUNCTION":
            continue

        wt_label = 1 if go_term in wt_go_terms else 0
        var_s    = var_probs[idx].item()

        if wt_label == 1 and var_s < (1 - threshold):
            status = "LOSS"
        elif wt_label == 0 and var_s > threshold:
            status = "GAIN"
        else:
            status = "NEUTRAL"

        results[go_term] = {
            "wt":     wt_label,
            "var":    round(var_s, 4),
            "status": status,
        }

        if wt_label == 1:
            annotated_var_scores.append(var_s)

    if annotated_var_scores:
        fitness = sum(annotated_var_scores) / len(annotated_var_scores)
    else:
        fitness = 0.0

    results["ContVAR_fitness"] = round(fitness, 4)

    return results
