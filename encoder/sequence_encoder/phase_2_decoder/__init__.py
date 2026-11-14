"""
phase_2_decoder — Phase 3: GO Function Decoder (Sekans Tabanlı)

ESM-2 mean-pooled protein embedding'lerinden GO fonksiyon anotasyonlarını
tahmin eden decoder modülü.

Pipeline:
    Phase 0 (GO Pretrain) → Phase 1 (DMS Fine-tune) → Phase 3 (Decoder) ← Biz buradayız

Kullanım:
    from phase_2_decoder.config import DecoderConfig
    from phase_2_decoder.models.ffn_decoder import FFNDecoder
    from phase_2_decoder.train import train
    from phase_2_decoder.inference import predict_function_change
"""

__version__ = "0.1.0"
