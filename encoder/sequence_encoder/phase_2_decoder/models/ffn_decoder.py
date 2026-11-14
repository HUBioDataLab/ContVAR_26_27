"""
FFNDecoder — Feed-Forward Network Decoder

Encoder embedding'den N_go_terms adet logit üretir.

Mimari:
    Linear(in → 512) → BN → ReLU → Dropout(0.30)
    Linear(512 → 1024) → BN → ReLU → Dropout(0.30)
    Linear(1024 → 512) → BN → ReLU → Dropout(0.20)
    Linear(512 → N_GO)

forward() sigmoid UYGULAMAZ:
    - Eğitimde BCEWithLogitsLoss sigmoid'i kendi içinde uygular (numerik kararlılık)
    - Inference'ta torch.sigmoid() açıkça çağrılır
"""

import torch
import torch.nn as nn


class FFNDecoder(nn.Module):

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims: list[int] = None, dropout: float = 0.3):
        """
        Args:
            input_dim:   Encoder çıktı boyutu (ESM-2 650M için 1280)
            n_classes:   GO vocab boyutu (NULL_FUNCTION dahil)
            hidden_dims: Gizli katman boyutları, default [512, 1024, 512]
            dropout:     Dropout oranı (son katmanda %67'si uygulanır)
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 1024, 512]

        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            # Son gizli katmanda daha düşük dropout
            p = dropout if i < len(hidden_dims) - 1 else dropout * 0.67
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(p),
            ]
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]

        Returns:
            [B, n_classes] — raw logits (sigmoid uygulanmamış)
        """
        return self.net(x)
