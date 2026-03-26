from __future__ import annotations

import torch
from torch import nn


class TemporalProsodicTransformer(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encoder(sequence)
