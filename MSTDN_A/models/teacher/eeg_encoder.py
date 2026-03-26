from __future__ import annotations

import torch
from torch import nn


class EEGNetBlock(nn.Module):
    def __init__(self, in_channels: int = 20, hidden: int = 64) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.project = nn.Conv1d(hidden, 256, kernel_size=1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self.temporal(eeg)
        x = self.project(x).transpose(1, 2)
        x = self.transformer(x).transpose(1, 2)
        return self.pool(x).squeeze(-1)
