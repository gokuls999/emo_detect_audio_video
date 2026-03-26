from __future__ import annotations

import torch
from torch import nn


class DilatedSignalEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 128) -> None:
        super().__init__()
        hidden = 64
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden, output_dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        x = self.net(signal).squeeze(-1)
        return self.proj(x)
