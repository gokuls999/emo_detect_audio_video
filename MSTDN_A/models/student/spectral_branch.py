from __future__ import annotations

import torch
from torch import nn


class SpectralBranch(nn.Module):
    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(32, output_dim)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        x = spectrogram.unsqueeze(1)
        x = self.net(x).flatten(1)
        return self.proj(x)
