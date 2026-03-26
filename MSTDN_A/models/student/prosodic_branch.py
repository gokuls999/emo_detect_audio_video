from __future__ import annotations

import torch
from torch import nn


class ProsodicBranch(nn.Module):
    def __init__(self, in_channels: int = 40, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, prosodic_features: torch.Tensor) -> torch.Tensor:
        x = self.net(prosodic_features).squeeze(-1)
        return self.proj(x)
