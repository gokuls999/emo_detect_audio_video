from __future__ import annotations

import torch
from torch import nn


class SpeakerMemoryGRU(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_size: int = 256, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, sequence: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.gru(sequence, hidden)
        return output[:, -1], hidden
