from __future__ import annotations

import torch
from torch import nn


class TeacherFusion(nn.Module):
    def __init__(self, input_dims: list[int] | None = None, output_dim: int = 512) -> None:
        super().__init__()
        dims = input_dims or [256, 128, 128, 256, 256]
        self.projections = nn.ModuleList([nn.Linear(dim, 256) for dim in dims])
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(256, output_dim)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        tokens = [proj(emb) for proj, emb in zip(self.projections, embeddings)]
        x = torch.stack(tokens, dim=1)
        x = self.fusion(x)
        return self.output(x.mean(dim=1))
