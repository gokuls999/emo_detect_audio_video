from __future__ import annotations

import torch
from torch import nn

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None


class DeepAudioBranch(nn.Module):
    def __init__(self, model_name: str = "facebook/wav2vec2-base", output_dim: int = 256) -> None:
        super().__init__()
        self.enabled = AutoModel is not None
        if self.enabled:
            try:
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                for param in self.model.parameters():
                    param.requires_grad = False
                for layer in self.model.encoder.layers[-2:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                hidden = self.model.config.hidden_size
                self.proj = nn.Linear(hidden, output_dim)
            except Exception:
                self.enabled = False
        if not self.enabled:
            self.model = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Linear(128, output_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            outputs = self.model(input_values=waveform)
            hidden = outputs.last_hidden_state.mean(dim=1)
            return self.proj(hidden)
        x = self.model(waveform.unsqueeze(1)).squeeze(-1)
        return self.proj(x)
