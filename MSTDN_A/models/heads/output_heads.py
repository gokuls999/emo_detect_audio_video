from __future__ import annotations

import torch
from torch import nn


class OutputHeads(nn.Module):
    def __init__(self, input_dim: int = 256, num_classes: int = 11) -> None:
        super().__init__()
        self.primary_emotion = nn.Linear(input_dim, num_classes)
        self.emotion_distribution = nn.Linear(input_dim, num_classes)
        self.secondary_emotions = nn.Linear(input_dim, num_classes)
        self.valence = nn.Linear(input_dim, 1)
        self.arousal = nn.Linear(input_dim, 1)
        self.stress_score = nn.Linear(input_dim, 1)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "primary_logits": self.primary_emotion(embedding),
            "distribution_logits": self.emotion_distribution(embedding),
            "secondary_logits": self.secondary_emotions(embedding),
            "valence": torch.tanh(self.valence(embedding)),
            "arousal": torch.sigmoid(self.arousal(embedding)),
            "stress_score": torch.sigmoid(self.stress_score(embedding)),
        }
