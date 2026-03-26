from __future__ import annotations

import torch
from torch import nn

from models.heads.output_heads import OutputHeads
from models.student.deep_audio_branch import DeepAudioBranch
from models.student.prosodic_branch import ProsodicBranch
from models.student.speaker_gru import SpeakerMemoryGRU
from models.student.spectral_branch import SpectralBranch
from models.student.temporal_transformer import TemporalProsodicTransformer


class MSTDNAStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prosodic_branch = ProsodicBranch()
        self.spectral_branch = SpectralBranch()
        self.deep_audio_branch = DeepAudioBranch()
        self.fusion = nn.Linear(512, 256)
        self.temporal = TemporalProsodicTransformer()
        self.speaker_memory = SpeakerMemoryGRU()
        self.heads = OutputHeads(input_dim=256)

    @staticmethod
    def build_prosodic_proxy(waveform: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(waveform, chunks=20, dim=-1)
        stacked = []
        for chunk in chunks:
            stacked.append(torch.stack([chunk.mean(dim=-1), chunk.std(dim=-1)], dim=1))
        features = torch.cat(stacked, dim=1)
        return features.unsqueeze(-1).repeat(1, 1, 16)

    def encode(self, waveform: torch.Tensor, spectrogram: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        prosodic = self.build_prosodic_proxy(waveform)
        p = self.prosodic_branch(prosodic)
        s = self.spectral_branch(spectrogram)
        d = self.deep_audio_branch(waveform)
        fused = self.fusion(torch.cat([p, s, d], dim=-1))
        temporal_out = self.temporal(fused.unsqueeze(1))
        z_s, hidden = self.speaker_memory(temporal_out, hidden)
        return z_s, hidden

    def forward(self, waveform: torch.Tensor, spectrogram: torch.Tensor, hidden: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_s, hidden = self.encode(waveform, spectrogram, hidden)
        outputs = self.heads(z_s)
        outputs["embedding"] = z_s
        outputs["hidden"] = hidden
        return outputs
