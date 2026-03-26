from __future__ import annotations

import torch
from torch import nn

from models.heads.output_heads import OutputHeads
from models.teacher.audio_teacher import AudioTeacherEncoder
from models.teacher.caption_encoder import FrozenCaptionEncoder
from models.teacher.eeg_encoder import EEGNetBlock
from models.teacher.gsr_encoder import DilatedSignalEncoder
from models.teacher.ppg_encoder import PPGEncoder
from models.teacher.teacher_fusion import TeacherFusion


class MSTDNATeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eeg_encoder = EEGNetBlock()
        self.gsr_encoder = DilatedSignalEncoder()
        self.ppg_encoder = PPGEncoder()
        self.audio_encoder = AudioTeacherEncoder()
        self.caption_encoder = FrozenCaptionEncoder()
        self.fusion = TeacherFusion()
        self.heads = OutputHeads(input_dim=512)

    def encode(self, eeg: torch.Tensor, gsr: torch.Tensor, ppg: torch.Tensor, waveform: torch.Tensor, captions: list[str]) -> torch.Tensor:
        eeg_emb = self.eeg_encoder(eeg)
        gsr_emb = self.gsr_encoder(gsr)
        ppg_emb = self.ppg_encoder(ppg)
        audio_emb = self.audio_encoder(waveform)
        caption_emb = self.caption_encoder(captions).to(waveform.device)
        return self.fusion([eeg_emb, gsr_emb, ppg_emb, audio_emb, caption_emb])

    def forward(self, eeg: torch.Tensor, gsr: torch.Tensor, ppg: torch.Tensor, waveform: torch.Tensor, captions: list[str]) -> dict[str, torch.Tensor]:
        z_t = self.encode(eeg, gsr, ppg, waveform, captions)
        outputs = self.heads(z_t)
        outputs["embedding"] = z_t
        return outputs
