from __future__ import annotations

import random
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AugmentationConfig:
    rir_p: float = 0.5
    noise_p: float = 0.5
    mic_deg_p: float = 0.5
    tempo_pitch_p: float = 0.5


class AudioAugmentationPipeline:
    def __init__(self, sample_rate: int = 16000, config: AugmentationConfig | None = None) -> None:
        self.sample_rate = sample_rate
        self.config = config or AugmentationConfig()

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        audio = np.asarray(waveform, dtype=np.float32).copy()
        if random.random() < self.config.noise_p:
            noise = np.random.normal(0.0, 0.005, size=audio.shape).astype(np.float32)
            audio = audio + noise
        if random.random() < self.config.mic_deg_p:
            cutoff = random.uniform(3000.0, 8000.0)
            audio = librosa.effects.preemphasis(audio, coef=min(0.97, cutoff / self.sample_rate))
            audio = np.clip(audio, -1.0, 1.0)
        if random.random() < self.config.tempo_pitch_p:
            rate = random.uniform(0.8, 1.2)
            steps = random.uniform(-2.0, 2.0)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
        if random.random() < self.config.rir_p:
            decay = np.exp(-np.linspace(0.0, 4.0, 256)).astype(np.float32)
            audio = np.convolve(audio, decay / decay.sum(), mode="full")[: waveform.shape[-1]]
        if audio.shape[-1] < waveform.shape[-1]:
            audio = np.pad(audio, (0, waveform.shape[-1] - audio.shape[-1]))
        return audio[: waveform.shape[-1]].astype(np.float32)
