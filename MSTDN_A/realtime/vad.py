from __future__ import annotations

import numpy as np

try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except Exception:
    get_speech_timestamps = None
    load_silero_vad = None


class VADWrapper:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.model = load_silero_vad() if load_silero_vad is not None else None

    def is_speech(self, waveform: np.ndarray) -> bool:
        if self.model is not None:
            timestamps = get_speech_timestamps(waveform, self.model, sampling_rate=self.sample_rate)
            return bool(timestamps)
        energy = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        return energy > 0.01
