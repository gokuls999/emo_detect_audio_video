from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import librosa
import numpy as np
import torch

from models.student.student_model import MSTDNAStudent
from realtime.group_analytics import aggregate_group_metrics


@dataclass
class SpeakerState:
    hidden: torch.Tensor | None = None
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=48000))
    history: list[dict] = field(default_factory=list)
    last_seen: float = 0.0
    active: bool = True


class RealtimeInferenceEngine:
    def __init__(self, checkpoint_path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MSTDNAStudent().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        self.state: dict[str, SpeakerState] = defaultdict(SpeakerState)

    def update_speaker_audio(self, speaker_id: str, waveform: np.ndarray, timestamp: float) -> None:
        state = self.state[speaker_id]
        state.audio_buffer.extend(waveform.tolist())
        state.last_seen = timestamp
        state.active = True

    def infer_speaker(self, speaker_id: str) -> dict:
        state = self.state[speaker_id]
        audio = np.asarray(state.audio_buffer, dtype=np.float32)
        if audio.size < 48000:
            audio = np.pad(audio, (0, 48000 - audio.size))
        else:
            audio = audio[-48000:]
        spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
        with torch.no_grad():
            outputs = self.model(
                torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(spec, dtype=torch.float32, device=self.device).unsqueeze(0),
                hidden=state.hidden,
            )
        state.hidden = outputs["hidden"]
        probs = torch.softmax(outputs["primary_logits"], dim=-1).cpu().numpy()[0]
        secondary = torch.sigmoid(outputs["secondary_logits"]).cpu().numpy()[0]
        jitter_proxy = float(np.std(np.diff(audio[: min(len(audio), 4000)]))) if len(audio) > 1 else 0.0
        hnr_proxy = float(np.mean(np.abs(audio))) + 1e-6
        vocal_stress = float(np.clip((jitter_proxy + float(np.std(audio)) - hnr_proxy) * 10.0, 0.0, 1.0))
        engagement = float(np.clip(np.sqrt(np.mean(np.square(audio))) * 5.0, 0.0, 1.0))
        record = {
            "primary_emotion": int(probs.argmax()),
            "emotion_distribution": probs.tolist(),
            "secondary_emotions": secondary.tolist(),
            "valence": float(outputs["valence"].item()),
            "arousal": float(outputs["arousal"].item()),
            "stress_score": float(outputs["stress_score"].item()),
            "vocal_stress_index": vocal_stress,
            "engagement_score": engagement,
            "confidence": float(1.0 - np.var(probs)),
            "affect_trend": 0,
            "session_state": "calm",
            "active": state.active,
        }
        state.history.append(record)
        return record

    def summarize_room(self) -> dict:
        outputs = {speaker_id: self.infer_speaker(speaker_id) for speaker_id in self.state}
        return aggregate_group_metrics(outputs)
