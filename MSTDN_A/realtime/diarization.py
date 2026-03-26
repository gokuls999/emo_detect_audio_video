from __future__ import annotations

try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None


class SpeakerDiarizer:
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1") -> None:
        self.pipeline = Pipeline.from_pretrained(model_name) if Pipeline is not None else None

    def diarize(self, audio_path: str) -> list[dict[str, str | float]]:
        if self.pipeline is None:
            return [{"speaker_id": "SPK_001", "start": 0.0, "end": 3.0}]
        diarization = self.pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"speaker_id": str(speaker), "start": float(turn.start), "end": float(turn.end)})
        return segments
