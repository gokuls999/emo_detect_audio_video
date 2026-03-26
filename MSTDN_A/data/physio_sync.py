from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass
class ClipAlignment:
    clip_name: str
    start_seconds: float
    duration_seconds: float


def estimate_subject_clip_timeline(subject_dir: str | Path) -> list[ClipAlignment]:
    subject_path = Path(subject_dir)
    clips = sorted(subject_path.glob("*.mp4"), key=lambda p: int(p.stem))
    alignments: list[ClipAlignment] = []
    cursor = 0.0
    for clip in clips:
        wav_candidate = clip.with_suffix(".wav")
        duration = 3.0
        if wav_candidate.exists():
            info = sf.info(str(wav_candidate))
            duration = info.duration
        alignments.append(ClipAlignment(clip.name, cursor, duration))
        cursor += duration
    return alignments


def clip_start_lookup(subject_dir: str | Path) -> dict[str, ClipAlignment]:
    return {item.clip_name: item for item in estimate_subject_clip_timeline(subject_dir)}
