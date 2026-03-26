from __future__ import annotations

from collections import Counter

import numpy as np


def aggregate_group_metrics(speaker_outputs: dict[str, dict]) -> dict:
    active = [value for value in speaker_outputs.values() if value.get("active", True)]
    if not active:
        return {"speaking_stress_prevalence": 0.0, "mean_valence": 0.0, "mean_arousal": 0.0, "engagement_rate": 0.0, "emotion_distribution": {}}
    stress_scores = np.array([item["stress_score"] for item in active], dtype=float)
    valence = np.array([item["valence"] for item in active], dtype=float)
    arousal = np.array([item["arousal"] for item in active], dtype=float)
    engagement = np.array([item["engagement_score"] for item in active], dtype=float)
    emotions = Counter(item["primary_emotion"] for item in active)
    return {
        "speaking_stress_prevalence": float((stress_scores > 0.65).mean()),
        "mean_valence": float(valence.mean()),
        "mean_arousal": float(arousal.mean()),
        "engagement_rate": float((engagement > 0.5).mean()),
        "vocal_distress_count": int(sum(item["vocal_stress_index"] > 0.7 for item in active)),
        "emotion_distribution": dict(emotions),
        "alert_speakers": [sid for sid, value in speaker_outputs.items() if value["stress_score"] > 0.8],
        "silent_speakers": [sid for sid, value in speaker_outputs.items() if not value.get("active", True)],
    }
