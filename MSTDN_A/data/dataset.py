from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from data.physio_sync import clip_start_lookup
from utils.caption_parser import load_captions
from utils.label_parser import LABEL_TO_INDEX, load_annotation_table, load_multi_set, load_single_set, parse_split_file
from utils.mat_loader import load_physio_session, slice_signal


def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    peak = waveform.abs().max().clamp_min(1e-6)
    return waveform / peak


def _safe_numeric(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else default


@dataclass
class DatasetConfig:
    base_dir: str
    split_variant: str = "single/no_caption"
    fold: str = "set_1"
    sample_rate: int = 16000
    window_seconds: float = 3.0
    use_physio: bool = True
    use_caption: bool = True


class AudioPhysioDataset(Dataset):
    def __init__(self, config: DatasetConfig, split_name: str = "train", transform: Any | None = None) -> None:
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.sample_rate = config.sample_rate
        self.window_samples = int(config.sample_rate * config.window_seconds)
        self.transform = transform

        split_path = self.base_dir / "train_test_splits" / config.split_variant / config.fold / f"{split_name}.txt"
        self.split_df = parse_split_file(split_path)
        self.annotation_df = load_annotation_table(self.base_dir / "emotion_labels")
        self.single_df = load_single_set(self.base_dir / "emotion_labels")
        self.multi_df = load_multi_set(self.base_dir / "emotion_labels")
        self.caption_df = load_captions(self.base_dir / "emotion_labels")
        self.annotation_lookup = self.annotation_df.set_index("clip")
        self.caption_lookup = self.caption_df.set_index("clip") if not self.caption_df.empty else pd.DataFrame()

        physio_root = self.base_dir / "physiological_data"
        self.physio_sessions = {
            path.name: load_physio_session(path)
            for path in physio_root.iterdir()
            if path.is_dir() and path.name not in {"5", "58"} and (path / "datas.mat").exists()
        } if config.use_physio and physio_root.exists() else {}
        self.physio_timelines = {
            subject: clip_start_lookup(physio_root / subject)
            for subject in self.physio_sessions
        }
        self.subject_order = sorted(self.physio_sessions)

    def __len__(self) -> int:
        return len(self.split_df)

    def _load_audio(self, clip: str) -> torch.Tensor:
        wav_path = self.base_dir / "video_clips" / "audio" / clip.replace(".mp4", ".wav")
        if wav_path.exists():
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            waveform = torch.from_numpy(data)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=-1)
            if sr != self.sample_rate:
                waveform = torch.from_numpy(
                    librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=self.sample_rate)
                )
            waveform = _normalize_audio(waveform)
        else:
            waveform = torch.zeros(self.window_samples, dtype=torch.float32)
        if waveform.numel() < self.window_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.window_samples - waveform.numel()))
        return waveform[: self.window_samples]

    def _subject_for_index(self, index: int) -> str | None:
        if not self.subject_order:
            return None
        return self.subject_order[index % len(self.subject_order)]

    def _lookup_labels(self, clip: str, label_value: str) -> dict[str, Any]:
        label_idx = LABEL_TO_INDEX.get(label_value, 0)
        emotion_distribution = np.zeros(11, dtype=np.float32)
        secondary = np.zeros(11, dtype=np.float32)
        valence = 0.0
        arousal = 0.0
        stress = 0.0
        if clip in self.annotation_lookup.index:
            row = self.annotation_lookup.loc[clip]
            emotion_cols = [col for col in row.index if col in LABEL_TO_INDEX]
            values = pd.to_numeric(row[emotion_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            values = np.clip(values, 0.0, 1.0)  # -1 means "not rated", treat as 0
            emotion_distribution[: len(values)] = values[:11]
            secondary = (values > 0.5).astype(np.float32)[:11]
            valence = float(values[3] - values[5] - 0.5 * values[0])
            arousal = float(values[[0, 2, 6, 8]].mean()) if len(values) >= 9 else 0.0
            stress = float(values[[0, 8, 9, 10]].mean()) if len(values) >= 11 else 0.0
        if "," in label_value:
            for token in label_value.split(","):
                token = token.strip()
                if token.isdigit():
                    secondary[int(token) - 1] = 1.0
        else:
            secondary[label_idx] = max(secondary[label_idx], 1.0)
        return {
            "primary_label": label_idx,
            "emotion_distribution": emotion_distribution,
            "secondary_labels": secondary,
            "valence": np.float32(np.tanh(valence)),
            "arousal": np.float32(np.clip((arousal + 1.0) / 2.0, 0.0, 1.0)),
            "stress": np.float32(np.clip(stress, 0.0, 1.0)),
        }

    def _lookup_caption(self, clip: str, row: pd.Series) -> str:
        if "caption_en" in row and isinstance(row["caption_en"], str) and row["caption_en"]:
            return row["caption_en"]
        if not self.config.use_caption or self.caption_lookup.empty or clip not in self.caption_lookup.index:
            return ""
        value = self.caption_lookup.loc[clip]
        if isinstance(value, pd.DataFrame):
            return str(value.iloc[0]["caption_en"])
        return str(value["caption_en"])

    def _lookup_physio(self, index: int, clip: str) -> dict[str, np.ndarray]:
        subject_id = self._subject_for_index(index)
        if subject_id is None:
            return {
                "eeg": np.zeros((20, 900), dtype=np.float32),
                "gsr": np.zeros((3, 12), dtype=np.float32),
                "ppg": np.zeros((3, 300), dtype=np.float32),
                "stress_target": np.float32(0.0),
                "subject_id": "",
            }
        session = self.physio_sessions[subject_id]
        timeline = self.physio_timelines[subject_id]
        alignment = timeline.get(clip, next(iter(timeline.values()), None))
        start_seconds = alignment.start_seconds if alignment else 0.0
        eeg = slice_signal(session.eeg, start_seconds, self.config.window_seconds, session.fs_eeg)
        gsr = slice_signal(session.gsr, start_seconds, self.config.window_seconds, session.fs_gsr)
        ppg = slice_signal(session.ppg, start_seconds, self.config.window_seconds, session.fs_ppg)
        stress_signal = np.maximum(gsr[0], 0.0)
        stress_target = float(np.clip(stress_signal.mean() / max(stress_signal.max(initial=1.0), 1e-6), 0.0, 1.0))
        return {
            "eeg": eeg.astype(np.float32),
            "gsr": gsr.astype(np.float32),
            "ppg": ppg.astype(np.float32),
            "stress_target": np.float32(stress_target),
            "subject_id": subject_id,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.split_df.iloc[index]
        clip = row["clip"]
        label_value = str(row["label"])
        waveform = self._load_audio(clip)
        if self.transform is not None:
            waveform = torch.from_numpy(self.transform(waveform.numpy()))
        labels = self._lookup_labels(clip, label_value)
        physio = self._lookup_physio(index, clip) if self.config.use_physio else {
            "eeg": np.zeros((20, 900), dtype=np.float32),
            "gsr": np.zeros((3, 12), dtype=np.float32),
            "ppg": np.zeros((3, 300), dtype=np.float32),
            "stress_target": np.float32(labels["stress"]),
            "subject_id": "",
        }
        return {
            "clip_id": clip.replace(".mp4", ""),
            "waveform": waveform.float(),
            "spectrogram": torch.tensor(librosa.feature.melspectrogram(y=waveform.numpy(), sr=self.sample_rate, n_mels=80), dtype=torch.float32),
            "caption_en": self._lookup_caption(clip, row),
            "eeg": torch.tensor(physio["eeg"], dtype=torch.float32),
            "gsr": torch.tensor(physio["gsr"], dtype=torch.float32),
            "ppg": torch.tensor(physio["ppg"], dtype=torch.float32),
            "primary_label": torch.tensor(labels["primary_label"], dtype=torch.long),
            "emotion_distribution": torch.tensor(labels["emotion_distribution"], dtype=torch.float32),
            "secondary_labels": torch.tensor(labels["secondary_labels"], dtype=torch.float32),
            "valence": torch.tensor(labels["valence"], dtype=torch.float32),
            "arousal": torch.tensor(labels["arousal"], dtype=torch.float32),
            "stress_score": torch.tensor(physio["stress_target"], dtype=torch.float32),
            "subject_id": physio["subject_id"],
        }
