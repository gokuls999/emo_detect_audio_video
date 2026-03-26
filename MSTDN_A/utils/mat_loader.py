from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


@dataclass
class PhysioSession:
    subject_id: str
    eeg: np.ndarray
    gsr: np.ndarray
    ppg: np.ndarray
    fs_eeg: int
    fs_gsr: int
    fs_ppg: int
    raw: dict[str, Any]


def _read_scalar(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).squeeze().item())
    return int(value)


def load_physio_session(subject_dir: str | Path) -> PhysioSession:
    subject_path = Path(subject_dir)
    mat_path = subject_path / "datas.mat"
    mat = loadmat(mat_path)
    return PhysioSession(
        subject_id=subject_path.name,
        eeg=np.asarray(mat["eeg_datas"], dtype=np.float32),
        gsr=np.asarray(mat["gsr_datas"], dtype=np.float32),
        ppg=np.asarray(mat["ppg_datas"], dtype=np.float32),
        fs_eeg=_read_scalar(mat["fs_eeg"]),
        fs_gsr=_read_scalar(mat["fs_gsr"]),
        fs_ppg=_read_scalar(mat["fs_ppg"]),
        raw=mat,
    )


def slice_signal(signal: np.ndarray, start_seconds: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    start = max(0, int(round(start_seconds * sample_rate)))
    end = max(start + 1, int(round((start_seconds + duration_seconds) * sample_rate)))
    end = min(end, signal.shape[-1])
    sliced = signal[..., start:end]
    target = int(round(duration_seconds * sample_rate))
    if sliced.shape[-1] >= target:
        return sliced[..., :target]
    pad_width = [(0, 0)] * sliced.ndim
    pad_width[-1] = (0, target - sliced.shape[-1])
    return np.pad(sliced, pad_width, mode="constant")
