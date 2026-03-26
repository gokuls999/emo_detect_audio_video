from __future__ import annotations

import os

import torch


def choose_torch_device(preferred: str = "cuda") -> torch.device:
    if preferred != "cuda" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = torch.zeros(1, device="cuda")
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def force_transformers_offline() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
