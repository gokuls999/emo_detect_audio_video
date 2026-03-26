from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.runtime import choose_torch_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_merge(base: dict, override: dict) -> None:
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if "inherits" in cfg:
        parent_path = config_path.parent / cfg.pop("inherits")
        base = load_config(parent_path)
        _deep_merge(base, cfg)
        return base
    return cfg


def resolve_device(device_name: str) -> torch.device:
    return choose_torch_device(device_name)
