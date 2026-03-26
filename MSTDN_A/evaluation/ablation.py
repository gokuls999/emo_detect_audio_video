from __future__ import annotations

import json


def ablation_plan() -> dict[str, list[str]]:
    return {
        "a": ["prosodic"],
        "b": ["spectral"],
        "c": ["deep_audio"],
        "d": ["prosodic", "spectral"],
        "e": ["prosodic", "deep_audio"],
        "f": ["prosodic", "spectral", "deep_audio"],
        "g": ["full_without_distillation"],
        "h": ["full_with_distillation"],
    }


if __name__ == "__main__":
    print(json.dumps(ablation_plan(), indent=2))
