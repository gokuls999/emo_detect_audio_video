from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_captions(labels_dir: str | Path) -> pd.DataFrame:
    path = Path(labels_dir) / "descriptive_text.xlsx"
    df = pd.read_excel(path, header=None, names=["clip", "caption_zh", "caption_en"])
    df["clip"] = df["clip"].astype(str).map(lambda x: x if x.endswith(".mp4") else f"{x}.mp4")
    return df
