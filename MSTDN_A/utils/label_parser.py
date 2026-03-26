from __future__ import annotations

from pathlib import Path

import pandas as pd

EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
    "contempt",
    "anxiety",
    "helplessness",
    "disappointment",
]
LABEL_TO_INDEX = {name: idx for idx, name in enumerate(EMOTIONS)}


def _normalize_clip(value: str) -> str:
    value = str(value).strip()
    return value if value.endswith(".mp4") else f"{value}.mp4"


def load_annotation_table(labels_dir: str | Path) -> pd.DataFrame:
    path = Path(labels_dir) / "annotation.xlsx"
    df = pd.read_excel(path)
    clip_col = "clip" if "clip" in df.columns else df.columns[0]
    df[clip_col] = df[clip_col].map(_normalize_clip)
    if clip_col != "clip":
        df = df.rename(columns={clip_col: "clip"})
    return df


def load_single_set(labels_dir: str | Path) -> pd.DataFrame:
    path = Path(labels_dir) / "single-set.xlsx"
    df = pd.read_excel(path)
    df["clip"] = df["clip"].map(_normalize_clip)
    df["single_label"] = df["single_label"].astype(int) - 1
    return df


def load_multi_set(labels_dir: str | Path) -> pd.DataFrame:
    path = Path(labels_dir) / "multi-set.xlsx"
    df = pd.read_excel(path)
    df["clip"] = df["clip"].map(_normalize_clip)
    return df


def parse_split_file(split_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    path = Path(split_path)
    encoding = "utf-8"
    try:
        raw = path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        raw = path.read_text(encoding="cp1252", errors="replace")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            clip, label, *captions = line.split("\t")
            rows.append({"clip": clip, "label": label, "caption_zh": captions[0] if len(captions) > 0 else "", "caption_en": captions[1] if len(captions) > 1 else ""})
        else:
            clip, label = line.split(maxsplit=1)
            rows.append({"clip": clip, "label": label})
    return pd.DataFrame(rows)
