"""
prepare_english_datasets.py
============================
Builds a clean, structured English emotion dataset from RAVDESS, CREMA-D and MELD.

Output layout
-------------
english_datasets/
    audio/
        train/   ← 16 kHz mono WAV
        val/
        test/
    labels/
        train.csv
        val.csv
        test.csv
    meta.json

Label mapping  (English -> MAFW 11-class)
-----------------------------------------
0  anger       | 1  disgust  | 2  fear      | 3  happiness
4  neutral     | 5  sadness  | 6  surprise  | 7  contempt
8  anxiety     | 9  helplessness | 10  disappointment

Run
---
    py -3.11 prepare_english_datasets.py
"""

import os, csv, json, shutil, subprocess, glob
from pathlib import Path

# -- paths ---------------------------------------------------------------------
BASE        = Path(r"C:\Users\ADMIN\Desktop\Bineetha - emoDet")
RAVDESS_DIR = BASE / "RAVDESS"
CREMAD_DIR  = BASE / "CREMA-D" / "AudioWAV"
MELD_RAW    = BASE / "MELD" / "MELD-RAW" / "MELD.Raw"
OUT_DIR     = BASE / "english_datasets"

FFMPEG = Path(r"C:\Users\ADMIN\AppData\Roaming\Python\Python314\site-packages"
              r"\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")

# -- label maps ----------------------------------------------------------------
# MAFW 11-class indices
EMO = {
    "anger":0, "disgust":1, "fear":2, "happiness":3,
    "neutral":4, "sadness":5, "surprise":6, "contempt":7,
    "anxiety":8, "helplessness":9, "disappointment":10,
}

# RAVDESS: digit 3 in filename (1-indexed emotion code -> MAFW index)
RAVDESS_MAP = {
    "01": (4, "neutral"),   # neutral
    "02": (4, "neutral"),   # calm -> neutral (closest)
    "03": (3, "happiness"), # happy
    "04": (5, "sadness"),   # sad
    "05": (0, "anger"),     # angry
    "06": (2, "fear"),      # fearful
    "07": (1, "disgust"),   # disgust
    "08": (6, "surprise"),  # surprised
}

# CREMA-D: code in position 3 of filename
CREMAD_MAP = {
    "ANG": (0, "anger"),
    "DIS": (1, "disgust"),
    "FEA": (2, "fear"),
    "HAP": (3, "happiness"),
    "NEU": (4, "neutral"),
    "SAD": (5, "sadness"),
}

# MELD: raw label string -> MAFW index
MELD_MAP = {
    "anger":   (0, "anger"),
    "disgust": (1, "disgust"),
    "fear":    (2, "fear"),
    "joy":     (3, "happiness"),
    "neutral": (4, "neutral"),
    "sadness": (5, "sadness"),
    "surprise":(6, "surprise"),
}

# -- split boundaries for RAVDESS (by actor number 1-24) ----------------------
# 80 % train (1-19) | 10 % val (20-22) | 10 % test (23-24)
def ravdess_split(actor_num: int) -> str:
    if actor_num <= 19: return "train"
    if actor_num <= 22: return "val"
    return "test"

# CREMA-D actor IDs 1001-1091  (91 actors)
# 80 % = first 73 | 10 % = next 9 | 10 % = last 9
def cremad_split(actor_id: int) -> str:
    idx = actor_id - 1001          # 0-based
    if idx < 73:  return "train"
    if idx < 82:  return "val"
    return "test"

# -- helpers -------------------------------------------------------------------
def ffmpeg_to_wav(src: Path, dst: Path) -> bool:
    """Convert any audio/video file to 16 kHz mono WAV using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(FFMPEG), "-y", "-i", str(src),
        "-vn", "-ar", "16000", "-ac", "1",
        "-acodec", "pcm_s16le", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0

def copy_resample_wav(src: Path, dst: Path) -> bool:
    """Resample existing WAV to 16 kHz mono via ffmpeg."""
    return ffmpeg_to_wav(src, dst)

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file","emotion_idx","emotion_name","source"])
        w.writeheader()
        w.writerows(rows)

# -- create output folders -----------------------------------------------------
for split in ("train", "val", "test"):
    (OUT_DIR / "audio" / split).mkdir(parents=True, exist_ok=True)
(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

records = {"train": [], "val": [], "test": []}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  RAVDESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- RAVDESS ----------------------------------------------------------")
ravdess_ok = ravdess_skip = 0

for actor_dir in sorted(RAVDESS_DIR.iterdir()):
    if not actor_dir.is_dir():
        continue
    actor_num = int(actor_dir.name.replace("Actor_", ""))
    split = ravdess_split(actor_num)

    for wav in sorted(actor_dir.glob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) < 7:
            ravdess_skip += 1
            continue
        emo_code = parts[2]
        if emo_code not in RAVDESS_MAP:
            ravdess_skip += 1
            continue
        emo_idx, emo_name = RAVDESS_MAP[emo_code]
        out_name = f"ravdess_{wav.stem}.wav"
        dst = OUT_DIR / "audio" / split / out_name

        if not dst.exists():
            ok = copy_resample_wav(wav, dst)
            if not ok:
                ravdess_skip += 1
                continue
        records[split].append({
            "file": f"audio/{split}/{out_name}",
            "emotion_idx": emo_idx,
            "emotion_name": emo_name,
            "source": "ravdess",
        })
        ravdess_ok += 1

print(f"  Processed : {ravdess_ok}  |  Skipped: {ravdess_skip}")
for s in ("train","val","test"):
    print(f"    {s}: {sum(1 for r in records[s] if r['source']=='ravdess')}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CREMA-D
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- CREMA-D ----------------------------------------------------------")
cremad_ok = cremad_skip = 0

for wav in sorted(CREMAD_DIR.glob("*.wav")):
    parts = wav.stem.split("_")
    if len(parts) < 3:
        cremad_skip += 1
        continue
    actor_id  = int(parts[0])
    emo_code  = parts[2]
    if emo_code not in CREMAD_MAP:
        cremad_skip += 1
        continue
    emo_idx, emo_name = CREMAD_MAP[emo_code]
    split    = cremad_split(actor_id)
    out_name = f"cremad_{wav.stem}.wav"
    dst      = OUT_DIR / "audio" / split / out_name

    if not dst.exists():
        ok = copy_resample_wav(wav, dst)
        if not ok:
            cremad_skip += 1
            continue
    records[split].append({
        "file": f"audio/{split}/{out_name}",
        "emotion_idx": emo_idx,
        "emotion_name": emo_name,
        "source": "cremad",
    })
    cremad_ok += 1

print(f"  Processed : {cremad_ok}  |  Skipped: {cremad_skip}")
for s in ("train","val","test"):
    print(f"    {s}: {sum(1 for r in records[s] if r['source']=='cremad')}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  MELD  (MP4 -> WAV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- MELD -------------------------------------------------------------")
meld_ok = meld_skip = 0

MELD_SPLITS = {
    "train": (MELD_RAW / "train", MELD_RAW / "train" / "train_sent_emo.csv", "train"),
    "dev":   (MELD_RAW / "dev",   MELD_RAW / "dev_sent_emo.csv",             "val"),
    "test":  (MELD_RAW / "test",  MELD_RAW / "test_sent_emo.csv",            "test"),
}

for meld_split, (clip_dir, csv_path, out_split) in MELD_SPLITS.items():
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping {meld_split}")
        continue

    with open(csv_path, encoding="utf-8") as f:
        rows_csv = list(csv.DictReader(f))

    split_ok = split_skip = 0
    for row in rows_csv:
        dia  = row["Dialogue_ID"]
        utt  = row["Utterance_ID"]
        emo  = row["Emotion"].strip().lower()
        if emo not in MELD_MAP:
            split_skip += 1
            continue
        emo_idx, emo_name = MELD_MAP[emo]

        # find the MP4 — could be in split root or in sub-folder *_splits_complete
        mp4_name = f"dia{dia}_utt{utt}.mp4"
        mp4_candidates = list(clip_dir.rglob(mp4_name))
        if not mp4_candidates:
            split_skip += 1
            continue
        mp4 = mp4_candidates[0]

        out_name = f"meld_{meld_split}_dia{dia}_utt{utt}.wav"
        dst = OUT_DIR / "audio" / out_split / out_name

        if not dst.exists():
            ok = ffmpeg_to_wav(mp4, dst)
            if not ok:
                split_skip += 1
                continue
        records[out_split].append({
            "file": f"audio/{out_split}/{out_name}",
            "emotion_idx": emo_idx,
            "emotion_name": emo_name,
            "source": "meld",
        })
        split_ok += 1
        meld_ok  += 1

    print(f"  {meld_split:5s} -> {out_split:5s}: {split_ok} ok, {split_skip} skipped")
    meld_skip += split_skip

print(f"  Total processed: {meld_ok}  |  Skipped: {meld_skip}")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Write label CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n-- Writing label CSVs -----------------------------------------------")
for split, rows in records.items():
    write_csv(OUT_DIR / "labels" / f"{split}.csv", rows)
    print(f"  {split}.csv  ->  {len(rows)} clips")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Write meta.json
# ══════════════════════════════════════════════════════════════════════════════
meta = {
    "description": "English emotion audio dataset — RAVDESS + CREMA-D + MELD",
    "sample_rate": 16000,
    "channels": 1,
    "format": "WAV PCM 16-bit",
    "emotion_classes": EMO,
    "splits": {s: len(r) for s, r in records.items()},
    "sources": {
        "ravdess": ravdess_ok,
        "cremad":  cremad_ok,
        "meld":    meld_ok,
    },
    "total": ravdess_ok + cremad_ok + meld_ok,
}
with open(OUT_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n-- Summary ----------------------------------------------------------")
print(f"  RAVDESS : {ravdess_ok:>6}")
print(f"  CREMA-D : {cremad_ok:>6}")
print(f"  MELD    : {meld_ok:>6}")
print(f"  TOTAL   : {ravdess_ok+cremad_ok+meld_ok:>6}")
print(f"\n  Output  : {OUT_DIR}")
print("\nDone.")
