# Dataset Analysis

This document summarizes the datasets currently present in this workspace based on local file inspection on March 13, 2026.

## Workspace Inventory

| Path | Purpose | File count | Size |
| --- | --- | ---: | ---: |
| `dataset_11194571/` | Main extracted video + physiological signals dataset | 1491 | 7.541 GB |
| `labels/` | Manual label spreadsheets | 4 | 0.002 GB |
| `labels_auto/` | Auto-generated gender and landmarks metadata | 2 | 0.157 GB |
| `train_test_set/` | Predefined train/test folds | 40 | 0.015 GB |
| `data_clips/` | Split archive parts for clip data | 2 | 7.428 GB |
| `Extracted/` | Split archive parts for aligned data | 3 | 14.940 GB |

## 1. Main Extracted Dataset: `dataset_11194571/`

### Structure

- `47` subject folders are present.
- `1442` `.mp4` clips are available in total.
- Video counts per subject are not uniform across the extracted folders:
  - `45` subjects contain `32` clips each.
  - Subject `58` contains only `2` clips: `0.mp4`, `1.mp4`.
  - Subject `5` exists but contains `0` video clips.
- Non-video files present:
  - `45` `.mat` files named `datas.mat`
  - `2` `.fig` files
  - `2` `.tif` files

### Physiological Signal Files

A sampled `datas.mat` file contains:

- `eeg_datas`: shape `(20, 316500)`
- `fs_eeg`: shape `(1, 1)`
- `fs_gsr`: shape `(1, 1)`
- `fs_ppg`: shape `(1, 1)`
- `gsr_datas`: shape `(3, 4220)`
- `ppg_datas`: shape `(3, 105500)`

This indicates the dataset is multimodal: video + EEG + GSR + PPG.

### Notes

- `dataset_11194571/37/` additionally contains `gsr.fig`, `gsr.tif`, `ppg.fig`, and `ppg.tif`.
- The subject IDs are not continuous from `1` to `58`; only the folders currently extracted on disk can be assumed available.

## 2. Manual Label Files: `labels/Labels/`

Four Excel files define the label space and textual metadata.

### `annotation.xlsx`

- Rows: `10045`
- Columns: `13`
- Columns:
  - `clip`
  - `invalid`
  - `anger`
  - `disgust`
  - `fear`
  - `happiness`
  - `neutral`
  - `sadness`
  - `surprise`
  - `contempt`
  - `anxiety`
  - `helplessness`
  - `disappointment`

Observed `invalid` values:

- `0`: `7613` clips
- `-1`: `2431` clips
- `0.93`: `1` clip

The `0.93` value in `invalid` is a data anomaly and should be treated carefully in preprocessing.

Positive annotation counts by emotion column:

| Label id | Emotion | Positive clips |
| ---: | --- | ---: |
| 1 | anger | 1597 |
| 2 | disgust | 1778 |
| 3 | fear | 978 |
| 4 | happiness | 1377 |
| 5 | neutral | 1141 |
| 6 | sadness | 1762 |
| 7 | surprise | 1532 |
| 8 | contempt | 431 |
| 9 | anxiety | 1724 |
| 10 | helplessness | 1255 |
| 11 | disappointment | 331 |

### `single-set.xlsx`

- Rows: `9172` labeled clips
- Columns: `13`
- One row per clip
- Final column: `single_label`

Single-label distribution:

| Label id | Emotion | Clips |
| ---: | --- | ---: |
| 1 | anger | 1390 |
| 2 | disgust | 639 |
| 3 | fear | 625 |
| 4 | happiness | 1242 |
| 5 | neutral | 1138 |
| 6 | sadness | 1470 |
| 7 | surprise | 1072 |
| 8 | contempt | 236 |
| 9 | anxiety | 916 |
| 10 | helplessness | 262 |
| 11 | disappointment | 182 |

Observations:

- All clip IDs are unique.
- This file exactly matches the clip universe used in `train_test_set/single/no_caption/`.

### `multi-set.xlsx`

- Rows: `4058` labeled clips
- Columns: `13`
- Final column: `multi_label`

Emotion participation counts inside multi-label clips:

| Label id | Emotion | Occurrences |
| ---: | --- | ---: |
| 1 | anger | 1127 |
| 2 | disgust | 1412 |
| 3 | fear | 770 |
| 4 | happiness | 258 |
| 6 | sadness | 1105 |
| 7 | surprise | 971 |
| 8 | contempt | 278 |
| 9 | anxiety | 1230 |
| 10 | helplessness | 1036 |
| 11 | disappointment | 235 |

Top multi-label combinations:

| Combination | Clips |
| --- | ---: |
| `1,2` | 883 |
| `6,10` | 542 |
| `3,7` | 392 |
| `7,9` | 232 |
| `3,9` | 218 |
| `2,8` | 208 |
| `4,7` | 206 |
| `6,9` | 185 |
| `9,10` | 156 |
| `2,9` | 131 |

Observations:

- All clip IDs are unique.
- The multi-label set overlaps completely with the clip universe used in `train_test_set/compound/no_caption/`, but that split contains additional single-label clips as well.

### `descriptive_text.xlsx`

- Rows: `8034`
- Columns: `3`
- No header row
- Columns are:
  - clip id
  - Chinese description
  - English description

Example row pattern:

- `00025.mp4`
- Chinese natural-language description of the scene/expression
- English natural-language description of the scene/expression

Key relationship:

- This file contains exactly all non-neutral clips from `single-set.xlsx`.
- Missing from this file: all `1138` `neutral` single-label clips.

## 3. Auto Labels: `labels_auto/Labels(auto)/`

### `gender.txt`

- Rows: `10045`
- Format: `clip_id_without_extension: label`
- Example:
  - `00019: 1`
  - `00020: 1`

Observed label counts:

| Label | Clips |
| --- | ---: |
| `1` | 5841 |
| `0` | 4204 |

The mapping of `0` and `1` to semantic genders is not documented in the file itself and must not be assumed without external source confirmation.

### `landmarks.rar`

- Present but not extracted
- Size: `168,766,477` bytes

This likely contains landmark annotations aligned to the clip IDs, but the internal structure has not yet been inspected because it remains archived.

## 4. Predefined Splits: `train_test_set/Train & Test Set/`

The split package contains `5` folds for each of the following views:

- `single/no_caption`
- `single/with_caption`
- `compound/no_caption`
- `compound/with_caption`

### `single/no_caption`

- Encoding: UTF-8
- Line format: `clip.mp4 label`
- Total clips per fold: `9172`

| Fold | Train | Test |
| --- | ---: | ---: |
| set_1 | 7333 | 1839 |
| set_2 | 7335 | 1837 |
| set_3 | 7339 | 1833 |
| set_4 | 7340 | 1832 |
| set_5 | 7341 | 1831 |

This split exactly covers `single-set.xlsx`.

### `single/with_caption`

- Encoding: effectively non-UTF8; `cp1252` read succeeds because lines contain mixed English text plus stored Chinese bytes
- Line format: `clip.mp4<TAB>label<TAB>Chinese caption<TAB>English caption`
- Total clips per fold: `8034`

| Fold | Train | Test |
| --- | ---: | ---: |
| set_1 | 6423 | 1611 |
| set_2 | 6425 | 1609 |
| set_3 | 6429 | 1605 |
| set_4 | 6429 | 1605 |
| set_5 | 6430 | 1604 |

This split exactly covers `descriptive_text.xlsx`, which means:

- all non-neutral single-label clips are included
- all neutral clips are excluded

### `compound/no_caption`

- Encoding: UTF-8
- Line format: `clip.mp4 label_or_label_combo`
- Total clips per fold: `8996`

| Fold | Train | Test |
| --- | ---: | ---: |
| set_1 | 7178 | 1818 |
| set_2 | 7187 | 1809 |
| set_3 | 7198 | 1798 |
| set_4 | 7208 | 1788 |
| set_5 | 7213 | 1783 |

Important relationship:

- All `4058` clips from `multi-set.xlsx` are included.
- An additional `4938` single-label clips are also included.
- `176` single-label clips from `single-set.xlsx` are not part of the compound split.

Class distribution for `compound/no_caption` set_1:

| Label | Clips |
| --- | ---: |
| neutral | 1138 |
| happiness | 1075 |
| anger_disgust | 883 |
| sadness | 610 |
| sadness_helplessness | 542 |
| surprise | 497 |
| anger | 425 |
| anxiety | 408 |
| fear_surprise | 392 |
| disgust | 310 |

The remaining classes are lower-frequency combinations and minority single labels.

### `compound/with_caption`

- Encoding: same practical issue as `single/with_caption`
- Line format: `clip.mp4<TAB>label_or_label_combo<TAB>Chinese caption<TAB>English caption`
- Total clips per fold: `7858`

| Fold | Train | Test |
| --- | ---: | ---: |
| set_1 | 6268 | 1590 |
| set_2 | 6277 | 1581 |
| set_3 | 6288 | 1570 |
| set_4 | 6297 | 1561 |
| set_5 | 6302 | 1556 |

Important relationship:

- This is the captioned subset of the compound split.
- It contains no neutral-only clips because neutral clips have no description entries in `descriptive_text.xlsx`.
- It is smaller than `single/with_caption` by `176` clips because those `176` single-label clips are not included in the compound split definition.

## 5. Dataset Relationships Summary

The current metadata can be viewed as a hierarchy:

1. `annotation.xlsx` is the broadest clip-level label table with `10045` clips.
2. `single-set.xlsx` is a filtered single-label subset with `9172` clips.
3. `multi-set.xlsx` is a distinct multi-label subset with `4058` clips.
4. `descriptive_text.xlsx` is the caption subset and matches all non-neutral single-label clips exactly (`8034` clips).
5. Split files package these clip universes into 5 predefined folds.

## 6. Data Quality and Processing Risks

- `invalid` in `annotation.xlsx` contains a non-binary outlier value: `0.93`.
- Captioned split files are not safely UTF-8 encoded; decoding assumptions will break.
- `gender.txt` lacks an explicit semantic legend for labels `0` and `1`.
- `dataset_11194571/` is incomplete or irregular at the subject level:
  - subject `5` has no videos
  - subject `58` has only two videos
- Landmark data is still archived and cannot yet be cross-checked against the clip IDs.

## 7. Practical Recommendation Before Modeling

Use the following as the primary source of truth depending on task:

- Single-label classification without captions:
  - `labels/Labels/single-set.xlsx`
  - `train_test_set/Train & Test Set/single/no_caption/`
- Single-label classification with captions:
  - `labels/Labels/descriptive_text.xlsx`
  - `train_test_set/Train & Test Set/single/with_caption/`
- Multi-label or compound emotion modeling:
  - `labels/Labels/multi-set.xlsx`
  - `train_test_set/Train & Test Set/compound/`
- Multimodal physiological modeling:
  - `dataset_11194571/`
  - verify subject/clip alignment before training, because the raw extracted subject folders do not line up one-to-one with the clip-level label tables by simple count alone.
