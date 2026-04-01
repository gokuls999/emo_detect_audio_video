# MSTDN-A — Project Reference for Claude

## Project Identity

**Full name:** MSTDN — Multimodal Spatio-Temporal Distillation Network (Audio-Primary)
**Full title:** Affective State Inference in Unconstrained Multi-Person Video Streams via Cross-Modal Physiological Knowledge Distillation and Temporal Dynamics Modeling
**Level:** PhD research project
**Goal:** Detect emotions + stress in real-time from audio and face in classroom/office settings
**Status:** Live dashboard deployed; multi-emotion display being added

---

## 11 MAFW Emotion Classes (in index order)

| Index | Emotion | Emoji |
|-------|---------|-------|
| 0 | Anger | 😠 |
| 1 | Disgust | 🤢 |
| 2 | Fear | 😨 |
| 3 | Happiness | 😊 |
| 4 | Neutral | 😐 |
| 5 | Sadness | 😢 |
| 6 | Surprise | 😲 |
| 7 | Contempt | 😒 |
| 8 | Anxiety | 😰 |
| 9 | Helplessness | 😞 |
| 10 | Disappointment | 😔 |

---

## Model Architecture — MSTDNAStudent

**File:** `MSTDN_A/models/student/student_model.py`

```
waveform + spectrogram
    │
    ├── ProsodicBranch     → 128d  (Conv1d, 40 prosodic channels)
    ├── SpectralBranch     → 128d  (Conv2d on mel-spec, 80 mels)
    └── DeepAudioBranch    → 256d  (wav2vec2-base or fallback CNN)
            │
        Concat 512d → fusion Linear → 256d
            │
        TemporalProsodicTransformer  (4 layers, 4 heads, RoPE)
            │
        SpeakerMemoryGRU  (2-layer GRU, 256d hidden — persists across session)
            │
        z_s (256d embedding)
            │
        OutputHeads
            ├── primary_emotion      Linear(256,11)  → softmax → top-1 emotion
            ├── emotion_distribution Linear(256,11)  → softmax → probability vector
            ├── secondary_emotions   Linear(256,11)  → sigmoid → multi-label (threshold 0.30)
            ├── valence              Linear(256,1)   → tanh → [-1,+1]
            ├── arousal              Linear(256,1)   → sigmoid → [0,1]
            └── stress_score         Linear(256,1)   → sigmoid → [0,1]
```

**Key method:** `model.encode(wav, spec, hidden)` → returns `(z_s, new_hidden)` — can call heads directly on z_s

**Checkpoint file:** `MSTDN_A/checkpoints/english_finetune_r2.pt`
**Online-tuned checkpoint:** `MSTDN_A/checkpoints/online_tuned.pt`

---

## Face Detection & Emotion

### Face ID (`MSTDN_A/dashboard/face_id.py`)
- **Library:** InsightFace buffalo_l (ArcFace), ONNX Runtime
- **Provider:** CUDAExecutionProvider (cudnn_conv_algo_search=HEURISTIC), fallback CPU
- **Embedding:** 512-dim ArcFace, normalized
- **Matching:** cosine distance, `THRESHOLD = 0.40`
- **Key functions:** `extract_embedding()`, `detect_faces()`, `identify_all()`

### Face Emotion (`MSTDN_A/dashboard/face_emotion.py`)
- **Model:** FERPlus ONNX (8 classes), auto-downloaded on first run
- **Input:** 64×64 grayscale face crop
- **Mapping:** 8 FERPlus classes → 11 MAFW classes (see `_FERPLUS_TO_MAFW`)
- **Temperature scaling:** T=2.5 to flatten neutral-heavy distributions
- **Key functions:** `analyze_frame()`, `analyze_crop()` (use crop when InsightFace already detected face)

---

## Dashboard Server (`MSTDN_A/dashboard/server.py`)

### How to run
```bash
cd MSTDN_A
py -3 -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
```

### Key constants
```python
SAMPLE_RATE     = 16000       # Audio sample rate (Hz)
MAX_SAMPLES     = 64000       # 4 seconds of audio per inference chunk
INFER_EVERY     = 1.0         # Inference cadence (seconds)
SILENCE_THRESH  = 0.015       # RMS below this = silence, skip audio inference
AUDIO_TEMP      = 1.0         # Temperature scaling on primary_logits
STRESS_ALPHA    = 0.25        # EMA smoothing for stress score
STRESS_ALERT_TH = 0.80        # Threshold for triggering stress alert
```

### Two inference pipelines

**Dashboard (live camera + mic):**
- `_inference_loop()` — runs every INFER_EVERY seconds
- `_face_loop()` — runs every ~0.5s (InsightFace detection + FERPlus emotion)
- Blend: `probs = (aw * audio_p + fw * face_p) / (aw + fw)`
- WebSocket: `/ws` → broadcasts `emotion_update`

**YouTube test:**
- `_yt_infer()` — processes YouTube audio chunks
- `_yt_face_reader()` — face emotion from camera while YouTube plays
- Same blend formula, same secondary_logits extraction
- WebSocket: `/ws/yt` → broadcasts emotion results

### emotion_update WS message fields
```json
{
  "type":       "emotion_update",
  "emotion":    "Neutral",
  "emoji":      "😐",
  "confidence": 0.72,
  "stress":     0.23,
  "valence":    -0.1,
  "arousal":    0.4,
  "probs":      [0.05, 0.02, ...],   // 11-element float array
  "secondary":  ["Anxiety"],         // multi-label from secondary_logits >= 0.30
  "source":     "audio",             // "audio" | "face" | "audio+face" | "none"
  "rms":        0.045,
  "alert":      false,
  "face_name":  null,
  "timestamp":  "2026-03-30T..."
}
```

### Blend slider API
- `POST /api/blend` `{"audio": 0.7, "face": 0.3}` — set weights (dashboard)
- `POST /api/yt/blend` `{"audio": 0.7, "face": 0.3}` — set weights (YT test)

### Online teach (fast, ~5ms per click)
- Buffer stores `out["embedding"]` (256-dim z_s) — NOT raw audio
- `do_teach_step(emotion_idx, source)` — BCE on `primary_logits` with cached embedding
- `do_multi_teach_step(emotion_indices, source)` — BCE on `secondary_logits` for multi-label
- Endpoints: `POST /api/teach`, `POST /api/teach/multi`, `POST /api/yt/teach`, `POST /api/yt/teach/multi`
- Auto-learn: `POST /api/teach/auto_learn {"enabled": true}` — auto-teaches when conf >= 0.62, cooldown 12s

---

## Database (`MSTDN_A/dashboard/database.py`)

SQLite at `MSTDN_A/dashboard/dashboard.db`

| Table | Key columns | Purpose |
|-------|-------------|---------|
| `sessions` | id, name, location, started_at, ended_at | Session records |
| `participants` | id, session_id, name, role, department | People in session |
| `readings` | session_id, participant_id, emotion, confidence, stress, valence, arousal, probs(JSON) | Emotion history |
| `alerts` | session_id, participant_id, alert_type, message | Stress events |
| `face_embeddings` | participant_id, embedding(BLOB 512-dim float32) | ArcFace embeddings |

---

## Dashboard UI

### Main Dashboard (`static/index.html`)
**Layout:** 4-cell grid + nav + blend bar

- **Top nav:** Logo, session status chip, timer, mic activity, theme toggle, face-ID toggle, PDF button, Start/End session
- **Blend bar:** Audio% slider + Face% slider (sends to `/api/blend`)
- **Top-left:** Camera feed, LIVE badge, face name overlay, RMS bars
- **Top-right (emotion card):**
  - View toggle: Single ↔ Multi emotion mode
  - Primary emotion label + emoji + confidence
  - Secondary emotion chips (multi-mode only)
  - Source badge (Audio / Face / Both)
  - Metrics: Stress, Valence, Arousal
  - Distribution bars (2-column, 11 emotions)
  - Teach section (single-click in single mode, toggle+submit in multi mode)
  - Auto-learn checkbox + buffer counter + Save button
- **Bottom-left:** Participants list (add/remove, face register, focus speaker)
- **Bottom-right tabs:** Analytics | Alerts | Sessions

### YouTube Test (`static/yt_test.html`)
**Layout:** Left (video+waveform) + Right (emotion analysis)

- Same emotion card with view toggle
- Same teach section (single/multi mode)
- Same blend sliders (sends to `/api/yt/blend`)
- Dark mode toggle (saves to localStorage `mstdn-theme`)

---

## View Modes

### Single Mode (default)
- Shows top-1 primary emotion
- Teach: 11 single-click buttons (each click immediately teaches that one emotion)
- Auto-learn: trains `primary_emotion` head

### Multi Mode
- Shows primary emotion + secondary emotion chips (threshold 0.30 on sigmoid)
- Teach: 11 toggle-select buttons → Submit Correction button → teaches `secondary_emotions` head
- Auto-learn: trains both `primary_emotion` and `secondary_emotions` heads

Toggle state saved in localStorage (`mstdn-view-mode`).

---

## Development History / Corrections Made

### Face detection (deepface → InsightFace)
**Problem:** deepface requires tensorflow which can't install on Python 3.14 (ResolutionImpossible).
**Fix:** Replaced deepface with InsightFace (buffalo_l ArcFace) for face detection/recognition, FERPlus ONNX for emotion.

### Face emotion not blending
**Problem:** `face_emotion.analyze_frame()` used its own Haar cascade which failed to find the same face InsightFace detected, so `face_emotion_probs` was always None.
**Fix:** Feed InsightFace crops directly to `face_emotion.analyze_crop()` — guarantees emotion is computed for every detected face.

### Face embedding dimension mismatch
**Problem:** Old embeddings from Haar cascade (4096-dim) stored in DB conflicted with ArcFace (512-dim).
**Fix:** `database.get_face_embeddings()` skips any embedding not exactly 512-dim with a warning.

### GPU for ONNX Runtime
**Problem:** `onnxruntime` (CPU-only package) was installed; InsightFace and FERPlus ran on CPU.
**Fix:** Uninstalled `onnxruntime`, installed `onnxruntime-gpu` → CUDAExecutionProvider active.
Also: Changed `cudnn_conv_algo_search` from `EXHAUSTIVE` (slow startup) to `HEURISTIC`, added warmup inference at init.

### Teach button lag (500ms → 5ms)
**Problem:** Every teach click ran the full Wav2Vec2 forward+backward (~500ms on GPU).
**Fix:** Inference loop now caches `out["embedding"]` (256-dim z_s) in the buffer. `do_teach_step` runs only `model.heads(emb)` — 3 tiny linear layers, ~5ms.
Also removed second forward pass that was computing `new_prediction` after each teach step.

### Teach button stuck at 0 (JS busy flag)
**Problem:** Client-side `_teachBusy` flag stuck `true` if first teach failed (empty buffer / no session). Every subsequent click silently blocked.
**Fix:** Removed client-side busy flag entirely. Server handles deduplication via `teach_lock`.

### UI compaction
- Distribution bars → 2-column CSS grid (11 bars in 2 columns, not 1)
- Teach buttons moved into emotion card (always visible, no scroll needed)
- Dark mode added to main dashboard (was only on YT test)
- Blend sliders added to both dashboards

---

## UI Rules — MUST FOLLOW

- **No scrolling to reach controls.** Distribution bars and teach buttons MUST be visible without scrolling on the YT test page. All headers (topnav, blend bar, URL bar) must be as compact as possible. If adding new UI elements, they must NOT push bars/buttons below the fold.
- **Buttons must be compact.** Use small padding, short labels. Never enlarge buttons or add tall bars that eat vertical space.
- **Never change internal settings** (inference, training, model, LR, replay buffer, etc.) when working on UI/PDF/history features. Only touch what was asked.

## Known Issues

- `secondary_logits` (multi-label emotions) are computed by the model but were not surfaced to UI until multi-emotion feature was added.
- The model's `emotion_distribution` head output is not separately displayed — only the blended `probs` vector is shown in bars.
- Teacher model (EEG/GSR/PPG-based) not trained yet — student uses audio+face only.

---

## Project File Map

```
MSTDN_A/
├── checkpoints/
│   ├── english_finetune_r2.pt   ← main inference checkpoint
│   └── online_tuned.pt          ← online-corrected weights (created after first teach)
├── models/
│   ├── heads/output_heads.py    ← 3 emotion heads + regression
│   └── student/
│       ├── student_model.py     ← MSTDNAStudent main class
│       ├── deep_audio_branch.py ← wav2vec2 branch
│       ├── spectral_branch.py   ← CNN mel-spec branch
│       ├── prosodic_branch.py   ← Conv1d prosodic branch
│       ├── temporal_transformer.py ← 4-layer TransformerEncoder
│       └── speaker_gru.py       ← 2-layer GRU memory
├── dashboard/
│   ├── server.py                ← FastAPI server (~1400 lines)
│   ├── database.py              ← SQLite (5 tables)
│   ├── face_id.py               ← InsightFace detection + ArcFace matching
│   ├── face_emotion.py          ← FERPlus ONNX emotion (8→11 class)
│   ├── capture.py               ← VideoCapture + AudioCapture
│   ├── pdf_report.py            ← Session PDF export
│   └── static/
│       ├── index.html           ← Main dashboard
│       └── yt_test.html         ← YouTube test page
└── documents/
    └── CLAUDE.md                ← Full PhD research specification
```
