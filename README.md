# MSTDN-A: Multimodal Emotion & Stress Detection

**Real-time emotion detection from audio and video using deep learning with interactive online learning.**

MSTDN-A (Multimodal Spatio-Temporal Distillation Network — Audio) detects 11 emotions and stress levels from live audio/video streams. It uses a teacher-student knowledge distillation architecture where physiological signals (EEG, GSR, PPG) guide training, but only audio and face video are needed at inference time.

---

## Features

- **11 Emotion Classes**: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Contempt, Anxiety, Helplessness, Disappointment
- **Stress Detection**: Continuous stress score (0–1) with alert thresholds
- **Real-Time Dashboard**: Live webcam + microphone emotion monitoring with WebSocket updates
- **YouTube Analysis**: Analyze emotions from any YouTube video URL
- **Interactive Online Learning**: Teach the model correct emotions in real-time — click the correct emotion button while audio plays, and the model learns immediately
- **Face + Audio Fusion**: Combines facial expression analysis with audio emotion recognition
- **Session Reports**: PDF/Excel reports with emotion timelines and statistics

## Architecture

```
┌─────────────────── Teacher (training only) ───────────────────┐
│  EEG Transformer + GSR/PPG CNN + CLIP Caption Encoder         │
│  → Cross-modal Fusion → Teacher embedding Z_t                 │
└───────────────────────────────────────────────────────────────┘
                         ↓ Knowledge Distillation
┌─────────────────── Student (inference) ───────────────────────┐
│  wav2vec2 Deep Audio + Mel-Spectrogram + Prosodic Features    │
│  → Temporal Transformer (RoPE) → Speaker GRU Memory           │
│  → Multi-task Heads (emotion, stress, valence, arousal)       │
└───────────────────────────────────────────────────────────────┘
```

### Training Pipeline (5 Stages)

1. **Stage 1** — VideoMAE self-supervised pretraining
2. **Stage 2** — Multimodal teacher training (EEG + GSR + PPG + audio + captions)
3. **Stage 3** — Student knowledge distillation (audio only, teacher frozen)
4. **Stage 4** — Caption-CLIP refinement
5. **Stage 5** — Domain augmentation (noise, occlusion, varied conditions)
6. **Fine-tuning** — English audio datasets (CREMA-D, RAVDESS, MELD)

## Project Structure

```
MSTDN_A/
├── configs/              # YAML training configs for each stage
├── dashboard/
│   ├── server.py         # FastAPI server (main app)
│   ├── static/
│   │   ├── index.html    # Main dashboard (webcam + mic)
│   │   └── yt_test.html  # YouTube video analysis page
│   ├── capture.py        # Audio capture utilities
│   ├── database.py       # SQLite session storage
│   ├── face_id.py        # Face detection & recognition
│   └── pdf_report.py     # Session report generation
├── data/
│   ├── dataset.py        # PyTorch dataset classes
│   ├── loaders.py        # DataLoader builders
│   ├── augmentations.py  # Audio augmentation transforms
│   └── physio_sync.py    # Physiological signal alignment
├── models/
│   ├── student/          # Student model (audio-based)
│   │   ├── student_model.py
│   │   ├── deep_audio_branch.py    # wav2vec2 feature extractor
│   │   ├── spectral_branch.py      # Mel-spectrogram CNN
│   │   ├── prosodic_branch.py      # Pitch/energy features
│   │   ├── temporal_transformer.py  # Temporal attention
│   │   └── speaker_gru.py          # Speaker identity memory
│   ├── teacher/          # Teacher model (multimodal)
│   │   ├── teacher_model.py
│   │   ├── eeg_encoder.py
│   │   ├── gsr_encoder.py
│   │   ├── ppg_encoder.py
│   │   └── caption_encoder.py
│   ├── heads/output_heads.py  # Classification/regression heads
│   └── distillation.py       # KD loss functions
├── training/
│   ├── stage1_ssl.py → stage5_augment.py  # Training scripts
│   ├── finetune_english.py    # English dataset fine-tuning
│   ├── losses.py              # Loss functions
│   └── common.py              # Shared training utilities
├── evaluation/
│   ├── evaluate.py       # Full evaluation pipeline
│   ├── metrics.py        # Classification/regression metrics
│   └── ablation.py       # Ablation study runner
├── realtime/
│   ├── inference.py      # Real-time inference engine
│   ├── vad.py            # Voice activity detection
│   ├── diarization.py    # Speaker diarization
│   └── group_analytics.py
├── utils/                # Label parsing, audio loading, etc.
├── requirements.txt
├── start_dashboard.bat   # Windows launcher for dashboard
└── start_live_demo.bat   # Windows launcher for live demo
```

## Quick Start

### Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with CUDA support (12GB+ VRAM recommended)
- **FFmpeg** (for audio/video processing)
- **yt-dlp** (for YouTube analysis)

### Installation

```bash
# Clone the repository
git clone https://github.com/gokuls999/emo_detect_audio_video.git
cd emo_detect_audio_video

# Create virtual environment
python -m venv MSTDN_A/.venv
MSTDN_A/.venv/Scripts/activate   # Windows
# source MSTDN_A/.venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r MSTDN_A/requirements.txt

# Additional dashboard dependencies
pip install fastapi uvicorn[standard] websockets python-multipart yt-dlp
pip install deepface tf-keras  # For face emotion detection
```

### Model Checkpoints

Model checkpoint files (`.pt`) are **not included** in this repository due to their size (~377MB each). You need to either:

1. **Train from scratch** using the training pipeline (requires the MAFW dataset + physiological data)
2. **Fine-tune on English datasets** (CREMA-D, RAVDESS, MELD) using `training/finetune_english.py`
3. **Contact the authors** for pre-trained weights

Place checkpoint files in `MSTDN_A/checkpoints/`:
```
MSTDN_A/checkpoints/
├── english_finetune_r2.pt    # Main inference checkpoint
└── online_tuned.pt           # Online-learning refined (created at runtime)
```

### Running the Dashboard

```bash
cd MSTDN_A
python -m uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
```

Then open:
- **Main Dashboard**: http://localhost:8000 — live webcam + microphone emotion detection
- **YouTube Analysis**: http://localhost:8000/yt_test — analyze emotions from YouTube videos

### Online Learning (Interactive Teaching)

Both dashboard pages include a **Teach Panel** that lets you correct the model in real-time:

1. Play audio/video with a known emotion
2. Click the correct emotion button in the teach panel
3. The model updates its weights immediately (single-step gradient descent)
4. After teaching, click "Save Checkpoint" to persist improvements

This is particularly useful for improving detection of under-represented emotions like Anger.

## Dataset

This project was trained on the **MAFW (Multi-modal Affective dataset in the Wild)** dataset:
- 10,045 video clips with 11 emotion labels
- Physiological signals (EEG, GSR, PPG) from 45 subjects
- Single-label and multi-label annotations

Additional fine-tuning used English audio emotion datasets:
- **CREMA-D** — Crowd-sourced Emotional Multimodal Actors Dataset
- **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- **MELD** — Multimodal EmotionLines Dataset (from Friends TV series)

## Outputs

For each detected person/audio stream, the system produces:

| Output | Description |
|--------|-------------|
| Primary Emotion | Top-1 predicted emotion (11 classes) |
| Emotion Distribution | Probability across all 11 emotions |
| Stress Score | 0.0–1.0 continuous stress level |
| Valence | Positive/negative emotional polarity |
| Arousal | Low/high emotional activation |

## Tech Stack

- **PyTorch** — Deep learning framework
- **wav2vec2** (HuggingFace) — Pre-trained audio feature extractor
- **FastAPI** + **WebSocket** — Real-time web dashboard
- **DeepFace** — Face emotion detection
- **librosa** — Audio feature extraction
- **yt-dlp** — YouTube video downloading
- **FFmpeg** — Audio/video stream processing

## License

This project is part of ongoing PhD research. Please contact the authors before using in production or academic publications.

## Citation

If you use this work, please cite:

```
@misc{mstdn2025,
  title={MSTDN: Multimodal Spatio-Temporal Distillation Network for Affective State Inference},
  author={Bineetha},
  year={2025}
}
```
