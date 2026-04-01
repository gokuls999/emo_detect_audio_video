# MSTDN-A — Build & Deployment Guide

## What Gets Packaged

### Included (both GPU & CPU)
| Component | Path in ZIP | Size |
|-----------|------------|------|
| Model code | `models/` | 217 KB |
| Dashboard server | `dashboard/server.py, database.py, face_id.py, face_emotion.py, capture.py, pdf_report.py` | ~240 KB |
| Dashboard UI | `dashboard/static/index.html, yt_test.html` | ~140 KB |
| FERPlus ONNX | `dashboard/models/emotion-ferplus-8.onnx` | 33 MB |
| InsightFace buffalo_l | `insightface_models/buffalo_l/` (5 ONNX files) | 326 MB |
| wav2vec2-base cache | `hf_cache/models--facebook--wav2vec2-base/` | 726 MB |
| Base checkpoint | `checkpoints/english_finetune_r2.pt` | 377 MB |
| Online-tuned checkpoint | `checkpoints/online_tuned.pt` | 377 MB |
| Setup script | `setup.bat` | auto |
| Run script | `run.bat` | auto |
| Launcher | `MSTDN-A.bat` (double-click to run) | auto |
| Requirements | `requirements-gpu.txt` or `requirements-cpu.txt` | auto |

**Total ZIP size: ~1.8 GB (compressed ~1.2 GB)**

### Excluded
- Training datasets (CREMA-D, MELD, RAVDESS, english_datasets, video_clips, emotion_labels, etc.)
- Physiological data (EEG/GSR/PPG)
- Training scripts and logs
- Old session reports (*.pdf, *.xlsx in root)
- Stage checkpoints (stage2_*, stage3_*, stage4_*, stage5_*, *_resume.pt)
- `.venv/`, `__pycache__/`, `.git/`
- `dashboard.db` (recreated on first run)
- `local_videos/`, `pdf_reports/`, `dash_pdf_reports/` (user-generated, recreated)

## GPU vs CPU Differences

| Feature | GPU Version | CPU Version |
|---------|-------------|-------------|
| PyTorch | `torch` + CUDA | `torch` (CPU-only, smaller) |
| ONNX Runtime | `onnxruntime-gpu` | `onnxruntime` |
| InsightFace provider | CUDAExecutionProvider | CPUExecutionProvider |
| Inference speed | ~100ms per cycle | ~500-1000ms per cycle |
| RAM needed | 4 GB + GPU VRAM | 4 GB minimum |
| Works on | Any NVIDIA GPU (GTX 1050+) | Any CPU |

## Target System Requirements
- **OS:** Windows 10/11
- **Python:** 3.10 or 3.11 (NOT 3.14 — many packages don't support it yet)
- **GPU version:** Any NVIDIA GPU with CUDA support + CUDA 12.x drivers
- **CPU version:** 4 GB RAM minimum
- **Disk:** ~4 GB free space after extraction

## How the Installer Works
1. User extracts ZIP
2. Double-clicks `setup.bat` → installs Python packages
3. Double-clicks `MSTDN-A.bat` → starts server + opens browser
4. Dashboard at http://localhost:8000

## Updating
To push an update to a deployed version:
1. Replace changed files (usually `dashboard/server.py`, `dashboard/static/*.html`)
2. If model code changed: replace `models/` folder
3. If checkpoint changed: replace `checkpoints/online_tuned.pt`
4. Re-run `setup.bat` only if new Python packages are needed
5. Restart via `MSTDN-A.bat`

## Build Commands
```bash
# From MSTDN_A/ directory:
python build_package.py --variant gpu    # Creates MSTDN-A_GPU.zip
python build_package.py --variant cpu    # Creates MSTDN-A_CPU.zip
```
