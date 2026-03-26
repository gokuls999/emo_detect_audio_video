# MSTDN-A

MSTDN-A is a PhD-oriented research implementation for physiologically grounded speech emotion and stress estimation using multimodal teacher training and audio-only student inference.

## What is included

- Data parsing for labels, captions, physio sessions, and split files
- Audio-physio dataset and dataloaders
- Teacher model with EEG, GSR, PPG, audio, and caption branches
- Audio-only student model with prosodic, spectral, and deep audio branches
- Distillation losses and five training stage entrypoints
- Evaluation metrics, evaluation runner, and ablation scaffold
- Realtime inference, VAD, diarization wrapper, and group analytics scaffold

## Project layout

- `configs/`: stage configs
- `data/`: datasets, sync logic, augmentations
- `models/`: teacher, student, heads, and distillation
- `training/`: stage entrypoints and losses
- `evaluation/`: metrics and experiment runners
- `realtime/`: live inference pipeline components
- `utils/`: data readers and parsers

## Install

```powershell
py -3 -m pip install -r requirements.txt
```

## Run training stages

```powershell
python training/stage2_teacher.py --config configs/stage2_teacher.yaml
python training/stage3_distill.py --config configs/stage3_distill.yaml
python training/stage4_caption.py --config configs/stage4_caption.yaml
python training/stage5_augment.py --config configs/stage5_augment.yaml
```

## Evaluate

```powershell
python evaluation/evaluate.py --checkpoint checkpoints/stage5_final.pt --base-dir "C:/Users/ADMIN/Desktop/Bineetha - emoDet"
```

## Notes

- Heavy external models such as CLIP, wav2vec2, PyAnnote, and Silero VAD are wrapped with fallbacks so the codebase can still import in a partially provisioned environment.
- The code is structured to match the provided build prompt closely while staying practical for incremental experimentation.
