"""
finetune_english_r2.py
=======================
Round 2: load english_finetune.pt, fix class imbalance with weighted CE.

Changes from Round 1
--------------------
- Loads english_finetune.pt (Round 1 best checkpoint)
- Computes class weights from train.csv  (rare emotions count more)
- LR lowered to 1e-5
- Label smoothing reduced to 0.05
- 20 epochs, cosine annealing, 1 warmup epoch
- Auto-resumes from english_finetune_r2_resume.pt
- Saves best to english_finetune_r2.pt

Run
---
    cd MSTDN_A
    py -3.11 training/finetune_english_r2.py
"""
from __future__ import annotations

import csv, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.student.student_model import MSTDNAStudent

# -- paths --------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent.parent
ENG_DIR     = BASE_DIR / "english_datasets"
CKPT_DIR    = Path(__file__).parent.parent / "checkpoints"
LOAD_FROM   = CKPT_DIR / "english_finetune.pt"       # Round 1 best
SAVE_TO     = CKPT_DIR / "english_finetune_r2.pt"
RESUME_PATH = CKPT_DIR / "english_finetune_r2_resume.pt"
STATUS_FILE = BASE_DIR / "finetune_status.json"

# -- hyperparams --------------------------------------------------------------
SAMPLE_RATE   = 16_000
MAX_SAMPLES   = int(SAMPLE_RATE * 4.0)
N_MELS        = 80
EPOCHS        = 20
WARMUP_EPOCHS = 1
BATCH_SIZE    = 16             # doubled for speed (halves steps/epoch)
LR            = 1e-5          # lower than Round 1
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
LABEL_SMOOTH  = 0.05          # lighter smoothing (weights handle balance)
NUM_WORKERS   = 2             # parallel data loading
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
STATUS_EVERY  = 50
NUM_CLASSES   = 11            # MAFW output size (English uses 0-6)


# -- class weights ------------------------------------------------------------
def compute_class_weights(train_csv: Path, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights for classes 0-6; 1.0 for unused 7-10."""
    counts = [0] * NUM_CLASSES
    with open(train_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["emotion_idx"])
            if idx < NUM_CLASSES:
                counts[idx] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (NUM_CLASSES * c))
        else:
            weights.append(1.0)   # unused class -- neutral weight

    w = torch.tensor(weights, dtype=torch.float32).to(device)
    print("Class weights:")
    labels = ["anger","disgust","fear","happiness","neutral","sadness","surprise",
              "c7","c8","c9","c10"]
    for i, (lbl, wt) in enumerate(zip(labels, w.tolist())):
        print(f"  [{i}] {lbl:<12} count={counts[i]:>5}  weight={wt:.3f}")
    return w


# -- status writer ------------------------------------------------------------
def write_status(
    epoch, total_epochs, batch, total_batches, phase,
    loss, train_acc, val_acc, best_val_acc, lr, epoch_start, run_start,
):
    now = time.time()
    elapsed_total = now - run_start
    elapsed_epoch = now - epoch_start

    if batch > 0 and phase == "train":
        secs_per_batch = elapsed_epoch / batch
        batches_left = (total_epochs - epoch) * total_batches - batch
        eta_secs = int(secs_per_batch * batches_left)
    else:
        eta_secs = None

    def fmt(s):
        if s is None: return "--"
        h, r = divmod(int(s), 3600)
        m, s = divmod(r, 60)
        return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

    status = {
        "phase":            phase,
        "epoch":            epoch + 1,
        "total_epochs":     total_epochs,
        "batch":            batch,
        "total_batches":    total_batches,
        "loss":             round(loss, 4) if loss is not None else None,
        "train_acc_pct":    round(train_acc * 100, 1),
        "val_acc_pct":      round(val_acc * 100, 1),
        "best_val_acc_pct": round(best_val_acc * 100, 1),
        "lr":               f"{lr:.2e}",
        "elapsed":          fmt(elapsed_total),
        "elapsed_epoch":    fmt(elapsed_epoch),
        "eta":              fmt(eta_secs),
        "eta_seconds":      eta_secs,
        "updated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "device":           DEVICE,
        "status":           "training" if phase != "done" else "done",
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception:
        pass


# -- dataset ------------------------------------------------------------------
class EnglishEmoDataset(Dataset):
    def __init__(self, split: str):
        self.base = ENG_DIR
        self.rows: list[dict] = []
        with open(ENG_DIR / "labels" / f"{split}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        try:
            wav, _ = librosa.load(str(self.base / row["file"]), sr=SAMPLE_RATE, mono=True)
        except Exception:
            return None

        if len(wav) < MAX_SAMPLES:
            wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
        else:
            wav = wav[:MAX_SAMPLES]

        rms = np.sqrt(np.mean(wav ** 2)) + 1e-8
        wav = np.clip(wav / max(rms, 0.01), -1.0, 1.0)
        spec = np.log1p(
            librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=N_MELS)
        ).astype(np.float32)

        return {
            "waveform":    torch.tensor(wav,  dtype=torch.float32),
            "spectrogram": torch.tensor(spec, dtype=torch.float32),
            "label":       torch.tensor(int(row["emotion_idx"]), dtype=torch.long),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# -- freeze strategy (same as Round 1) ----------------------------------------
def configure_trainable(model: MSTDNAStudent):
    for p in model.parameters():
        p.requires_grad = False
    dab = model.deep_audio_branch
    if dab.enabled:
        for layer in dab.model.encoder.layers[-4:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in dab.proj.parameters(): p.requires_grad = True
    for p in model.fusion.parameters(): p.requires_grad = True
    for p in model.heads.parameters():  p.requires_grad = True


# -- main ---------------------------------------------------------------------
def train():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    train_ds = EnglishEmoDataset("train")
    val_ds   = EnglishEmoDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
                              pin_memory=(DEVICE == "cuda"), persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(DEVICE == "cuda"), persistent_workers=(NUM_WORKERS > 0))

    total_batches = len(train_loader)
    print(f"Train: {len(train_ds)} clips ({total_batches} batches)  |  Val: {len(val_ds)} clips")

    # class weights
    class_weights = compute_class_weights(ENG_DIR / "labels" / "train.csv", device)

    model = MSTDNAStudent().to(device)
    if LOAD_FROM.exists():
        model.load_state_dict(torch.load(LOAD_FROM, map_location=device), strict=False)
        print(f"Loaded weights: {LOAD_FROM.name}")
    else:
        print(f"WARNING: {LOAD_FROM} not found — starting from scratch")

    configure_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    start_epoch  = 0
    best_val_acc = 0.0
    val_acc      = 0.0

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        val_acc      = best_val_acc
        print(f"Resumed from epoch {start_epoch}/{EPOCHS}  best_val_acc={best_val_acc:.3f}")

    run_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        # warmup LR
        if epoch < WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * (epoch + 1) / WARMUP_EPOCHS

        # -- train ------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_correct = train_total = skipped = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")):
            if batch is None:
                skipped += 1
                continue

            wav    = batch["waveform"].to(device)
            spec   = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out  = model(wav, spec)
                loss = F.cross_entropy(
                    out["primary_logits"], labels,
                    weight=class_weights,
                    label_smoothing=LABEL_SMOOTH,
                )

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item()
            preds          = out["primary_logits"].argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            if (i + 1) % STATUS_EVERY == 0:
                avg_l = train_loss / max(i + 1 - skipped, 1)
                t_acc = train_correct / max(train_total, 1)
                write_status(epoch, EPOCHS, i + 1, total_batches,
                             "train", avg_l, t_acc, val_acc, best_val_acc,
                             current_lr, epoch_start, run_start)

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        n_batches = max(total_batches - skipped, 1)
        avg_loss  = train_loss / n_batches
        train_acc = train_correct / max(train_total, 1)

        # -- val --------------------------------------------------------------
        write_status(epoch, EPOCHS, total_batches, total_batches,
                     "validating", avg_loss, train_acc, val_acc, best_val_acc,
                     current_lr, epoch_start, run_start)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "):
                if batch is None: continue
                wav    = batch["waveform"].to(device)
                spec   = batch["spectrogram"].to(device)
                labels = batch["label"].to(device)
                out    = model(wav, spec)
                preds  = out["primary_logits"].argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        print(f"  loss={avg_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}  lr={current_lr:.2e}")
        if skipped:
            print(f"  [warn] skipped {skipped} NaN batches")

        # -- checkpoint -------------------------------------------------------
        if all(torch.isfinite(p).all() for p in model.state_dict().values()):
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": max(best_val_acc, val_acc),
            }, RESUME_PATH)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), SAVE_TO)
                print(f"  ** Best val_acc={best_val_acc:.3f} -- saved {SAVE_TO.name}")
        else:
            print("  [warn] NaN weights -- checkpoint NOT written")

        write_status(epoch, EPOCHS, total_batches, total_batches,
                     "train", avg_loss, train_acc, val_acc, best_val_acc,
                     current_lr, epoch_start, run_start)

    # done
    write_status(EPOCHS - 1, EPOCHS, total_batches, total_batches,
                 "done", 0.0, train_acc, val_acc, best_val_acc,
                 current_lr, time.time(), run_start)
    print(f"\nDone.  Best val_acc={best_val_acc:.3f}  Saved: {SAVE_TO}")


if __name__ == "__main__":
    train()
