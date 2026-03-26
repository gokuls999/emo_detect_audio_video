from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from models.distillation import info_nce_alignment, mse_alignment, relational_kd
from models.student.student_model import MSTDNAStudent
from models.teacher.teacher_model import MSTDNATeacher
from training.common import load_config, resolve_device, set_seed
from training.losses import multi_task_loss

CHECKPOINT_DIR = Path("checkpoints")
RESUME_PATH    = CHECKPOINT_DIR / "stage3_resume.pt"
FINAL_PATH     = CHECKPOINT_DIR / "stage3_student_distilled.pt"


def _clean(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["training"]["device"])
    ds_cfg = DatasetConfig(
        base_dir=cfg["project"]["base_dir"],
        split_variant="single/no_caption",
        fold="set_1",
        use_physio=True,
        use_caption=True,
    )
    loader = build_dataloader(ds_cfg, "train", batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"])

    teacher = MSTDNATeacher().to(device)
    teacher_path = CHECKPOINT_DIR / "stage2_teacher.pt"
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print(f"Loaded teacher from {teacher_path}")
    else:
        print("WARNING: stage2_teacher.pt not found — teacher is random init")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student   = MSTDNAStudent().to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"].get("weight_decay", 1e-4))
    alpha     = cfg["loss"]["alpha"]
    beta      = cfg["loss"]["beta"]
    gamma     = cfg["loss"]["gamma"]
    total_epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    start_epoch = 0

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        if all(torch.isfinite(p).all() for p in ckpt["model"].values()):
            student.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming stage3 from epoch {start_epoch}/{total_epochs}")
        else:
            print("Stage3 checkpoint has NaN — starting fresh.")
    else:
        print("Starting stage3 fresh.")

    for epoch in range(start_epoch, total_epochs):
        student.train()
        totals: dict[str, float] = {}
        skipped = 0
        for batch in tqdm(loader, desc=f"stage3 epoch {epoch + 1}"):
            optimizer.zero_grad()
            batch_t = {k: (_clean(v.to(device)) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                teacher_outputs = teacher(batch_t["eeg"], batch_t["gsr"], batch_t["ppg"], batch_t["waveform"], batch["caption_en"])
            student_outputs = student(batch_t["waveform"], batch_t["spectrogram"])
            task  = multi_task_loss(student_outputs, batch_t)["loss"]
            align = mse_alignment(student_outputs["embedding"], teacher_outputs["embedding"])
            rkd   = relational_kd(student_outputs["embedding"], teacher_outputs["embedding"][:, : student_outputs["embedding"].shape[-1]])
            nce   = info_nce_alignment(student_outputs["embedding"], teacher_outputs["embedding"][:, : student_outputs["embedding"].shape[-1]])
            loss  = task + alpha * align + beta * rkd + gamma * nce
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item())

        scheduler.step()
        n = max(len(loader) - skipped, 1)
        avg = totals.get("loss", 0.0) / n
        print(f"Epoch {epoch + 1}/{total_epochs}  loss={avg:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        if skipped:
            print(f"  [warn] skipped {skipped} NaN batches")

        if all(torch.isfinite(p).all() for p in student.state_dict().values()):
            torch.save({"epoch": epoch, "model": student.state_dict(), "optimizer": optimizer.state_dict()}, RESUME_PATH)
        else:
            print(f"  [warn] NaN weights at epoch {epoch + 1} — checkpoint NOT overwritten")

    torch.save(student.state_dict(), FINAL_PATH)
    print(f"Stage 3 complete. Saved to {FINAL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage3_distill.yaml")
    run(parser.parse_args().config)
