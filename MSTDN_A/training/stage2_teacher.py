from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from models.teacher.teacher_model import MSTDNATeacher
from training.common import load_config, resolve_device, set_seed
from training.losses import multi_task_loss

CHECKPOINT_DIR = Path("checkpoints")
RESUME_PATH    = CHECKPOINT_DIR / "stage2_resume.pt"
FINAL_PATH     = CHECKPOINT_DIR / "stage2_teacher.pt"


def _clean(t: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf in any input tensor with 0."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def train_epoch(model: MSTDNATeacher, loader, optimizer, device: torch.device, epoch: int) -> dict[str, float]:
    model.train()
    totals:   dict[str, float] = {}
    skipped = 0
    for batch in tqdm(loader, desc=f"stage2 epoch {epoch}"):
        optimizer.zero_grad()

        # Sanitize all tensor inputs to prevent NaN propagation
        eeg      = _clean(batch["eeg"].to(device))
        gsr      = _clean(batch["gsr"].to(device))
        ppg      = _clean(batch["ppg"].to(device))
        waveform = _clean(batch["waveform"].to(device))

        outputs = model(eeg, gsr, ppg, waveform, batch["caption_en"])
        loss_dict = multi_task_loss(
            outputs,
            {k: (_clean(v.to(device)) if torch.is_tensor(v) else v) for k, v in batch.items()},
        )

        # Skip batch if loss is NaN or Inf — don't corrupt weights
        if not torch.isfinite(loss_dict["loss"]):
            skipped += 1
            continue

        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())

    n = max(len(loader) - skipped, 1)
    if skipped:
        print(f"  [warn] skipped {skipped} NaN batches this epoch")
    return {key: value / n for key, value in totals.items()}


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
    loader = build_dataloader(
        ds_cfg, "train",
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    model     = MSTDNATeacher().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=1e-6,
    )

    start_epoch  = 0
    total_epochs = cfg["training"]["epochs"]
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Resume only if checkpoint has valid (non-NaN) weights
    if RESUME_PATH.exists():
        ckpt     = torch.load(RESUME_PATH, map_location=device)
        params   = list(ckpt["model"].values())
        is_valid = all(torch.isfinite(p).all() for p in params)
        if is_valid:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from {RESUME_PATH} — epoch {start_epoch}/{total_epochs}")
        else:
            print("Checkpoint has NaN weights — starting fresh.")
    else:
        print("Starting fresh training.")

    for epoch in range(start_epoch, total_epochs):
        metrics  = train_epoch(model, loader, optimizer, device, epoch + 1)
        scheduler.step()
        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        lr_now   = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{total_epochs}  {loss_str}  lr={lr_now:.2e}")

        # Only save checkpoint if weights are still healthy
        params    = list(model.state_dict().values())
        is_finite = all(torch.isfinite(p).all() for p in params)
        if is_finite:
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, RESUME_PATH)
        else:
            print(f"  [warn] NaN weights at epoch {epoch + 1} — checkpoint NOT overwritten")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Training complete. Final model saved to {FINAL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_teacher.yaml")
    run(parser.parse_args().config)
