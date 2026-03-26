from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from models.student.student_model import MSTDNAStudent
from models.teacher.caption_encoder import FrozenCaptionEncoder
from training.common import load_config, resolve_device, set_seed
from training.losses import multi_task_loss

CHECKPOINT_DIR = Path("checkpoints")
RESUME_PATH    = CHECKPOINT_DIR / "stage4_resume.pt"
FINAL_PATH     = CHECKPOINT_DIR / "stage4_caption_refined.pt"


def _clean(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["training"]["device"])
    ds_cfg = DatasetConfig(
        base_dir=cfg["project"]["base_dir"],
        split_variant="single/with_caption",
        fold="set_1",
        use_physio=False,
        use_caption=True,
    )
    loader = build_dataloader(ds_cfg, "train", batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"])

    model = MSTDNAStudent().to(device)
    prior = CHECKPOINT_DIR / "stage3_student_distilled.pt"
    if prior.exists():
        model.load_state_dict(torch.load(prior, map_location=device))
        print(f"Loaded student from {prior}")
    else:
        print("WARNING: stage3 checkpoint not found — student is random init")

    caption_encoder = FrozenCaptionEncoder().to(device)
    projector       = nn.Linear(256, 256).to(device)
    optimizer       = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    total_epochs = cfg["training"]["epochs"]
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    start_epoch = 0

    if RESUME_PATH.exists():
        ckpt = torch.load(RESUME_PATH, map_location=device)
        if all(torch.isfinite(p).all() for p in ckpt["model"].values()):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming stage4 from epoch {start_epoch}/{total_epochs}")
        else:
            print("Stage4 checkpoint has NaN — starting fresh.")
    else:
        print("Starting stage4 fresh.")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        totals: dict[str, float] = {}
        skipped = 0
        for batch in tqdm(loader, desc=f"stage4 epoch {epoch + 1}"):
            optimizer.zero_grad()
            batch_t = {k: (_clean(v.to(device)) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs     = model(batch_t["waveform"], batch_t["spectrogram"])
            losses      = multi_task_loss(outputs, batch_t)
            audio_emb   = F.normalize(projector(outputs["embedding"]), dim=-1)
            text_emb    = F.normalize(caption_encoder(batch["caption_en"]).to(device), dim=-1)
            logits      = audio_emb @ text_emb.T / 0.07
            labels      = torch.arange(logits.size(0), device=device)
            caption_loss = F.cross_entropy(logits, labels)
            loss        = losses["loss"] + cfg["loss"]["caption_refine"] * caption_loss
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), max_norm=1.0)
            optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item())

        scheduler.step()
        n   = max(len(loader) - skipped, 1)
        avg = totals.get("loss", 0.0) / n
        print(f"Epoch {epoch + 1}/{total_epochs}  loss={avg:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        if skipped:
            print(f"  [warn] skipped {skipped} NaN batches")

        if all(torch.isfinite(p).all() for p in model.state_dict().values()):
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, RESUME_PATH)
        else:
            print(f"  [warn] NaN weights at epoch {epoch + 1} — checkpoint NOT overwritten")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Stage 4 complete. Saved to {FINAL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4_caption.yaml")
    run(parser.parse_args().config)
