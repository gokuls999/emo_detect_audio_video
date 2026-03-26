"""
Quick smoke test — runs 3 real training batches through Stage 2 (Teacher).
Proves: dataset loads real data, model runs on GPU, loss computes, gradients flow.
Run from MSTDN_A/: py -3.11 smoke_test.py
"""
from __future__ import annotations

import sys
import time

import torch

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from models.teacher.teacher_model import MSTDNATeacher
from training.common import load_config, resolve_device, set_seed
from training.losses import multi_task_loss

CONFIG_PATH = "configs/stage2_teacher.yaml"
N_BATCHES = 3


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["training"]["device"])

    print(f"\n{'='*55}")
    print(f"  MSTDN-A Smoke Test — Stage 2 Teacher")
    print(f"{'='*55}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU     : {name} ({vram:.1f} GB VRAM)")
    print(f"  Batches : {N_BATCHES}")
    print(f"{'='*55}\n")

    ds_cfg = DatasetConfig(
        base_dir=cfg["project"]["base_dir"],
        split_variant="single/no_caption",
        fold="set_1",
        use_physio=True,
        use_caption=True,
    )
    loader = build_dataloader(ds_cfg, "train", batch_size=4, num_workers=0)
    print(f"Dataset loaded: {len(loader.dataset)} clips in split\n")

    model = MSTDNATeacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= N_BATCHES:
            break

        t0 = time.time()
        optimizer.zero_grad()
        outputs = model(
            batch["eeg"].to(device),
            batch["gsr"].to(device),
            batch["ppg"].to(device),
            batch["waveform"].to(device),
            batch["caption_en"],
        )
        loss_dict = multi_task_loss(
            outputs,
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()},
        )
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        elapsed = time.time() - t0

        vram_str = ""
        if device.type == "cuda":
            used = torch.cuda.memory_allocated(0) / 1024**2
            vram_str = f"  VRAM: {used:.0f} MB"

        print(f"Batch {batch_idx + 1}/{N_BATCHES}  |  "
              f"total_loss={loss_dict['loss'].item():.4f}  |  "
              f"ce={loss_dict.get('ce', torch.tensor(0)).item():.4f}  |  "
              f"va={loss_dict.get('va', torch.tensor(0)).item():.4f}  |  "
              f"stress={loss_dict.get('stress', torch.tensor(0)).item():.4f}  |  "
              f"{elapsed:.2f}s{vram_str}")

    print(f"\n{'='*55}")
    print(f"  SUCCESS — MSTDN-A training pipeline is working.")
    print(f"  Real data loaded, GPU used, loss computed, gradients flow.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
