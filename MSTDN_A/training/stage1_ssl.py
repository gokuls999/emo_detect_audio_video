from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from training.common import load_config, resolve_device, set_seed


class AudioSSLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=2, padding=5),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.GELU(),
        )
        self.decoder = nn.ConvTranspose1d(128, 1, kernel_size=8, stride=4, padding=2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(waveform.unsqueeze(1))
        return self.decoder(latent).squeeze(1)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["training"]["device"])
    ds_cfg = DatasetConfig(base_dir=cfg["project"]["base_dir"], use_physio=True)
    loader = build_dataloader(ds_cfg, "train", batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["num_workers"])
    model = AudioSSLModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    for _ in range(cfg["training"]["epochs"]):
        for batch in tqdm(loader, desc="stage1_audio_ssl"):
            waveform = batch["waveform"].to(device)
            mask = torch.rand_like(waveform) > 0.5
            masked = waveform * mask
            recon = model(masked)
            loss = F.l1_loss(recon[..., : waveform.shape[-1]], waveform)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    out = Path("checkpoints") / "stage1_audio_ssl.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_ssl.yaml")
    run(parser.parse_args().config)
