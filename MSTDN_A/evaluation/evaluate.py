from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from data.dataset import DatasetConfig
from data.loaders import build_dataloader
from evaluation.metrics import classification_metrics, multilabel_metrics, regression_metrics
from models.student.student_model import MSTDNAStudent


def run(checkpoint_path: str, base_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = DatasetConfig(base_dir=base_dir, split_variant="single/no_caption", fold="set_5", use_physio=False, use_caption=False)
    loader = build_dataloader(ds_cfg, "test", batch_size=8, num_workers=0)
    model = MSTDNAStudent().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_multi_true: list[np.ndarray] = []
    y_multi_prob: list[np.ndarray] = []
    stress_true: list[float] = []
    stress_pred: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            outputs = model(batch["waveform"].to(device), batch["spectrogram"].to(device))
            y_true.extend(batch["primary_label"].numpy().tolist())
            y_pred.extend(outputs["primary_logits"].argmax(dim=-1).cpu().numpy().tolist())
            y_multi_true.extend(batch["secondary_labels"].numpy())
            y_multi_prob.extend(torch.sigmoid(outputs["secondary_logits"]).cpu().numpy())
            stress_true.extend(batch["stress_score"].numpy().tolist())
            stress_pred.extend(outputs["stress_score"].squeeze(-1).cpu().numpy().tolist())
    report = {
        "classification": classification_metrics(np.asarray(y_true), np.asarray(y_pred)),
        "multilabel": multilabel_metrics(np.asarray(y_multi_true), np.asarray(y_multi_prob)),
        "stress": regression_metrics(np.asarray(stress_true), np.asarray(stress_pred)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/stage5_final.pt")
    parser.add_argument("--base-dir", default="C:/Users/ADMIN/Desktop/Bineetha - emoDet")
    args = parser.parse_args()
    run(args.checkpoint, args.base_dir)
