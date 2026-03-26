from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_task_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    ce = F.cross_entropy(outputs["primary_logits"], batch["primary_label"])
    bce = F.binary_cross_entropy_with_logits(outputs["secondary_logits"], batch["secondary_labels"])
    distribution = torch.softmax(outputs["distribution_logits"], dim=-1)
    target_distribution = batch["emotion_distribution"]
    dist = F.mse_loss(distribution, target_distribution)
    valence = F.mse_loss(outputs["valence"].squeeze(-1), batch["valence"])
    arousal = F.mse_loss(outputs["arousal"].squeeze(-1), batch["arousal"])
    stress = F.mse_loss(outputs["stress_score"].squeeze(-1), batch["stress_score"])
    total = ce + 0.5 * bce + 0.5 * (valence + arousal) + 0.5 * stress + 0.25 * dist
    return {
        "loss": total,
        "ce": ce,
        "bce": bce,
        "dist": dist,
        "valence": valence,
        "arousal": arousal,
        "stress": stress,
    }
