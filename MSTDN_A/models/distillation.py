from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_alignment(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if teacher.shape[-1] != student.shape[-1]:
        teacher = teacher[..., : student.shape[-1]]
    return F.mse_loss(student, teacher)


def relational_kd(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_dist = torch.cdist(student, student, p=2)
    teacher_dist = torch.cdist(teacher, teacher, p=2)
    return F.mse_loss(student_dist, teacher_dist)


def info_nce_alignment(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    logits = student @ teacher.T / temperature
    targets = torch.arange(student.size(0), device=student.device)
    return F.cross_entropy(logits, targets)
