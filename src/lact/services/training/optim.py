"""Optimizers and per-step learning-rate schedules: linear warmup then cosine, constant, or exponential decay."""
from __future__ import annotations

import math

import torch

from ...config import TrainConfig


def build_optimizer(model: torch.nn.Module, t: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    lr, wd = float(t.lr), float(t.weight_decay)
    betas = (float(t.betas[0]), float(t.betas[1]))
    name = t.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(t.momentum), weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {t.optimizer}")


def lr_lambda_factory(total_steps: int, warmup_steps: int, lr: float, lr_min: float, schedule: str = "cosine"):
    ratio = lr_min / lr if lr > 0 else 0.0

    def f(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "constant":
            return 1.0
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        t = min(max(t, 0.0), 1.0)
        if schedule == "exp":           # uniform decay on a log scale from lr to lr_min (FBPConvNet: 1e-2 -> 1e-3)
            return ratio ** t
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    return f


def build_scheduler(opt: torch.optim.Optimizer, t: TrainConfig, steps_per_epoch: int, total_steps: int):
    warm = round(float(t.warmup_epochs) * steps_per_epoch)
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, warm, float(t.lr), float(t.lr_min),
                                                                    t.schedule))
