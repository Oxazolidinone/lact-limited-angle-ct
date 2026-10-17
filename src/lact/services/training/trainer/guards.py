"""Helpers of the training loop: divergence guard, GPU out-of-memory detection, memory and epoch-progress tracking."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_mem_mb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def _jobs_per_gpu() -> int:
    try:
        return max(1, int(os.environ.get("LACT_JOBS_PER_GPU", "1")))
    except ValueError:
        return 1


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@dataclass
class DivergenceGuard:
    """Diverged when the loss is not finite or > factor x the mean of the first 50 finite losses (after >= 10)."""
    factor: float = 100.0
    ref_losses: list[float] = field(default_factory=list)

    def check(self, loss: float) -> tuple[bool, float]:
        if len(self.ref_losses) < 50 and math.isfinite(loss):
            self.ref_losses.append(loss)
        ref = float(np.mean(self.ref_losses)) if self.ref_losses else loss
        diverged = not math.isfinite(loss) or (len(self.ref_losses) >= 10 and loss > self.factor * ref)
        return diverged, ref


@dataclass
class EpochProgress:
    """State of the running epoch; saved with mid-epoch checkpoints."""
    epoch: int = 0
    batch: int = 0              # batches completed in this epoch
    loss_sum: float = 0.0
    loss_count: int = 0

    @property
    def mean_loss(self) -> float:
        return self.loss_sum / self.loss_count if self.loss_count else float("nan")
