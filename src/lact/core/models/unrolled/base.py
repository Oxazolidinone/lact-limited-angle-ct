"""Common interface of the unrolled reconstruction models used by training and evaluation."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class Unrolled(nn.Module):
    """Common interface of the unrolled models (LEARN with the update rules below, RegFormer) used by training and
    evaluation: ``forward(y, x0, return_stages=False, return_terms=False)`` plus the methods below."""

    K: int
    grad_checkpoint: bool
    projector: Any = None          # SparseFanBeam; not a submodule, so never part of the state_dict

    @property
    def uses_cg(self) -> bool:
        """Whether each stage solves a linear system with CG (``last_cg_iters``, ``cg_iters``, ``cg_tol`` apply)."""
        return False

    def cg_iters_mean(self) -> float:
        """Mean CG iterations per stage in the latest forward pass (0 when the model does not use CG)."""
        return 0.0

    def set_projector(self, projector):
        self.projector = projector
        return self

    def stage_params(self) -> dict[str, torch.Tensor]:
        """Learned per-stage scalar parameters by name (for logging and plots)."""
        raise NotImplementedError

    def regularizer_params(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _norm(t: torch.Tensor) -> float:
        return t.flatten(1).norm(dim=1).mean().item()


def val_forward(model: nn.Module):
    """Inference function used for validation during training (and when re-checking validation PSNR). Models with
    an expensive non-learned inference step (DPMA: 300 refinement iterations) define ``validation_forward`` so that
    checkpoint selection runs only the learned part; other models use ``forward`` itself. Likewise, models with their
    own loss define ``training_loss(y, x0, x)``, which the trainer uses instead of the output MSE."""
    return getattr(model, "validation_forward", model)
