"""Deep-unrolling framework based on LEARN (Chen et al., 2018), with two update rules for each stage
k = 0..K-1 (x^0 = FBP(y)).

``update = "gd"`` - LEARN, first order:

    x^{k+1} = x^k - alpha_k * A^T (A x^k - y) - beta_k * R_theta(x^k)

``update = "lbfgs"`` - quasi-Newton on the data gradient. The first stage has no memory pairs and equals ``gd``;
afterwards:

    g       = A^T (A x^k - y)
    d       = H_k g                         H_k approximates (A^T A)^{-1} by L-BFGS (H_0 = I) over earlier stages
    x^{k+1} = x^k - alpha_k * d - beta_k * R_theta(x^k)

  The regularizer and beta share their names with ``gd``, so first-order checkpoints load; only the alpha_k step has
  its own name (``a_qn``) and restarts at ``alpha0``: the learned ``gd`` alpha reaches 2-4.6 in the last stages and
  overshoots when multiplied by the Newton direction (tried on the 256 grid: PSNR collapses to 3.5 dB at once; with
  alpha = 1 it stays at 24-25 dB). The (s, y) pairs are treated as constants in backpropagation (like QN-Mixer's
  BFGS update): the gradient flows through g but not through 1 / s^T y.

- A and y are normalised (||A|| = 1) inside the model; the dataset returns the raw y.
- alpha_k, beta_k > 0 through softplus, learned per stage.
- ``share_theta=True``: one regularizer shared by all stages (the thesis setting);
  ``False``: K separate regularizers (as in the original LEARN / reference [2]).
- The projector is NOT a submodule, so it is not saved in the state_dict; set it with ``set_projector``.
- With ``update = "gd"`` the parameter names (``a``, ``b``, ``reg``) are unchanged, so older checkpoints still load.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...solvers.lbfgs import lbfgs_direction
from .base import Unrolled

UPDATES = ("gd", "lbfgs")


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


class UnrolledLEARN(Unrolled):
    def __init__(self, make_regularizer: Callable[[], nn.Module], K: int = 14, share_theta: bool = True,
                 alpha0: float = 1.0, beta0: float = 1.0, grad_checkpoint: bool = False, projector=None,
                 update: str = "gd", lbfgs_memory: int = 13, lbfgs_damping: float = 0.0, lbfgs_curv_min: float = 0.0):
        super().__init__()
        if update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}, got {update!r}")
        if update == "lbfgs" and int(lbfgs_memory) < 1:
            raise ValueError("lbfgs_memory must be >= 1")
        self.K = int(K)
        self.share_theta = bool(share_theta)
        self.update = update
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_damping, self.lbfgs_curv_min = float(lbfgs_damping), float(lbfgs_curv_min)
        if self.share_theta:
            self.reg = make_regularizer()
        else:
            self.regs = nn.ModuleList(make_regularizer() for _ in range(self.K))
        if update == "gd":
            self.a = nn.Parameter(torch.full((self.K,), _inv_softplus(alpha0)))
        else:
            self.a_qn = nn.Parameter(torch.full((self.K,), _inv_softplus(alpha0)))
        self.b = nn.Parameter(torch.full((self.K,), _inv_softplus(beta0)))
        self.grad_checkpoint = bool(grad_checkpoint)
        self.projector = projector

    def regularizer(self, k: int) -> nn.Module:
        return self.reg if self.share_theta else self.regs[k]

    # ------------------------------------------------------------------ per-stage parameters
    @property
    def alphas(self) -> torch.Tensor:
        return F.softplus(self.a if self.update == "gd" else self.a_qn)

    @property
    def betas(self) -> torch.Tensor:
        return F.softplus(self.b)

    def stage_params(self) -> dict[str, torch.Tensor]:
        """Learned per-stage scalar parameters by name (for logging and plots)."""
        return {"alphas": self.alphas, "betas": self.betas}

    def regularizer_params(self) -> int:
        mods = [self.reg] if self.share_theta else list(self.regs)
        return sum(p.numel() for m in mods for p in m.parameters())

    # ------------------------------------------------------------------ one stage
    def _stage(self, x: torch.Tensor, y_n: torch.Tensor, k: int) -> torch.Tensor:
        P = self.projector
        dc = P.adjoint(P.forward(x) - y_n)
        return x - self.alphas[k] * dc - self.betas[k] * self.regularizer(k)(x)

    def _data_grad(self, x: torch.Tensor, y_n: torch.Tensor) -> torch.Tensor:
        P = self.projector
        return P.adjoint(P.forward(x) - y_n)

    def _stage_lbfgs(self, x: torch.Tensor, d: torch.Tensor, k: int) -> torch.Tensor:
        return x - self.alphas[k] * d - self.betas[k] * self.regularizer(k)(x)

    # ------------------------------------------------------------------ full model
    def forward(self, y: torch.Tensor, x0: torch.Tensor, return_stages: bool = False,
                return_terms: bool = False):
        """y: raw sinogram (B,1,V,D); x0: FBP image (B,1,H,W).

        return_stages: return the list [x^0, ..., x^K].
        return_terms: also return the per-stage norms of the data-consistency and regularizer steps.
        """
        if self.projector is None:
            raise RuntimeError("No projector set (call set_projector)")
        y_n = self.projector.normalize_data(y)
        x = x0
        prev_x: torch.Tensor | None = None
        prev_g: torch.Tensor | None = None
        s_hist: list[torch.Tensor] = []
        y_hist: list[torch.Tensor] = []
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        xs: list[torch.Tensor] = [x0]
        terms: list[dict] = []
        for k in range(self.K):
            if self.update == "lbfgs":
                g = self._data_grad(x, y_n)
                if prev_x is not None and prev_g is not None:
                    s_hist.append((x - prev_x).detach())
                    y_hist.append((g - prev_g).detach())
                    if len(s_hist) > self.lbfgs_memory:
                        s_hist.pop(0)
                        y_hist.pop(0)
                d = g if not s_hist else lbfgs_direction(g, torch.stack(s_hist), torch.stack(y_hist),
                                                         self.lbfgs_damping, self.lbfgs_curv_min)
                if return_terms:
                    with torch.no_grad():
                        rg = self.betas[k] * self.regularizer(k)(x)
                        terms.append({"k": k, "dc_norm": self._norm(self.alphas[k] * d),
                                      "reg_norm": self._norm(rg),
                                      "alpha": self.alphas[k].item(), "beta": self.betas[k].item()})
                prev_x, prev_g = x, g
                if use_ckpt:
                    x = checkpoint(self._stage_lbfgs, x, d, k, use_reentrant=False)
                else:
                    x = self._stage_lbfgs(x, d, k)
            else:
                if return_terms:
                    with torch.no_grad():
                        P = self.projector
                        dc = self.alphas[k] * P.adjoint(P.forward(x) - y_n)
                        rg = self.betas[k] * self.regularizer(k)(x)
                        terms.append({"k": k, "dc_norm": dc.flatten(1).norm(dim=1).mean().item(),
                                      "reg_norm": rg.flatten(1).norm(dim=1).mean().item(),
                                      "alpha": self.alphas[k].item(), "beta": self.betas[k].item()})
                if use_ckpt:
                    x = checkpoint(self._stage, x, y_n, k, use_reentrant=False)
                else:
                    x = self._stage(x, y_n, k)
            if return_stages:
                xs.append(x)
        out = xs if return_stages else x
        return (out, terms) if return_terms else out
