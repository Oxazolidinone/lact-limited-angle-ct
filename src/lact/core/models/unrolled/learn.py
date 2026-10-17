"""Deep-unrolling framework based on LEARN (Chen et al., 2018), with four update rules for each stage
k = 0..K-1 (x^0 = FBP(y)).

``update = "gd"`` - LEARN, first order:

    x^{k+1} = x^k - alpha_k * A^T (A x^k - y) - beta_k * R_theta(x^k)

``update = "cg"`` - second order with the surrogate Hessian H_k = lam_k A^T A + B_k (MoDL / CG-GLORE style). The
network proposes v, then the data step solves a least-squares problem with conjugate gradients (CG):

    v       = x^k - beta_k * R_theta(x^k)
    x^{k+1} = argmin_x  lam_k/2 ||A x - y||^2 + 1/2 (x - v)^T B_k (x - v)
            = v + H_k^{-1} lam_k A^T (y - A v)

``update = "admm"`` - as above plus a dual variable u (u^0 = 0) that remembers the gap between the data-consistent
solution and the proposal:

    v       = w - beta_k * R_theta(w),   w = x^k + u^k
    x^{k+1} = (v - u^k) + H_k^{-1} lam_k A^T (y - A (v - u^k))
    u^{k+1} = u^k + gamma_k * (x^{k+1} - v)

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

``metric``: B_k = I ("identity", as in CG-GLORE) or B_k = I + rho_k D^T D ("sobolev", D = first differences). The
Sobolev metric penalises departures from the proposal more at high frequencies: trust the network for detail and the
data for large structures; rho_k -> 0 recovers the identity metric.

- A and y are normalised (||A|| = 1) inside the model; the dataset returns the raw y.
- alpha_k, beta_k, gamma_k > 0 through softplus; lam_k, rho_k > 0 through a log parameterisation (spanning orders of
  magnitude). Each of these is learned per stage.
- CG runs at most ``cg_iters`` iterations per stage and stops once the residual has dropped by ``cg_tol`` relative to
  the start (``cg_tol = 0``: always run every iteration). Gradients flow through every CG iteration run.
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

from ...solvers.cg import cg_solve, grad_gram
from ...solvers.lbfgs import lbfgs_direction
from .base import Unrolled

UPDATES = ("gd", "cg", "admm", "lbfgs")
METRICS = ("identity", "sobolev")


def _inv_softplus(v: float) -> float:
    return math.log(math.expm1(v))


class UnrolledLEARN(Unrolled):
    def __init__(self, make_regularizer: Callable[[], nn.Module], K: int = 14, share_theta: bool = True,
                 alpha0: float = 1.0, beta0: float = 1.0, grad_checkpoint: bool = False, projector=None,
                 update: str = "gd", metric: str = "identity", lambda0: float = 100.0, rho0: float = 1.0,
                 gamma0: float = 1.0, cg_iters: int = 10, cg_tol: float = 0.0, lbfgs_memory: int = 13,
                 lbfgs_damping: float = 0.0, lbfgs_curv_min: float = 0.0):
        super().__init__()
        if update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}, got {update!r}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if update != "gd" and (int(cg_iters) < 1 or cg_tol < 0 or lambda0 <= 0 or rho0 <= 0 or gamma0 <= 0):
            raise ValueError("need cg_iters >= 1, cg_tol >= 0 and lambda0, rho0, gamma0 > 0")
        if update == "lbfgs" and int(lbfgs_memory) < 1:
            raise ValueError("lbfgs_memory must be >= 1")
        self.K = int(K)
        self.share_theta = bool(share_theta)
        self.update, self.metric = update, metric
        self.cg_iters, self.cg_tol = int(cg_iters), float(cg_tol)
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_damping, self.lbfgs_curv_min = float(lbfgs_damping), float(lbfgs_curv_min)
        if self.share_theta:
            self.reg = make_regularizer()
        else:
            self.regs = nn.ModuleList(make_regularizer() for _ in range(self.K))
        if update == "gd":
            self.a = nn.Parameter(torch.full((self.K,), _inv_softplus(alpha0)))
        elif update == "lbfgs":
            self.a_qn = nn.Parameter(torch.full((self.K,), _inv_softplus(alpha0)))
        self.b = nn.Parameter(torch.full((self.K,), _inv_softplus(beta0)))
        if update in ("cg", "admm"):
            self.log_lam = nn.Parameter(torch.full((self.K,), math.log(lambda0)))
            if metric == "sobolev":
                self.log_rho = nn.Parameter(torch.full((self.K,), math.log(rho0)))
            if update == "admm":
                self.g = nn.Parameter(torch.full((self.K,), _inv_softplus(gamma0)))
        self.grad_checkpoint = bool(grad_checkpoint)
        self.projector = projector
        self.last_cg_iters: list[int] = [0] * self.K     # CG iterations per stage in the latest forward pass

    @property
    def uses_cg(self) -> bool:
        return self.update in ("cg", "admm")

    def regularizer(self, k: int) -> nn.Module:
        return self.reg if self.share_theta else self.regs[k]

    # ------------------------------------------------------------------ per-stage parameters
    @property
    def alphas(self) -> torch.Tensor:
        if self.update not in ("gd", "lbfgs"):
            raise AttributeError("alphas exist only for update = 'gd' or 'lbfgs' (see stage_params)")
        return F.softplus(self.a if self.update == "gd" else self.a_qn)

    @property
    def betas(self) -> torch.Tensor:
        return F.softplus(self.b)

    @property
    def lambdas(self) -> torch.Tensor:
        return self.log_lam.exp()

    @property
    def rhos(self) -> torch.Tensor:
        return self.log_rho.exp()

    @property
    def gammas(self) -> torch.Tensor:
        return F.softplus(self.g)

    def stage_params(self) -> dict[str, torch.Tensor]:
        """Learned per-stage scalar parameters of the active update rule, by name (for logging and plots)."""
        if self.update in ("gd", "lbfgs"):
            return {"alphas": self.alphas, "betas": self.betas}
        out = {"betas": self.betas, "lambdas": self.lambdas}
        if self.metric == "sobolev":
            out["rhos"] = self.rhos
        if self.update == "admm":
            out["gammas"] = self.gammas
        return out

    def regularizer_params(self) -> int:
        mods = [self.reg] if self.share_theta else list(self.regs)
        return sum(p.numel() for m in mods for p in m.parameters())

    def cg_iters_mean(self) -> float:
        return float(sum(self.last_cg_iters)) / max(1, self.K) if self.uses_cg else 0.0

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

    def _metric(self, p: torch.Tensor, k: int) -> torch.Tensor:
        """B_k p."""
        return p if self.metric == "identity" else p + self.rhos[k] * grad_gram(p)

    def _split(self, x: torch.Tensor, u: torch.Tensor | None, y_n: torch.Tensor, k: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variable-splitting stage (cg, admm). Returns (new x, proposal v, point z = v - u for the data step)."""
        P, lam = self.projector, self.lambdas[k]
        w = x if u is None else x + u
        v = w - self.betas[k] * self.regularizer(k)(w)
        z = v if u is None else v - u
        # x = z + d with H d = lam A^T (y - A z): the B z term cancels on the right-hand side, so CG starts at d = 0
        rhs = lam * P.adjoint(y_n - P.forward(z))
        d, n = cg_solve(lambda p: lam * P.normal(p) + self._metric(p, k), rhs, self.cg_iters, self.cg_tol)
        self.last_cg_iters[k] = n
        return z + d, v, z

    def _stage_cg(self, x: torch.Tensor, y_n: torch.Tensor, k: int) -> torch.Tensor:
        return self._split(x, None, y_n, k)[0]

    def _stage_admm(self, x: torch.Tensor, u: torch.Tensor, y_n: torch.Tensor, k: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_new, v, _ = self._split(x, u, y_n, k)
        return x_new, u + self.gammas[k] * (x_new - v)

    def _split_with_terms(self, x: torch.Tensor, u: torch.Tensor | None, y_n: torch.Tensor, k: int
                          ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        x_new, v, z = self._split(x, u, y_n, k)
        with torch.no_grad():
            w = x if u is None else x + u
            t = {"k": k, "dc_norm": self._norm(x_new - z), "reg_norm": self._norm(w - v),
                 "cg_iters": self.last_cg_iters[k],
                 **{name[:-1]: vals[k].item() for name, vals in self.stage_params().items()}}
        if u is not None:
            u = u + self.gammas[k] * (x_new - v)
        return x_new, u, t

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
        u = torch.zeros_like(x0) if self.update == "admm" else None
        prev_x: torch.Tensor | None = None
        prev_g: torch.Tensor | None = None
        s_hist: list[torch.Tensor] = []
        y_hist: list[torch.Tensor] = []
        self.last_cg_iters = [0] * self.K
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
            elif self.update == "gd":
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
            elif return_terms:
                x, u, t = self._split_with_terms(x, u, y_n, k)
                terms.append(t)
            elif self.update == "cg":
                if use_ckpt:
                    x = checkpoint(self._stage_cg, x, y_n, k, use_reentrant=False)
                else:
                    x = self._stage_cg(x, y_n, k)
            else:
                assert u is not None
                if use_ckpt:
                    x, u = checkpoint(self._stage_admm, x, u, y_n, k, use_reentrant=False)
                else:
                    x, u = self._stage_admm(x, u, y_n, k)
            if return_stages:
                xs.append(x)
        out = xs if return_stages else x
        return (out, terms) if return_terms else out
