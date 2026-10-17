"""Learned Primal-Dual (Adler & Öktem, IEEE TMI 2018).

NOT part of the LEARN framework: K = 10 iterations with their own weights; each iteration updates the dual variable h
(sinogram domain) and then the primal variable f (image domain), each with 5 memory channels:

    h <- h + Gamma_k(h, A f[1], y)            f <- f + Lambda_k(f, A^T h[0])            output = f[0]

Gamma_k, Lambda_k: three 3x3 convolutions, 32 channels, PReLU. A is the normalised operator with ||A|| = 1 (the
original paper also divides A by its norm) and A^T is its exact transpose. 0.253M parameters (the CG-GLORE paper
reports 0.251M for this 10-iteration configuration).

Kept from the original code (github.com/adler-j/learned_primal_dual): the channels fed to A and A^T, per-channel PReLU
initialised to 0, Xavier weights, zero biases. Different: by default every channel of f starts from x^0 = FBP
(``init = "fbp"``), like the other methods in the comparison table and the DIVAL re-implementation; the original
starts from zero (``init = "zero"``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..unrolled import Unrolled


def _block(cin: int, cout: int, channels: int, kernel: int) -> nn.Sequential:
    p = kernel // 2
    net = nn.Sequential(
        nn.Conv2d(cin, channels, kernel, padding=p), nn.PReLU(channels, init=0.0),
        nn.Conv2d(channels, channels, kernel, padding=p), nn.PReLU(channels, init=0.0),
        nn.Conv2d(channels, cout, kernel, padding=p),
    )
    for m in net:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net


class LearnedPrimalDual(Unrolled):
    def __init__(self, K: int = 10, n_primal: int = 5, n_dual: int = 5, channels: int = 32, kernel: int = 3,
                 init: str = "fbp", grad_checkpoint: bool = False, projector=None):
        super().__init__()
        if init not in ("fbp", "zero"):
            raise ValueError(f"init must be 'fbp' or 'zero', got {init!r}")
        if K < 1 or n_primal < 2 or n_dual < 1:
            raise ValueError("Learned Primal-Dual needs K >= 1, n_primal >= 2 and n_dual >= 1")
        self.K, self.n_primal, self.n_dual, self.init = int(K), int(n_primal), int(n_dual), init
        self.dual = nn.ModuleList(_block(self.n_dual + 2, self.n_dual, channels, kernel) for _ in range(self.K))
        self.primal = nn.ModuleList(_block(self.n_primal + 1, self.n_primal, channels, kernel)
                                    for _ in range(self.K))
        self.grad_checkpoint = bool(grad_checkpoint)
        self.projector = projector

    def stage_params(self) -> dict[str, torch.Tensor]:
        return {}

    def regularizer_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def regularization_only(self, x: torch.Tensor) -> None:
        """Run only the networks of all K iterations (zero image and sinogram of the right size), no projections."""
        if self.projector is None:
            raise RuntimeError("No projector set (the sinogram size is needed)")
        f = x.expand(-1, self.n_primal + 1, -1, -1)
        h = x.new_zeros(x.shape[0], self.n_dual + 2, *self.projector.sino_shape)
        for k in range(self.K):
            self.dual[k](h)
            self.primal[k](f)

    def _stage(self, f: torch.Tensor, h: torch.Tensor, y_n: torch.Tensor, k: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
        h = h + self.dual[k](torch.cat([h, self.projector.forward(f[:, 1:2]), y_n], dim=1))
        f = f + self.primal[k](torch.cat([f, self.projector.adjoint(h[:, 0:1])], dim=1))
        return f, h

    def forward(self, y: torch.Tensor, x0: torch.Tensor, return_stages: bool = False, return_terms: bool = False):
        """y: raw sinogram (B,1,V,D); x0: FBP image (B,1,H,W). Return options as in ``UnrolledLEARN.forward``."""
        if self.projector is None:
            raise RuntimeError("No projector set (call set_projector)")
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        y_n = self.projector.normalize_data(y)
        f = x0.repeat(1, self.n_primal, 1, 1) if self.init == "fbp" else x0.new_zeros(
            x0.shape[0], self.n_primal, *x0.shape[-2:])
        h = y_n.new_zeros(y_n.shape[0], self.n_dual, *y_n.shape[-2:])
        xs: list[torch.Tensor] = [x0]
        terms: list[dict] = []
        for k in range(self.K):
            prev = f
            f, h = checkpoint(self._stage, f, h, y_n, k, use_reentrant=False) if use_ckpt \
                else self._stage(f, h, y_n, k)
            if return_terms:        # the two steps interleave, so only the output-image update size is recorded
                with torch.no_grad():
                    terms.append({"k": k, "dc_norm": 0.0, "reg_norm": self._norm(f[:, 0:1] - prev[:, 0:1])})
            if return_stages:
                xs.append(f[:, 0:1])
        out = xs if return_stages else f[:, 0:1]
        return (out, terms) if return_terms else out
