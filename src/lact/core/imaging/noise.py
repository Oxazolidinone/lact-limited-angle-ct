"""Measurement noise added to simulated sinograms (off for noise-free datasets, enabled by the ``noise`` section).

Poisson noise follows the Beer-Lambert law: each detector element counts N ~ Poisson(I0 * exp(-p)) photons, where
p is the line integral of the attenuation coefficient; the noisy measurement is -ln(N / I0). Its variance is about
exp(p) / I0, so rays through thick or dense regions are noisier. I0 = 1e6 is the usual normal-dose level in
sparse-view CT papers (GloReDi, QN-Mixer); 1e5 corresponds to a lower dose (Huang et al., 2019, limited-angle CT).
"""
from __future__ import annotations

import torch

NOISE_TYPES = ("none", "poisson", "gaussian")


def add_noise(y: torch.Tensor, kind: str, mu_ref: float, gen: torch.Generator, I0: float = 1e5,
              sigma: float = 0.01) -> torch.Tensor:
    """``y`` is on the normalised scale (normalised intensity x mm).

    - poisson: count Poisson(I0 * exp(-y * mu_ref)) photons, then take the log.
    - gaussian: add Gaussian noise with standard deviation ``sigma * max|y|``.
    """
    if kind == "none":
        return y
    if kind == "poisson":
        p = (y * mu_ref).double()
        counts = torch.poisson(I0 * torch.exp(-p), generator=gen).clamp_min(1.0)
        return (-torch.log(counts / I0) / mu_ref).float()
    if kind == "gaussian":
        s = sigma * float(y.abs().max())
        return y + s * torch.randn(y.shape, generator=gen, device=y.device)
    raise ValueError(f"Unsupported noise type: {kind} (available: {NOISE_TYPES})")
