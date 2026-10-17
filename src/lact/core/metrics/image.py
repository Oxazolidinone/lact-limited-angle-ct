"""Image quality metrics (batched, PyTorch).

- Images are on the normalised [0, 1] scale with data_range = 1. Predictions are clipped to [0, 1] by default.
- SSIM matches ``skimage.metrics.structural_similarity(..., gaussian_weights=True, sigma=1.5,
  use_sample_covariance=False, data_range=1)``: 11x11 Gaussian window, 5 border pixels dropped from the mean.
- ``psnr_minmax`` / ``ssim_minmax``: the protocol of the CG-GLORE paper (BMVC 2026) and QN-Mixer, where each slice is
  min-max normalised to [0, 1] and scored with ``torchmetrics`` using a data range taken from the data. Here the
  (prediction, reference) pair is normalised with the min and max of the REFERENCE slice at scoring time and the
  prediction is not clipped: the same number without rescaling the training data (PSNR is invariant when both images
  undergo the same affine map). These two functions are NOT part of ``all_metrics``: every thesis table, including
  the comparison with the CG-GLORE paper, uses ``psnr`` / ``ssim`` above (decided 2026-10-13); they are kept for
  cross-checking.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _prep(x: torch.Tensor, ref: torch.Tensor, clip: bool):
    x = x.float()
    ref = ref.float()
    if clip:
        x = x.clamp(0.0, 1.0)
    return x, ref


def mse(x, ref, clip: bool = True, mask: torch.Tensor | None = None) -> torch.Tensor:
    x, ref = _prep(x, ref, clip)
    d2 = (x - ref) ** 2
    if mask is None:
        return d2.flatten(1).mean(1)
    m = mask.float().expand_as(d2)
    return (d2 * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)


def rmse(x, ref, clip: bool = True, mask=None) -> torch.Tensor:
    return mse(x, ref, clip, mask).sqrt()


def psnr(x, ref, data_range: float = 1.0, clip: bool = True, mask=None) -> torch.Tensor:
    m = mse(x, ref, clip, mask).clamp_min(1e-20)
    return 10.0 * torch.log10(data_range ** 2 / m)


def rmse_hu(x, ref, mu_ref: float, mu_water: float, clip: bool = True, mask=None) -> torch.Tensor:
    """RMSE in HU: HU = (x * mu_ref / mu_water - 1) * 1000, so the HU error = 1000 * mu_ref / mu_water * error in x."""
    return rmse(x, ref, clip, mask) * (1000.0 * mu_ref / mu_water)


def _gauss_kernel(sigma: float = 1.5, truncate: float = 3.5, device=None) -> torch.Tensor:
    r = int(truncate * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _filt(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = (k.numel() - 1) // 2
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return F.conv2d(x, k.view(1, 1, -1, 1))


def ssim_map(x, ref, data_range: float | torch.Tensor = 1.0, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03,
             clip: bool = True) -> torch.Tensor:
    """``data_range``: a number, or a (B,1,1,1) tensor when each image has its own range."""
    x, ref = _prep(x, ref, clip)
    k = _gauss_kernel(sigma, device=x.device)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    ux, uy = _filt(x, k), _filt(ref, k)
    uxx, uyy, uxy = _filt(x * x, k), _filt(ref * ref, k), _filt(x * ref, k)
    vx, vy, vxy = uxx - ux * ux, uyy - uy * uy, uxy - ux * uy
    return ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))


def ssim(x, ref, data_range: float | torch.Tensor = 1.0, sigma: float = 1.5, clip: bool = True,
         mask=None) -> torch.Tensor:
    S = ssim_map(x, ref, data_range, sigma, clip=clip)
    r = int(3.5 * sigma + 0.5)
    S = S[..., r:-r, r:-r]
    if mask is None:
        return S.flatten(1).mean(1)
    m = mask.float()[..., r:-r, r:-r].expand_as(S)
    return (S * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp_min(1)


def minmax_pair(x: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each reference slice to [0, 1] by its own min and max; apply the same map to the prediction (no clip)."""
    x, ref = x.float(), ref.float()
    lo = ref.amin(dim=(1, 2, 3), keepdim=True)
    span = (ref.amax(dim=(1, 2, 3), keepdim=True) - lo).clamp_min(1e-12)
    return (x - lo) / span, (ref - lo) / span


def psnr_minmax(x, ref) -> torch.Tensor:
    """PSNR as in the CG-GLORE paper: peak = (max - min) of the reference slice, i.e. 1 after per-slice min-max."""
    xn, rn = minmax_pair(x, ref)
    return 10.0 * torch.log10(1.0 / ((xn - rn) ** 2).flatten(1).mean(1).clamp_min(1e-20))


def ssim_minmax(x, ref, sigma: float = 1.5) -> torch.Tensor:
    """SSIM as in the CG-GLORE paper (``torchmetrics`` defaults on per-slice min-max normalised images): 11x11
    Gaussian window, data_range = the larger value range of the two images of the pair."""
    xn, rn = minmax_pair(x, ref)
    x_span = xn.amax(dim=(1, 2, 3), keepdim=True) - xn.amin(dim=(1, 2, 3), keepdim=True)
    return ssim(xn, rn, data_range=torch.maximum(x_span, torch.ones_like(x_span)), sigma=sigma, clip=False)


def all_metrics(x, ref, mu_ref: float | None = None, mu_water: float | None = None, mask=None,
                clip: bool = True) -> dict[str, torch.Tensor]:
    out = {"psnr": psnr(x, ref, clip=clip), "ssim": ssim(x, ref, clip=clip), "rmse": rmse(x, ref, clip=clip)}
    if mu_ref is not None and mu_water is not None:
        out["rmse_hu"] = rmse_hu(x, ref, mu_ref, mu_water, clip=clip)
    if mask is not None:
        out["psnr_body"] = psnr(x, ref, clip=clip, mask=mask)
        out["ssim_body"] = ssim(x, ref, clip=clip, mask=mask)
        out["rmse_body"] = rmse(x, ref, clip=clip, mask=mask)
    return out


def psnr_from_rmse(r: float, data_range: float = 1.0) -> float:
    return 20.0 * math.log10(data_range / max(r, 1e-12))
