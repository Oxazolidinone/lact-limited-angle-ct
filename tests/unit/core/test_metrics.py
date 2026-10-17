import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lact.core.metrics import body_mask_np, psnr, rmse, rmse_hu, ssim


def test_match_skimage():
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.random((48, 64)).astype(np.float32)
        b = np.clip(a + 0.05 * rng.standard_normal(a.shape), 0, 1).astype(np.float32)
        t = lambda z: torch.from_numpy(z)[None, None]
        s_ref = structural_similarity(a, b, data_range=1, gaussian_weights=True, sigma=1.5,
                                      use_sample_covariance=False)
        assert abs(float(ssim(t(b), t(a))) - s_ref) < 1e-4
        assert abs(float(psnr(t(b), t(a))) - peak_signal_noise_ratio(a, b, data_range=1)) < 1e-3


def test_rmse_hu_scaling():
    x = torch.full((1, 1, 8, 8), 0.3)
    r = torch.full((1, 1, 8, 8), 0.2)
    assert abs(float(rmse(x, r)) - 0.1) < 1e-6
    assert abs(float(rmse_hu(x, r, mu_ref=0.05, mu_water=0.02)) - 0.1 * 1000 * 0.05 / 0.02) < 1e-3


def test_body_mask():
    img = np.zeros((64, 64), np.float32)
    img[16:48, 12:52] = 0.25
    img[30:34, 30:34] = 0.0   # an inner air pocket must be filled
    m = body_mask_np(img, mu_ref=0.08, mu_water=0.0192)
    assert m[32, 32] and not m[2, 2]


def test_minmax_metrics_follow_the_cg_glore_paper():
    """psnr_minmax / ssim_minmax: each slice min-max normalised by the reference slice, prediction not clipped (the
    CG-GLORE paper's torchmetrics protocol). Matches skimage on normalised images and is invariant when both images
    are rescaled. Kept for cross-checking, not part of ``all_metrics``."""
    sk = pytest.importorskip("skimage.metrics")
    from lact.core.metrics import all_metrics, psnr_minmax, ssim_minmax

    rng = np.random.default_rng(0)
    ref = rng.random((3, 1, 64, 64)).astype(np.float32) * np.array([0.6, 1.0, 1.4], np.float32).reshape(3, 1, 1, 1)
    ref += 0.02
    pred = ref + rng.normal(0, 0.03, ref.shape).astype(np.float32)
    pred[2, 0, :5, :5] += 1.0                                    # prediction exceeds the reference range
    R, P = torch.from_numpy(ref), torch.from_numpy(pred)
    pm, sm = psnr_minmax(P, R).numpy(), ssim_minmax(P, R).numpy()
    for i in range(3):
        lo, hi = ref[i].min(), ref[i].max()
        rn, pn = (ref[i, 0] - lo) / (hi - lo), (pred[i, 0] - lo) / (hi - lo)
        dr = max(float(pn.max() - pn.min()), 1.0)
        assert pm[i] == pytest.approx(sk.peak_signal_noise_ratio(rn, pn, data_range=1.0), abs=1e-3)
        assert sm[i] == pytest.approx(sk.structural_similarity(rn, pn, gaussian_weights=True, sigma=1.5,
                                                               use_sample_covariance=False, data_range=dr), abs=1e-4)
    assert np.abs(psnr_minmax(P * 3.7 + 0.2, R * 3.7 + 0.2).numpy() - pm).max() < 1e-3
    assert np.abs(ssim_minmax(P * 3.7, R * 3.7).numpy() - sm).max() < 1e-4
    m = all_metrics(P, R)                                        # thesis tables use only the thesis psnr / ssim
    assert {"psnr", "ssim"} <= set(m) and not {"psnr_mm", "ssim_mm"} & set(m)
    assert not torch.allclose(m["psnr"], torch.from_numpy(pm))
