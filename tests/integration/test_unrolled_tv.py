"""Unrolled models and TV-PDHG on a small operator."""
import numpy as np
import torch

from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.metrics import psnr
from lact.core.models import build_unrolled
from lact.core.operators import FanBeamFBP
from lact.core.solvers import tv_pdhg
from lact.infra.operators.operator_store import load_matrix


def _data(op_dir, geom):
    A64, _ = load_matrix(geom, 64, op_dir)
    ph = shepp_logan(64)
    x = torch.from_numpy(block_mean(ph, 2))[None, None]
    y = torch.from_numpy((A64 @ ph.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    return y, FanBeamFBP(geom, 32)(y), x


def _cnn(P, **kw):
    channels = kw.pop("channels", 48)
    return build_unrolled("cnn3", {"channels": channels, "kernel": 5}, img_size=32, projector=P, **kw)


def test_zero_regularizer_decreases_residual(op_dir, small_geom, small_projector):
    P = small_projector
    y, x0, _ = _data(op_dir, small_geom)
    with torch.no_grad():
        xs = _cnn(P, K=6)(y, x0, return_stages=True)
    res = [float(P.data_residual(xk, y)) for xk in xs]
    assert all(b <= a + 1e-7 for a, b in zip(res, res[1:]))
    assert res[-1] < res[0]


def test_overfit_one_slice(op_dir, small_geom, small_projector):
    torch.manual_seed(0)
    y, x0, x = _data(op_dir, small_geom)
    m = _cnn(small_projector, K=4, channels=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    p0 = float(psnr(x0, x))
    for _ in range(60):
        loss = torch.mean((m(y, x0) - x) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert float(psnr(m(y, x0), x)) > p0 + 3.0


def test_unshared_and_checkpoint(op_dir, small_geom, small_projector):
    y, x0, x = _data(op_dir, small_geom)
    m = _cnn(small_projector, K=3, share_theta=False, grad_checkpoint=True, channels=8)
    assert len(m.regs) == 3
    m.train()
    torch.mean((m(y, x0) - x) ** 2).backward()
    assert m.a.grad is not None
    assert "projector" not in "".join(m.state_dict().keys())


def test_tv_improves_over_fbp(op_dir, small_geom, small_projector):
    y, x0, x = _data(op_dir, small_geom)
    xt = tv_pdhg(small_projector, y, 1e-4, 200, x_init=x0)
    assert float(psnr(xt, x)) > float(psnr(x0, x))
