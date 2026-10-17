"""The L-BFGS data step of the LEARN framework on a small operator, and the first-order parameter names."""
import numpy as np
import pytest
import torch

from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.models import build_unrolled
from lact.core.operators import FanBeamFBP
from lact.infra.operators.operator_store import load_matrix

N = 32


def _data(op_dir, geom):
    A64, _ = load_matrix(geom, 64, op_dir)
    ph = shepp_logan(64)
    x = torch.from_numpy(block_mean(ph, 2))[None, None]
    y = torch.from_numpy((A64 @ ph.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    return y, FanBeamFBP(geom, N)(y), x


def _model(P, update, **kw):
    return build_unrolled("cnn3", {"channels": 8, "kernel": 5}, img_size=N, projector=P, update=update, **kw)


def test_lbfgs_starts_as_gradient_step_then_uses_curvature(op_dir, small_geom, small_projector):
    """The first stage has no memory pairs and equals LEARN; later stages use curvature and depart from the gradient
    step. Regularizer and beta share the first-order names (checkpoints load) while alpha is a separate, re-initialised
    parameter."""
    y, x0, x = _data(op_dir, small_geom)
    gd = build_unrolled("cnn3", {"channels": 8, "kernel": 5}, img_size=N, projector=small_projector, K=3,
                        alpha0=2.5)                              # like the large alpha learned by a first-order run
    lb = _model(small_projector, "lbfgs", K=3, lbfgs_memory=4)
    assert {k for k in lb.state_dict() if not k.startswith("reg.")} == {"a_qn", "b"}
    state, own = gd.state_dict(), lb.state_dict()
    report = lb.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
    assert report.missing_keys == ["a_qn"] and torch.allclose(lb.alphas, torch.ones(3))     # 2.5 is not inherited
    with torch.no_grad():
        gd.a.copy_(lb.a_qn)                                      # same alpha = 1 to compare the two rules
        a = gd(y, x0, return_stages=True)
        b = lb(y, x0, return_stages=True)
    assert torch.allclose(a[1], b[1], atol=1e-5)
    assert not torch.allclose(a[2], b[2], atol=1e-4)
    res = [float(small_projector.data_residual(v, y)) for v in b]
    # R = 0 at initialisation: L-BFGS still reduces the residual
    assert res[-1] < res[0]
    lb.grad_checkpoint = True
    lb.train()
    torch.mean((lb(y, x0) - x) ** 2).backward()
    assert torch.isfinite(lb.a_qn.grad).all() and float(lb.a_qn.grad.abs().max()) > 0
    assert list(lb.stage_params()) == ["alphas", "betas"]


def test_stage_terms_and_gd_keys_unchanged(op_dir, small_geom, small_projector):
    y, x0, _ = _data(op_dir, small_geom)
    for update, alpha in (("gd", "a"), ("lbfgs", "a_qn")):
        m = _model(small_projector, update, K=2)
        assert {k for k in m.state_dict() if not k.startswith("reg.")} == {alpha, "b"}  # old checkpoints still load
        assert list(m.stage_params()) == ["alphas", "betas"]
        with torch.no_grad():
            xs, terms = m(y, x0, return_stages=True, return_terms=True)
            plain = m(y, x0)
        assert len(xs) == 3 and torch.allclose(xs[-1], plain, atol=1e-6)
        assert set(terms[0]) == {"k", "dc_norm", "reg_norm", "alpha", "beta"} and terms[0]["dc_norm"] > 0
    with pytest.raises(ValueError):
        _model(small_projector, "cg")


def test_lbfgs_overfits_one_slice(op_dir, small_geom, small_projector):
    y, x0, x = _data(op_dir, small_geom)
    torch.manual_seed(0)
    m = _model(small_projector, "lbfgs", K=3, lbfgs_damping=0.5)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    start = float(torch.mean((m(y, x0) - x) ** 2))
    for _ in range(60):
        opt.zero_grad()
        loss = torch.mean((m(y, x0) - x) ** 2)
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * start
