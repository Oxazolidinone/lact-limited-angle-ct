"""Second-order updates (update = cg | admm): the CG solver, the Sobolev metric and unrolled models on a small
operator."""
import numpy as np
import pytest
import torch

from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.metrics import psnr
from lact.core.models import build_unrolled
from lact.core.operators import FanBeamFBP
from lact.core.solvers import cg_solve, div2d, grad2d, grad_gram
from lact.infra.operators.operator_store import load_matrix

N = 32


def _data(op_dir, geom):
    A64, _ = load_matrix(geom, 64, op_dir)
    ph = shepp_logan(64)
    x = torch.from_numpy(block_mean(ph, 2))[None, None]
    y = torch.from_numpy((A64 @ ph.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    return y, FanBeamFBP(geom, N)(y), x


def _model(P, update, metric="identity", **kw):
    kw.setdefault("channels", 8)
    channels = kw.pop("channels")
    return build_unrolled("cnn3", {"channels": channels, "kernel": 5}, img_size=N, projector=P, update=update,
                          metric=metric, **kw)


def _dense(apply, n=N * N):
    """Dense matrix of a linear image operator (applied to each unit vector, batched)."""
    eye = torch.eye(n).view(n, 1, N, N)
    return apply(eye).reshape(n, n).t().double()


def test_grad_gram_is_dtd():
    x = torch.randn(3, 1, 9, 11, generator=torch.Generator().manual_seed(0))
    y = torch.randn(3, 1, 9, 11, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(grad_gram(x), -div2d(grad2d(x)), atol=1e-5)
    assert torch.allclose((grad_gram(x) * y).sum(), (x * grad_gram(y)).sum(), rtol=1e-4)       # symmetric
    assert float((x * grad_gram(x)).sum()) >= 0.0                                             # positive semi-definite
    assert float(grad_gram(torch.ones(1, 1, 9, 11)).abs().max()) == 0.0                       # constants in the kernel


@pytest.mark.parametrize("rho", [0.0, 2.0])
def test_cg_matches_dense_solve(small_projector, rho):
    P, lam = small_projector, 50.0
    H = lambda p: lam * P.normal(p) + p + rho * grad_gram(p)
    rhs = torch.randn(2, 1, N, N, generator=torch.Generator().manual_seed(0))
    ref = torch.linalg.solve(_dense(H), rhs.reshape(2, -1).double().t()).t().reshape(2, 1, N, N)
    d, n = cg_solve(H, rhs, 400, tol=1e-6)
    assert n < 400                                                                             # stops on the threshold
    assert float((d.double() - ref).norm() / ref.norm()) < 1e-4
    d5, n5 = cg_solve(H, rhs, 5)
    assert n5 == 5 and float((d5.double() - ref).norm()) > float((d.double() - ref).norm())    # stopping early is worse
    # zero right-hand side: no NaN
    zero, _ = cg_solve(H, torch.zeros_like(rhs), 3)
    assert torch.isfinite(zero).all() and float(zero.abs().max()) == 0.0


@pytest.mark.parametrize("metric", ["identity", "sobolev"])
def test_cg_stage_solves_the_weighted_least_squares(op_dir, small_geom, small_projector, metric):
    """Zero regularizer (at initialisation): x^1 = argmin lam/2 ||A x - y||^2 + 1/2 (x - x0)^T B (x - x0)."""
    P = small_projector
    y, x0, _ = _data(op_dir, small_geom)
    lam, rho = 30.0, 1.5
    m = _model(P, "cg", metric, K=1, lambda0=lam, rho0=rho, cg_iters=500, cg_tol=1e-7)
    with torch.no_grad():
        x1 = m(y, x0)
    B = lambda p: p if metric == "identity" else p + rho * grad_gram(p)
    Hd = _dense(lambda p: lam * P.normal(p) + B(p))
    b = lam * P.adjoint(P.normalize_data(y)) + B(x0)
    ref = torch.linalg.solve(Hd, b.reshape(-1).double()).reshape(1, 1, N, N)
    assert float((x1.double() - ref).norm() / ref.norm()) < 1e-4
    assert 0 < m.last_cg_iters[0] < 500 and m.cg_iters_mean() == m.last_cg_iters[0]


def test_cg_stages_decrease_residual_and_admm_equals_cg_without_regularizer(op_dir, small_geom, small_projector):
    P = small_projector
    y, x0, _ = _data(op_dir, small_geom)
    with torch.no_grad():
        xs = _model(P, "cg", K=5, cg_iters=40)(y, x0, return_stages=True)
        xa = _model(P, "admm", K=5, cg_iters=40)(y, x0, return_stages=True)
        few = _model(P, "cg", K=5, cg_iters=3)(y, x0, return_stages=True)
    for seq in (xs, few):
        res = [float(P.data_residual(xk, y)) for xk in seq]
        assert all(b <= a + 1e-7 for a, b in zip(res, res[1:])) and res[-1] < 0.5 * res[0]
    # R = 0: the dual variable only records the last step and the pull-back point z = v - u is still x^k => x matches cg
    # (CG solves nearly to convergence; stopping early amplifies the rounding of (x + u) - u at small-residual stages)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(xs, xa))


def test_tolerance_stops_early(op_dir, small_geom, small_projector):
    y, x0, _ = _data(op_dir, small_geom)
    m = _model(small_projector, "cg", K=3, cg_iters=50, cg_tol=0.3)
    with torch.no_grad():
        m(y, x0)
    assert all(0 < n < 50 for n in m.last_cg_iters)


@pytest.mark.parametrize(("update", "metric"), [("cg", "identity"), ("cg", "sobolev"), ("admm", "identity"),
                                           ("admm", "sobolev")])
def test_parameters_and_gradients(op_dir, small_geom, small_projector, update, metric):
    torch.manual_seed(0)
    y, x0, x = _data(op_dir, small_geom)
    m = _model(small_projector, update, metric, K=3, cg_iters=4)
    names = {k for k, _ in m.named_parameters() if not k.startswith("reg.")}
    want = {"b", "log_lam"} | ({"log_rho"} if metric == "sobolev" else set()) | ({"g"} if update == "admm" else set())
    assert names == want and set(m.stage_params()) == {
        "betas", "lambdas", *(["rhos"] if metric == "sobolev" else []), *(["gammas"] if update == "admm" else [])}
    # last layer starts at 0: let gradients reach the network
    torch.nn.init.normal_(m.reg.net[-1].weight, std=1e-2)
    m.train()
    torch.mean((m(y, x0) - x) ** 2).backward()
    grads = {k: p.grad.clone() for k, p in m.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert all(float(grads[k].abs().max()) > 0 for k in want)
    # per-stage gradient checkpointing must not change the gradient
    m.zero_grad()
    m.grad_checkpoint = True
    torch.mean((m(y, x0) - x) ** 2).backward()
    assert all(torch.allclose(p.grad, grads[k], rtol=1e-4, atol=1e-9) for k, p in m.named_parameters())


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
    assert list(lb.stage_params()) == ["alphas", "betas"] and not lb.uses_cg


def test_stage_terms_and_gd_keys_unchanged(op_dir, small_geom, small_projector):
    y, x0, _ = _data(op_dir, small_geom)
    gd = build_unrolled("cnn3", {"channels": 8, "kernel": 5}, img_size=N, projector=small_projector, K=2)
    assert {k for k in gd.state_dict() if not k.startswith("reg.")} == {"a", "b"}      # old checkpoints still load
    assert list(gd.stage_params()) == ["alphas", "betas"] and gd.cg_iters_mean() == 0.0
    m = _model(small_projector, "admm", "sobolev", K=2, cg_iters=3)
    with torch.no_grad():
        xs, terms = m(y, x0, return_stages=True, return_terms=True)
        plain = m(y, x0)
    assert len(xs) == 3 and torch.allclose(xs[-1], plain, atol=1e-6)
    assert set(terms[0]) == {"k", "dc_norm", "reg_norm", "cg_iters", "beta", "lambda", "rho", "gamma"}
    assert terms[0]["cg_iters"] == 3 and terms[0]["dc_norm"] > 0
    with pytest.raises(AttributeError):
        _ = m.alphas
    with pytest.raises(ValueError):
        _model(small_projector, "newton")


@pytest.mark.parametrize(("update", "metric"), [("cg", "identity"), ("admm", "sobolev")])
def test_overfit_one_slice(op_dir, small_geom, small_projector, update, metric):
    torch.manual_seed(0)
    y, x0, x = _data(op_dir, small_geom)
    m = _model(small_projector, update, metric, K=3, channels=16, cg_iters=5, lambda0=30.0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    p0 = float(psnr(x0, x))
    for _ in range(60):
        loss = torch.mean((m(y, x0) - x) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        assert float(psnr(m(y, x0), x)) > p0 + 3.0
