"""Two-loop L-BFGS (H0 = I): no pairs gives the gradient; a Newton step on stored directions, gradient elsewhere."""
import torch

from lact.core.solvers.lbfgs import lbfgs_direction


def _img(values, grad=False):
    t = torch.tensor(values, dtype=torch.float64).view(1, 1, 1, -1)
    return t.requires_grad_(grad) if grad else t


def test_no_history_is_the_gradient_and_a_flat_pair_is_skipped():
    g = _img([2.0, 4.0], grad=True)
    s = _img([1.0, 0.0], grad=True)[None]
    y = _img([0.0, 0.0], grad=True)[None]
    d = lbfgs_direction(g, s, y)
    assert torch.allclose(d, g)
    d.sum().backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


def test_one_pair_is_newton_on_that_direction_and_gradient_elsewhere():
    # s = (1, 0), y = (2, 0) => curvature 2 on the first axis: g = (2, 4) => (2 / 2, 4). The unstored axis is unchanged
    # (H0 = I).
    g = _img([2.0, 4.0])
    s = _img([1.0, 0.0])[None]
    y = _img([2.0, 0.0])[None]
    d = lbfgs_direction(g, s, y)
    assert torch.allclose(d.view(-1), torch.tensor([1.0, 4.0], dtype=torch.float64))


def test_secant_equation_and_descent_on_a_quadratic():
    """Quadratic f = x'Ax/2 with arbitrary steps: H y_new = s_new (secant equation), H symmetric positive definite."""
    gen = torch.Generator().manual_seed(0)
    n, m = 12, 5
    L = torch.randn(n, n, generator=gen, dtype=torch.float64)
    A = L @ L.T / n + 0.05 * torch.eye(n, dtype=torch.float64)
    S = torch.randn(m, n, generator=gen, dtype=torch.float64)
    s = S.view(m, 1, 1, 1, n)
    y = (S @ A).view(m, 1, 1, 1, n)
    assert torch.allclose(lbfgs_direction(y[-1], s, y), s[-1], atol=1e-9)
    H = torch.stack([lbfgs_direction(e.view(1, 1, 1, n), s, y).view(n) for e in torch.eye(n, dtype=torch.float64)])
    assert torch.allclose(H, H.T, atol=1e-9) and float(torch.linalg.eigvalsh(H).min()) > 0
    g = torch.randn(1, 1, 1, n, generator=gen, dtype=torch.float64)
    assert float((g * lbfgs_direction(g, s, y)).sum()) > 0                      # descent direction


def test_batches_do_not_mix():
    g = torch.tensor([2.0, 4.0, 3.0, 5.0], dtype=torch.float64).view(2, 1, 1, 2)
    s = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64).view(1, 2, 1, 1, 2)
    y = torch.tensor([2.0, 0.0, 0.0, 0.0], dtype=torch.float64).view(1, 2, 1, 1, 2)
    d = lbfgs_direction(g, s, y)
    assert torch.allclose(d[0].view(-1), torch.tensor([1.0, 4.0], dtype=torch.float64))
    assert torch.allclose(d[1], g[1])          # the second sample's pair has no curvature


def test_damping_bounds_the_step_on_a_nearly_flat_pair():
    """A pair with near-zero curvature (a step in an almost unmeasured direction) makes H scale that direction by
    1/κ; Powell damping (relative to B0 = I) caps it at 1/η and leaves sufficiently curved pairs unchanged. This keeps
    from-scratch training from blowing up."""
    import pytest

    torch.manual_seed(0)
    s = torch.zeros(1, 2, 1, 4, 4, dtype=torch.float64)
    s[0, :, 0, 0, 0] = 1.0                                          # direction e1, two samples
    g = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    g[:, 0, 0, 0] = 1.0
    flat = 1e-6 * s                                                 # y = κ s, κ = 1e-6
    raw = lbfgs_direction(g, s, flat)
    assert torch.allclose(raw[:, 0, 0, 0], torch.full((2,), 1e6, dtype=torch.float64))
    damped = lbfgs_direction(g, s, flat, damping=0.2)
    assert torch.allclose(damped[:, 0, 0, 0], torch.full((2,), 5.0, dtype=torch.float64))        # exactly 1/η
    rest = torch.ones_like(g, dtype=torch.bool)
    rest[:, 0, 0, 0] = False
    # other directions: H0 = I
    assert torch.allclose(damped[rest], g[rest]) and torch.allclose(raw[rest], g[rest])
    curved = 0.5 * s                                                # κ = 0.5 >= η: unchanged
    assert torch.allclose(lbfgs_direction(g, s, curved, damping=0.2), lbfgs_direction(g, s, curved))
    # cautious update: drop pairs that are too flat, keep sufficiently curved ones
    assert torch.allclose(lbfgs_direction(g, s, flat, curv_min=1e-3), g)
    assert torch.allclose(lbfgs_direction(g, s, curved, curv_min=1e-3), lbfgs_direction(g, s, curved))
    for bad in ({"damping": 1.0}, {"damping": -0.1}, {"curv_min": -1.0}):
        with pytest.raises(ValueError):
            lbfgs_direction(g, s, flat, **bad)


def test_damping_reaches_the_models_from_config(small_projector):
    """model.lbfgs_damping / lbfgs_curv_min flow from the configuration into both LEARN and RegFormer and change the
    result."""
    import pytest

    from lact.config import ConfigError, load_config
    from lact.services.factories import model_for

    base = ["data.img_size_rec=32", "model.K=4", "model.update=lbfgs"]
    for files, small in ((["configs/model/learn_cnn.yaml"], ["model.regularizer.channels=8"]),
                         (["configs/model/regformer.yaml"], ["model.regularizer.dim=8", "model.regularizer.heads=2"])):
        plain = model_for(load_config(files, base + small), small_projector)
        # a large eta makes sure some pair is damped (the pure data step is fairly curved; a small eta changes nothing)
        cfg = load_config(files, base + small + ["model.lbfgs_damping=0.9", "model.lbfgs_curv_min=0.001"])
        damped = model_for(cfg, small_projector)
        assert (plain.lbfgs_damping, plain.lbfgs_curv_min) == (0.0, 0.0)
        assert (damped.lbfgs_damping, damped.lbfgs_curv_min) == (0.9, 0.001)
        damped.load_state_dict(plain.state_dict())
        torch.manual_seed(0)
        x = torch.rand(2, 1, 32, 32)
        y = small_projector.raw_forward(x)
        with torch.no_grad():
            a, b = plain.eval()(y, torch.zeros_like(x)), damped.eval()(y, torch.zeros_like(x))
        assert torch.isfinite(b).all() and not torch.allclose(a, b)
    with pytest.raises(ConfigError):
        load_config(["configs/model/learn_cnn.yaml"], ["model.update=lbfgs", "model.lbfgs_damping=1.0"])
