"""Sparse projection operator: adjoint, normalisation, autograd gradient, finite differences, ASTRA conventions."""
import numpy as np
import pytest
import torch

from lact.core.imaging.geometry import FanBeamGeometry
from lact.core.operators import opnorm_torch
from lact.infra.operators.operator_store import load_matrix


def test_adjoint_inner_product(small_projector):
    P = small_projector
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 1, P.img_size, P.img_size, generator=g)
    v = torch.rand(2, 1, P.n_views, P.n_det, generator=g)
    lhs = (P.forward(x) * v).double().sum()
    rhs = (x * P.adjoint(v)).double().sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-5


def test_normalized_operator_norm(small_projector):
    P = small_projector
    n, _ = opnorm_torch(P.forward, P.adjoint, (1, 1, P.img_size, P.img_size))
    assert abs(n - 1.0) < 1e-3


def test_autograd_uses_exact_transpose(small_projector):
    P = small_projector
    x = torch.rand(1, 1, P.img_size, P.img_size, requires_grad=True)
    y = torch.rand(1, 1, P.n_views, P.n_det)
    (0.5 * ((P.forward(x) - y) ** 2).sum()).backward()
    ref = P.adjoint(P.forward(x.detach()) - y)
    assert torch.allclose(x.grad, ref, rtol=1e-4, atol=1e-6)


def test_raw_vs_normalized(small_projector):
    P = small_projector
    x = torch.rand(1, 1, P.img_size, P.img_size)
    assert torch.allclose(P.raw_forward(x) * P.scale, P.forward(x), rtol=1e-5, atol=1e-6)


def test_finite_difference_float64(op_dir, small_geom):
    A, _ = load_matrix(small_geom, 32, op_dir)
    A = A.astype(np.float64)
    rng = np.random.default_rng(0)
    x = rng.random(A.shape[1])
    y = A @ rng.random(A.shape[1])
    g = A.T @ (A @ x - y)

    def f(z):
        return 0.5 * np.sum((A @ z - y) ** 2)

    for _ in range(3):
        d = rng.standard_normal(A.shape[1])
        eps = 1e-4
        fd = (f(x + eps * d) - f(x - eps * d)) / (2 * eps)
        assert abs(fd - g @ d) / abs(fd) < 1e-6


def test_vectors_match_astra():
    pytest.importorskip("astra")
    from lact.infra.operators.astra_matrix import astra_vectors

    g = FanBeamGeometry(span_deg=120, step_deg=10, n_det=16, det_spacing_mm=60)
    assert np.allclose(astra_vectors(g, 32), g.vectors())


def _chord(src, det, h):
    """Length of the src->det ray segment inside the square [-h, h]^2 (slab method)."""
    d = det - src
    lo, hi = 0.0, 1.0
    for k in range(2):
        if d[k] == 0:
            if not -h < src[k] < h:
                return 0.0
            continue
        t0, t1 = sorted(((-h - src[k]) / d[k], (h - src[k]) / d[k]))
        lo, hi = max(lo, t0), min(hi, t1)
    return max(hi - lo, 0.0) * float(np.hypot(*d))


@pytest.mark.parametrize(("span", "center"), [(90.0, 0.0), (120.0, 37.0), (360.0, 0.0)])
def test_siddon_ray_sums_are_chord_lengths(span, center):
    """Row sums of A (Siddon) = chord length of the ray through the field of view, for every ray."""
    from lact.infra.operators.siddon import build_matrix_siddon

    g = FanBeamGeometry(span_deg=span, step_deg=span / 12, center_deg=center, n_det=40, det_spacing_mm=24.0)
    n = 24
    A = build_matrix_siddon(g, n)
    rows = np.asarray(A.sum(axis=1)).ravel()
    vec = g.vectors()
    off = np.arange(g.n_det) - 0.5 * (g.n_det - 1)
    ref = np.array([_chord(v[:2], v[2:4] + o * v[4:6], 0.5 * g.fov_mm) for v in vec for o in off])
    assert np.allclose(rows, ref, rtol=1e-5, atol=1e-3)
    assert (ref > 0).sum() > 0.5 * ref.size


def test_siddon_matches_astra_line_fanflat():
    pytest.importorskip("astra")
    import scipy.sparse.linalg as spla

    from lact.infra.operators.astra_matrix import build_matrix
    from lact.infra.operators.siddon import build_matrix_siddon

    g = FanBeamGeometry(span_deg=120, step_deg=2.0, center_deg=11.0, n_det=96, det_spacing_mm=10.0)
    A, B = build_matrix(g, 48), build_matrix_siddon(g, 48)
    assert abs(A.nnz - B.nnz) <= 10
    assert spla.norm(A - B) / spla.norm(A) < 1e-3


@pytest.mark.parametrize("threads", [1, 3])
def test_scipy_backend_matches_torch(op_dir, small_geom, threads):
    """The SciPy backend (CPU, parallel row blocks) gives the same results and gradients as the PyTorch backend."""
    from lact.core.operators import SparseFanBeam

    A, meta = load_matrix(small_geom, 32, op_dir)
    Pt = SparseFanBeam(A, small_geom, 32, norm=meta["opnorm"], backend="torch")
    Ps = SparseFanBeam(A, small_geom, 32, norm=meta["opnorm"], backend="scipy", threads=threads)
    assert Ps.backend == "scipy" and Pt.backend == "torch"
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 1, 32, 32, generator=g)
    y = torch.rand(3, 1, Pt.n_views, Pt.n_det, generator=g)
    assert torch.allclose(Ps.forward(x), Pt.forward(x), rtol=1e-5, atol=1e-6)
    assert torch.allclose(Ps.adjoint(y), Pt.adjoint(y), rtol=1e-5, atol=1e-6)
    grads = []
    for P in (Pt, Ps):
        xr = x.clone().requires_grad_(True)
        (0.5 * ((P.forward(xr) - y) ** 2).sum() + P.adjoint(y).mul(xr).sum()).backward()
        grads.append(xr.grad)
    assert torch.allclose(grads[0], grads[1], rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError):
        SparseFanBeam(A, small_geom, 32, backend="cuda-magic")
