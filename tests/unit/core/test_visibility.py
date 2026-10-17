"""Visibility operator (Parseval frame + microlocal gate) used to split errors into measured / unmeasured directions."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lact.core.analysis.visibility import VisibilityOperator, directional_frame, visibility_gates
from lact.core.imaging.geometry import FanBeamGeometry

N = 32


def geom(span: float = 90.0) -> FanBeamGeometry:
    return FanBeamGeometry(name=f"t{int(span)}", span_deg=span, step_deg=2.0, n_det=48, det_spacing_mm=20.0)


@pytest.mark.parametrize("n", [32, 33])
def test_frame_is_parseval(n):
    H = directional_frame(n, 12)
    assert np.abs((H ** 2).sum(0) - 1).max() < 1e-6
    op = VisibilityOperator(geom(), n, 12)
    x = torch.randn(2, 1, n, n)
    assert (op.synthesis(op.analysis(x)) - x).abs().max() < 1e-5          # T^H T = I


def test_visibility_operator_is_selfadjoint_psd_and_splits_identity():
    op = VisibilityOperator(geom(), N, 12)
    x, y = torch.randn(3, 1, N, N, dtype=torch.float64), torch.randn(3, 1, N, N, dtype=torch.float64)
    px, py = op.vis(x.float()).double(), op.vis(y.float()).double()
    assert abs(float((px * y).sum() - (x * py).sum())) < 1e-4 * float((x * y).abs().sum())
    q = float((px * x).sum() / (x * x).sum())
    assert 0.0 <= q <= 1.0
    assert (op.vis(x.float()) + op.inv(x.float()) - x.float()).abs().max() < 1e-5
    # not idempotent (redundant frame + per-pixel gate): a visibility operator rather than a projection
    v = op.vis(x.float())
    assert (op.vis(v) - v).norm() > 1e-3 * v.norm()


@pytest.mark.parametrize("span", [60.0, 90.0, 120.0, 150.0])
def test_center_gate_matches_angular_span(span):
    """At the rotation centre, the ray from the source at angle beta has direction beta + 90°, so the measured
    frequency direction is beta itself: the visible-direction fraction = span/180 (independent of the
    implementation)."""
    G = visibility_gates(geom(span), 64, 18)
    frac = float(G[:, 32, 32].mean())
    assert abs(frac - span / 180.0) <= 1.5 / 18


def test_fan_beam_visibility_varies_with_position():
    G = visibility_gates(geom(90.0), 64, 18).mean(axis=0)
    assert G.std() > 0.01                       # fan beam is not shift-invariant: the gate must vary with the pixel
    full = visibility_gates(FanBeamGeometry(name="full", span_deg=360.0, step_deg=4.0, n_det=48,
                                            det_spacing_mm=20.0), 32, 12)
    assert full[:, 8:24, 8:24].min() == 1.0     # full scan: every direction is visible in the central region
