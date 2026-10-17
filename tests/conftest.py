"""Shared fixtures: a small geometry so tests run fast on the CPU."""
from __future__ import annotations

import pytest

from lact.core.imaging.geometry import FanBeamGeometry


@pytest.fixture(scope="session")
def small_geom() -> FanBeamGeometry:
    return FanBeamGeometry(name="t120", span_deg=120.0, step_deg=2.0, n_det=96, det_spacing_mm=10.0)


@pytest.fixture(scope="session")
def op_dir(tmp_path_factory, small_geom):
    # Without ASTRA (e.g. Linux aarch64), build_and_save falls back to the pure-NumPy Siddon builder (same line_fanflat
    # model).
    from lact.infra.operators.operator_store import build_and_save

    d = tmp_path_factory.mktemp("ops")
    for n in (32, 64):
        build_and_save(small_geom, n, d)
    return d


@pytest.fixture(scope="session")
def small_projector(op_dir, small_geom):
    from lact.infra.operators.operator_store import load_projector

    return load_projector(small_geom, 32, op_dir)
