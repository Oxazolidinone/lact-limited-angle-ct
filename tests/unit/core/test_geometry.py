import numpy as np
import pytest

from lact.config import load_geometry
from lact.core.imaging.geometry import FanBeamGeometry


def test_views_and_angles():
    for span, nv in ((90, 360), (120, 480), (240, 960)):
        g = load_geometry(f"fan_a{span}")
        assert g.n_views == nv
        a = g.angles_deg()
        assert np.isclose(a.mean(), 0.0)                       # same centre of the scan range
        assert np.isclose(a[0], -span / 2 + 0.125)             # end point excluded, offset by half a step
        assert np.allclose(np.diff(a), 0.25)


def test_short_scan_and_coverage():
    g = load_geometry("fan_a120")
    assert abs(g.fan_half_angle_deg - 28.335) < 0.01
    assert abs(g.short_scan_deg - 236.67) < 0.01
    assert g.fov_radius_covered_mm >= 0.5 * g.fov_mm * np.sqrt(2) - 1.0


def test_bad_step_raises():
    with pytest.raises(ValueError):
        _ = FanBeamGeometry(span_deg=100, step_deg=0.3).n_views


def test_dict_roundtrip():
    g = load_geometry("fan_a120")
    assert FanBeamGeometry.from_dict(g.to_dict()) == g
