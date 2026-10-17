"""360° and short-scan (Parker) FBP must reconstruct a fine-grid simulated phantom correctly: orientation and scale."""
import numpy as np
import pytest
import torch

from lact.core.imaging.geometry import FanBeamGeometry
from lact.core.imaging.intensity import block_mean
from lact.core.imaging.phantom import shepp_logan
from lact.core.operators import FanBeamFBP

astra = pytest.importorskip("astra")
from lact.infra.operators.astra_matrix import build_matrix  # noqa: E402


def _psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((np.clip(a, 0, 1) - b) ** 2))


def _geom(span):
    return FanBeamGeometry(name="t", span_deg=span, step_deg=1.0, n_det=128, det_spacing_mm=7.5)


def _project(g, img):
    return torch.from_numpy((build_matrix(g, img.shape[0]) @ img.ravel()).astype(np.float32)).view(
        1, 1, g.n_views, g.n_det)


@pytest.mark.parametrize(("span", "parker"), [(360.0, False), (237.0, True)])
def test_fbp_reconstructs_phantom(span, parker):
    g = _geom(span)
    ph = shepp_logan(128)
    ref = block_mean(ph, 2)
    f = FanBeamFBP(g, 64)
    assert f.parker == parker
    rec = f(_project(g, ph))[0, 0].numpy()
    assert _psnr(rec, ref) > 28.0
    assert abs(rec.mean() / ref.mean() - 1.0) < 0.05


def test_limited_angle_worse_than_full():
    ph = shepp_logan(128)
    ref = block_mean(ph, 2)
    out = {span: _psnr(FanBeamFBP(_geom(span), 64)(_project(_geom(span), ph))[0, 0].numpy(), ref)
           for span in (360.0, 90.0)}
    assert out[360.0] > out[90.0] + 5.0


def test_orientation_point_source():
    """An off-centre bright point must be reconstructed at the right position (no flip/rotation)."""
    g = _geom(360.0)
    img = np.zeros((128, 128), np.float32)
    img[20:24, 90:94] = 1.0
    rec = FanBeamFBP(g, 64)(_project(g, img))[0, 0].numpy()
    r, c = np.unravel_index(np.argmax(rec), rec.shape)
    assert abs(r - 10.5) <= 1.5 and abs(c - 45.5) <= 1.5
