"""ASTRA Toolbox adapter: builds the sparse projection matrix A of a FanBeamGeometry.

Row = view * n_det + channel; column = image row * n + image column. Entry = intersection length (mm).
Only this module imports ``astra``; the rest of the program works with scipy matrices.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ...core.imaging.geometry import FanBeamGeometry

PROJECTORS = ("line_fanflat", "strip_fanflat")


def is_available() -> bool:
    """Whether the ASTRA toolbox can be imported on this machine."""
    try:
        import astra  # noqa: F401
    except ImportError:
        return False
    return True


def astra_geometries(geom: FanBeamGeometry, img_size: int):
    """ASTRA (vol_geom, proj_geom) for the geometry and image grid."""
    import astra

    h = 0.5 * geom.fov_mm
    vol = astra.create_vol_geom(img_size, img_size, -h, h, -h, h)
    proj = astra.create_proj_geom("fanflat", geom.det_spacing_mm, geom.n_det, geom.angles_rad(),
                                  geom.dso_mm, geom.dod_mm)
    return vol, proj


def build_matrix(geom: FanBeamGeometry, img_size: int, projector: str = "line_fanflat") -> sp.csr_matrix:
    import astra

    if projector not in PROJECTORS:
        raise ValueError(f"projector must be one of {PROJECTORS}")
    vol, proj = astra_geometries(geom, img_size)
    pid = astra.create_projector(projector, proj, vol)
    try:
        mid = astra.projector.matrix(pid)
        A = astra.matrix.get(mid)
        astra.matrix.delete(mid)
    finally:
        astra.projector.delete(pid)
    A = sp.csr_matrix(A, dtype=np.float32)
    A.sum_duplicates()
    A.eliminate_zeros()
    expected = (geom.n_views * geom.n_det, img_size * img_size)
    if A.shape != expected:
        raise RuntimeError(f"Matrix shape {A.shape} differs from the expected {expected}")
    return A


def astra_vectors(geom: FanBeamGeometry, img_size: int = 32) -> np.ndarray:
    """ASTRA geometry vectors (used to check the conventions of FanBeamGeometry.vectors())."""
    import astra

    _, proj = astra_geometries(geom, img_size)
    return astra.geom_2vec(proj)["Vectors"]
