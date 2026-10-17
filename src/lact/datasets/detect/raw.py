"""2DeteCT raw data: scanner constants, the measured sinogram (-log of flat/dark-corrected counts, binned) and the
reference image in mu, read from the ``slice#####.npz`` files written by the downloader."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SOD_MM, SDD_MM = 431.019989, 529.000488
DET_PIX_MM, N_DET_RAW, N_PROJ_RAW, STEP_RAW_DEG = 0.0748, 1912, 3601, 0.1
SHIFT_BEFORE = 2830                 # LION: slices < 2830 shift the detector by 1 pixel (detector changed mid-campaign)
FOV_MM = N_DET_RAW * DET_PIX_MM * SOD_MM / SDD_MM     # field of view of the 1024² reference images (as in LION)
REC_VOXEL_MM = FOV_MM / 1024


def slice_name(i: int) -> str:
    """File stem of slice ``i`` in ``raw_dir``."""
    return f"slice{i:05d}"


TRANSFORMS = {
    "none": lambda a: a,
    "fliplr": lambda a: a[..., :, ::-1],
    "flipud": lambda a: a[..., ::-1, :],
    "rot90": lambda a: np.rot90(a, 1, axes=(-2, -1)),
    "rot180": lambda a: np.rot90(a, 2, axes=(-2, -1)),
    "rot270": lambda a: np.rot90(a, 3, axes=(-2, -1)),
    "transpose": lambda a: np.swapaxes(a, -1, -2),
    "antitranspose": lambda a: np.rot90(np.swapaxes(a, -1, -2), 2, axes=(-2, -1)),
}


def first_angle_deg(angle_bin: int) -> float:
    """Angle (the project's ASTRA ``fanflat`` convention) of the first projection after binning ``angle_bin`` raw
    projections.

    Raw projection j corresponds to the project angle 180° + j * 0.1°, with the reference image rotated by 180° and the
    detector axis NOT flipped (data.detect.image_transform = rot180, flip_det = false): chosen experimentally on
    full-360° slices (residual ||cAx - y|| / ||y|| about 3 %, c about 1)."""
    return 180.0 + 0.5 * (angle_bin - 1) * STEP_RAW_DEG


def _shift1(a: np.ndarray) -> np.ndarray:
    """Shift by 1 pixel along the detector as in LION: value at i taken from i + 1, linear extrapolation at the edge."""
    out = np.empty_like(a)
    out[..., :-1] = a[..., 1:]
    out[..., -1] = 2 * a[..., -1] - a[..., -2]
    return out


def measured_sinogram(z, n_views: int, det_bin: int, angle_bin: int, flip_det: bool) -> np.ndarray:
    """Measured sinogram (n_views, 1912 / det_bin) = -log((S - D) / (F - D)) after binning detector elements and
    projections on the counts (binning before the log, like detector pixel binning). Units: line integrals of mu
    (dimensionless)."""
    i = int(z["slice"])
    rows = n_views * angle_bin
    if z["sino"].shape[0] < rows:
        raise ValueError(f"slice {i}: has {z['sino'].shape[0]} projections, {rows} needed")
    s = z["sino"][:rows].astype(np.float64)
    d = z["dark"].astype(np.float64)
    f = 0.5 * (z["flat1"].astype(np.float64) + z["flat2"].astype(np.float64))
    if i < SHIFT_BEFORE:
        s, d, f = _shift1(s), _shift1(d), _shift1(f)
    nd = N_DET_RAW // det_bin
    S = s.reshape(n_views, angle_bin, nd, det_bin).sum(axis=(1, 3))
    D = d.reshape(nd, det_bin).sum(axis=1) * angle_bin
    F = f.reshape(nd, det_bin).sum(axis=1) * angle_bin
    y = -np.log(np.clip((S - D) / (F - D), 1e-6, None))
    return (y[:, ::-1] if flip_det else y).astype(np.float32)


def reference_mu(z, transform: str) -> np.ndarray:
    """Mode-2 1024² reference image converted to mu (mm^-1): stored values are mu x voxel size (LION's geometry is
    scaled so that a voxel is 1 unit long)."""
    return np.ascontiguousarray(TRANSFORMS[transform](z["rec"].astype(np.float64) / REC_VOXEL_MM))


def load_slice(raw_dir: Path, i: int):
    """The raw arrays of slice ``i`` (sino, dark, flat1, flat2, rec, slice)."""
    return np.load(raw_dir / f"{slice_name(i)}.npz")
