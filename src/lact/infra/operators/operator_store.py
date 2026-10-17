"""Projection-matrix store: save/load ``A_{geom}_n{size}.npz`` + ``.json`` metadata (shape, nnz, ||A||, SHA-256)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ...core.imaging.geometry import FanBeamGeometry
from ...core.operators import SparseFanBeam, opnorm_sparse
from ..io.files import load_json, save_json, sha256_file

_GEOM_KEYS = ("span_deg", "step_deg", "center_deg", "n_det", "det_spacing_mm", "dso_mm", "dod_mm", "fov_mm")


def operator_path(geom: FanBeamGeometry, img_size: int, op_dir: str | Path) -> Path:
    return Path(op_dir) / f"A_{geom.name}_n{img_size}.npz"


def build_matrix_any(geom: FanBeamGeometry, img_size: int,
                     projector: str = "line_fanflat") -> tuple[sp.csr_matrix, str]:
    """Build A with ASTRA; without ASTRA and with projector = line_fanflat, fall back to the pure-NumPy Siddon
    builder (same intersection-length model, matches ASTRA to float32 precision). Returns (A, backend)."""
    from .astra_matrix import build_matrix, is_available

    if not is_available():
        if projector != "line_fanflat":
            raise ImportError(f"Projector {projector!r} needs ASTRA, which is not installed")
        from .siddon import build_matrix_siddon
        return build_matrix_siddon(geom, img_size), "siddon"
    return build_matrix(geom, img_size, projector), "astra"


def build_and_save(geom: FanBeamGeometry, img_size: int, op_dir: str | Path, projector: str = "line_fanflat",
                   compute_norm: bool = True, overwrite: bool = False) -> dict:
    """Build A (ASTRA, or Siddon without ASTRA), compute ||A||_2 and save .npz + .json.

    Returns the metadata; an existing matrix is kept unless ``overwrite``.
    """
    path = operator_path(geom, img_size, op_dir)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not overwrite:
        return load_json(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    A, backend = build_matrix_any(geom, img_size, projector)
    t_build = time.time() - t0
    norm, it = opnorm_sparse(A) if compute_norm else (float("nan"), 0)
    sp.save_npz(path, A, compressed=False)
    meta = {
        "geometry": geom.to_dict(), "img_size": img_size, "projector": projector, "backend": backend,
        "shape": list(A.shape),
        "nnz": int(A.nnz), "memory_mb_csr": round((A.nnz * 8 + (A.shape[0] + 1) * 4) / 2**20, 1),
        "opnorm": norm, "opnorm_iters": it, "build_seconds": round(t_build, 2), "file": path.name,
        "sha256": sha256_file(path),
    }
    save_json(meta, meta_path)
    return meta


def load_matrix(geom: FanBeamGeometry, img_size: int, op_dir: str | Path, build_if_missing: bool = False,
                verify_hash: bool = False) -> tuple[sp.csr_matrix, dict]:
    path = operator_path(geom, img_size, op_dir)
    meta_path = path.with_suffix(".json")
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"Matrix {path} does not exist. Run: lact ops build")
        build_and_save(geom, img_size, op_dir)
    meta = load_json(meta_path) if meta_path.exists() else {}
    if verify_hash and meta.get("sha256") and sha256_file(path) != meta["sha256"]:
        raise RuntimeError(f"SHA-256 of {path} does not match its metadata")
    saved = meta.get("geometry")
    if saved:
        for k in _GEOM_KEYS:
            if abs(float(saved[k]) - float(getattr(geom, k))) > 1e-9:
                raise RuntimeError(f"Matrix {path.name} was built with {k}={saved[k]}, "
                                   f"the configuration has {getattr(geom, k)}")
    return sp.load_npz(path).tocsr().astype(np.float32), meta


def load_projector(geom: FanBeamGeometry, img_size: int, op_dir: str | Path, device="cpu",
                   with_adjoint: bool = True, build_if_missing: bool = False) -> SparseFanBeam:
    A, meta = load_matrix(geom, img_size, op_dir, build_if_missing=build_if_missing)
    return SparseFanBeam(A, geom, img_size, norm=meta.get("opnorm"), device=device, with_adjoint=with_adjoint)
