"""HDF5 data stores.

- ``images.h5``: /{pid}/img512, /{pid}/img256 (normalised [0,1] images), z, instance, sop_uid;
  attrs mu_ref, mu_water_mm.
- ``sino_{geom}.h5``: /{pid}/y (N,V,D) raw sinogram, /{pid}/fbp (N,H,W), /{pid}/dc_floor (N,);
  augmented variants carry a suffix (y_hflip, ...). attrs: geometry (JSON), grid sizes, matrix SHA-256.
Writes are atomic: data goes to a .tmp file that is renamed on success.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _suffix(variant: str) -> str:
    return "" if variant == "orig" else f"_{variant}"


class _AtomicH5:
    def __init__(self, path: str | Path, attrs: dict[str, Any] | None = None):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.attrs = attrs or {}
        self.h5: h5py.File | None = None

    @property
    def file(self) -> h5py.File:
        if self.h5 is None:
            raise RuntimeError("HDF5 file is not open (use it inside a with block)")
        return self.h5

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.tmp, "w")
        self.h5.attrs.update(self.attrs)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.file.close()
        if exc_type is None:
            self.tmp.replace(self.path)
        return False


class ImageWriter(_AtomicH5):
    def add_patient(self, pid: str, img512: np.ndarray, img256: np.ndarray, z: np.ndarray, instance: np.ndarray,
                    sop_uid: list[str], attrs: dict[str, Any] | None = None) -> None:
        g = self.file.create_group(pid)
        n_sim, n_rec = img512.shape[-1], img256.shape[-1]
        g.create_dataset("img512", data=img512.astype(np.float32), chunks=(1, n_sim, n_sim), compression="lzf")
        g.create_dataset("img256", data=img256.astype(np.float32), chunks=(1, n_rec, n_rec), compression="lzf")
        g.create_dataset("z", data=np.asarray(z, np.float64))
        g.create_dataset("instance", data=np.asarray(instance, np.int32))
        g.create_dataset("sop_uid", data=np.array(sop_uid, dtype="S64"))
        g.attrs.update(attrs or {})


class ImageStore:
    """Read images.h5 (the file is opened and closed per read, safe across processes)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist. Run: lact data preprocess")
        with h5py.File(self.path, "r") as h:
            self.attrs = dict(h.attrs)
            self._n = {pid: int(h[pid]["img256"].shape[0]) for pid in h}

    @property
    def mu_ref(self) -> float:
        return float(self.attrs["mu_ref"])

    @property
    def mu_water(self) -> float:
        return float(self.attrs["mu_water_mm"])

    def patients(self) -> list[str]:
        return list(self._n)

    def n_slices(self, pid: str) -> int:
        return self._n[pid]

    def read(self, pid: str, key: str, sl: slice | list[int] | int = slice(None)) -> np.ndarray:
        with h5py.File(self.path, "r") as h:
            return h[pid][key][sl]


class SinoWriter(_AtomicH5):
    def create(self, pid: str, variant: str, n: int, n_views: int, n_det: int, n_rec: int) -> _SinoSlots:
        g = self.file.require_group(pid)
        s = _suffix(variant)
        return _SinoSlots(
            g.create_dataset("y" + s, (n, n_views, n_det), np.float32, chunks=(1, n_views, n_det)),
            g.create_dataset("fbp" + s, (n, n_rec, n_rec), np.float32, chunks=(1, n_rec, n_rec)),
            g.create_dataset("dc_floor" + s, (n,), np.float32))


class _SinoSlots:
    def __init__(self, y, fbp, dc):
        self.y, self.fbp, self.dc = y, fbp, dc

    def write(self, start: int, stop: int, y: np.ndarray, fbp: np.ndarray, dc_floor: np.ndarray) -> None:
        self.y[start:stop] = y
        self.fbp[start:stop] = fbp
        self.dc[start:stop] = dc_floor


class SinoStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} does not exist. Run: lact data simulate")
        with h5py.File(self.path, "r") as h:
            self.attrs = dict(h.attrs)
            self._keys = {pid: set(h[pid].keys()) for pid in h}
            self._n = {pid: int(h[pid]["y"].shape[0]) for pid in h}

    @property
    def geometry(self) -> dict:
        return json.loads(self.attrs.get("geometry", "{}"))

    def patients(self) -> list[str]:
        return list(self._n)

    def n_slices(self, pid: str) -> int:
        return self._n[pid]

    def has_variant(self, pid: str, variant: str) -> bool:
        return ("y" + _suffix(variant)) in self._keys.get(pid, set())

    def read(self, pid: str, key: str, variant: str = "orig", sl: slice | int = slice(None)) -> np.ndarray:
        with h5py.File(self.path, "r") as h:
            return h[pid][key + _suffix(variant)][sl]
