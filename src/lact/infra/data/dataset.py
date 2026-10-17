"""PyTorch Dataset adapter over HDF5: returns (y, x0 = FBP, x = reference image) per slice."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ...core.imaging.augment import apply as apply_aug
from ..system.locks import file_lock
from .cache import ArrayCache, file_signature

CACHE_VERSION = 1


class LACTDataset(Dataset):
    """Each item: dict(y (1,V,D), x0 (1,H,W), x (1,H,W), pid, idx, var).

    y is the UNNORMALISED sinogram; models normalise it themselves with ``projector.normalize_data``.
    """

    def __init__(self, images_h5: str | Path, sino_h5: str | Path, patients: list[str],
                 variants: tuple[str, ...] = ("orig",), preload: bool = True, limit: int | None = None,
                 array_cache: ArrayCache | None = None):
        self.images_h5, self.sino_h5 = str(images_h5), str(sino_h5)
        self.array_cache = array_cache
        self.patients = list(patients)
        self.variants = tuple(variants)
        self._h_img = self._h_sino = None
        items: list[tuple[str, int, str]] = []
        with h5py.File(self.sino_h5, "r") as hs, h5py.File(self.images_h5, "r") as hi:
            self.geometry = hs.attrs.get("geometry", "{}")
            self.mu_ref = float(hi.attrs["mu_ref"])
            self.mu_water = float(hi.attrs["mu_water_mm"])
            for pid in self.patients:
                if pid not in hs:
                    raise KeyError(f"{pid} is not in {self.sino_h5}")
                n = hs[pid]["y"].shape[0]
                for var in self.variants:
                    key = "y" if var == "orig" else f"y_{var}"
                    if key not in hs[pid]:
                        continue
                    items.extend((pid, i, var) for i in range(n))
        if limit is not None:
            items = items[: int(limit)]
        self.items = items
        self.cache: dict | None = None
        if preload:
            self._preload()

    # ------------------------------------------------------------------ loading
    def _needed_blocks(self) -> dict[tuple[str, str], slice]:
        need: dict[tuple[str, str], list[int]] = {}
        for pid, i, var in self.items:
            need.setdefault((pid, var), []).append(i)
        return {k: slice(min(v), max(v) + 1) for k, v in need.items()}

    def _cache_key(self, blocks: dict[tuple[str, str], slice]) -> str:
        """Key that changes when the source files change (size, mtime) or when the set of slices to load changes."""
        return ArrayCache.make_key({
            "v": CACHE_VERSION,
            "images": [Path(self.images_h5).name, file_signature(self.images_h5)],
            "sino": [Path(self.sino_h5).name, file_signature(self.sino_h5)],
            "blocks": sorted(f"{p}|{v}|{sl.start}|{sl.stop}" for (p, v), sl in blocks.items()),
        })

    def _read_blocks(self, blocks: dict[tuple[str, str], slice]) -> dict:
        cache = {}
        with h5py.File(self.sino_h5, "r") as hs, h5py.File(self.images_h5, "r") as hi:
            for (pid, var), sl in blocks.items():
                sfx = "" if var == "orig" else f"_{var}"
                cache[(pid, var)] = {
                    "offset": sl.start,
                    "y": hs[pid]["y" + sfx][sl], "fbp": hs[pid]["fbp" + sfx][sl],
                    "x": apply_aug(hi[pid]["img256"][sl], var), "z": hi[pid]["z"][sl],
                }
        return cache

    def _preload(self):
        """Load the needed slices into RAM.

        With ``array_cache`` the first load writes .npy files and later loads memmap them.
        """
        blocks = self._needed_blocks()
        if self.array_cache is None:
            self.cache = self._read_blocks(blocks)
            return
        key = self._cache_key(blocks)
        # Only one worker reads the HDF5 and builds a given array set; the others open the finished memmap.
        with file_lock(self.array_cache.root / f".{key}.build.lock"):
            self._preload_cached(blocks, key)

    def _preload_cached(self, blocks: dict, key: str) -> None:
        assert self.array_cache is not None
        hit = self.array_cache.load(key)
        if hit is None:
            cache = self._read_blocks(blocks)
            arrays, offsets = {}, {}
            for n, ((pid, var), c) in enumerate(cache.items()):
                for name in ("y", "fbp", "x", "z"):
                    arrays[f"b{n}_{name}"] = c[name]
                offsets[f"b{n}"] = [pid, var, int(c["offset"])]
            self.array_cache.save(key, arrays, {"blocks": offsets})
            hit = self.array_cache.load(key)
            assert hit is not None
        arrays, extra = hit
        self.cache = {(pid, var): {"offset": off, **{name: arrays[f"{b}_{name}"] for name in ("y", "fbp", "x", "z")}}
                      for b, (pid, var, off) in extra["blocks"].items()}

    def _open(self):
        if self._h_img is None:
            self._h_img = h5py.File(self.images_h5, "r")
            self._h_sino = h5py.File(self.sino_h5, "r")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, k: int) -> dict:
        pid, i, var = self.items[k]
        if self.cache is not None:
            c = self.cache[(pid, var)]
            j = i - c["offset"]
            y, x0, x, z = c["y"][j], c["fbp"][j], c["x"][j], float(c["z"][j])
        else:
            self._open()
            assert self._h_sino is not None and self._h_img is not None
            sfx = "" if var == "orig" else f"_{var}"
            y = self._h_sino[pid]["y" + sfx][i]
            x0 = self._h_sino[pid]["fbp" + sfx][i]
            x = apply_aug(self._h_img[pid]["img256"][i], var)
            z = float(self._h_img[pid]["z"][i])
        return {
            # np.array(...) always makes a writable copy (memmapped cache arrays are read-only)
            "y": torch.from_numpy(np.array(y, dtype=np.float32))[None],
            "x0": torch.from_numpy(np.array(x0, dtype=np.float32))[None],
            "x": torch.from_numpy(np.array(x, dtype=np.float32))[None],
            "pid": pid, "idx": i, "var": var, "z": z,
        }

    def __del__(self):
        for h in (self._h_img, self._h_sino):
            try:
                if h is not None:
                    h.close()
            except Exception:
                pass


def collate(batch: list[dict]) -> dict:
    """Collate a batch: tensors are stacked, metadata stays as lists."""
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in ("y", "x0", "x")}
    for k in ("pid", "idx", "var", "z"):
        out[k] = [b.get(k) for b in batch]
    return out
