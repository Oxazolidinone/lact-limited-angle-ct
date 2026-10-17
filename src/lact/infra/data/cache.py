"""Local cache for input data.

On Colab the data lives on Google Drive (durable but slow to read through FUSE). This module provides:

- ``stage_file``: copy a file to the local disk once and reuse it while the original is unchanged
  (size + modification time, or SHA-256). Writes are atomic, so an interruption never leaves a partial file.
- ``ArrayCache``: store loaded NumPy arrays (e.g. a Dataset's whole training split) as .npy files; later runs memmap
  them, which is almost instant and uses no RAM until read.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..io.files import load_json, save_json, sha256_file
from ..system.locks import file_lock

log = logging.getLogger("lact")

STAMP_SUFFIX = ".stamp.json"
COMPLETE = "COMPLETE"


def file_signature(path: str | os.PathLike) -> dict:
    st = Path(path).stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _stamp_path(dst: Path) -> Path:
    return dst.with_name(dst.name + STAMP_SUFFIX)


def is_fresh(src: str | os.PathLike, dst: str | os.PathLike) -> bool:
    """Whether the copy ``dst`` still matches ``src`` (size + mtime stamp recorded at copy time)."""
    dst = Path(dst)
    stamp = _stamp_path(dst)
    if not (dst.exists() and stamp.exists()):
        return False
    try:
        sig = file_signature(src)
        info = load_json(stamp)
        return (info.get("source_path") == str(Path(src)) and info.get("source") == sig
                and dst.stat().st_size == sig["size"])
    except (OSError, ValueError):
        return False


def stage_file(src: str | os.PathLike, dst_dir: str | os.PathLike, expected_sha256: str | None = None,
               verify: str = "size") -> Path:
    """Copy ``src`` into ``dst_dir`` unless an up-to-date copy exists; return the local path.

    ``verify='sha256'`` recomputes the SHA-256 of the copy and compares it with ``expected_sha256`` (when given).
    """
    with file_lock(Path(dst_dir) / f".{Path(src).name}.lock"):
        return _stage_file(src, dst_dir, expected_sha256, verify)


def _stage_file(src, dst_dir, expected_sha256, verify) -> Path:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst = dst_dir / src.name
    if is_fresh(src, dst):
        if verify == "sha256" and expected_sha256 and cached_sha256(dst) != expected_sha256:
            raise RuntimeError(f"SHA-256 of {dst} does not match the manifest")
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    t0 = time.time()
    shutil.copyfile(src, tmp)
    sha = None
    if verify == "sha256":
        sha = sha256_file(tmp)
        if expected_sha256 and sha != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 of the copy of {src.name} does not match the manifest "
                               "(source corrupted or still being written?)")
    os.replace(tmp, dst)
    save_json({"source": file_signature(src), "source_path": str(src), "sha256": sha,
               "dst_size": dst.stat().st_size, "copied_at": time.strftime("%Y-%m-%d %H:%M:%S")}, _stamp_path(dst))
    mb = dst.stat().st_size / 2**20
    log.info("Copied %s (%.0f MB, %.1fs) -> %s", src.name, mb, time.time() - t0, dst_dir)
    return dst


def publish_file(local: str | os.PathLike, dst: str | os.PathLike, keep_dir: str | os.PathLike | None = None) -> Path:
    """Publish a file built on the local disk to durable storage (e.g. Drive): copy to ``dst.partial``, then rename.

    ``keep_dir``: keep the local copy in the cache directory (with a stamp matching ``dst``) to avoid copying it back.
    """
    local, dst = Path(local), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    t0 = time.time()
    shutil.copyfile(local, tmp)
    os.replace(tmp, dst)
    log.info("Saved %s (%.0f MB, %.1fs)", dst, dst.stat().st_size / 2**20, time.time() - t0)
    if keep_dir is not None:
        kept = Path(keep_dir) / dst.name
        kept.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local), str(kept))
        save_json({"source": file_signature(dst), "source_path": str(dst), "sha256": None,
                   "dst_size": kept.stat().st_size, "copied_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                  _stamp_path(kept))
    else:
        local.unlink(missing_ok=True)
    return dst


def cached_sha256(dst: str | os.PathLike) -> str:
    """SHA-256 of the copy: taken from the stamp when already computed, otherwise computed and stored in the stamp."""
    with file_lock(Path(dst).with_name(f".{Path(dst).name}.sha.lock")):
        return _cached_sha256(Path(dst))


def _cached_sha256(dst: Path) -> str:
    dst = Path(dst)
    stamp = _stamp_path(dst)
    info = load_json(stamp) if stamp.exists() else {}
    if info.get("sha256") and info.get("dst_size") in (None, dst.stat().st_size):
        return str(info["sha256"])
    sha = sha256_file(dst)
    if stamp.exists():
        info.update({"sha256": sha, "dst_size": dst.stat().st_size})
        save_json(info, stamp)
    return sha


class ArrayCache:
    """Keyed store of .npy arrays; each entry is a directory ``<root>/<key>/`` with a ``COMPLETE`` file once written."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    @staticmethod
    def make_key(parts: dict[str, Any]) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def path(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return (self.path(key) / COMPLETE).exists()

    def load(self, key: str, mmap: bool = True) -> tuple[dict[str, np.ndarray], dict] | None:
        d = self.path(key)
        if not self.has(key):
            return None
        meta = load_json(d / "meta.json")
        arrays = {name: np.load(d / f"{name}.npy", mmap_mode="r" if mmap else None) for name in meta["arrays"]}
        return arrays, meta.get("extra", {})

    def save(self, key: str, arrays: dict[str, np.ndarray], extra: dict | None = None) -> Path:
        """Write to a temporary directory, then rename: an interruption never leaves a partial entry."""
        with file_lock(self.root / f".{key}.write.lock"):
            if self.has(key):
                return self.path(key)
            return self._save(key, arrays, extra)

    def _save(self, key: str, arrays: dict[str, np.ndarray], extra: dict | None) -> Path:
        d = self.path(key)
        tmp = self.root / f".{key}.tmp{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for name, a in arrays.items():
            np.save(tmp / f"{name}.npy", np.ascontiguousarray(a))
        save_json({"arrays": list(arrays), "extra": extra or {}}, tmp / "meta.json")
        (tmp / COMPLETE).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        if d.exists():
            shutil.rmtree(d)
        os.replace(tmp, d)
        return d

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
