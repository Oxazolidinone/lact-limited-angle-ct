"""Stage input data into the local cache (when ``cache.enabled``) and check its integrity.

The configuration (and each run's ``config.yaml``) always keeps the original paths, e.g. on Google Drive. Whenever a
service needs to read an input file it calls ``local_input``: with the cache enabled the file is copied once to
``cache.dir`` (the running machine's disk) and the copy is used afterwards. A new Colab session therefore only needs
to copy the data again, with no configuration change, and results are still written to Drive.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ...config import Config, load_geometry
from ...core.imaging.geometry import FanBeamGeometry
from ...infra.data.cache import cached_sha256, publish_file, stage_file
from ...infra.io.files import sha256_file
from ...infra.operators.operator_store import operator_path
from ...infra.stores.manifest import load_manifest, manifest_key

log = logging.getLogger("lact")


def _expected_sha(cfg: Config, path: Path) -> str | None:
    return load_manifest(cfg.paths.manifest).get(manifest_key(path), {}).get("sha256")


def local_input(cfg: Config, path: str | Path, kind: str = "data") -> Path:
    """Path to read for input ``path``: the local copy when the cache is enabled, otherwise the path itself."""
    path = Path(path)
    if not cfg.cache.enabled or cfg.cache.dir is None or not path.exists():
        return path
    return stage_file(path, Path(cfg.cache.dir) / kind, expected_sha256=_expected_sha(cfg, path),
                      verify=cfg.cache.verify)


@contextmanager
def write_via_cache(cfg: Config, dst: str | Path, kind: str = "data") -> Iterator[Path]:
    """Build a large file on the local disk first, then publish it to ``dst`` (Drive writes large files slowly and
    unreliably through FUSE).

    Without the cache the file is written straight to ``dst``. The local copy is kept for later reads.
    """
    dst = Path(dst)
    if not cfg.cache.enabled or cfg.cache.dir is None:
        yield dst
        return
    local = Path(cfg.cache.dir) / "build" / dst.name
    local.parent.mkdir(parents=True, exist_ok=True)
    yield local
    if local.exists():
        publish_file(local, dst, keep_dir=Path(cfg.cache.dir) / kind)


def operator_dir(cfg: Config, geom: FanBeamGeometry, size: int) -> Path:
    """Directory holding the needed matrix (.npz + .json); copied into the cache when enabled."""
    npz = operator_path(geom, size, cfg.paths.operators)
    if not cfg.cache.enabled:
        return npz.parent
    local = local_input(cfg, npz, "operators")
    local_input(cfg, npz.with_suffix(".json"), "operators")
    return local.parent


def verify_inputs(cfg: Config, files: list[Path]) -> list[str]:
    """Compare the SHA-256 of the input files (the copies in use) with the manifest. Returns errors (empty = match)."""
    man = load_manifest(cfg.paths.manifest)
    if not man:
        return []
    errors = []
    for f in files:
        entry = man.get(manifest_key(f))
        if entry is None:
            errors.append(f"{manifest_key(f)}: not in the manifest")
            continue
        if not Path(f).exists():
            errors.append(f"{manifest_key(f)}: does not exist")
            continue
        local = local_input(cfg, f, "operators" if f.suffix == ".npz" else "data")
        sha = cached_sha256(local) if local != Path(f) else sha256_file(f)
        if sha != entry["sha256"]:
            errors.append(f"{manifest_key(f)}: SHA-256 mismatch")
    return errors


def training_inputs(cfg: Config, geom: FanBeamGeometry) -> list[Path]:
    return [cfg.data.images_h5, cfg.paths.sino_file(geom.name),
            operator_path(geom, cfg.data.img_size_rec, cfg.paths.operators)]


def stage_all(cfg: Config, geometries: list[str]) -> list[Path]:
    """Pre-copy every file needed to train/evaluate the given geometries (call at the start of each Colab session)."""
    if not cfg.cache.enabled:
        log.info("cache.enabled = false: nothing to copy")
        return []
    out = [local_input(cfg, cfg.data.images_h5)]
    for name in geometries:
        geom = load_geometry(name)
        sino = cfg.paths.sino_file(geom.name)
        if sino.exists():
            out.append(local_input(cfg, sino))
        else:
            log.warning("%s does not exist yet (run lact data simulate)", sino)
        if operator_path(geom, cfg.data.img_size_rec, cfg.paths.operators).exists():
            out.append(operator_dir(cfg, geom, cfg.data.img_size_rec))
    return out
