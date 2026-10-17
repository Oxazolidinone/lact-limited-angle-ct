"""Build the projection matrices A (pipeline step 02) and record them in the data manifest."""
from __future__ import annotations

import logging

from ...config import Config
from ...core.imaging.geometry import FanBeamGeometry
from ...infra.data.cache import publish_file
from ...infra.io.files import load_json
from ...infra.operators.operator_store import build_and_save, operator_path
from ...infra.stores.manifest import update_manifest
from ..data.staging import write_via_cache

log = logging.getLogger("lact")


def build_operators(cfg: Config, geoms: list[FanBeamGeometry], sizes: list[int] | None = None,
                    projector: str = "line_fanflat", overwrite: bool = False) -> list[dict]:
    """Build A for each (geometry, grid size), compute ||A||_2, write .npz/.json and update the manifest."""
    sizes = sizes or [cfg.data.img_size_rec, cfg.data.img_size_sim]
    op_dir = cfg.paths.operators
    metas, files = [], []
    for geom in geoms:
        log.info(geom.describe())
        for n in sizes:
            path = operator_path(geom, n, op_dir)
            if path.exists() and path.with_suffix(".json").exists() and not overwrite:
                meta = load_json(path.with_suffix(".json"))
                if meta.get("geometry") != geom.to_dict() or meta.get("projector") != projector:
                    raise ValueError(f"{path}: geometry/projector differ from the cache; "
                                     "use a separate operators directory.")
            else:
                # build on the local disk (when the cache is enabled), then publish to durable storage
                with write_via_cache(cfg, path, "operators") as target:
                    meta = build_and_save(geom, n, target.parent, projector=projector, overwrite=True)
                    if target != path:
                        publish_file(target.with_suffix(".json"), path.with_suffix(".json"),
                                     keep_dir=target.parent.parent / "operators")
            log.info("  n=%d: %s, nnz=%.1fM, %.0f MB, ||A||=%.3f, built in %.1fs", n, meta["shape"], meta["nnz"] / 1e6,
                     meta["memory_mb_csr"], meta["opnorm"], meta["build_seconds"])
            files += [path, path.with_suffix(".json")]
            metas.append(meta)
    update_manifest(cfg.paths.manifest, files)
    return metas
