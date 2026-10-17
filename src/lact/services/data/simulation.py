"""Simulation use case: limited-angle sinograms from fine-grid images (A_sim) + FBP on the reconstruction grid.

Sinograms are generated with A_sim (512) while models reconstruct with A_rec (256): different grids avoid the
"inverse crime".
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import torch

from ...config import Config, ConfigError
from ...core.imaging.augment import apply as apply_aug
from ...core.imaging.augment import variants as aug_variants
from ...core.imaging.geometry import FanBeamGeometry
from ...core.imaging.noise import add_noise
from ...infra.io.files import load_json
from ...infra.io.hdf5_store import SinoStore, SinoWriter
from ...infra.operators.operator_store import operator_path
from ...infra.stores.manifest import load_manifest, manifest_key, update_manifest
from ...infra.system.runtime import git_hash
from ..factories import fbp_for, image_store, projector_for
from .staging import write_via_cache

log = logging.getLogger("lact")


@torch.no_grad()
def simulate_geometry(cfg: Config, geom: FanBeamGeometry, device="cpu", batch: int = 8,
                      overwrite: bool = False) -> Path:
    if cfg.data.format == "2detect":          # measured data: no simulation, only angle cropping and calibration
        from .detect import measure_geometry
        return measure_geometry(cfg, geom, device=device, batch=batch, overwrite=overwrite)
    out = cfg.paths.sino_file(geom.name)
    if out.exists() and not overwrite:
        saved = SinoStore(out)
        images = image_store(cfg)
        expected = {"img_size_sim": cfg.data.img_size_sim, "img_size_rec": cfg.data.img_size_rec,
                    "mu_ref": images.mu_ref, "fbp_filter": cfg.baseline.fbp.filter}
        for label, n in (("A_sim_sha256", cfg.data.img_size_sim), ("A_rec_sha256", cfg.data.img_size_rec)):
            expected[label] = load_json(operator_path(geom, n, cfg.paths.operators).with_suffix(".json"))["sha256"]
        if (saved.geometry != geom.to_dict() or any(saved.attrs.get(k) != v for k, v in expected.items())
                or json.loads(saved.attrs.get("noise", "{}")) != vars(cfg.noise)):
            raise ConfigError(f"{out}: sinogram does not match the configuration; use a separate dataset/cache.")
        for pid in images.patients():
            variants = aug_variants(cfg.data.augment) if pid in cfg.split.train else ["orig"]
            if (pid not in saved.patients() or images.n_slices(pid) != saved.n_slices(pid)
                    or any(not saved.has_variant(pid, v) for v in variants)):
                raise ConfigError(f"{out}: missing slices/variants for {pid}; simulate again with --overwrite.")
        log.info("Skipping %s (already present)", out)
        return out
    n_sim, n_rec = cfg.data.img_size_sim, cfg.data.img_size_rec
    P_sim = projector_for(cfg, geom, n_sim, device, with_adjoint=False)
    P_rec = projector_for(cfg, geom, n_rec, device)
    fbp = fbp_for(cfg, geom, device)
    images = image_store(cfg)
    nz = cfg.noise
    gen = torch.Generator(device="cpu").manual_seed(nz.seed)
    op_meta = {n: load_json(operator_path(geom, n, cfg.paths.operators).with_suffix(".json")) for n in (n_sim, n_rec)}
    attrs = {"geometry": json.dumps(geom.to_dict()), "img_size_sim": n_sim, "img_size_rec": n_rec,
             "fbp_filter": fbp.filt, "noise": json.dumps(vars(nz)), "mu_ref": images.mu_ref,
             "A_sim_sha256": op_meta[n_sim].get("sha256", ""), "A_rec_sha256": op_meta[n_rec].get("sha256", ""),
             "A_rec_opnorm": P_rec.norm, "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "git": git_hash()}
    with write_via_cache(cfg, out) as target, SinoWriter(target, attrs) as w:
        for pid in images.patients():
            n = images.n_slices(pid)
            variants = aug_variants(cfg.data.augment) if pid in cfg.split.train else ["orig"]
            for var in variants:
                slots = w.create(pid, var, n, geom.n_views, geom.n_det, n_rec)
                for s in range(0, n, batch):
                    e = min(n, s + batch)
                    x_sim = torch.from_numpy(apply_aug(images.read(pid, "img512", slice(s, e)), var))
                    x_rec = torch.from_numpy(apply_aug(images.read(pid, "img256", slice(s, e)), var))
                    y = P_sim.raw_forward(x_sim.unsqueeze(1).to(device))
                    y = add_noise(y.cpu(), nz.type, images.mu_ref, gen, nz.I0, nz.sigma).to(device)
                    slots.write(s, e, y[:, 0].cpu().numpy(), fbp(y)[:, 0].cpu().numpy(),
                                P_rec.data_residual(x_rec.unsqueeze(1).to(device), y).cpu().numpy())
            log.info("  %s %s: %d slices, variants %s", geom.name, pid, n, variants)
    return out


def simulate_all(cfg: Config, geoms: list[FanBeamGeometry], device="cpu", batch: int = 8,
                 overwrite: bool = False) -> list[Path]:
    files = [simulate_geometry(cfg, g, device, batch, overwrite) for g in geoms]
    # Data variants (e.g. added noise) reuse the images.h5 of another outputs tree but keep their own manifest:
    # record the reference images in this manifest too, so training can verify their SHA-256.
    man = load_manifest(cfg.paths.manifest)
    shared = [p for p in (cfg.data.images_h5,) if manifest_key(p) not in man]
    update_manifest(cfg.paths.manifest, files + shared)
    return files
