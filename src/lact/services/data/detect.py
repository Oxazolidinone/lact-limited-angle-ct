"""2DeteCT use cases: normalise the reference images into the image store and turn the measured sinograms into the
sinogram stores (with their FBP), using the raw-data adapters of ``lact.datasets.detect``."""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import numpy as np

from ...datasets.detect import first_angle_deg, load_slice, measured_sinogram, read_slices, reference_mu
from ...datasets.detect.raw import (
    DET_PIX_MM,
    FOV_MM,
    N_DET_RAW,
    REC_VOXEL_MM,
    SDD_MM,
    SOD_MM,
    STEP_RAW_DEG,
    slice_name,
)

log = logging.getLogger("lact")


def compute_mu_ref(cfg, rows: list[dict], seed: int = 0) -> dict:
    """mu_ref = the ``data.norm.percentile`` percentile of mu over pixels sampled from the TRAIN slices (as for
    Mayo)."""
    d, n = cfg.data.detect, cfg.data.norm
    if n.mode == "fixed":
        return {"mode": "fixed", "mu_ref": float(n.mu_ref)}
    if n.mode != "percentile":
        raise ValueError("2DeteCT supports only data.norm.mode = percentile or fixed")
    rng = np.random.default_rng(seed)
    vals = []
    for r in rows:
        if r["split"] == "train":
            mu = reference_mu(load_slice(d.raw_dir, r["slice"]), "none").ravel()
            vals.append(rng.choice(mu, min(n.sample_per_slice, mu.size), replace=False))
    allv = np.concatenate(vals)
    return {"mode": "percentile", "percentile": n.percentile, "mu_ref": float(np.percentile(allv, n.percentile)),
            "mu_max_sampled": float(allv.max()), "n_samples": int(allv.size)}


def preprocess(cfg, overwrite: bool = False) -> dict:
    """Reference image -> mu -> [0, 1] (divide by mu_ref, clip) on the 1024 grid -> block means down to img512 and
    img256 -> images.h5, stats.json, slices.json. Each "patient" is a sample holding only its drawn slices."""
    from ...core.imaging.intensity import block_mean
    from ...infra.io.files import save_json
    from ...infra.io.hdf5_store import ImageWriter
    from ...infra.stores.manifest import update_manifest
    from ...infra.system.runtime import git_hash
    from .staging import write_via_cache

    data, d = cfg.data, cfg.data.detect
    if data.images_h5.exists() and not overwrite:
        raise FileExistsError(f"{data.images_h5} already exists (use --overwrite to rewrite it)")
    rows = read_slices(d.slices_csv)
    stats = compute_mu_ref(cfg, rows)
    mu_ref = stats["mu_ref"]
    stats.update({"dataset": "2detect", "img_size_sim": data.img_size_sim, "img_size_rec": data.img_size_rec,
                  "rec_voxel_mm": REC_VOXEL_MM, "fov_mm": FOV_MM, "image_transform": d.image_transform,
                  "train_patients": cfg.split.train})
    log.info("2DeteCT: mu_ref = %.6f mm^-1 (percentile %s; sampled max %.6f)", mu_ref, stats.get("percentile"),
             stats.get("mu_max_sampled", float("nan")))
    by_pid: dict[str, list[int]] = {}
    for r in rows:
        by_pid.setdefault(r["patient"], []).append(r["slice"])
    patients = list(dict.fromkeys(data.patients or list(by_pid)))
    f_sim, f_rec = 1024 // data.img_size_sim, 1024 // data.img_size_rec
    attrs = {"mu_ref": mu_ref, "mu_water_mm": data.mu_water_mm, "norm_mode": stats["mode"], "dataset": "2detect",
             "img_size_sim": data.img_size_sim, "img_size_rec": data.img_size_rec,
             "created": _dt.datetime.now().isoformat(timespec="seconds"), "git": git_hash()}
    slices: dict[str, dict] = {}
    clipped: list[float] = []
    with write_via_cache(cfg, data.images_h5) as target, ImageWriter(target, attrs) as w:
        for pid in patients:
            ids = sorted(by_pid[pid])
            mu = np.stack([reference_mu(load_slice(d.raw_dir, i), d.image_transform) for i in ids])
            clipped.append(float((mu > mu_ref).mean()))
            x = np.clip(mu / mu_ref, 0.0, 1.0)
            w.add_patient(pid, block_mean(x, f_sim).astype(np.float32), block_mean(x, f_rec).astype(np.float32),
                          [float(i) for i in ids], ids, [slice_name(i) for i in ids], {"source": "2DeteCT mode2"})
            slices[pid] = {"n": len(ids), "z": [float(i) for i in ids], "sop_uid": [slice_name(i) for i in ids]}
    stats["clipped_fraction_mean"] = float(np.mean(clipped))
    save_json(stats, data.stats_json)
    save_json({"split": {k: cfg.split.patients(k) for k in ("train", "val", "test")}, "patients": slices},
              data.slices_json)
    update_manifest(cfg.paths.manifest, [data.images_h5, data.stats_json, data.slices_json])
    log.info("2DeteCT: images.h5 done, %d samples, %d slices", len(slices), sum(v["n"] for v in slices.values()))
    return {"stats": stats}


def check_geometry(cfg, geom) -> None:
    d = cfg.data.detect
    want = {"n_det": N_DET_RAW // d.det_bin, "det_spacing_mm": DET_PIX_MM * d.det_bin, "dso_mm": SOD_MM,
            "dod_mm": SDD_MM - SOD_MM, "step_deg": STEP_RAW_DEG * d.angle_bin, "fov_mm": FOV_MM}
    bad = {k: (getattr(geom, k), v) for k, v in want.items() if abs(float(getattr(geom, k)) - v) > 1e-4}
    a0 = float(geom.angles_deg()[0])
    if abs(a0 - first_angle_deg(d.angle_bin)) > 1e-6:
        bad["first_angle_deg"] = (a0, first_angle_deg(d.angle_bin))
    if geom.n_views * d.angle_bin > d.n_rows and not geom.is_full_scan:
        bad["n_views"] = (geom.n_views, f"<= {d.n_rows // d.angle_bin}")
    if bad:
        raise ValueError(f"Geometry {geom.name} does not match the 2DeteCT data (have, need): {bad}")


def measure_geometry(cfg, geom, device="cpu", batch: int = 8, overwrite: bool = False) -> Path:
    """Replaces ``simulate_geometry`` for measured data: the measured sinogram (divided by mu_ref to share the scale of
    A x for normalised images), the FBP x^0 and the reference image's data residual, in the simulated sinogram
    format."""
    import torch

    from ...infra.io.files import load_json
    from ...infra.io.hdf5_store import SinoStore, SinoWriter
    from ...infra.operators.operator_store import operator_path
    from ...infra.stores.manifest import update_manifest
    from ...infra.system.runtime import git_hash
    from ..factories import fbp_for, image_store, projector_for
    from .staging import write_via_cache

    d = cfg.data.detect
    check_geometry(cfg, geom)
    out = cfg.paths.sino_file(geom.name)
    images = image_store(cfg)
    key = {"source": "2detect-measured", "det_bin": d.det_bin, "angle_bin": d.angle_bin, "flip_det": d.flip_det,
           "image_transform": d.image_transform}
    if out.exists() and not overwrite:
        a = SinoStore(out).attrs
        if (abs(float(a.get("mu_ref", -1)) - images.mu_ref) < 1e-12 and json.loads(a.get("measured", "{}")) == key
                and set(SinoStore(out).patients()) == set(images.patients())):
            log.info("Skipping %s (already present)", out)
            return out
        raise ValueError(f"{out} exists but with different settings; run again with --overwrite")
    n_rec = cfg.data.img_size_rec
    P_rec = projector_for(cfg, geom, n_rec, device)
    fbp = fbp_for(cfg, geom, device)
    meta = load_json(operator_path(geom, n_rec, cfg.paths.operators).with_suffix(".json"))
    attrs = {"geometry": json.dumps(geom.to_dict()), "img_size_sim": cfg.data.img_size_sim, "img_size_rec": n_rec,
             "fbp_filter": fbp.filt, "noise": json.dumps({"type": "measured"}), "measured": json.dumps(key),
             "mu_ref": images.mu_ref, "A_sim_sha256": "", "A_rec_sha256": meta.get("sha256", ""),
             "A_rec_opnorm": P_rec.norm, "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "git": git_hash()}
    with write_via_cache(cfg, out) as target, SinoWriter(target, attrs) as w:
        for pid in images.patients():
            ids = [int(v) for v in images.read(pid, "instance")]
            n = len(ids)
            slots = w.create(pid, "orig", n, geom.n_views, geom.n_det, n_rec)
            for s in range(0, n, batch):
                e = min(n, s + batch)
                y = np.stack([measured_sinogram(load_slice(d.raw_dir, i), geom.n_views, d.det_bin, d.angle_bin,
                                                d.flip_det) for i in ids[s:e]]) / images.mu_ref
                yt = torch.from_numpy(y.astype(np.float32)).unsqueeze(1).to(device)
                x = torch.from_numpy(images.read(pid, "img256", slice(s, e))).unsqueeze(1).to(device)
                with torch.no_grad():
                    slots.write(s, e, y.astype(np.float32), fbp(yt)[:, 0].cpu().numpy(),
                                P_rec.data_residual(x, yt).cpu().numpy())
            log.info("  %s %s: %d slices (measured)", geom.name, pid, n)
    update_manifest(cfg.paths.manifest, [out])
    return out
