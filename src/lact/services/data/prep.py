"""Data preparation use case: inspect raw data, compute mu_ref, normalise images and write images.h5."""
from __future__ import annotations

import datetime as _dt
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Config, SplitConfig
from ...core.imaging.intensity import block_mean, hu_to_mu, mu_to_hu, normalize_mu, resize_square
from ...datasets import deeplesion
from ...datasets.dicom import find_series_files, read_series, summarize_series
from ...infra.io.files import load_json, save_json
from ...infra.io.hdf5_store import ImageStore, ImageWriter
from ...infra.stores.manifest import update_manifest
from ...infra.system.runtime import git_hash
from .staging import write_via_cache

log = logging.getLogger("lact")


@lru_cache(maxsize=4)
def _deeplesion_index(root: Path, info_csv: Path | None) -> tuple[dict, dict]:
    """(series per patient, DL_info.csv information): the directory is scanned once per preprocessing run."""
    return deeplesion.find_series(root), deeplesion.load_info(info_csv)


def patient_source(cfg: Config, pid: str) -> tuple[str, Path]:
    """(format, root directory) of one patient. In a joint store (``data.format = mixed``) DLxxxxxx ids are DeepLesion
    (read from ``data.deeplesion_root``) and the others DICOM (``data.dicom_root``)."""
    d = cfg.data
    fmt, root, key = d.format, d.dicom_root, "data.dicom_root"
    if fmt == "mixed":
        fmt = "deeplesion" if deeplesion.is_patient_id(pid) else "dicom"
        if fmt == "deeplesion":
            root, key = d.deeplesion_root, "data.deeplesion_root"
    if root is None:
        raise ValueError(f"{key} is not set (see configs/local.yaml)")
    return fmt, root


def _read_patient(cfg: Config, pid: str):
    d = cfg.data
    fmt, root = patient_source(cfg, pid)
    lo, hi = d.hu_clip
    if fmt == "deeplesion":
        series, info = _deeplesion_index(root, d.info_csv)
        if pid not in series:
            raise FileNotFoundError(f"Patient {pid} not found in {root}")
        hu, meta = deeplesion.read_patient(series[pid], info, d.max_slices_per_patient, d.img_size_sim)
        return np.clip(hu, lo, hi), meta, len(meta)
    files = find_series_files(root, pid, d.series_glob)
    hu, meta = read_series(files, d.expected_slice_thickness_mm)
    return np.clip(hu, lo, hi), meta, len(files)


def inspect_dicom(cfg: Config) -> pd.DataFrame:
    """Slice count, thickness, kernel and HU range per patient; writes dicom_inspect.csv/.json."""
    owner = cfg.split.owner()
    rows = []
    for pid in cfg.data.patients:
        hu, meta, n_files = _read_patient(cfg, pid)
        s = summarize_series(pid, hu, meta)
        s.update({"n_files": n_files, "split": owner.get(pid, "-")})
        rows.append(s)
        log.info("%s [%s]: %d files -> %d slices, thickness %s, spacing %s, kernel %s, HU [%.0f, %.0f]", pid,
                 s["split"],
                 n_files, s["n_slices"], s["slice_thickness"], s["pixel_spacing"], s["kernel"], s["hu_min"],
                 s["hu_max"])
    df = pd.DataFrame(rows)
    cfg.paths.data.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.paths.data / "dicom_inspect.csv", index=False)
    save_json(rows, cfg.paths.data / "dicom_inspect.json")
    return df


def compute_mu_ref(cfg: Config, seed: int = 0) -> dict:
    """mu_ref according to data.norm: a percentile over pixels sampled from the TRAINING split, a fixed HU window, or
    a given value (``fixed``: reuse another dataset's mu_ref so both datasets share one intensity scale)."""
    n, mu_w = cfg.data.norm, cfg.data.mu_water_mm
    if n.mode == "hu_window":
        return {"mode": n.mode, "mu_ref": float(hu_to_mu(np.array(n.hu_max), mu_w)), "hu_max": n.hu_max}
    if n.mode == "fixed":
        assert n.mu_ref is not None
        return {"mode": n.mode, "mu_ref": float(n.mu_ref),
                "hu_at_mu_ref": float(mu_to_hu(np.array(n.mu_ref), mu_w))}
    rng = np.random.default_rng(seed)
    samples = []
    for pid in cfg.split.train:
        hu, _, _ = _read_patient(cfg, pid)
        mu = hu_to_mu(hu, mu_w).reshape(hu.shape[0], -1)
        idx = rng.integers(0, mu.shape[1], size=(mu.shape[0], min(n.sample_per_slice, mu.shape[1])))
        samples.append(np.take_along_axis(mu, idx, axis=1).ravel())
        log.info("  mu_ref: sampled %s (%d slices)", pid, hu.shape[0])
    allv = np.concatenate(samples)
    mu_ref = float(np.percentile(allv, n.percentile))
    return {"mode": n.mode, "mu_ref": mu_ref, "percentile": n.percentile, "n_samples": int(allv.size),
            "hu_at_mu_ref": float(mu_to_hu(np.array(mu_ref), mu_w))}


def check_uid_disjoint(cfg: Config) -> None:
    """No SOP/Series UID may appear in two different splits."""
    info = load_json(cfg.data.slices_json)["patients"]
    owner: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for pid in cfg.split.patients(name):
            for uid in list(info.get(pid, {}).get("sop_uid", [])) + list(info.get(pid, {}).get("series_uid", [])):
                if uid and owner.setdefault(uid, name) != name:
                    raise ValueError(f"UID {uid} appears in both '{owner[uid]}' and '{name}'")


def count_slices(store: ImageStore, split: SplitConfig) -> dict:
    return {s: {p: store.n_slices(p) for p in split.patients(s) if p in store.patients()}
            for s in ("train", "val", "test")}


def preprocess(cfg: Config, overwrite: bool = False) -> dict:
    """DICOM -> HU -> mu -> [0,1] on the simulation and reconstruction grids -> images.h5, stats.json, slices.json."""
    if cfg.data.format == "2detect":
        from . import detect
        return detect.preprocess(cfg, overwrite)
    d = cfg.data
    if d.images_h5.exists() and not overwrite:
        raise FileExistsError(f"{d.images_h5} already exists (use --overwrite to rewrite it)")
    log.info("Step 1/2: computing mu_ref from the training split %s", cfg.split.train)
    stats = compute_mu_ref(cfg)
    stats.update({"mu_water_mm": d.mu_water_mm, "hu_clip": list(d.hu_clip), "train_patients": cfg.split.train,
                  "img_size_sim": d.img_size_sim, "img_size_rec": d.img_size_rec})
    mu_ref = stats["mu_ref"]
    log.info("  mu_ref = %.6f mm^-1", mu_ref)

    log.info("Step 2/2: normalising and writing %s", d.images_h5)
    patients = list(dict.fromkeys(d.patients or (cfg.split.train + cfg.split.val + cfg.split.test)))
    slices, summaries = {}, {}
    attrs = {"mu_ref": mu_ref, "mu_water_mm": d.mu_water_mm, "norm_mode": stats["mode"],
             "img_size_sim": d.img_size_sim, "img_size_rec": d.img_size_rec,
             "created": _dt.datetime.now().isoformat(timespec="seconds"), "git": git_hash()}
    with write_via_cache(cfg, d.images_h5) as target, ImageWriter(target, attrs) as w:
        for pid in patients:
            hu, meta, _ = _read_patient(cfg, pid)
            summ = summarize_series(pid, hu, meta)
            summaries[pid] = summ
            x = normalize_mu(hu_to_mu(hu, d.mu_water_mm), mu_ref)
            x_sim = resize_square(x, d.img_size_sim).astype(np.float32)
            x_rec = block_mean(x_sim, d.img_size_sim // d.img_size_rec).astype(np.float32)
            w.add_patient(pid, x_sim, x_rec, [m["z"] for m in meta], [m["instance"] for m in meta],
                          [m["sop_uid"] for m in meta],
                          {"series_uid": ";".join(summ["series_uid"]), "pixel_spacing": str(summ["pixel_spacing"]),
                           "kernel": ";".join(summ["kernel"])})
            slices[pid] = {"n": int(x.shape[0]), "z": [m["z"] for m in meta], "sop_uid": [m["sop_uid"] for m in meta],
                           "series_uid": summ["series_uid"]}
            log.info("  %s: %d slices, HU [%.0f, %.0f]", pid, x.shape[0], hu.min(), hu.max())
    save_json(stats, d.stats_json)
    save_json({"split": {k: cfg.split.patients(k) for k in ("train", "val", "test")}, "patients": slices},
              d.slices_json)
    save_json(summaries, d.stats_json.with_name("dicom_summary.json"))
    check_uid_disjoint(cfg)
    update_manifest(cfg.paths.manifest, [d.images_h5, d.stats_json, d.slices_json])
    log.info("Slice counts: %s", count_slices(ImageStore(d.images_h5), cfg.split))
    return {"stats": stats, "summaries": summaries}
