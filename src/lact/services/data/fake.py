"""Generate a SYNTHETIC DICOM dataset (abdominal-slice phantoms) with a Mayo-like directory layout for pipeline tests.

NEVER use this data for thesis results.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...core.imaging.phantom import random_body_phantom
from ...datasets.deeplesion import write_png_hu
from ...datasets.dicom import new_uid, write_ct_slice

log = logging.getLogger("lact")

FAKE_PATIENTS = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310", "L333", "L506"]


def make_fake_dicom(root: str | Path, n_slices: int = 6, size: int = 128, seed: int = 0,
                    patients: list[str] | None = None) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    for pid in patients or FAKE_PATIENTS:
        d = root / pid / "full_3mm"
        d.mkdir(parents=True, exist_ok=True)
        study, series = new_uid(), new_uid()
        for k in range(n_slices):
            x = random_body_phantom(size, rng)
            hu = -1000.0 + x / 0.24 * 1040.0   # 0 -> -1000 HU (air), 0.24 -> ~40 HU (soft tissue)
            write_ct_slice(d / f"{pid}_FD_3_1.CT.{k + 1:04d}.IMA", hu.astype(np.float32), pid, series, study,
                           z=-100.0 + 3.0 * k, inst=k + 1, spacing=400.0 / size)
        log.info("%s: %d slices -> %s", pid, n_slices, d)
    return root


def make_fake_deeplesion(root: str | Path, n_patients: int = 12, n_slices: int = 6, size: int = 128,
                         seed: int = 0) -> Path:
    """SYNTHETIC directory tree + DL_info.csv in the DeepLesion format (16-bit PNG, HU + 32768), 8/2/2 patient split."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = ["File_name,Patient_index,Possibly_noisy,Spacing_mm_px_,Image_size,Train_Val_Test"]
    for p in range(1, n_patients + 1):
        split = 1 if p <= n_patients - 4 else 2 if p <= n_patients - 2 else 3
        for s in (1, 2):                                  # two series per patient
            series = f"{p:06d}_01_{s:02d}"
            for k in range(n_slices // 2):
                idx = 10 * s + k
                x = random_body_phantom(size, rng)
                write_png_hu(root / "Images_png" / series / f"{idx:03d}.png", -1000.0 + x / 0.24 * 1040.0)
            rows.append(f'{series}_{10 * s:03d}.png,{p},0,"0.8, 0.8, 5",\"{size}, {size}\",{split}')
    (root / "DL_info.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    log.info("Synthetic DeepLesion: %d patients -> %s", n_patients, root)
    return root
