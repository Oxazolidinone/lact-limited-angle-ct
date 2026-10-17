"""Download the Mayo Clinic / AAPM Low Dose CT Grand Challenge (2016) dataset from AAPM's public Box folder.

Only the full-dose, 3 mm, B30-kernel images of the 10 patients (L067 ... L506) are needed: one ~716 MB zip file that
unpacks to ``<dest>/full_3mm/<pid>/full_3mm/*.IMA`` (2378 slices), exactly the layout ``data.dicom_root`` expects.
Source: https://www.aapm.org/grandchallenge/lowdosect/ ("2021 Update" section). Cite reference [7] of the proposal.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import shutil
import zipfile
from pathlib import Path

from ..infra.io.files import load_json, save_json
from .box import BoxShare

log = logging.getLogger("lact")

MAYO_SHARE = "eaw4jddb53keg1bptavvvd1sf4x3pe9h"
MAYO_HOST = "aapm.app.box.com"
MAYO_FILES = {
    "full_3mm": "Patient_Data/Training_Image_Data/3mm B30/full_3mm.zip",
}
# slice counts verified in full_3mm.zip (2026-09)
EXPECTED_SLICES = {"L067": 224, "L096": 330, "L109": 128, "L143": 234, "L192": 240, "L286": 210, "L291": 343,
                   "L310": 214, "L333": 244, "L506": 211}
DONE_MARKER = "DOWNLOAD_OK.json"


def count_slices(root: Path) -> dict[str, int]:
    """Number of DICOM files per patient in ``root/<pid>/full_3mm``."""
    out = {}
    for pid in EXPECTED_SLICES:
        d = root / pid / "full_3mm"
        out[pid] = sum(1 for f in d.iterdir() if f.is_file()) if d.exists() else 0
    return out


def _extract(zip_path: Path, dest: Path, top: str) -> Path:
    """Unpack into a temporary directory next to the target, then rename: an interruption never leaves a half-unpacked
    directory under the real name."""
    final = dest / top
    tmp = dest / f".{top}.extracting"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        for i, m in enumerate(members, 1):
            z.extract(m, tmp)                 # the CRC is checked as each file is read
            if i % 500 == 0:
                log.info("  unpacked %d/%d files", i, len(members))
    src = tmp / top
    if final.exists():
        shutil.rmtree(final)
    os.replace(src, final)
    shutil.rmtree(tmp, ignore_errors=True)
    return final


def download_mayo(dest: str | os.PathLike, variant: str = "full_3mm", extract: bool = True,
                  keep_zip: bool = True, force: bool = False) -> Path:
    """Download (resumable) and unpack; skipped when complete. Returns the directory to use as ``data.dicom_root``."""
    dest = Path(dest)
    root = dest / variant
    marker = root / DONE_MARKER
    if marker.exists() and not force:
        counts = count_slices(root)
        if counts == EXPECTED_SLICES:
            log.info("Data already complete at %s (%d slices)", root, sum(counts.values()))
            return root
        log.warning("Slices missing compared with the previous download (%s); downloading/unpacking again", counts)
    box = BoxShare(MAYO_SHARE, MAYO_HOST)
    item = box.find(MAYO_FILES[variant])
    zip_path = box.download(item, dest / "zips" / item.name)
    if not extract:
        return zip_path
    log.info("Unpacking %s -> %s", zip_path.name, dest)
    root = _extract(zip_path, dest, variant)
    counts = count_slices(root)
    if counts != EXPECTED_SLICES:
        raise RuntimeError(f"Slice counts after unpacking do not match: {counts} (expected {EXPECTED_SLICES})")
    save_json({"source": f"https://{MAYO_HOST}/s/{MAYO_SHARE}", "path": MAYO_FILES[variant],
               "zip_bytes": item.size, "slices": counts, "total": sum(counts.values()),
               "downloaded_at": _dt.datetime.now().isoformat(timespec="seconds")}, marker)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
    log.info("Done: %d slices of %d patients at %s", sum(counts.values()), len(counts), root)
    return root


def download_info(dest: str | os.PathLike, variant: str = "full_3mm") -> dict:
    marker = Path(dest) / variant / DONE_MARKER
    return load_json(marker) if marker.exists() else {}
