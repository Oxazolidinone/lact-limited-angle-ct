"""DeepLesion adapter (NIH Clinical Center; Yan et al., 2018): reads 16-bit PNG slices as HU.

Layout after unpacking ``Images_png_XX.zip``:

    <root>/Images_png/<patient 6 digits>_<study 2 digits>_<series 2 digits>/<slice index>.png

Each series directory holds the slices adjacent to a lesion slice (about 30 mm on each side). HU = pixel - 32768.
``DL_info.csv`` (optional) gives the pixel/slice spacing, the series flagged as noisy and the official
patient-level train/val/test split.
"""
from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("lact")

SERIES_RE = re.compile(r"^(\d{6})_(\d{2})_(\d{2})$")
HU_OFFSET = 32768
INFO_CSV = "DL_info.csv"
OFFICIAL_SPLIT = {1: "train", 2: "val", 3: "test"}
PID_PREFIX = "DL"       # project id: DL000001 (an all-digit string would be read by YAML as a number)
_NAN = float("nan")     # a single object so that sets/statistics never count NaN more than once


def patient_id(series_name: str) -> str:
    return PID_PREFIX + series_name[:6]


def is_patient_id(pid: str) -> bool:
    """Project id of a DeepLesion patient (DL + 6 digits); distinct from Mayo ids (L067, ...) in a joint store."""
    return bool(re.fullmatch(PID_PREFIX + r"\d{6}", str(pid)))


def find_series(root: str | Path) -> dict[str, list[Path]]:
    """{patient id: [series directories, sorted]} for every series found under ``root`` (at any depth)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"DeepLesion directory not found: {root}")
    out: dict[str, list[Path]] = {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and SERIES_RE.match(p.name)):
        out.setdefault(patient_id(d.name), []).append(d)
    return out


def series_slices(series_dir: Path) -> list[Path]:
    """Slice files of a series, sorted by slice index."""
    files = [f for f in series_dir.iterdir() if f.suffix.lower() == ".png" and f.stem.isdigit()]
    return sorted(files, key=lambda f: int(f.stem))


def read_png_hu(path: str | Path) -> np.ndarray:
    """One 16-bit PNG slice -> HU (float32)."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim != 2 or arr.dtype.itemsize < 2:
        raise ValueError(f"{path}: expected a 16-bit grayscale PNG, got shape {arr.shape}, dtype {arr.dtype}")
    return arr.astype(np.float32) - HU_OFFSET


def write_png_hu(path: str | Path, hu: np.ndarray) -> None:
    """Write one HU slice as a 16-bit PNG in the DeepLesion convention (used to create synthetic test data)."""
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(hu) + HU_OFFSET, 0, 65535).astype(np.uint16)).save(path)


def _floats(text) -> list[float]:
    return [float(t) for t in re.split(r"[,\s]+", str(text).strip()) if t]


def load_info(csv_path: str | Path | None) -> dict[str, dict]:
    """Read DL_info.csv -> {series name: {spacing, noisy, split}}; returns an empty dict when the file is missing."""
    if csv_path is None or not Path(csv_path).is_file():
        return {}
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    info: dict[str, dict] = {}
    for r in df.to_dict("records"):
        name = str(r.get("File_name", ""))
        series = name.rsplit("_", 1)[0]
        if not SERIES_RE.match(series):
            continue
        e = info.setdefault(series, {"spacing": None, "noisy": False, "split": None})
        try:
            sp = _floats(r.get("Spacing_mm_px_", ""))
            e["spacing"] = sp if len(sp) == 3 else e["spacing"]
        except ValueError:
            pass
        e["noisy"] = e["noisy"] or str(r.get("Possibly_noisy", "0")).strip() == "1"
        with contextlib.suppress(TypeError, ValueError):
            e["split"] = OFFICIAL_SPLIT.get(int(r.get("Train_Val_Test")), e["split"])
    return info


def official_split(info: dict[str, dict]) -> dict[str, str]:
    """{patient id: train | val | test} from the Train_Val_Test column (patient-level split)."""
    out: dict[str, str] = {}
    for series, e in info.items():
        if e.get("split"):
            pid = patient_id(series)
            if out.setdefault(pid, e["split"]) != e["split"]:
                raise ValueError(f"DL_info.csv: patient {pid} belongs to two different splits")
    return out


def patient_slices(series_dirs: list[Path], info: dict[str, dict] | None = None, size: int = 512,
                   skip_noisy: bool = True) -> list[tuple[Path, Path]]:
    """Usable [(series directory, slice file)] of one patient: drops series flagged as noisy and series that are not
    ``size`` x ``size`` (checked on the first slice of each series)."""
    from PIL import Image

    info = info or {}
    out = []
    for d in series_dirs:
        if skip_noisy and info.get(d.name, {}).get("noisy"):
            continue
        files = series_slices(d)
        if not files:
            continue
        with Image.open(files[0]) as im:
            if im.size != (size, size):
                continue
        out += [(d, f) for f in files]
    return out


def pick_evenly(n: int, k: int | None) -> list[int]:
    """Indices of at most ``k`` evenly spaced elements out of ``n`` (order preserved)."""
    if k is None or n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def read_patient(series_dirs: list[Path], info: dict[str, dict] | None = None, max_slices: int | None = None,
                 size: int = 512) -> tuple[np.ndarray, list[dict]]:
    """Return (hu: (N, size, size) float32, meta: list[dict]) in the same form as ``dicom.read_series``.

    ``max_slices``: keep at most that many EVENLY SPACED slices over all slices of the patient (adjacent slices,
    especially in 1 mm series, are nearly identical). z is the slice position (mm) within a series, accumulated
    across series.
    """
    info = info or {}
    items = patient_slices(series_dirs, info, size)
    if not items:
        raise RuntimeError(f"No usable {size}x{size} slice in {[d.name for d in series_dirs]}")
    chosen = [items[i] for i in pick_evenly(len(items), max_slices)]
    hu, meta = [], []
    z0, prev_series, last_z = 0.0, None, 0.0
    for k, (d, f) in enumerate(chosen):
        sp = info.get(d.name, {}).get("spacing") or [_NAN] * 3
        dz = sp[2] if np.isfinite(sp[2]) and sp[2] > 0 else 1.0
        if d.name != prev_series:                       # new series: continue after the previous one
            z0, prev_series = (last_z + 10.0 if prev_series is not None else 0.0) - int(f.stem) * dz, d.name
        last_z = z0 + int(f.stem) * dz
        hu.append(read_png_hu(f))
        meta.append({"file": str(f), "sop_uid": f"{d.name}/{f.stem}", "series_uid": d.name, "z": float(last_z),
                     "instance": k + 1, "slice_thickness": sp[2], "pixel_spacing": [sp[0], sp[1]],
                     "kernel": "", "kvp": _NAN, "rows": size, "cols": size})
    return np.stack(hu).astype(np.float32), meta
