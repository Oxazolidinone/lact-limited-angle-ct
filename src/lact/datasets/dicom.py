"""DICOM adapter: reads the full-dose 3 mm series of Mayo/AAPM 2016 and writes CT slices (for synthetic data)."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("lact")


def _is_candidate_dir(name: str) -> bool:
    n = name.lower().replace("-", "_")
    return ("full" in n or "fd" in n.split("_")) and ("3mm" in n or "3_mm" in n or "_3_" in f"_{n}_")


def find_series_files(root: str | Path, pid: str, series_glob: str | None = None) -> list[Path]:
    """Find the full-dose 3 mm DICOM files of patient ``pid``.

    - With ``series_glob`` (e.g. ``"{pid}/full_3mm/**/*"``): use exactly that pattern.
    - Otherwise: look for a sub-directory (under a directory whose name contains pid) whose name contains
      "full"/"FD" and "3mm".
    """
    root = Path(root)
    if series_glob:
        pattern = str(root / series_glob.format(pid=pid))
        files = [Path(p) for p in glob.glob(pattern, recursive=True)]
        return sorted(p for p in files if p.is_file())
    pdirs = [p for p in root.rglob("*") if p.is_dir() and pid.lower() in p.name.lower()]
    pdirs = pdirs or ([root / pid] if (root / pid).exists() else [])
    cands: list[Path] = []
    for pdir in pdirs:
        for d in [pdir, *[q for q in pdir.rglob("*") if q.is_dir()]]:
            if _is_candidate_dir(d.name):
                cands.append(d)
    if not cands:
        raise FileNotFoundError(
            f"No full-dose 3 mm directory for {pid} under {root}. Set data.series_glob.")
    # pick the directory with the most files (skips empty parent directories)
    best = max(cands, key=lambda d: sum(1 for f in d.iterdir() if f.is_file()))
    return sorted(f for f in best.rglob("*") if f.is_file())


def _read_one(path: Path):
    import pydicom

    try:
        ds = pydicom.dcmread(str(path), force=True)
    except Exception:
        return None
    if "PixelData" not in ds:
        return None
    return ds


def read_series(files: list[Path], expected_thickness: float | None = None, tol: float = 0.25):
    """Read the slices and sort them by z. Returns (hu: (N,H,W) float32, meta: list[dict])."""
    items = []
    for f in files:
        ds = _read_one(f)
        if ds is None:
            continue
        thick = float(getattr(ds, "SliceThickness", np.nan) or np.nan)
        if expected_thickness is not None and np.isfinite(thick) and abs(thick - expected_thickness) > tol:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = float(ipp[2]) if ipp is not None else float(getattr(ds, "SliceLocation", np.nan))
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + inter
        ps = getattr(ds, "PixelSpacing", [np.nan, np.nan])
        items.append((z, int(getattr(ds, "InstanceNumber", 0) or 0), hu, {
            "file": str(f),
            "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "z": z,
            "instance": int(getattr(ds, "InstanceNumber", 0) or 0),
            "slice_thickness": thick,
            "pixel_spacing": [float(ps[0]), float(ps[1])],
            "kernel": str(getattr(ds, "ConvolutionKernel", "")),
            "kvp": float(getattr(ds, "KVP", np.nan) or np.nan),
            "rows": int(ds.Rows), "cols": int(ds.Columns),
        }))
    if not items:
        raise RuntimeError("No DICOM slice could be read")
    items.sort(key=lambda t: (t[0] if np.isfinite(t[0]) else t[1], t[1]))
    # drop duplicate slices (same SOP UID or same z)
    seen_uid, seen_z, keep = set(), set(), []
    for it in items:
        uid, z = it[3]["sop_uid"], round(it[0], 3)
        if (uid and uid in seen_uid) or (np.isfinite(z) and z in seen_z):
            log.warning("Dropping duplicate slice: %s", it[3]["file"])
            continue
        seen_uid.add(uid)
        seen_z.add(z)
        keep.append(it)
    shapes = {it[2].shape for it in keep}
    if len(shapes) != 1:
        raise RuntimeError(f"Slices have different shapes: {shapes}")
    hu = np.stack([it[2] for it in keep]).astype(np.float32)
    meta = [it[3] for it in keep]
    return hu, meta


def summarize_series(pid: str, hu: np.ndarray, meta: list[dict]) -> dict:
    zs = np.array([m["z"] for m in meta])
    dz = np.diff(zs) if len(zs) > 1 else np.array([np.nan])
    return {
        "patient": pid,
        "n_slices": int(hu.shape[0]),
        "shape": list(hu.shape[1:]),
        "slice_thickness": sorted({m["slice_thickness"] for m in meta}),
        "z_spacing_median": float(np.nanmedian(np.abs(dz))),
        "pixel_spacing": sorted({tuple(m["pixel_spacing"]) for m in meta}),
        "kernel": sorted({m["kernel"] for m in meta}),
        "kvp": sorted({m["kvp"] for m in meta}),
        "series_uid": sorted({m["series_uid"] for m in meta}),
        "hu_min": float(hu.min()),
        "hu_p01": float(np.percentile(hu, 1)),
        "hu_p99": float(np.percentile(hu, 99)),
        "hu_max": float(hu.max()),
    }


def write_ct_slice(path: Path, hu: np.ndarray, pid: str, series_uid: str, study_uid: str, z: float, inst: int,
                   spacing: float) -> None:
    """Write one CT slice (HU) as a valid DICOM file; used to create synthetic test data."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    sop = generate_uid()
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = pid
    ds.Modality = "CT"
    ds.Rows, ds.Columns = hu.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.SliceThickness = 3.0
    ds.PixelSpacing = [spacing, spacing]
    ds.ImagePositionPatient = [-hu.shape[1] * spacing / 2, -hu.shape[0] * spacing / 2, z]
    ds.InstanceNumber = inst
    ds.ConvolutionKernel = "B30f"
    ds.KVP = 100
    ds.PixelData = np.clip(hu + 1024, 0, 65535).astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def new_uid() -> str:
    from pydicom.uid import generate_uid

    return str(generate_uid())
