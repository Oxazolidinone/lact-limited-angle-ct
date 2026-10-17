"""Data sections: raw data and intensity normalisation (``data``), patient split (``split``), simulated noise."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...core.imaging.noise import NOISE_TYPES
from ..loader import resolve_path
from .base import ConfigError


@dataclass
class NormConfig:
    mode: str = "percentile"          # percentile | hu_window | fixed
    percentile: float = 99.9
    hu_max: float = 3071.0
    sample_per_slice: int = 20000
    mu_ref: float | None = None       # fixed: given mu_ref (mm^-1), e.g. Mayo's, so two datasets share one scale

    def __post_init__(self):
        if self.mode not in ("percentile", "hu_window", "fixed"):
            raise ConfigError(f"data.norm.mode must be percentile, hu_window or fixed, got {self.mode}")
        if self.mode == "fixed" and not (self.mu_ref and self.mu_ref > 0):
            raise ConfigError("data.norm.mode = fixed requires data.norm.mu_ref > 0")


@dataclass
class DetectConfig:
    """2DeteCT (lact/datasets/detect): measured data, mode 2, limited angle = the first projections, as in LION."""
    raw_dir: Path = field(default_factory=lambda: resolve_path("data_raw/2detect"))
    records_csv: Path = field(default_factory=lambda: resolve_path("data_raw/2detect_meta/default_data_records.csv"))
    slices_csv: Path = field(default_factory=lambda: resolve_path("configs/data/detect_slices.csv"))
    n_rows: int = 1200                # number of leading projections (0.1° steps) downloaded per slice
    det_bin: int = 8                  # detector binning: 1912 -> 239 (0.5984 mm; detector/pixel ratio as LION at 1024²)
    angle_bin: int = 2                # projection binning: 0.1° -> 0.2° steps
    flip_det: bool = False            # flip the detector axis (see experiments/analysis/detect_geometry_check.py)
    image_transform: str = "rot180"   # flip/rotation of the reference image into the projector frame (verified)

    def __post_init__(self):
        if self.n_rows < 1 or self.det_bin < 1 or self.angle_bin < 1 or N_DET_2DETECT % self.det_bin:
            raise ConfigError(f"data.detect: n_rows, det_bin, angle_bin must be >= 1 and det_bin must divide "
                              f"{N_DET_2DETECT}")
        if self.image_transform not in IMAGE_TRANSFORMS:
            raise ConfigError(f"data.detect.image_transform must be one of {IMAGE_TRANSFORMS}")


N_DET_2DETECT = 1912
IMAGE_TRANSFORMS = ("none", "fliplr", "flipud", "rot90", "rot180", "rot270", "transpose", "antitranspose")


@dataclass
class DataConfig:
    # dicom (Mayo/AAPM) | deeplesion (16-bit PNGs of NIH DeepLesion) | mixed (both in one store: patients with a
    # DLxxxxxx id are read from deeplesion_root, the rest as DICOM from dicom_root) | 2detect (measured CT, data.detect)
    format: str = "dicom"
    dicom_root: Path | None = None    # raw data directory (per-patient DICOM, or the parent of Images_png/)
    deeplesion_root: Path | None = None   # mixed: directory holding DeepLesion's Images_png/
    info_csv: Path | None = None      # deeplesion: DL_info.csv (slice spacing, noisy series)
    max_slices_per_patient: int | None = None   # deeplesion: keep at most N evenly spaced slices per patient
    series_glob: str | None = None
    patients: list[str] = field(default_factory=list)
    expected_slice_thickness_mm: float | None = 3.0
    hu_clip: list[float] = field(default_factory=lambda: [-1024.0, 3071.0])
    mu_water_mm: float = 0.0192
    norm: NormConfig = field(default_factory=NormConfig)
    img_size_sim: int = 512
    img_size_rec: int = 256
    images_h5: Path = field(default_factory=lambda: resolve_path("outputs/data/images.h5"))
    stats_json: Path = field(default_factory=lambda: resolve_path("outputs/data/stats.json"))
    slices_json: Path = field(default_factory=lambda: resolve_path("outputs/data/slices.json"))
    augment: str = "none"             # none | hflip
    detect: DetectConfig = field(default_factory=DetectConfig)

    def __post_init__(self):
        if self.img_size_sim % self.img_size_rec:
            raise ConfigError("data.img_size_sim must be a multiple of data.img_size_rec")
        if self.augment not in ("none", "hflip"):
            raise ConfigError(f"data.augment must be none or hflip, got {self.augment}")
        if self.format not in ("dicom", "deeplesion", "mixed", "2detect"):
            raise ConfigError(f"data.format must be dicom, deeplesion, mixed or 2detect, got {self.format}")
        if self.max_slices_per_patient is not None and self.max_slices_per_patient < 1:
            raise ConfigError("data.max_slices_per_patient must be >= 1 or null")


@dataclass
class SplitConfig:
    name: str = "split"
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.check_disjoint()

    def check_disjoint(self) -> None:
        seen: dict[str, str] = {}
        for name in ("train", "val", "test"):
            for pid in getattr(self, name):
                if pid in seen:
                    raise ConfigError(f"Data leakage: {pid} is in both '{seen[pid]}' and '{name}'")
                seen[pid] = name

    def patients(self, split: str) -> list[str]:
        if split not in ("train", "val", "test"):
            raise ConfigError(f"Invalid split: {split}")
        return list(getattr(self, split))

    def owner(self) -> dict[str, str]:
        return {p: s for s in ("train", "val", "test") for p in getattr(self, s)}


@dataclass
class NoiseConfig:
    """Measurement noise added to sinograms during simulation (``lact data simulate``); none by default."""
    type: str = "none"                # none | poisson | gaussian
    I0: float = 1e5                   # poisson: photons reaching each detector element without an object
    sigma: float = 0.01               # gaussian: standard deviation as a fraction of max|y|
    seed: int = 1234

    def __post_init__(self):
        if self.type not in NOISE_TYPES:
            raise ConfigError(f"noise.type must be one of {NOISE_TYPES}, got {self.type}")
        if self.I0 <= 0 or self.sigma < 0:
            raise ConfigError("noise.I0 must be > 0 and noise.sigma must be >= 0")
