"""2-D fan-beam geometry with a flat detector (ASTRA ``fanflat`` conventions). A pure value object, no I/O.

Conventions (matching ``astra.geom_2vec`` for ``fanflat``), for projection angle theta:
    source             S = ( sin(theta) * DSO, -cos(theta) * DSO)
    detector centre    D = (-sin(theta) * DOD,  cos(theta) * DOD)
    detector pixel u   u = ( cos(theta),  sin(theta)) * det_spacing
    detector pixel i lies at D + (i - (n_det - 1) / 2) * u

Angles: theta_i = center - span/2 + (i + 0.5) * step, i = 0..n_views-1 (end point excluded, symmetric about the
centre of the scan range). The field of view is the square [-fov/2, fov/2]^2 (mm).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FanBeamGeometry:
    name: str = "fan"
    span_deg: float = 120.0
    step_deg: float = 0.25
    center_deg: float = 0.0
    n_det: int = 512
    det_spacing_mm: float = 1.875
    dso_mm: float = 600.0
    dod_mm: float = 290.0
    fov_mm: float = 400.0
    extra: dict = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------ construction / conversion
    @classmethod
    def from_dict(cls, d: dict) -> FanBeamGeometry:
        d = dict(d.get("geometry", d))
        names = {f.name for f in dataclasses.fields(cls)} - {"extra"}
        kw = {k: d[k] for k in names if k in d}
        extra = {k: v for k, v in d.items() if k not in names}
        for k in ("span_deg", "step_deg", "center_deg", "det_spacing_mm", "dso_mm", "dod_mm", "fov_mm"):
            if k in kw:
                kw[k] = float(kw[k])
        if "n_det" in kw:
            kw["n_det"] = int(kw["n_det"])
        return cls(**kw, extra=extra)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("extra", None)
        d["n_views"] = self.n_views
        return d

    def replace(self, **kw) -> FanBeamGeometry:
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------ derived quantities
    @property
    def n_views(self) -> int:
        n = self.span_deg / self.step_deg
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"span_deg={self.span_deg} is not a multiple of step_deg={self.step_deg}")
        return round(n)

    @property
    def dsd_mm(self) -> float:
        return self.dso_mm + self.dod_mm

    @property
    def fan_half_angle_deg(self) -> float:
        return math.degrees(math.atan(0.5 * self.n_det * self.det_spacing_mm / self.dsd_mm))

    @property
    def short_scan_deg(self) -> float:
        """Fan-beam data-sufficiency condition: 180° + 2 * gamma_m."""
        return 180.0 + 2.0 * self.fan_half_angle_deg

    @property
    def is_full_scan(self) -> bool:
        return self.span_deg >= 360.0 - 1e-9

    @property
    def fov_radius_covered_mm(self) -> float:
        """Radius of the region crossed by every ray of the fan (DSO * sin(gamma_m))."""
        return self.dso_mm * math.sin(math.radians(self.fan_half_angle_deg))

    def angles_deg(self) -> np.ndarray:
        i = np.arange(self.n_views, dtype=np.float64)
        return self.center_deg - 0.5 * self.span_deg + (i + 0.5) * self.step_deg

    def angles_rad(self) -> np.ndarray:
        return np.deg2rad(self.angles_deg())

    def pixel_size_mm(self, img_size: int) -> float:
        return self.fov_mm / img_size

    def det_coords_mm(self) -> np.ndarray:
        """Detector pixel centre coordinates (mm, on the physical detector)."""
        return (np.arange(self.n_det) - 0.5 * (self.n_det - 1)) * self.det_spacing_mm

    def vectors(self) -> np.ndarray:
        """Array (V, 6): [srcX, srcY, dX, dY, uX, uY] in the ASTRA fanflat_vec convention."""
        th = self.angles_rad()
        s, c = np.sin(th), np.cos(th)
        return np.stack([s * self.dso_mm, -c * self.dso_mm,
                         -s * self.dod_mm, c * self.dod_mm,
                         c * self.det_spacing_mm, s * self.det_spacing_mm], axis=1)

    def describe(self) -> str:
        return (f"{self.name}: span={self.span_deg}°, step={self.step_deg}°, views={self.n_views}, "
                f"det={self.n_det}x{self.det_spacing_mm}mm, DSO={self.dso_mm}, DOD={self.dod_mm}, "
                f"fov={self.fov_mm}mm, gamma_m={self.fan_half_angle_deg:.2f}°, "
                f"short-scan={self.short_scan_deg:.2f}°")
