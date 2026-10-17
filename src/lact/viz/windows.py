"""HU display windows for CT images."""
from __future__ import annotations

import numpy as np

WINDOWS = {
    "soft": (400.0, 40.0),     # abdominal soft tissue W400 / L40
    "bone": (1500.0, 300.0),   # bone W1500 / L300
    "lung": (1500.0, -600.0),  # lung W1500 / L-600
    "wide": (2000.0, 0.0),
}


def to_hu(x: np.ndarray, mu_ref: float, mu_water: float) -> np.ndarray:
    return (np.asarray(x, np.float64) * mu_ref / mu_water - 1.0) * 1000.0


def apply_window(hu: np.ndarray, width: float, level: float) -> np.ndarray:
    lo, hi = level - width / 2.0, level + width / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def window_image(x: np.ndarray, mu_ref: float, mu_water: float, name: str = "soft") -> np.ndarray:
    W, L = WINDOWS[name]
    return apply_window(to_hu(x, mu_ref, mu_water), W, L)
