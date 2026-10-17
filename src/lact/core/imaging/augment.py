"""Data augmentation applied to IMAGES before sinograms are simulated (keeps image and measurement consistent).

Pure NumPy functions.
"""
from __future__ import annotations

import numpy as np

AUGMENTS = ("none", "hflip")


def variants(augment: str) -> list[str]:
    """Variants generated ahead of time for the training set."""
    if augment not in AUGMENTS:
        raise ValueError(f"augment must be one of {AUGMENTS}")
    return ["orig"] if augment == "none" else ["orig", "hflip"]


def apply(img: np.ndarray, variant: str) -> np.ndarray:
    """img: (..., H, W). 'hflip' = left-right flip (column axis)."""
    if variant == "orig":
        return img
    if variant == "hflip":
        return np.ascontiguousarray(img[..., ::-1])
    raise ValueError(variant)
