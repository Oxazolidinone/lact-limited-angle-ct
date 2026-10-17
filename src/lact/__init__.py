"""lact: benchmarking regularization blocks in LEARN deep unrolling for limited-angle CT.

Layers, with dependencies pointing downwards only (see docs/ARCHITECTURE.md):

    cli -> services -> datasets -> (infra | viz) -> config -> core

The main objects are importable from the top level and are loaded on first use, so ``import lact.core...`` stays
light::

    from lact import Config, load_config, FanBeamGeometry, SparseFanBeam, FanBeamFBP, model_for
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

# public name -> module that defines it (imported lazily by __getattr__)
_API = {
    "Config": "lact.config",
    "load_config": "lact.config",
    "FanBeamGeometry": "lact.core.imaging.geometry",
    "SparseFanBeam": "lact.core.operators",
    "FanBeamFBP": "lact.core.operators",
    "model_for": "lact.services.factories",
}

__all__ = ["Config", "FanBeamFBP", "FanBeamGeometry", "SparseFanBeam", "__version__", "load_config", "model_for"]

if TYPE_CHECKING:
    from .config import Config, load_config
    from .core.imaging.geometry import FanBeamGeometry
    from .core.operators import FanBeamFBP, SparseFanBeam
    from .services.factories import model_for


def __getattr__(name: str) -> Any:
    if name in _API:
        value = getattr(importlib.import_module(_API[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lact' has no attribute {name!r}")
