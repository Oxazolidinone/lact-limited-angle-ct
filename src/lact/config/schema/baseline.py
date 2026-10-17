"""Classical baselines: FBP filter and the TV-PDHG parameter grid."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FBPConfig:
    filter: str = "ram-lak"
    cutoff: float = 1.0


@dataclass
class TVConfig:
    lambdas: list[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3])
    iters: list[int] = field(default_factory=lambda: [100, 200, 500, 1000])
    n_val_slices: int = 30
    batch_size: int = 8
    init: str = "fbp"


@dataclass
class BaselineConfig:
    fbp: FBPConfig = field(default_factory=FBPConfig)
    tv: TVConfig = field(default_factory=TVConfig)
