"""Root configuration of one run, composed of every section."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any

from ...core.imaging.geometry import FanBeamGeometry
from .base import ConfigError, _from_dict, to_plain
from .baseline import BaselineConfig
from .data import DataConfig, NoiseConfig, SplitConfig
from .model import ModelConfig
from .storage import CacheConfig, PathsConfig
from .train import PrecisionConfig, TrainConfig


@dataclass
class Config:
    """Root configuration of one run."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    geometry: FanBeamGeometry | None = None
    model: ModelConfig | None = None
    exp: str = "dev"
    seed: int = 0
    device: str = "auto"
    run_id: str | None = None
    verify_data: bool = True
    derived: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ load / save
    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return _from_dict(cls, d, "config")

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "geometry" and v is not None:
                out[f.name] = v.to_dict()
            else:
                out[f.name] = to_plain(v)
        return out

    def replace(self, **kw) -> Config:
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------ required sections
    def require_geometry(self) -> FanBeamGeometry:
        if self.geometry is None:
            raise ConfigError("Missing 'geometry' section (add -c configs/geometry/<...>.yaml)")
        return self.geometry

    def require_model(self) -> ModelConfig:
        if self.model is None:
            raise ConfigError("Missing 'model' section (add -c configs/model/<...>.yaml)")
        return self.model
