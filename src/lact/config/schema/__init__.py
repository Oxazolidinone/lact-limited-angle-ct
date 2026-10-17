"""Typed configuration (dataclasses) for the whole application, one module per group of sections."""
from .base import ConfigError, to_plain
from .baseline import BaselineConfig, FBPConfig, TVConfig
from .data import IMAGE_TRANSFORMS, N_DET_2DETECT, DataConfig, DetectConfig, NoiseConfig, NormConfig, SplitConfig
from .model import COMPARED_TAGS, ModelConfig, RegularizerConfig
from .root import Config
from .storage import CacheConfig, PathsConfig
from .train import PrecisionConfig, TrainConfig

__all__ = [
    "COMPARED_TAGS", "IMAGE_TRANSFORMS", "N_DET_2DETECT", "BaselineConfig", "CacheConfig", "Config", "ConfigError",
    "DataConfig", "DetectConfig", "FBPConfig", "ModelConfig", "NoiseConfig", "NormConfig", "PathsConfig",
    "PrecisionConfig", "RegularizerConfig", "SplitConfig", "TVConfig", "TrainConfig", "to_plain",
]
