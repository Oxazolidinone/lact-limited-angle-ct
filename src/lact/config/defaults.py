"""Configuration entry point: default files -> configs/local.yaml (if present) -> extra files -> overrides."""
from __future__ import annotations

from collections.abc import Iterable

from ..core.imaging.geometry import FanBeamGeometry
from .loader import compose, load_yaml, resolve_path
from .schema import Config, ConfigError

DEFAULT_FILES = [
    "configs/paths.yaml",
    "configs/data/mayo_3mm.yaml",
    "configs/data/split_7_1_2.yaml",
    "configs/train/default.yaml",
    "configs/baseline/fbp.yaml",
    "configs/baseline/tv.yaml",
]
LOCAL_FILE = "configs/local.yaml"


def config_files(extra: Iterable[str] = (), use_defaults: bool = True) -> list[str]:
    files = list(DEFAULT_FILES) if use_defaults else []
    if use_defaults and resolve_path(LOCAL_FILE).exists():
        files.append(LOCAL_FILE)
    return files + list(extra)


def load_raw(extra: Iterable[str] = (), overrides: Iterable[str] = (), use_defaults: bool = True) -> dict:
    return compose(config_files(extra, use_defaults), overrides)


def load_config(extra: Iterable[str] = (), overrides: Iterable[str] = (), use_defaults: bool = True) -> Config:
    """Load and validate a configuration, returning a typed ``Config``."""
    return Config.from_dict(load_raw(extra, overrides, use_defaults))


def load_config_file(path) -> Config:
    """Load an already-merged config.yaml (e.g. the one saved with a training run)."""
    return Config.from_dict(load_yaml(path))


def geometry_file(name_or_path: str) -> str:
    """'fan_a120' -> 'configs/geometry/fan_a120.yaml'; .yaml paths are returned unchanged."""
    return name_or_path if name_or_path.endswith(".yaml") else f"configs/geometry/{name_or_path}.yaml"


def load_geometry(name_or_path: str) -> FanBeamGeometry:
    d = load_yaml(geometry_file(name_or_path))
    if "geometry" not in d:
        raise ConfigError(f"{name_or_path}: missing 'geometry' section")
    return FanBeamGeometry.from_dict(d["geometry"])


def model_file(name_or_path: str) -> str:
    return name_or_path if name_or_path.endswith(".yaml") else f"configs/model/{name_or_path}.yaml"
