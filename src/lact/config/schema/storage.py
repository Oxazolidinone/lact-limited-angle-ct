"""Where results go (``paths``) and the optional local cache of slow storage (``cache``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..loader import resolve_path
from .base import ConfigError


@dataclass
class PathsConfig:
    outputs: Path = field(default_factory=lambda: resolve_path("outputs"))
    data: Path = field(default_factory=lambda: resolve_path("outputs/data"))
    operators: Path = field(default_factory=lambda: resolve_path("outputs/operators"))
    runs: Path = field(default_factory=lambda: resolve_path("outputs/runs"))
    eval: Path = field(default_factory=lambda: resolve_path("outputs/eval"))
    tables: Path = field(default_factory=lambda: resolve_path("outputs/tables"))
    figures: Path = field(default_factory=lambda: resolve_path("outputs/figures"))
    baselines: Path = field(default_factory=lambda: resolve_path("outputs/baselines"))

    @property
    def checks(self) -> Path:
        return self.outputs / "checks"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.json"

    def sino_file(self, geom_name: str) -> Path:
        return self.data / f"sino_{geom_name}.h5"


@dataclass
class CacheConfig:
    """Local cache: copy data from slow storage (Google Drive) to the local disk and store loaded arrays."""
    enabled: bool = False
    dir: Path | None = None           # e.g. /content/lact_cache on Colab
    arrays: bool = True               # save the Dataset's preloaded arrays as .npy (reloaded with memmap)
    verify: str = "size"              # size | sha256: how to check that the copy still matches the original

    def __post_init__(self):
        if self.verify not in ("size", "sha256"):
            raise ConfigError(f"cache.verify must be size or sha256, got {self.verify}")
        if self.enabled and self.dir is None:
            raise ConfigError("cache.enabled = true but cache.dir is missing")
