"""Read and merge YAML files into a raw dict (the step before conversion to typed dataclasses in schema.py).

- Every YAML file may declare ``_base_: [file1.yaml, ...]`` (paths relative to that file); base files are merged
  first and the current file overrides them.
- Dicts are deep-merged; lists and scalars are replaced.
- Command-line overrides look like ``train.lr=3e-4`` or ``model.regularizer.channels=[16,32,64,128]``.
- Relative paths in a configuration are resolved against the repository root (or the LACT_ROOT environment
  variable); environment variables in paths are expanded, e.g. ``${LACT_DRIVE}/outputs`` (used on Colab).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, overload

import yaml

REPO_ROOT = Path(os.environ.get("LACT_ROOT", Path(__file__).resolve().parents[3]))


@overload
def resolve_path(p: None) -> None: ...
@overload
def resolve_path(p: str | os.PathLike) -> Path: ...
def resolve_path(p: str | os.PathLike | None) -> Path | None:
    """Return an absolute path; relative paths are resolved against the repository root."""
    if p is None:
        return None
    q = Path(os.path.expanduser(os.path.expandvars(str(p))))
    return q if q.is_absolute() else (REPO_ROOT / q)


def deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge b into a copy of a (b wins)."""
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | os.PathLike, _seen: set | None = None) -> dict:
    """Read one YAML file, resolving ``_base_`` recursively."""
    path = resolve_path(path)
    _seen = set() if _seen is None else _seen
    if path in _seen:
        raise ValueError(f"Cyclic _base_ at {path}")
    _seen.add(path)
    with open(path, encoding="utf-8") as f:
        cfg = _fix_numbers(yaml.safe_load(f) or {})
    bases = cfg.pop("_base_", []) or []
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        bp = Path(b)
        bp = bp if bp.is_absolute() else (path.parent / bp)
        merged = deep_merge(merged, load_yaml(bp, _seen))
    return deep_merge(merged, cfg)


def set_by_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def get_by_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _fix_numbers(v: Any) -> Any:
    """YAML 1.1 reads '3e-4' as a string; convert numeric-looking strings to float."""
    if isinstance(v, str) and _NUM_RE.match(v.strip()):
        return float(v)
    if isinstance(v, list):
        return [_fix_numbers(x) for x in v]
    if isinstance(v, dict):
        return {k: _fix_numbers(x) for k, x in v.items()}
    return v


def parse_value(s: str) -> Any:
    return _fix_numbers(yaml.safe_load(s))


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``key=value`` overrides; each value is parsed as YAML (numbers, lists, bools, null, ...)."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Invalid override (expected key=value): {ov}")
        k, v = ov.split("=", 1)
        set_by_dotted(cfg, k.strip(), parse_value(v))
    return cfg


def compose(files: Iterable[str | os.PathLike], overrides: Iterable[str] | None = None) -> dict:
    """Merge configuration files in order, then apply the overrides."""
    cfg: dict = {}
    for f in files:
        cfg = deep_merge(cfg, load_yaml(f))
    return apply_overrides(cfg, overrides)


def save_yaml(cfg: dict, path: str | os.PathLike) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def config_hash(cfg: dict, n: int = 10) -> str:
    """Stable hash of a configuration (used to compare runs)."""
    s = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:n]


def diff_configs(a: dict, b: dict, prefix: str = "") -> dict:
    """Return {key: (value in a, value in b)} for the keys that differ."""
    out = {}
    keys = set(a) | set(b)
    for k in sorted(keys):
        ka = f"{prefix}{k}"
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(diff_configs(va, vb, ka + "."))
        elif va != vb:
            out[ka] = (va, vb)
    return out
