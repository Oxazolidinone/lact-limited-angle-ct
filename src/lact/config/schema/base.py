"""Error type and the dict <-> dataclass conversion behind every configuration section.

YAML is read ONCE at the boundary (``load_config``) and converted into the dataclasses of this package; the rest of the
code uses typed attributes (``cfg.train.lr``) rather than strings (``cfg["train"]["lr"]``). Unknown keys and wrong types
are reported at load time together with the key path.
"""
from __future__ import annotations

import types
import typing
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from ..loader import REPO_ROOT, resolve_path


class ConfigError(ValueError):
    """Invalid configuration (unknown key, wrong type, missing required section)."""


def _is_optional(t) -> tuple[bool, Any]:
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(t)) == 2:
            return True, args[0]
    return False, t


def _coerce(t, v, where: str):
    optional, inner = _is_optional(t)
    if v is None:
        if optional or t is Any:
            return None
        raise ConfigError(f"{where}: must not be empty")
    t = inner
    if t is Any:
        return v
    if isinstance(t, type) and hasattr(t, "from_dict") and isinstance(v, dict):
        return t.from_dict(v, where) if _accepts_where(t) else t.from_dict(v)
    if is_dataclass(t):
        return _from_dict(t, v, where)
    origin = typing.get_origin(t)
    if origin in (list, tuple):
        if not isinstance(v, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(v).__name__}")
        (arg,) = typing.get_args(t)[:1] or (Any,)
        return [_coerce(arg, x, f"{where}[{i}]") for i, x in enumerate(v)]
    if origin is dict or t is dict:
        if not isinstance(v, dict):
            raise ConfigError(f"{where}: expected a dict")
        return dict(v)
    if t is Path:
        return resolve_path(v)
    if t is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{where}: expected true/false, got {v!r}")
        return v
    if t is int:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise ConfigError(f"{where}: expected an integer, got {v!r}")
        return int(v)
    if t is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {v!r}")
        return float(v)
    if t is str:
        return str(v)
    return v


def _accepts_where(cls) -> bool:
    import inspect

    try:
        return "where" in inspect.signature(cls.from_dict).parameters
    except (TypeError, ValueError):
        return False


def _from_dict(cls, d: dict | None, where: str):
    d = {} if d is None else d
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - names)
    if unknown:
        raise ConfigError(f"{where}: invalid keys {unknown}; allowed: {sorted(names)}")
    kw = {f.name: _coerce(hints[f.name], d[f.name], f"{where}.{f.name}") for f in fields(cls) if f.name in d}
    try:
        return cls(**kw)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def _path_str(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def to_plain(obj: Any) -> Any:
    """Dataclass/Path -> plain YAML types (used to save each run's config.yaml)."""
    if hasattr(obj, "to_dict") and not isinstance(obj, type):
        return to_plain(obj.to_dict())
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return _path_str(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj
