"""SHA-256 manifest of data files and matrices, checked before training."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...config.loader import REPO_ROOT, resolve_path
from ..io.files import load_json, save_json, sha256_file
from ..system.locks import wait_or_warn


def manifest_key(p: str | Path) -> str:
    """Manifest key of a file: path relative to the repository root, or absolute when outside the repository."""
    p = resolve_path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p.resolve())


def load_manifest(manifest_json: str | Path) -> dict:
    manifest_json = resolve_path(manifest_json)
    return load_json(manifest_json) if manifest_json.exists() else {}


_rel = manifest_key


def update_manifest(manifest_json: str | Path, files: Sequence[str | Path]) -> dict:
    """Add/update the SHA-256 of ``files``. Safe under concurrent updates (parallel pipeline steps): files are hashed
    outside the lock (can be slow), then the manifest is read-merged-written inside the lock so no entry is lost."""
    manifest_json = resolve_path(manifest_json)
    entries = {}
    for f in files:
        f = resolve_path(f)
        if f.exists():
            entries[_rel(f)] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    with wait_or_warn(manifest_json.with_name(f".{manifest_json.name}.lock")):
        man = load_json(manifest_json) if manifest_json.exists() else {}
        man.update(entries)
        save_json(man, manifest_json)
    return man


def verify_manifest(manifest_json: str | Path, only: Sequence[str | Path] | None = None) -> list[str]:
    """Return a list of errors (empty when every file matches)."""
    manifest_json = resolve_path(manifest_json)
    if not manifest_json.exists():
        return [f"No manifest at {manifest_json}"]
    man = load_json(manifest_json)
    keys = man.keys() if only is None else [_rel(resolve_path(f)) for f in only]
    errors = []
    for k in keys:
        if k not in man:
            errors.append(f"{k}: not in the manifest")
            continue
        p = resolve_path(k)
        if not p.exists():
            errors.append(f"{k}: does not exist")
        elif sha256_file(p) != man[k]["sha256"]:
            errors.append(f"{k}: SHA-256 mismatch")
    return errors
