"""Package the source code as dist/kltn-lact.zip for transfer to a server.

    python scripts/pack_source.py              # -> dist/kltn-lact.zip (+ CODE_VERSION inside and at the project root)

Takes only code, configuration, documentation and tests; skips data, results and caches. CODE_VERSION = date-time +
a short SHA-256 of the content, recorded in each training run's env.json (instead of a git hash when there is no .git).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "configs", "tests", "docs", "scripts", "experiments", "pyproject.toml", "requirements.txt",
           "README.md", "Makefile", ".gitignore"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".import_linter_cache", "outputs",
              "outputs_smoke", "data_raw", "dist", ".git"}
SKIP_SUFFIX = {".pyc", ".h5", ".npz", ".pt", ".zip", ".ipynb"}


def files() -> list[Path]:
    out = []
    for name in INCLUDE:
        p = ROOT / name
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_PARTS)
                    and f.suffix not in SKIP_SUFFIX and not f.name.endswith(".egg-info")]
    return [f for f in out if ".egg-info" not in str(f) and f != ROOT / "configs/local.yaml"
            and not f.name.startswith(".env") and not f.is_symlink()]


def main() -> int:
    fs = files()
    h = hashlib.sha256()
    for f in fs:
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    version = f"{_dt.datetime.now():%Y%m%d-%H%M}-{h.hexdigest()[:10]}"
    out = ROOT / "dist" / "kltn-lact.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in fs:
            z.write(f, f.relative_to(ROOT).as_posix())
        z.writestr("CODE_VERSION", version + "\n")
    # the working copy carries the same version code (rsyncing the project directly to a machine still records it)
    (ROOT / "CODE_VERSION").write_text(version + "\n", encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB, {len(fs)} file, CODE_VERSION {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
