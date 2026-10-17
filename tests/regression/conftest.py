"""Shared fixtures for the regression suite.

The suite pins the behaviour of the code that produced the thesis results, so that refactoring cannot silently
change a number. It needs the finished runs, which live outside the repository:

* ``LACT_RESULTS_DIR``: directory holding the ``vast_results_<date>`` folders (default: ``../ket-qua`` next to the
  repository). Tests that need a missing run are skipped, so the suite still passes on machines without results.
* ``LACT_UPDATE_GOLDEN=1``: rewrite the golden files instead of comparing against them. Only do this on purpose,
  after confirming that a behaviour change is intended.

Projection matrices are rebuilt locally (Siddon when ASTRA is unavailable) and cached in ``.regression_cache/`` so
that repeated runs are fast and use exactly the same operators.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"


def results_root() -> Path:
    return Path(os.environ.get("LACT_RESULTS_DIR", REPO.parent / "ket-qua")).resolve()


def update_golden() -> bool:
    return os.environ.get("LACT_UPDATE_GOLDEN", "") == "1"


def require(path: Path) -> Path:
    """Return ``path`` or skip the calling test when the result it refers to is not on this machine."""
    if not path.exists():
        pytest.skip(f"missing result: {path}")
    return path


@pytest.fixture(scope="session")
def results() -> Path:
    root = results_root()
    if not root.is_dir():
        pytest.skip(f"results directory not found: {root} (set LACT_RESULTS_DIR)")
    return root


@pytest.fixture(scope="session")
def operators_dir() -> Path:
    d = REPO / ".regression_cache" / "operators"
    d.mkdir(parents=True, exist_ok=True)
    return d
