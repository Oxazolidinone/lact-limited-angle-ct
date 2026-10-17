"""Every saved run configuration must still load, and load to the same typed configuration.

Finished runs store the merged configuration in ``config.yaml``; evaluation, reporting and resuming rebuild models
from it. A renamed key, a changed default or a stricter validator would break old runs, so the canonical JSON of each
loaded configuration is pinned by its SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lact.config import load_config_file, to_plain

from .conftest import GOLDEN, update_golden

GOLDEN_FILE = GOLDEN / "run_configs.json"
GRIDS = {p.stem for p in Path(__file__).resolve().parents[2].glob("configs/experiment/*.yaml")}
MODELS = ("learn_densenet", "regformer", "learn_orig", "fbpconvnet", "lpd", "dpma")


def _in_scope(rel: str) -> bool:
    """Runs of the grids in configs/experiment, with the models of this repository."""
    exp, run = Path(rel).parts[-3:-1]
    return exp in GRIDS and run.startswith(tuple(f"{exp}_{m}_" for m in MODELS))


def _digest(cfg) -> str:
    return hashlib.sha256(json.dumps(to_plain(cfg), sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.regression
def test_saved_run_configs_load_unchanged(results):
    files = [f for f in sorted(results.glob("vast_results_*/outputs*/runs/*/*/config.yaml"))
             if _in_scope(str(f.relative_to(results)))]
    if not files:
        pytest.skip("no saved run configurations found")
    digests = {str(f.relative_to(results)): _digest(load_config_file(f)) for f in files}
    if update_golden():
        GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"golden digests written for {len(digests)} runs")
    expected = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    changed = [k for k, v in expected.items() if k in digests and digests[k] != v]
    assert not changed, f"{len(changed)} run configurations load differently, e.g. {changed[:3]}"
