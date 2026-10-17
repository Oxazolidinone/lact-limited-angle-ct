"""The paper tables generated from the finished evaluation must not change.

Runs the paper-number generator on the per-slice evaluation results and compares its LaTeX output with the copy
recorded before the refactor. ``numbers.tex`` starts with a generation timestamp, which is ignored.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import GOLDEN, REPO, require, update_golden

SCRIPT = REPO / "experiments" / "paper" / "make_numbers.py"
D16 = "vast_results_2026-10-16"
OUTPUTS = ("tab_main.tex", "tab_qn.tex", "numbers.tex")


def _body(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[1:] if path.name == "numbers.tex" else lines


@pytest.mark.regression
def test_paper_tables_are_unchanged(results, tmp_path):
    evals = [require(results / D16 / d / "eval" / "per_slice") for d in ("outputs_joint", "outputs_detect")]
    subprocess.run([sys.executable, str(SCRIPT), "--eval", *map(str, evals), "--out", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    gold = GOLDEN / "paper"
    if update_golden():
        gold.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:
            (gold / name).write_text((tmp_path / name).read_text(encoding="utf-8"), encoding="utf-8")
        pytest.skip("golden paper tables written")
    for name in OUTPUTS:
        assert _body(tmp_path / name) == _body(require(gold / name)), f"{name} changed"
