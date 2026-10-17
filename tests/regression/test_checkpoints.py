"""Trained checkpoints must load and reproduce their outputs bit for bit (up to float noise).

One run per method of the comparison table (2DeteCT, 90 degrees). Each case rebuilds the model from the run's own
``config.yaml``, loads ``best.pt``, reconstructs a fixed synthetic body phantom and compares the result with the
golden output recorded before the refactor. This catches renamed module attributes (``state_dict`` keys), changed
layer order, and any change in the projector, FBP or data-consistency steps.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from lact.config import load_config_file
from lact.core.imaging.phantom import random_body_phantom
from lact.infra.operators.operator_store import build_and_save
from lact.infra.stores.run_store import RunStore
from lact.services.factories import fbp_for, model_for, projector_for

from .conftest import GOLDEN, require, update_golden

IMG = 256
D16 = "vast_results_2026-10-16"

# (golden name, run directory relative to the results root, attribute overrides applied after loading)
CASES = [
    ("lbfgs_densenet_dt", f"{D16}/outputs_detect/runs/lbfgs_dt/lbfgs_dt_learn_densenet_dt_a90_lr0.0003_s0", {}),
    ("regformer_dt", f"{D16}/outputs_detect/runs/base_dt/base_dt_regformer_dt_a90_lr0.0001_s0", {}),
    ("learn_orig_dt", f"{D16}/outputs_detect/runs/cmp_dt/cmp_dt_learn_orig_dt_a90_lr0.0001_s0", {}),
    ("lpd_dt", f"{D16}/outputs_detect/runs/cmp_dt/cmp_dt_lpd_dt_a90_lr0.0001_s0", {}),
    ("fbpconvnet_dt", f"{D16}/outputs_detect/runs/cmp_dt/cmp_dt_fbpconvnet_dt_a90_lr0.0001_s0", {}),
    # Three refinement iterations exercise the inference-time refinement path without its full CPU cost.
    ("dpma_dt", f"{D16}/outputs_detect/runs/dpma_dt/dpma_dt_dpma_dt_a90_lr0.0001_s0", {"refine_iters": 3}),
]

PARAMS = [pytest.param(*c, id=c[0]) for c in CASES]


def _local_config(run_dir: Path, operators: Path):
    """The run's configuration with operators read from (and built into) the local regression cache."""
    cfg = load_config_file(run_dir / "config.yaml")
    return cfg.replace(paths=dataclasses.replace(cfg.paths, operators=operators),
                       cache=dataclasses.replace(cfg.cache, enabled=False))


def _build_operators(cfg, operators: Path) -> None:
    """Build every projection matrix the model needs (DPMA also needs its extended-range target geometry)."""
    geom = cfg.require_geometry()
    needed = [geom]
    m = cfg.require_model()
    if m.framework == "dpma":
        params = m.regularizer.params
        name, span = str(params.get("target_name", "fan_a240")), float(params.get("target_span_deg", 240.0))
        if name == "auto":
            name = f"{geom.name}_to{span:g}"
        needed.append(geom.replace(name=name, span_deg=span))
    for g in needed:
        build_and_save(g, IMG, operators)


@torch.no_grad()
def _reconstruct(run_dir: Path, operators: Path, overrides: dict) -> np.ndarray:
    cfg = _local_config(run_dir, operators)
    _build_operators(cfg, operators)
    torch.manual_seed(0)
    model = model_for(cfg)
    RunStore(run_dir).load_checkpoint("best", model, map_location="cpu")
    for k, v in overrides.items():
        setattr(model, k, v)
    projector = projector_for(cfg, device="cpu")
    model.set_projector(projector).eval()
    x = torch.from_numpy(random_body_phantom(IMG, np.random.default_rng(0)).astype(np.float32))[None, None]
    y = projector.raw_forward(x)
    x0 = fbp_for(cfg)(y)
    return model(y, x0)[0, 0].numpy()


@pytest.mark.regression
@pytest.mark.parametrize(("name", "rel", "overrides"), PARAMS)
def test_checkpoint_output_is_unchanged(results, operators_dir, name, rel, overrides):
    run_dir = require(results / rel)
    require(run_dir / "best.pt")
    out = _reconstruct(run_dir, operators_dir, overrides)
    assert np.isfinite(out).all()
    golden = GOLDEN / "checkpoints" / f"{name}.npz"
    if update_golden():
        golden.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(golden, out=out)
        pytest.skip(f"golden output written: {golden.name}")
    ref = np.load(require(golden))["out"]
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)
