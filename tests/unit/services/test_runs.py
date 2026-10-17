"""Tests of running training jobs, stopping them and caching data; they do not replace numerical ASTRA/GPU tests."""
import json
import signal
import sys
import threading
import time

import h5py
import numpy as np
import pandas as pd
import pytest

from lact.cli.common import load_cfg
from lact.cli.main import build_parser
from lact.config import ConfigError, load_config
from lact.infra.io.files import save_json
from lact.infra.operators.operator_store import operator_path
from lact.infra.stores.results_store import ResultsStore
from lact.infra.system.processes import INTERRUPTED, ProcessRunner
from lact.services.data.simulation import simulate_geometry
from lact.services.scheduling import expand, run_jobs


@pytest.fixture
def overrides(tmp_path):
    out = tmp_path / "outputs"
    sets = [f"paths.{key}={out / key}" for key in ("data", "operators", "runs", "eval", "baselines")]
    return sets + [f"paths.outputs={out}", f"data.images_h5={out}/data/images.h5",
                   f"data.stats_json={out}/data/stats.json", f"data.slices_json={out}/data/slices.json"]


@pytest.fixture
def cfg(overrides):
    return load_config(overrides=overrides)


def test_cli_device_reaches_config():
    args = build_parser().parse_args(["train", "run", "--device", "cuda:1"])
    assert load_cfg(args).device == "cuda:1"
    args = build_parser().parse_args(["train", "run", "-s", "device=cpu"])
    assert load_cfg(args).device == "cpu"


def test_grid_time_budget_does_not_report_success(overrides):
    jobs = expand("configs/experiment/base_mayo.yaml", overrides=overrides)
    assert jobs and all(j.status == "todo" for j in jobs)
    assert run_jobs(jobs, max_hours=0) == [INTERRUPTED]


def test_process_queue_really_overlaps_without_duplicate_launch(tmp_path):
    # The two children must see each other running before exiting; a sequential run would fail.
    script = """
import pathlib, sys, time
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
(root / name).write_text('started')
deadline = time.monotonic() + 5
while len(list(root.glob('job*'))) < 2 and time.monotonic() < deadline:
    time.sleep(.02)
assert len(list(root.glob('job*'))) == 2
"""
    commands = [(f"job{i}", [sys.executable, "-c", script, str(tmp_path), f"job{i}"]) for i in range(2)]
    with ProcessRunner(tmp_path) as runner:
        assert runner.run(commands, workers=2) == {"job0": 0, "job1": 0}


@pytest.mark.parametrize("stop", ["deadline", "signal"])
def test_stop_forwards_sigterm_and_waits_for_checkpoint(tmp_path, stop):
    checkpoint = tmp_path / "checkpoint"
    ready = tmp_path / "ready"
    script = """
import pathlib, signal, sys, time
def save(*_):
    time.sleep(.15)
    pathlib.Path(sys.argv[1]).write_text('saved')
    sys.exit(3)
signal.signal(signal.SIGTERM, save)
pathlib.Path(sys.argv[2]).touch()
while True: time.sleep(.05)
"""
    with ProcessRunner(tmp_path, max_hours=1 / 3600 if stop == "deadline" else None, grace_s=3) as runner:
        def interrupt_when_ready():
            end = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < end:
                time.sleep(.02)
            # Call the parent's own SIGTERM handler without affecting pytest outside the context.
            runner._interrupt(signal.SIGTERM, None)

        thread = threading.Thread(target=interrupt_when_ready) if stop == "signal" else None
        if thread:
            thread.start()
        codes = runner.run([("active", [sys.executable, "-c", script, str(checkpoint), str(ready)]),
                            ("must_not_start", [sys.executable, "-c", "raise AssertionError()"] )])
        if thread:
            thread.join(timeout=6)
        assert runner.interrupted and codes == {"active": INTERRUPTED}
    assert checkpoint.read_text() == "saved"


def test_evaluation_partial_csv_is_not_complete(tmp_path):
    store = ResultsStore(tmp_path)
    info = {"run_id": "r", "split": "test", "geometry_test": "fan_a90"}
    path = store.csv_path("r__test__fan_a90")
    path.parent.mkdir()
    path.write_text("psnr\n30\n")
    assert not store.exists(info)
    store.save(info, pd.DataFrame({"psnr": [30]}))
    assert store.exists(info) and not store.exists(info, require_pred=True)


def test_simulation_reuses_matching_cache_but_rejects_changed_matrix(cfg, small_geom):
    pid = cfg.split.val[0]
    cfg.data.images_h5.parent.mkdir(parents=True)
    with h5py.File(cfg.data.images_h5, "w") as f:
        f.attrs.update(mu_ref=1., mu_water_mm=.02)
        f[f"{pid}/img256"] = np.zeros((1, 4, 4), np.float32)
    # Only metadata/cache are checked; no projection is computed in this test.
    for n in (cfg.data.img_size_rec, cfg.data.img_size_sim):
        save_json({"sha256": "original"}, operator_path(small_geom, n, cfg.paths.operators).with_suffix(".json"))
    sino = cfg.paths.sino_file(small_geom.name)
    with h5py.File(sino, "w") as f:
        f.attrs.update(geometry=json.dumps(small_geom.to_dict()), img_size_rec=cfg.data.img_size_rec,
                       img_size_sim=cfg.data.img_size_sim, mu_ref=1., fbp_filter=cfg.baseline.fbp.filter,
                       noise=json.dumps(vars(cfg.noise)), A_sim_sha256="original", A_rec_sha256="original")
        f[f"{pid}/y"] = np.zeros((1, 2, 2), np.float32)
    assert simulate_geometry(cfg, small_geom) == sino
    save_json({"sha256": "changed"},
              operator_path(small_geom, cfg.data.img_size_rec, cfg.paths.operators).with_suffix(".json"))
    with pytest.raises(ConfigError, match="sinogram does not match"):
        simulate_geometry(cfg, small_geom)
