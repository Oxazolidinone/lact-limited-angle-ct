"""Cache and run status (no ASTRA needed): environment variables in paths, file/array caches, grid filtering, run
status markers, command-line parsing."""
from __future__ import annotations

import os

import numpy as np
import pytest

from lact.cli.main import build_parser
from lact.config import load_config
from lact.config.loader import resolve_path
from lact.infra.data.cache import ArrayCache, is_fresh, publish_file, stage_file
from lact.infra.stores.run_store import RunStore
from lact.services.scheduling import Job, filter_jobs


def test_resolve_path_expands_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LACT_DRIVE", str(tmp_path / "drive"))
    assert resolve_path("${LACT_DRIVE}/outputs/runs") == tmp_path / "drive" / "outputs" / "runs"


def test_server_config_loads():
    """Configuration for .venv runs on servers / rented machines: cache on, mid-epoch checkpoints, GPU required."""
    c = load_config(["configs/env/server.yaml"])
    assert c.cache.enabled and c.cache.dir is not None and c.cache.verify == "sha256"
    assert c.train.checkpoint_every_min and c.train.keep_prev_checkpoint and c.train.require_gpu
    assert c.device == "cuda" and c.paths.runs.parts[-2:] == ("outputs", "runs")


def test_stage_and_publish(tmp_path):
    src = tmp_path / "drive" / "a.bin"
    src.parent.mkdir()
    src.write_bytes(os.urandom(1000))
    local = stage_file(src, tmp_path / "cache", verify="sha256")
    assert local.read_bytes() == src.read_bytes() and is_fresh(src, local)
    src.write_bytes(os.urandom(1200))                 # the source changed => copied again
    assert not is_fresh(src, local)
    assert stage_file(src, tmp_path / "cache").read_bytes() == src.read_bytes()

    built = tmp_path / "build" / "b.bin"
    built.parent.mkdir()
    built.write_bytes(b"x" * 10)
    dst = publish_file(built, tmp_path / "drive" / "b.bin", keep_dir=tmp_path / "cache")
    assert dst.read_bytes() == b"x" * 10 and not built.exists()
    assert is_fresh(dst, tmp_path / "cache" / "b.bin")


def test_array_cache_roundtrip(tmp_path):
    ac = ArrayCache(tmp_path)
    key = ac.make_key({"f": "x", "n": 1})
    assert ac.load(key) is None
    ac.save(key, {"y": np.arange(6, dtype=np.float32).reshape(2, 3)}, {"n": 2})
    arrays, extra = ac.load(key)
    assert extra == {"n": 2} and np.array_equal(arrays["y"], np.arange(6).reshape(2, 3))


def test_filter_jobs():
    jobs = [Job("e3_learn_cnn_fan_a90_s0", "todo"), Job("e3_learn_convnext_fan_a120_s0", "done")]
    assert filter_jobs(jobs, None) == jobs
    assert [j.run_id for j in filter_jobs(jobs, ["convnext"])] == ["e3_learn_convnext_fan_a120_s0"]
    assert not jobs[1].pending and Job("x", "oom").pending is False


def test_run_status_markers(tmp_path):
    run = RunStore(tmp_path / "r")
    assert run.status == "todo"
    run.dir.mkdir(parents=True, exist_ok=True)
    run.checkpoint("last_prev").write_bytes(b"")
    assert run.status == "partial"
    run.mark({"status": "oom"})
    assert run.status == "oom" and run.finished
    run.write_heartbeat(step=3)
    run.log_event("x", a=1)
    assert run.heartbeat()["step"] == 3 and run.events()[-1]["event"] == "x"


@pytest.mark.parametrize("argv", [
    ["data", "download-mayo", "--dest", "/tmp/x", "--no-extract"], ["data", "stage", "-g", "fan_a90"],
    ["train", "status"], ["train", "grid", "configs/experiment/e3_main.yaml", "--match", "cnn", "--max-hours", "2"],
])
def test_new_cli_parses(argv):
    assert callable(build_parser().parse_args(argv).func)
