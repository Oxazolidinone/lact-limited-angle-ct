"""Parallel runs on several GPUs: idle-GPU selection, one GPU per job, queue order, shared manifest writes.

No real GPU needed: nvidia-smi is faked and child jobs are small Python scripts reading CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lact.cli.main import build_parser
from lact.config import ConfigError, load_config
from lact.infra.io.files import load_json
from lact.infra.system import gpus as G
from lact.infra.system.processes import BLOCKED, ProcessRunner
from lact.services.scheduling import expand, order_jobs

CSV = ("0, GPU-a, RTX 2080 Ti, 11264, 6146, 35\n"
       "1, GPU-b, RTX 2080 Ti, 11264, 1458, 0\n"
       "2, GPU-c, RTX 2080 Ti, 11264, 3, 0\n"
       "3, GPU-d, RTX 2080 Ti, 11264, 1, 2\n"
       "4, GPU-e, RTX 2080 Ti, 11264, 0, 0\n")
APPS = "GPU-a, 111\nGPU-b, 222\n"


@pytest.fixture(autouse=True)
def clean_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def fake_smi(args):
    return CSV if any("query-gpu" in a for a in args) else APPS


def status():
    return G.query_gpus(samples=1, run=fake_smi)


def test_pick_skips_busy_gpus_and_orders_by_load():
    gpus = status()
    assert [g.procs for g in gpus] == [1, 1, 0, 0, 0] and gpus[4].util == 0
    # ordered by (process count, % load, used memory): e (0 MiB) before c (3 MiB), d (2% load) last
    assert G.pick_free_gpus(2, gpus) == ["GPU-e", "GPU-c"]
    assert G.pick_free_gpus(3, gpus) == ["GPU-e", "GPU-c", "GPU-d"]
    with pytest.raises(G.GpuSelectionError, match="Only 3/4"):
        G.pick_free_gpus(4, gpus)                                 # busy GPUs a, b are never taken
    assert "GPU-b" in G.pick_free_gpus(4, gpus, allow_shared=True)
    assert G.pick_free_gpus(1, gpus, allowed=["3"]) == ["GPU-d"]


def test_max_over_samples_is_conservative():
    seq = iter(["0, GPU-x, T, 100, 5, 0\n", "0, GPU-x, T, 100, 90, 70\n"])
    gpus = G.query_gpus(samples=2, interval_s=0,
                        run=lambda a: next(seq) if any("query-gpu" in x for x in a) else "")
    assert gpus[0].mem_used_mb == 90 and gpus[0].util == 70 and not G.is_idle(gpus[0])


def test_resolve_spec_forms():
    assert G.resolve_gpus(["auto"], 2, query=status) == ["GPU-e", "GPU-c"]
    assert G.resolve_gpus(["all"], query=status) == ["GPU-a", "GPU-b", "GPU-c", "GPU-d", "GPU-e"]
    assert G.resolve_gpus(["3", "GPU-c"], query=status) == ["GPU-d", "GPU-c"]
    with pytest.raises(G.GpuSelectionError):
        G.resolve_gpus(["9"], query=status)
    with pytest.raises(G.GpuSelectionError, match="listed twice"):
        G.resolve_gpus(["2", "GPU-c"], query=status)


def test_all_and_explicit_gpus_respect_inherited_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-c,GPU-d")
    assert G.resolve_gpus(["all"], query=status) == ["GPU-c", "GPU-d"]
    assert G.resolve_gpus(["auto"], query=status) == ["GPU-c", "GPU-d"]
    with pytest.raises(G.GpuSelectionError):
        G.resolve_gpus(["GPU-e"], query=status)


@pytest.mark.parametrize("env", ["", "-1"])
@pytest.mark.parametrize("spec", [["auto"], ["all"], ["GPU-c"]])
def test_hidden_gpus_are_never_selected(monkeypatch, env, spec):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", env)
    with pytest.raises(G.GpuSelectionError):
        G.resolve_gpus(spec, query=status)


def test_visibility_order_and_invalid_terminator(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 3,2,-1,4")
    assert G.resolve_gpus(["all"], query=status) == ["GPU-d", "GPU-c"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-")  # an ambiguous abbreviated UUID
    with pytest.raises(G.GpuSelectionError):
        G.resolve_gpus(["all"], query=status)


@pytest.mark.parametrize(("total", "used", "util"), [("[N/A]", "0", "0"), ("11264", "[N/A]", "0"),
                                           ("11264", "0", "[N/A]")])
def test_unknown_measurements_are_not_idle(total, used, util):
    row = f"0, GPU-x, RTX 2080 Ti, {total}, {used}, {util}\n"
    gpus = G.query_gpus(samples=1, run=lambda a: row if any("query-gpu" in x for x in a) else "")
    assert not G.is_idle(gpus[0]) and "?" in G.format_table(gpus)
    with pytest.raises(G.GpuSelectionError):
        G.pick_free_gpus(1, gpus)


def test_unreadable_process_list_cannot_confirm_idle():
    def smi(args):
        if any("query-gpu" in x for x in args):
            return "0, GPU-x, RTX 2080 Ti, 11264, 0, 0\n"
        raise subprocess.CalledProcessError(1, ["nvidia-smi"])

    gpus = G.query_gpus(samples=1, run=smi)
    with pytest.raises(G.GpuSelectionError):
        G.pick_free_gpus(1, gpus)


def test_unknown_sample_is_not_hidden_by_later_valid_sample():
    rows = iter(["0, GPU-x, T, 11264, [N/A], 0\n", "0, GPU-x, T, 11264, 0, 0\n"])
    gpus = G.query_gpus(samples=2, interval_s=0,
                       run=lambda a: next(rows) if any("query-gpu" in x for x in a) else "")
    assert gpus[0].mem_used_mb is None and not G.is_idle(gpus[0])


@pytest.mark.parametrize("dev", ["cpu", "cuda:1"])
@pytest.mark.parametrize("source", ["yaml", "override"])
def test_gpu_device_is_checked_after_merging_config(tmp_path, dev, source):
    from lact.cli.common import resolve_gpus

    cfg_file = tmp_path / "device.yaml"
    cfg_file.write_text(f"device: {dev}\n")
    cfg_args = ["-c", str(cfg_file)] if source == "yaml" else ["-s", f"device={dev}"]
    args = build_parser().parse_args(["train", "grid", "configs/experiment/base_mayo.yaml", "--run", "--gpus", "all",
                                      *cfg_args])
    with pytest.raises(ConfigError, match="device after merging"):
        resolve_gpus(args)


def test_grid_checks_device_overrides_from_experiment(tmp_path, monkeypatch):
    from lact.cli.commands.train import cmd_grid
    from lact.services import scheduling as experiments

    jobs = experiments.expand("configs/experiment/base_mayo.yaml", overrides=["device=cpu"])
    monkeypatch.setattr(experiments, "expand", lambda *a, **kw: jobs)
    monkeypatch.setattr(experiments, "run_jobs", lambda *a, **kw: pytest.fail("CPU jobs must not run"))
    args = build_parser().parse_args(["train", "grid", "configs/experiment/base_mayo.yaml", "--run", "--gpus", "all"])
    with pytest.raises(ConfigError, match="device after merging"):
        cmd_grid(args)


JOB = """
import json, os, pathlib, sys, time
out = pathlib.Path(sys.argv[1])
t0 = time.time(); time.sleep(0.4)
out.write_text(json.dumps({"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "order": os.environ.get("CUDA_DEVICE_ORDER"),
                           "per": os.environ.get("LACT_JOBS_PER_GPU"), "t0": t0, "t1": time.time()}))
"""


def _overlap(a, b):
    return a["t0"] < b["t1"] and b["t0"] < a["t1"]


@pytest.mark.parametrize("per_gpu", [1, 2])
def test_each_job_sees_one_gpu_and_gpu_capacity_is_respected(tmp_path, per_gpu):
    n = 6
    cmds = [(f"j{i}", [sys.executable, "-c", JOB, str(tmp_path / f"j{i}.json")]) for i in range(n)]
    with ProcessRunner(tmp_path, gpus=["GPU-c", "GPU-e"], per_gpu=per_gpu) as runner:
        assert runner.slots == 2 * per_gpu
        assert runner.run(cmds) == {f"j{i}": 0 for i in range(n)}
    rec = [json.loads((tmp_path / f"j{i}.json").read_text()) for i in range(n)]
    assert {r["gpu"] for r in rec} == {"GPU-c", "GPU-e"}
    assert all(r["order"] == "PCI_BUS_ID" and r["per"] == str(per_gpu) for r in rec)
    for gpu in ("GPU-c", "GPU-e"):
        same = [r for r in rec if r["gpu"] == gpu]
        # whenever a job starts, fewer than per_gpu other jobs are running on the same GPU
        assert max(sum(_overlap(a, b) for b in same if b is not a) for a in same) <= per_gpu - 1
    # the two GPUs really run at the same time
    assert any(_overlap(a, b) for a in rec for b in rec if a["gpu"] != b["gpu"])


def _run_weighted(tmp_path, names, weights, gpus=("GPU-c", "GPU-e")):
    cmds = [(k, [sys.executable, "-c", JOB, str(tmp_path / f"{k}.json")]) for k in names]
    with ProcessRunner(tmp_path, gpus=list(gpus), per_gpu=2) as runner:
        assert runner.run(cmds, weights=weights) == {k: 0 for k in names}
    return {k: json.loads((tmp_path / f"{k}.json").read_text()) for k in names}


def test_heavy_jobs_run_alone_and_light_jobs_share_a_gpu(tmp_path):
    """Memory-aware packing: runs needing more than half a GPU run alone; light runs share a GPU two at a time."""
    names = ["h1", "h2", "l1", "l2", "l3", "l4"]
    rec = _run_weighted(tmp_path, names, {"h1": 0.84, "h2": 1.0, **{f"l{i}": 0.3 for i in range(1, 5)}})
    for h in ("h1", "h2"):
        others = [r for k, r in rec.items() if k != h and r["gpu"] == rec[h]["gpu"]]
        assert not any(_overlap(rec[h], o) for o in others)                    # heavy runs share no GPU
        assert rec[h]["per"] == "1"
    light = [rec[f"l{i}"] for i in range(1, 5)]
    assert all(r["per"] == "2" for r in light)
    assert any(_overlap(a, b) for a in light for b in light if a is not b and a["gpu"] == b["gpu"])


def test_heavy_job_at_queue_head_is_not_starved_by_light_jobs(tmp_path):
    """A heavy run waiting at the head of the queue gets a reserved GPU; light runs behind it cannot take that GPU."""
    names = ["l1", "l2", "l3", "l4", "heavy", "l5", "l6"]
    rec = _run_weighted(tmp_path, names, {"heavy": 1.0})                      # runs without measurements: 1/per_gpu
    h = rec["heavy"]
    assert not any(_overlap(h, r) for k, r in rec.items() if k != "heavy" and r["gpu"] == h["gpu"])
    assert h["t0"] <= min(rec["l5"]["t0"], rec["l6"]["t0"]) + 0.3             # l5, l6 do not jump ahead


def test_solo_gpu_takes_one_job_at_a_time(tmp_path):
    """A weak (solo) GPU runs one job at a time; the other GPU still runs two."""
    names = [f"j{i}" for i in range(6)]
    cmds = [(k, [sys.executable, "-c", JOB, str(tmp_path / f"{k}.json")]) for k in names]
    with ProcessRunner(tmp_path, gpus=["GPU-c", "GPU-e"], per_gpu=2, solo_gpus=["GPU-e"]) as runner:
        assert runner.run(cmds) == {k: 0 for k in names}
    rec = [json.loads((tmp_path / f"{k}.json").read_text()) for k in names]
    solo = [r for r in rec if r["gpu"] == "GPU-e"]
    shared = [r for r in rec if r["gpu"] == "GPU-c"]
    assert solo and all(r["per"] == "1" for r in solo) and all(r["per"] == "2" for r in shared)
    assert not any(_overlap(a, b) for a in solo for b in solo if a is not b)
    assert any(_overlap(a, b) for a in shared for b in shared if a is not b)


def test_dependent_jobs_wait_for_their_source_and_blocked_ones_are_reported(tmp_path):
    """ready: dependent jobs run only after their source job (even when queued first); when the source fails the
    dependent job gets BLOCKED once every other job has finished, without hanging the queue."""
    job = lambda k: (k, [sys.executable, "-c", JOB, str(tmp_path / f"{k}.json")])
    cmds = [job("dep_a"), job("src_a"), job("dep_bad"), ("src_bad", [sys.executable, "-c", "raise SystemExit(1)"]),
            job("free")]
    asked = []

    def ready(key):
        asked.append(key)
        return (tmp_path / f"{key.replace('dep', 'src')}.json").exists() if key.startswith("dep") else True

    seen = []
    with ProcessRunner(tmp_path, gpus=["GPU-c", "GPU-e"], per_gpu=1) as runner:
        codes = runner.run(cmds, on_exit=lambda k, c: seen.append((k, c)), ready=ready)
    assert codes == {"src_a": 0, "free": 0, "dep_a": 0, "src_bad": 1, "dep_bad": BLOCKED}
    rec = {k: json.loads((tmp_path / f"{k}.json").read_text()) for k in ("src_a", "dep_a", "free")}
    assert rec["dep_a"]["t0"] >= rec["src_a"]["t1"] and rec["free"]["t0"] < rec["dep_a"]["t0"]
    assert seen[-1] == ("dep_bad", BLOCKED) and not (tmp_path / "dep_bad.json").exists()
    assert len(asked) < 40                   # the condition is re-checked only when a job ends, not on every loop
    # without ready: the old behaviour, every job runs in queue order
    with ProcessRunner(tmp_path, gpus=["GPU-c"], per_gpu=1) as runner:
        assert runner.run([job("x1"), job("x2")]) == {"x1": 0, "x2": 0}


def test_mem_tables_merge_by_model_and_use_the_smallest_gpu(tmp_path, monkeypatch):
    """--mem-table accepts several cost tables (one per grid and geometry): each architecture takes the LARGEST
    value."""
    from lact.cli.common import mem_tables

    (tmp_path / "a.csv").write_text("model,sec_per_step,peak_mem_mb\nregformer,0.40,17000\nlearn_densenet,0.10,2800\n")
    (tmp_path / "b.csv").write_text("model,sec_per_step,peak_mem_mb\nregformer,0.46,17900\nlearn_densenet,0.09,\n")
    real = G.query_gpus
    monkeypatch.setattr(G, "query_gpus", lambda samples=1: real(samples=samples, run=fake_smi))
    mem, sec, gpu_mem = mem_tables([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")], ["GPU-c", "GPU-e"])
    assert mem == {"regformer": 17900.0, "learn_densenet": 2800.0}
    assert sec == {"regformer": 0.46, "learn_densenet": 0.10}
    assert gpu_mem == 11264.0


def test_start_deadline_starts_nothing_new(tmp_path):
    cmds = [("j", [sys.executable, "-c", "raise SystemExit(0)"])]
    with ProcessRunner(tmp_path, gpus=["GPU-c"], start_hours=0) as runner:
        assert runner.run(cmds) == {} and runner.stopped_starting and not runner.interrupted


def test_sighup_is_a_graceful_stop_request(tmp_path):
    before = signal.getsignal(signal.SIGHUP)
    with ProcessRunner(tmp_path) as runner:
        os.kill(os.getpid(), signal.SIGHUP)
        time.sleep(0.05)
        assert runner.interrupted
    assert signal.getsignal(signal.SIGHUP) == before


class _Store:
    def __init__(self, status, beat=None):
        self.status, self._beat = status, beat or {}

    def heartbeat(self):
        return self._beat


def test_queue_order_partial_first_then_longest(tmp_path):
    out = tmp_path / "outputs"
    sets = [f"paths.{k}={out / k}" for k in ("data", "operators", "runs", "eval", "baselines")]
    sets.append(f"paths.outputs={out}")
    jobs, regs = [], []
    for grid in ("base_mayo", "cmp_mayo"):                   # one run per architecture
        for j in expand(f"configs/experiment/{grid}.yaml", overrides=sets):
            reg = load_config(j.files, j.overrides, use_defaults=False).require_model().regularizer.name
            if reg not in regs:
                jobs.append(j)
                regs.append(reg)
    assert len(jobs) == 5
    stores = {j.run_id: _Store("todo") for j in jobs}
    assert order_jobs(jobs, stores) == jobs                     # no measurements: keep the grid order
    sec = {r: float(i + 1) for i, r in enumerate(regs)}      # the last run of the grid is the slowest
    assert [j.run_id for j in order_jobs(jobs, stores, sec)] == [j.run_id for j in reversed(jobs)]
    stores[jobs[1].run_id] = _Store("partial", {"step": 5, "steps_per_epoch": 10})
    assert order_jobs(jobs, stores, sec)[0] is jobs[1]         # partial runs always go first


def test_manifest_updates_from_parallel_processes_are_not_lost(tmp_path, monkeypatch):
    script = """
import pathlib, sys
from lact.infra.stores.manifest import update_manifest
root, w = pathlib.Path(sys.argv[1]), sys.argv[2]
for i in range(15):
    f = root / f"{w}_{i}.bin"; f.write_bytes(bytes([i]) * 10)
    update_manifest(root / "manifest.json", [f])
"""
    # children run in tmp_path: point PYTHONPATH at the repository's src with an absolute path
    monkeypatch.setenv("PYTHONPATH", str(Path(G.__file__).parents[3]))  # .../src
    cmds = [(w, [sys.executable, "-c", script, str(tmp_path), w]) for w in ("a", "b", "c")]
    with ProcessRunner(tmp_path) as runner:
        assert runner.run(cmds, workers=3) == {"a": 0, "b": 0, "c": 0}
    man = load_json(tmp_path / "manifest.json")
    assert len([k for k in man if k.endswith(".bin")]) == 45
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def test_cli_accepts_gpu_options_and_rejects_pinned_device(monkeypatch):
    parser = build_parser()
    a = parser.parse_args(["train", "grid", "configs/experiment/base_mayo.yaml", "--gpus", "auto", "--n-gpus", "2",
                           "--per-gpu", "1", "--dry-run"])
    assert a.gpus == ["auto"] and a.n_gpus == 2
    a = parser.parse_args(["train", "grid", "configs/experiment/base_mayo.yaml", "configs/experiment/lbfgs_mayo.yaml",
                           "--run", "--gpus", "2", "3"])
    assert len(a.experiment) == 2 and a.gpus == ["2", "3"]
    from lact.cli.common import resolve_gpus
    a = parser.parse_args(["train", "grid", "configs/experiment/base_mayo.yaml", "--gpus", "2", "--device", "cuda:1"])
    with pytest.raises(ConfigError, match="--device"):
        resolve_gpus(a)
