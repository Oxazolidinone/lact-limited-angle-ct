"""Tests of experiment orchestration; they do not replace numerical ASTRA/GPU tests."""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time

import h5py
import numpy as np
import pandas as pd
import pytest

from lact.cli.common import load_cfg
from lact.cli.main import build_parser, main
from lact.config import ConfigError, load_config, save_yaml
from lact.infra.io.files import load_json, save_json
from lact.infra.operators.operator_store import operator_path
from lact.infra.stores.results_store import ResultsStore
from lact.infra.system.processes import INTERRUPTED, ProcessRunner
from lact.services.data.simulation import simulate_geometry
from lact.services.orchestration import pipeline as mod
from lact.services.orchestration.pipeline import Pipeline, Stage
from lact.services.reporting.aggregate import select_best_lr
from lact.services.scheduling import expand, run_jobs
from lact.services.training.trainer import run_store_for


@pytest.fixture
def pipeline(tmp_path):
    out = tmp_path / "outputs"
    sets = [f"paths.{key}={out / key}" for key in
            ("data", "operators", "runs", "eval", "tables", "figures", "baselines")]
    sets += [f"paths.outputs={out}", f"data.images_h5={out}/data/images.h5",
             f"data.stats_json={out}/data/stats.json", f"data.slices_json={out}/data/slices.json"]
    return Pipeline(load_config(overrides=sets), [], sets)


def complete(job, status="done", score=30.0):
    cfg = load_config(job.files, job.overrides, use_defaults=False)
    run = run_store_for(cfg)
    run.dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg.to_dict(), run.config_file)
    if status == "done":
        run.checkpoint("best").write_bytes(b"orchestration test: not a model")
    run.mark({"status": status, "regularizer": cfg.require_model().regularizer.name,
              "arch": cfg.require_model().arch, "lr": cfg.train.lr, "seed": cfg.seed,
              "best_val_psnr": score})
    return run


class FakeRunner:
    interrupted = False

    def __init__(self, action=lambda key, argv: 0):
        self.action = action
        self.calls = []

    def expired(self):
        return self.interrupted

    def run(self, commands, workers=1, on_exit=None):
        codes = {}
        for key, argv in commands:
            self.calls.append(key)
            code = self.action(key, argv)
            codes[key] = code
            if on_exit:
                on_exit(key, code)
            if code == INTERRUPTED:
                self.interrupted = True
                break
        return codes


def test_pipeline_cli_dry_run_does_not_write(pipeline, capsys, monkeypatch):
    monkeypatch.setattr("lact.cli.main.get_logger", lambda *_: logging.getLogger("test.pipeline"))
    assert main(["pipeline", "run", "--dry-run", "--workers", "2", "-s", *pipeline.overrides]) == 0
    assert not pipeline.path.exists()
    assert "E2 (45 runs)" in capsys.readouterr().out


def test_cli_device_reaches_config():
    args = build_parser().parse_args(["train", "run", "--device", "cuda:1"])
    assert load_cfg(args).device == "cuda:1"
    args = build_parser().parse_args(["train", "run", "-s", "device=cpu"])
    assert load_cfg(args).device == "cpu"


def test_lr_path_respects_overrides_and_is_required(pipeline):
    with pytest.raises(ConfigError, match="No lr chosen yet"):
        expand("configs/experiment/e3_main.yaml", overrides=pipeline.overrides, require_lr=True)
    save_json({"best_lr": {"cnn3": 0.001}}, pipeline.cfg.paths.runs / "e2_sweep/best_lr.json")
    jobs = expand("configs/experiment/e3_main.yaml", overrides=pipeline.overrides)
    cnn = [load_config(j.files, j.overrides, use_defaults=False) for j in jobs if "learn_cnn_" in j.run_id]
    assert cnn and all(c.train.lr == 0.001 for c in cnn)


def test_stages_resume_but_missing_artifact_reruns(pipeline, tmp_path):
    output = tmp_path / "report.json"
    source = tmp_path / "matrix.bin"
    source.write_bytes(b"first")
    pipeline._load_state()
    pipeline.runner = FakeRunner(lambda *_: output.write_text("{}") and 0)
    stage = Stage("e0", ["ops", "check"], [output], inputs=[source])
    assert pipeline.stage(stage) == 0
    assert pipeline.stage(stage) == 0
    assert pipeline.runner.calls == ["e0"]
    output.unlink()
    assert pipeline.stage(stage) == 0
    assert pipeline.runner.calls == ["e0", "e0"]
    source.write_bytes(b"matrix replaced")
    assert pipeline.stage(stage) == 0
    assert pipeline.runner.calls == ["e0", "e0", "e0"]


def test_failed_prerequisite_stops_before_sweep(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, "prepare", lambda: 1)
    monkeypatch.setattr(pipeline, "jobs", lambda _: pytest.fail("E2 must not start"))
    assert pipeline.run() == 1
    assert load_json(pipeline.path)["status"] == "failed"


def test_sweep_incomplete_cannot_select_lr_or_open_main(pipeline, monkeypatch):
    pipeline._load_state()
    pipeline.runner = FakeRunner()
    monkeypatch.setattr(pipeline, "prepare", lambda: 0)
    monkeypatch.setattr(mod, "select_lr", lambda *a, **kw: pytest.fail("E2 incomplete"))
    jobs = pipeline.jobs([mod.SWEEP])
    complete(jobs[0])
    assert pipeline._execute("all") == 1
    assert len(pipeline.runner.calls) == 44
    assert not any("e3_main" in key for key in pipeline.runner.calls)


def test_interrupted_sweep_continues_only_remaining_jobs(pipeline):
    pipeline._load_state()
    jobs = pipeline.jobs([mod.SWEEP])[:3]
    by_id = {j.run_id: j for j in jobs}
    complete(jobs[0])
    pipeline.runner = FakeRunner(lambda *_: INTERRUPTED)
    assert pipeline.train(jobs, sweep=True) == INTERRUPTED
    assert pipeline.runner.calls == [jobs[1].run_id]
    pipeline.runner = FakeRunner(lambda key, _: complete(by_id[key]) and 0)
    assert pipeline.train(jobs, sweep=True) == 0
    assert pipeline.runner.calls == [jobs[1].run_id, jobs[2].run_id]


def test_sweep_divergence_allowed_but_oom_blocks(pipeline):
    pipeline._load_state()
    pipeline.runner = FakeRunner()
    jobs = pipeline.jobs([mod.SWEEP])[:3]
    runs = [complete(jobs[0], score=31), complete(jobs[1], "diverged"), complete(jobs[2], score=32)]
    assert pipeline.train(jobs, sweep=True) == 0
    assert select_best_lr(runs)["best_lr"]["cnn3"] == 3e-4
    complete(jobs[2], "oom")
    (runs[2].dir / "DONE").unlink()
    assert pipeline.train(jobs, sweep=True) == 1


def test_no_finite_sweep_candidate_is_an_error(pipeline):
    job = pipeline.jobs([mod.SWEEP])[0]
    for status, score in (("diverged", 0), ("done", float("nan"))):
        with pytest.raises(ConfigError, match="finite PSNR"):
            select_best_lr([complete(job, status, score)])


def test_pipeline_orders_barriers_and_scopes_lr(pipeline, monkeypatch):
    events = []
    pipeline._load_state()
    pipeline.runner = FakeRunner()
    monkeypatch.setattr(pipeline, "prepare", lambda: events.append("prepare") or 0)
    actual_jobs = pipeline.jobs

    def jobs(names):
        events.append("expand:" + ",".join(names))
        return actual_jobs(names)

    monkeypatch.setattr(pipeline, "jobs", jobs)

    def train(jobs, sweep=False):
        events.append("sweep" if sweep else "main")
        for j in jobs:
            complete(j)
        return 0

    monkeypatch.setattr(pipeline, "train", train)
    monkeypatch.setattr(pipeline, "stage", lambda s: events.append(s.key) or 0)
    monkeypatch.setattr(pipeline, "finish", lambda jobs: events.append("eval+e6+report") or 0)
    assert pipeline._execute("all") == 0
    assert events == ["prepare", "expand:e2_sweep", "sweep", "freeze",
                      "expand:e3_main,e4_matched,e5_a150", "main", "eval+e6+report"]
    lr = load_json(pipeline.cfg.paths.runs / "e2_sweep/best_lr.json")
    assert len(lr["table"]) == 45 and len(lr["best_lr"]) == 15


def test_config_change_cannot_silently_resume(pipeline):
    pipeline._load_state()
    changed = Pipeline(load_config(overrides=pipeline.overrides + ["train.epochs=12"]), [], pipeline.overrides)
    with pytest.raises(ConfigError, match="configuration differs"):
        changed.run()
    same = Pipeline(pipeline.cfg.replace(device="cuda"), [], pipeline.overrides, workers=2)
    same._load_state()  # changing GPUs/workers does not change the experiment


def test_extend_adopts_outputs_but_old_runs_must_still_match(pipeline):
    pipeline._load_state()
    old = load_json(pipeline.path)["fingerprint"]
    complete(pipeline.jobs([mod.SWEEP])[0])
    sets = pipeline.overrides + ["train.epochs=12"]
    with pytest.raises(ConfigError, match="--extend"):
        Pipeline(load_config(overrides=sets), [], sets)._load_state()
    ext = Pipeline(load_config(overrides=sets), [], sets, extend=True)
    ext._load_state()
    state = load_json(pipeline.path)
    assert state["previous_fingerprints"] == [old] and state["fingerprint"] != old
    ext.runner = FakeRunner()
    with pytest.raises(ConfigError, match="configuration changed"):       # a run that ran must keep its configuration
        ext.train(ext.jobs([mod.SWEEP]), sweep=True)
    again = Pipeline(load_config(overrides=sets), [], sets)
    again._load_state()                                             # --extend is no longer needed afterwards
    assert load_json(pipeline.path)["previous_fingerprints"] == [old]


def test_new_runs_invalidate_cost_and_reports(pipeline, monkeypatch):
    groups = []
    monkeypatch.setattr(pipeline, "stages", lambda group: groups.append(group) or 0)
    jobs = pipeline.jobs([mod.SWEEP])[:2]
    assert pipeline.finish(jobs) == 0
    by_key = {st.key: st for group in groups for st in group}
    run_dirs = [run_store_for(load_config(j.files, j.overrides, use_defaults=False)).dir for j in jobs]
    assert by_key["cost"].inputs == [d / "summary.json" for d in run_dirs]
    index = pipeline.cfg.paths.eval / "index.json"
    assert all(index in by_key[k].inputs for k in ("report_tables", "report_figures", "report_analysis"))


def test_models_filter_limits_every_grid(pipeline):
    sets = pipeline.overrides
    part = Pipeline(load_config(overrides=sets), [], sets, models=["learn_unet", "learn_unet_regnet", "learn_cnn"])
    sweep = part.jobs([mod.SWEEP])
    assert len(sweep) == 9 and {j.arch for j in sweep} == {"learn_cnn", "learn_unet", "learn_unet_regnet"}
    for j in sweep:
        complete(j)
    mod.select_lr(part.cfg, mod.SWEEP, runs=[run_store_for(load_config(j.files, j.overrides, use_defaults=False))
                                              for j in sweep])
    main = part.jobs(mod.MAIN_EXPS)
    assert len(main) == 3 * 2 + 0 + 3 and not any("pm2m" in j.arch for j in main)      # E3 + E4 (not selected) + E5
    with pytest.raises(ConfigError, match="no model of the grid"):
        Pipeline(load_config(overrides=sets), [], sets, models=["learn_typo"]).jobs([mod.SWEEP])
    pipeline._load_state()
    # changing the model set = a different workflow
    with pytest.raises(ConfigError, match="configuration differs"):
        part._load_state()


def test_extra_seeds_are_kept_only_when_already_finished(pipeline):
    """Grids train only seed 0; seeds 1, 2 belong to the experiment if and only if the run was already DONE."""
    sets = pipeline.overrides
    sweep = pipeline.jobs([mod.SWEEP])
    for j in sweep:
        complete(j)
    mod.select_lr(pipeline.cfg, mod.SWEEP, runs=[run_store_for(load_config(j.files, j.overrides, use_defaults=False))
                                                  for j in sweep])
    e3 = "configs/experiment/e3_main.yaml"
    jobs = expand(e3, [], sets, require_lr=True)
    assert len(jobs) == 30 and {j.overrides[1] for j in jobs} == {"seed=0"}
    old = next(j for j in jobs if j.arch == "learn_resnet")
    for seed, status in ((1, "done"), (2, "diverged")):                       # two old runs: one done, one failed
        other = type(old)(old.run_id, "todo", old.files, [o if not o.startswith("seed=") else f"seed={seed}"
                                                           for o in old.overrides], old.arch)
        complete(other, status=status)
    again = expand(e3, [], sets, require_lr=True)
    extra = [j for j in again if j.overrides[1] != "seed=0"]
    assert len(again) == 31 and len(extra) == 1                               # only the DONE seed-1 run is added
    assert extra[0].arch == "learn_resnet" and extra[0].status == "done" and extra[0].overrides[1] == "seed=1"
    assert not any(j.pending for j in extra)                                  # seeds 1, 2 are never trained anew


def test_gpu_shares_follow_measured_memory(pipeline):
    """GPU share of each run = measured peak memory / budget; unmeasured runs run alone; unused at 1 run/GPU."""
    jobs = pipeline.jobs([mod.SWEEP])
    mem = {"learn_efficientnet": 23260.0, "learn_unet": 2873.0}
    assert Pipeline(pipeline.cfg, [], pipeline.overrides, gpus=["GPU-a"], per_gpu=1, job_mem_mb=mem,
                    gpu_mem_mb=32607.0).gpu_shares(jobs) == {}
    p2 = Pipeline(pipeline.cfg, [], pipeline.overrides, gpus=["GPU-a"], per_gpu=2, job_mem_mb=mem, gpu_mem_mb=32607.0)
    shares = p2.gpu_shares(jobs)
    by_arch = {j.arch: shares[j.run_id] for j in jobs}
    heavy = by_arch["learn_efficientnet"]
    assert heavy == pytest.approx(23260 / (0.85 * 32607)) and heavy > 0.5
    assert by_arch["learn_unet"] < 0.5 and by_arch["learn_cnn"] == 1.0          # no measurement: do not share


def test_lr_that_diverges_in_main_experiment_is_replaced_by_next_best(pipeline):
    """E2 picks the highest lr, but that lr made an E3 run diverge: the architecture switches to its best remaining
    lr."""
    sweep = pipeline.jobs([mod.SWEEP])
    score = {"3e-05": 28.0, "0.0001": 30.0, "0.0003": 32.0}
    for j in sweep:
        complete(j, score=score[j.run_id.split("_lr")[1].split("_s")[0]])
    runs = [run_store_for(load_config(j.files, j.overrides, use_defaults=False)) for j in sweep]
    assert mod.select_lr(pipeline.cfg, mod.SWEEP, runs=runs)["best_lr"]["mbconv_ed"] == 3e-4
    eff = next(j for j in pipeline.jobs(mod.MAIN_EXPS) if j.arch == "learn_efficientnet")
    assert "_lr0.0003_" in eff.run_id
    complete(eff, status="diverged")
    res = mod.select_lr(pipeline.cfg, mod.SWEEP, runs=runs)
    assert res["best_lr"]["mbconv_ed"] == 1e-4 and res["best_lr"]["unet_ed"] == 3e-4
    assert res["excluded_lr"] == {"mbconv_ed": [3e-4]} and "diverge" in res["notes"]["mbconv_ed"]
    again = [j for j in pipeline.jobs(mod.MAIN_EXPS) if j.arch == "learn_efficientnet"]
    assert len(again) == 3                                                    # 2 E3 runs + 1 E5 run
    assert all("_lr0.0001_" in j.run_id and j.status == "todo" for j in again)


def test_runs_needing_a_whole_gpu_are_queued_first(pipeline):
    """Memory-aware ordering: heavy runs (needing a whole GPU) are queued before light runs; the rest keeps its
    order."""
    mem = {j.arch: 2000.0 for j in pipeline.jobs([mod.SWEEP])}
    mem["learn_efficientnet"] = 23260.0
    p2 = Pipeline(pipeline.cfg, [], pipeline.overrides, gpus=["GPU-a", "GPU-b"], per_gpu=2, job_mem_mb=mem,
                  gpu_mem_mb=32607.0)
    p2._load_state()
    seen = {}

    def run(commands, workers=1, on_exit=None, weights=None):
        seen["order"] = [key for key, _ in commands]
        return {}

    runner = FakeRunner()
    runner.run = run
    p2.runner = runner
    p2.train(p2.jobs([mod.SWEEP]), sweep=True)
    heavy = [i for i, k in enumerate(seen["order"]) if "learn_efficientnet_fan" in k]
    assert heavy == [0, 1, 2] and len(seen["order"]) == 45


def test_training_config_mismatch_rejected(pipeline):
    pipeline._load_state()
    pipeline.runner = FakeRunner()
    job = pipeline.jobs([mod.SWEEP])[0]
    complete(job)
    job.overrides += ["train.epochs=99"]
    with pytest.raises(ConfigError, match="configuration changed"):
        pipeline.train([job], sweep=True)


def test_grid_time_budget_does_not_report_success(pipeline):
    assert run_jobs(pipeline.jobs([mod.SWEEP]), max_hours=0) == [INTERRUPTED]


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


def test_simulation_reuses_matching_cache_but_rejects_changed_matrix(pipeline, small_geom):
    cfg = pipeline.cfg
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
