"""Protection against lost work: a mid-epoch resume gives exactly the same result as an uninterrupted run, fallback
checkpoints, GPU out-of-memory handling, local cache (copied data + .npy arrays)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from lact.config import load_config
from lact.infra.stores.run_store import RunStore
from lact.services import operators
from lact.services.data import fake as fake_data
from lact.services.data import prep as data_prep
from lact.services.data import simulation
from lact.services.factories import dataset_for
from lact.services.training import trainer as training


@pytest.fixture(scope="module")
def cfg(tmp_path_factory, small_geom):
    pytest.importorskip("pydicom")
    tmp = tmp_path_factory.mktemp("resil")
    fake_data.make_fake_dicom(tmp / "dicom", n_slices=2, size=64)
    out = tmp / "out"
    paths = [f"paths.{k}={out / k}" for k in ("data", "operators", "runs", "eval", "tables", "figures", "baselines")]
    c = load_config(["configs/model/learn_cnn.yaml"], paths + [
        f"paths.outputs={out}", f"data.dicom_root={tmp / 'dicom'}", "data.img_size_sim=64", "data.img_size_rec=32",
        f"data.images_h5={out}/data/images.h5", f"data.stats_json={out}/data/stats.json",
        f"data.slices_json={out}/data/slices.json", "model.K=2", "model.regularizer.channels=4",
        "train.epochs=2", "train.limit_val=2", "train.val_batch_size=2", "train.tensorboard=false",
        "train.log_every=1", "train.checkpoint_every_min=null", "train.checkpoint_every_steps=3",
        "precision.deterministic=true", "exp=resil", f"cache.dir={tmp / 'cache'}", "cache.enabled=true"])
    c = c.replace(geometry=small_geom)
    data_prep.preprocess(c)
    operators.build_operators(c, [small_geom], sizes=[32, 64])
    simulation.simulate_all(c, [small_geom], batch=2)
    return c


class _StopAfter:
    """Simulate a stop request after n flag checks."""
    n = 4

    def __init__(self, *a, **k):
        self.calls = 0
        self.signal_name = "SIGINT"

    @property
    def requested(self):
        self.calls += 1
        return self.calls > _StopAfter.n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _weights(run: RunStore) -> dict:
    return torch.load(run.checkpoint("last"), weights_only=False)["model"]


def test_cache_used_and_arrays_reused(cfg):
    ds = dataset_for(cfg, "train")
    assert str(cfg.cache.dir) in ds.sino_h5 and str(cfg.cache.dir) in ds.images_h5
    assert any((cfg.cache.dir / "arrays").iterdir())
    ds2 = dataset_for(cfg, "train")
    assert torch.equal(ds[0]["y"], ds2[0]["y"])


def test_resume_mid_epoch_is_exact(cfg, tmp_path, monkeypatch):
    ref = training.Trainer(cfg.replace(run_id="ref"), RunStore(tmp_path / "ref"))
    assert ref.fit()["status"] == "done"

    monkeypatch.setattr(training.loop, "GracefulInterrupt", _StopAfter)
    r1 = training.Trainer(cfg.replace(run_id="cut"), RunStore(tmp_path / "cut"))
    out = r1.fit()
    assert out["status"] == "interrupted" and r1.run.status == "partial"
    assert r1.run.heartbeat()["status"] == "interrupted"
    monkeypatch.undo()

    r2 = training.Trainer(cfg.replace(run_id="cut"), RunStore(tmp_path / "cut"))
    assert r2.fit()["status"] == "done"
    for k, v in _weights(ref.run).items():
        assert torch.allclose(v, _weights(r2.run)[k], atol=1e-6), k
    a = pd.read_csv(ref.run.steps_csv).loss.to_numpy()
    b = pd.read_csv(r2.run.steps_csv).loss.to_numpy()
    assert np.allclose(a, b, rtol=1e-5)
    assert len(r2.run.sessions()) == 2


def test_corrupt_last_falls_back_to_prev(cfg, tmp_path):
    t = training.Trainer(cfg.replace(run_id="fb", train=dataclasses.replace(cfg.train, epochs=1)),
                         RunStore(tmp_path / "fb"))
    t.fit()
    run = t.run
    (run.dir / "DONE").unlink()
    run.checkpoint("last").write_bytes(b"hong")          # simulate a file truncated when the machine stopped
    assert run.load_latest("last") is not None
    assert any(e["event"] == "checkpoint_fallback" for e in run.events())


def test_oom_enables_grad_checkpoint(cfg, tmp_path, monkeypatch):
    t = training.Trainer(cfg.replace(run_id="oom", train=dataclasses.replace(cfg.train, epochs=1)),
                         RunStore(tmp_path / "oom"))
    t.setup()
    real = t._step_once
    calls = {"n": 0}

    def flaky(b, clip):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(b, clip)

    monkeypatch.setattr(t, "_step_once", flaky)
    s = t.fit()
    assert s["status"] == "done" and s["grad_checkpoint"] and s["oom_events"] == 1
    assert t.run.events()[0]["action"] == "enable_grad_checkpoint"


def test_second_process_cannot_train_same_run(cfg, tmp_path):
    """A leftover job of an old pipeline (SSH dropped) + a re-run: a run whose lock is held stops the second process."""
    from lact.infra.system.locks import file_lock

    run = RunStore(tmp_path / "busy")
    t = training.Trainer(cfg.replace(run_id="busy"), run)
    with file_lock(run.dir / ".train.lock", blocking=False), pytest.raises(FileExistsError):
        t.fit()
    assert not run.checkpoint("last").exists() and run.status == "todo"
