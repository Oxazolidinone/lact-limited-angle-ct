"""End-to-end pipeline run on small synthetic DICOM data, through the same services the CLI uses:
DICOM -> preprocessing -> matrices -> simulation -> dataset -> training -> evaluation -> aggregation."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from lact.config import ConfigError, SplitConfig, load_config
from lact.infra.data.dataset import LACTDataset
from lact.infra.stores.run_store import RunStore
from lact.services import operators
from lact.services.data import fake as fake_data
from lact.services.data import prep as data_prep
from lact.services.data import simulation
from lact.services.evaluation import evaluate as evaluation
from lact.services.factories import dataset_for, results_store
from lact.services.reporting import per_method, per_run
from lact.services.training.trainer import Trainer


@pytest.fixture(scope="module")
def cfg(tmp_path_factory, small_geom):
    pytest.importorskip("pydicom")
    tmp = tmp_path_factory.mktemp("pipe")
    fake_data.make_fake_dicom(tmp / "dicom", n_slices=3, size=64)
    out = tmp / "out"
    paths = [f"paths.{k}={out / k}" for k in ("data", "operators", "runs", "eval", "tables", "figures", "baselines")]
    c = load_config(["configs/model/learn_cnn.yaml"], paths + [
        f"paths.outputs={out}", f"data.dicom_root={tmp / 'dicom'}", "data.img_size_sim=64", "data.img_size_rec=32",
        f"data.images_h5={out}/data/images.h5", f"data.stats_json={out}/data/stats.json",
        f"data.slices_json={out}/data/slices.json", "data.augment=hflip",
        "model.K=3", "model.regularizer.channels=8", "train.epochs=2", "train.max_steps=6", "train.limit_val=4",
        "train.val_batch_size=2", "train.tensorboard=false", "train.log_every=1", "precision.deterministic=false",
        "exp=pytest"])
    c = c.replace(geometry=small_geom)
    data_prep.preprocess(c)
    # these functions also write the manifest, so the Trainer can verify SHA-256 as in a real pipeline
    operators.build_operators(c, [small_geom], sizes=[32, 64])
    simulation.simulate_all(c, [small_geom], batch=2)
    return c


def test_split_and_uids(cfg):
    data_prep.check_uid_disjoint(cfg)
    with pytest.raises(ConfigError):
        SplitConfig(train=["A"], val=["A"], test=[])


def test_dataset_shapes_and_augment(cfg, small_geom):
    tr = dataset_for(cfg, "train")
    va = dataset_for(cfg, "val")
    assert len(tr) == 7 * 3 * 2          # 7 patients x 3 slices x (original + flipped)
    assert len(va) == 3
    b = tr[0]
    assert b["y"].shape == (1, small_geom.n_views, small_geom.n_det)
    assert b["x0"].shape == b["x"].shape == (1, 32, 32)
    assert float(b["x"].min()) >= 0.0 and float(b["x"].max()) <= 1.0
    k_flip = next(i for i, it in enumerate(tr.items) if it[2] == "hflip")
    pid, idx, _ = tr.items[k_flip]
    k_orig = tr.items.index((pid, idx, "orig"))
    assert torch.allclose(tr[k_flip]["x"], tr[k_orig]["x"].flip(-1))
    lazy = LACTDataset(tr.images_h5, tr.sino_h5, [tr.items[0][0]], preload=False)
    assert torch.allclose(lazy[0]["y"], tr[0]["y"])


def test_train_resume_eval(cfg):
    t = Trainer(cfg)
    s = t.fit()
    assert s["status"] == "done" and t.run.status == "done" and t.run.checkpoint("best").exists()
    assert Trainer(cfg).fit()["status"] == "skipped"
    with pytest.raises(PermissionError):
        evaluation.evaluate_run(t.run, "test", device="cpu")
    evaluation.freeze_configs(cfg, "pytest")
    df = evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2)
    assert len(df) == 6 and np.isfinite(df.psnr).all()
    assert evaluation.evaluate_run(t.run, "test", device="cpu") is None      # already present -> skipped
    store = results_store(cfg)
    assert len(store.find(split="test", exp="pytest")) == 1
    assert len(store.load_slices(split="test")) == 6
    assert set(per_run(df).columns) >= {"psnr", "ssim", "rmse"}
    assert per_method(df).n_seeds.iloc[0] == 1


def test_determinism(cfg, tmp_path):
    losses = []
    for k in range(2):
        c = cfg.replace(run_id=f"det{k}", train=dataclasses.replace(cfg.train, epochs=1, max_steps=4))
        t = Trainer(c, RunStore(tmp_path / f"det{k}"))
        t.fit()
        losses.append(pd.read_csv(t.run.steps_csv).loss.to_numpy())
    assert np.allclose(losses[0], losses[1], rtol=1e-5, atol=1e-8)
