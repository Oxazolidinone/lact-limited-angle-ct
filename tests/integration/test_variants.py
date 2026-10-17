"""Data variants through the same pipeline services, without ASTRA (Siddon matrices):
synthetic DeepLesion -> split -> preprocessing -> matrices -> simulation (noise-free / Poisson noise) -> training of all
15 architectures -> evaluation -> aggregation."""
from __future__ import annotations

import dataclasses

import h5py
import numpy as np
import pytest

from lact.config import ConfigError, load_config
from lact.datasets.deeplesion import make_split
from lact.services import operators
from lact.services.data import fake as fake_data
from lact.services.data import prep as data_prep
from lact.services.data import simulation
from lact.services.evaluation import benchmark, recheck
from lact.services.evaluation import evaluate as evaluation
from lact.services.factories import dataset_for, results_store
from lact.services.reporting import per_method
from lact.services.training.trainer import Trainer
from lact.viz.style import ARCH_TO_REG

RN = [f"model.regularizer.regnet.{k}={v}" for k, v in (("w0", 8), ("wa", 12), ("wm", 2), ("depth", 6))]
# shrunk uniform layout; RN gives w = [8, 16, 32, 64], d = [1, 1, 2, 2]
TINY = ["model.regularizer.channels=[8,8,16,16]"]
VIT = ["model.regularizer.dim=32", "model.regularizer.depth=1", "model.regularizer.heads=2"]
SMALL = {"learn_cnn": ["model.regularizer.channels=8"],
         "learn_unet": TINY, "learn_unet_regnet": RN, "learn_resnet": TINY, "learn_resnet_regnet": RN,
         "learn_resnext": [*TINY, "model.regularizer.group_width=8"], "learn_regnet": RN,
         "learn_densenet": TINY, "learn_densenet_regnet": RN, "learn_efficientnet": TINY,
         "learn_efficientnet_regnet": RN, "learn_convnext": TINY, "learn_convnext_regnet": RN,
         "learn_vit": [*VIT, "model.regularizer.stem_channels=8"], "learn_vit_regnet": [*RN, *VIT]}
ARCHS = list(SMALL)


def _cfg(tmp, root, split, out_name, small_geom, extra=()):
    out = tmp / out_name
    paths = [f"paths.{k}={out / k}" for k in ("data", "runs", "eval", "tables", "figures", "baselines")]
    c = load_config(["configs/data/deeplesion.yaml", str(split)], paths + [
        f"paths.outputs={out}", f"paths.operators={tmp / 'operators'}", f"data.dicom_root={root}",
        f"data.info_csv={root}/DL_info.csv", "data.max_slices_per_patient=4", "data.img_size_sim=64",
        "data.img_size_rec=32", f"data.images_h5={tmp}/clean/data/images.h5",
        f"data.stats_json={tmp}/clean/data/stats.json", f"data.slices_json={tmp}/clean/data/slices.json",
        "train.epochs=1", "train.max_steps=3", "train.limit_val=2", "train.val_batch_size=2",
        "train.tensorboard=false", "train.log_every=1", "precision.deterministic=false", "exp=pytest", *extra])
    return c.replace(geometry=small_geom)


@pytest.fixture(scope="module")
def world(tmp_path_factory, small_geom):
    tmp = tmp_path_factory.mktemp("variants")
    root = fake_data.make_fake_deeplesion(tmp / "deeplesion", n_patients=12, n_slices=6, size=64)
    split = tmp / "split.yaml"
    make_split(root, split, n_train=4, n_val=2, n_test=2, min_slices=4, size=64)
    clean = _cfg(tmp, root, split, "clean", small_geom)
    noisy = _cfg(tmp, root, split, "noisy", small_geom, ["noise.type=poisson", "noise.I0=1e4"])
    data_prep.preprocess(clean)
    for c in (clean, noisy):
        operators.build_operators(c, [small_geom], sizes=[32, 64])     # shared matrices, separate manifest
        simulation.simulate_all(c, [small_geom], batch=2)
    return clean, noisy


def test_deeplesion_split_reaches_dataset(world, small_geom):
    clean, _ = world
    assert clean.split.name == "deeplesion_4_2_2" and all(p.startswith("DL") for p in clean.data.patients)
    tr, te = dataset_for(clean, "train"), dataset_for(clean, "test")
    assert len(tr) == 4 * 4 and len(te) == 2 * 4            # 4 evenly spaced slices per patient
    assert {b["pid"] for b in (te[i] for i in range(len(te)))} == set(clean.split.test)
    assert tr[0]["y"].shape == (1, small_geom.n_views, small_geom.n_det)
    data_prep.check_uid_disjoint(clean)


def test_noise_changes_measurements_but_not_references(world, small_geom):
    clean, noisy = world
    a, b = dataset_for(clean, "val"), dataset_for(noisy, "val")
    assert np.array_equal(a[0]["x"].numpy(), b[0]["x"].numpy())             # same reference image
    dy = (b[0]["y"] - a[0]["y"]).numpy()
    assert 0 < np.abs(dy).mean() < 0.05 * np.abs(a[0]["y"].numpy()).mean()  # noise small relative to the signal
    assert not np.array_equal(a[0]["x0"].numpy(), b[0]["x0"].numpy())       # FBP recomputed from noisy measurements
    with h5py.File(noisy.paths.sino_file(small_geom.name)) as f:
        assert '"poisson"' in f.attrs["noise"]
    # a sinogram file must not be reused for a different noise configuration
    other = noisy.replace(noise=dataclasses.replace(noisy.noise, I0=1e5))
    with pytest.raises(ConfigError, match="does not match the configuration"):
        simulation.simulate_geometry(other, small_geom)
    assert simulation.simulate_geometry(noisy, small_geom) == noisy.paths.sino_file(small_geom.name)


@pytest.mark.parametrize("arch", ARCHS)
def test_every_architecture_trains_and_evaluates_on_noisy_variant(world, arch):
    _, noisy = world
    model = load_config([f"configs/model/{arch}.yaml"], ["model.K=2", *SMALL[arch]]).require_model()
    cfg = noisy.replace(model=model)
    t = Trainer(cfg)
    s = t.fit()
    assert s["status"] == "done" and np.isfinite(s["best_val_psnr"]) and t.run.checkpoint("best").exists()
    evaluation.freeze_configs(cfg, "pytest")
    df = evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2)
    assert len(df) == 8 and np.isfinite(df.psnr).all() and set(df.pid) == set(cfg.split.test)


def test_results_aggregate_over_all_architectures(world):
    _, noisy = world
    df = results_store(noisy).load_slices(split="test")
    pm = per_method(df)
    assert sorted(pm.regularizer) == sorted(ARCH_TO_REG[a] for a in ARCHS) and len(pm) == 15
    assert (pm.n_seeds == 1).all() and np.isfinite(pm.psnr_mean).all()


def test_recheck_reproduces_recorded_validation_score(world):
    """Moving runs to another machine: re-evaluating the best checkpoint must give the recorded validation PSNR."""
    _, noisy = world
    df = recheck.recheck(noisy, ["pytest"], per_geometry=3, tol_db=1e-3)
    assert len(df) == 3 and df.arch.nunique() == 3 and df.ok.all()
    run = recheck.pick_runs(noisy, ["pytest"], per_geometry=1)[0]
    summary = run.summary()
    # as if the data on this machine differed
    run.mark({**summary, "best_val_psnr": summary["best_val_psnr"] + 0.5})
    try:
        bad = recheck.recheck(noisy, ["pytest"], per_geometry=1, tol_db=0.02)
        assert len(bad) == 1 and not bad.ok.any() and bad["diff"].iloc[0] == pytest.approx(-0.5, abs=1e-3)
    finally:
        run.mark(summary)
    assert recheck.recheck(noisy, ["khong_co"]).empty


def test_pilot_measures_and_survives_out_of_memory(world, monkeypatch):
    """Pilot: a model running out of GPU memory is re-measured with gradient checkpointing instead of failing the
    pilot."""
    clean, _ = world
    cfgs = {a: clean.replace(model=load_config([f"configs/model/{a}.yaml"], ["model.K=2", *SMALL[a]]).require_model())
            for a in ("learn_cnn", "learn_unet")}
    real, calls = benchmark._time_steps, []

    def flaky(model, opt, ds, steps, device):
        calls.append(model.grad_checkpoint)
        if len(calls) == 2:                                   # first measurement of the second model
            raise RuntimeError("CUDA out of memory (simulated)")
        return real(model, opt, ds, steps, device)

    monkeypatch.setattr(benchmark, "_time_steps", flaky)
    df = benchmark.pilot(cfgs, steps=2)
    assert calls == [False, False, True] and list(df.grad_checkpoint) == [False, True]
    assert np.isfinite(df.sec_per_step).all() and (df.n_train_slices == 16).all() and (df.gpu_hours_per_run > 0).all()

