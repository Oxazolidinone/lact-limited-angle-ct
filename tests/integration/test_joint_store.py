"""The joint Mayo + DeepLesion store on synthetic data, through the same services as the CLI (no ASTRA needed):
store on the Mayo intensity scale -> first-order, L-BFGS and compared runs -> evaluation on both datasets. Experiment
names in this file are only labels of synthetic runs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from lact.config import ConfigError, load_config
from lact.config.loader import save_yaml
from lact.infra.io.files import load_json
from lact.infra.io.hdf5_store import ImageStore
from lact.services import operators
from lact.services.data import fake as fake_data
from lact.services.data import prep as data_prep
from lact.services.data import simulation
from lact.services.evaluation import evaluate as evaluation
from lact.services.evaluation import recheck
from lact.services.factories import dataset_for, results_store
from lact.services.training.trainer import Trainer, find_init_run, run_store_for

MAYO = fake_data.FAKE_PATIENTS
MAYO_SPLIT = {"train": MAYO[:7], "val": ["L333"], "test": ["L310", "L506"]}
DL = [f"DL{p:06d}" for p in range(1, 9)]
DL_SPLIT = {"train": DL[:4], "val": DL[4:6], "test": DL[6:]}
COMMON = ["data.img_size_sim=64", "data.img_size_rec=32", "model.K=2",
          "train.epochs=1", "train.max_steps=4", "train.limit_val=4", "train.val_batch_size=2",
          "train.tensorboard=false", "train.log_every=1", "precision.deterministic=false"]
SMALL = {"regformer": ["model.regularizer.dim=8", "model.regularizer.heads=2"],      # scaled-down version of each model
         "fbpconvnet": ["model.regularizer.base=4"], "lpd": ["model.regularizer.channels=8"],
         "learn_orig": ["model.regularizer.channels=8"],
         "dpma": ["model.regularizer.width=4", "model.regularizer.chunk_size=4", "model.regularizer.target_name=t240",
                  "model.regularizer.refine_iters=4"]}


def _paths(out, tmp):
    return [f"paths.{k}={out / k}" for k in ("data", "runs", "eval", "tables", "figures", "baselines")] + [
        f"paths.outputs={out}", f"paths.operators={tmp / 'operators'}", f"data.images_h5={out}/data/images.h5",
        f"data.stats_json={out}/data/stats.json", f"data.slices_json={out}/data/slices.json"]


@pytest.fixture(scope="module")
def world(tmp_path_factory, small_geom):
    pytest.importorskip("pydicom")
    tmp = tmp_path_factory.mktemp("phase2")
    fake_data.make_fake_dicom(tmp / "dicom", n_slices=3, size=64)
    dl_root = fake_data.make_fake_deeplesion(tmp / "deeplesion", n_patients=8, n_slices=6, size=64)
    # 1) Mayo store: mu_ref from the Mayo training split
    mayo = load_config(["configs/model/learn_orig.yaml"], _paths(tmp / "mayo", tmp) + [
        f"data.dicom_root={tmp / 'dicom'}", "exp=mayo_x", *COMMON, *SMALL["learn_orig"]]).replace(geometry=small_geom)
    data_prep.preprocess(mayo)
    mu_ref = load_json(mayo.data.stats_json)["mu_ref"]
    # 2) joint store: same mu_ref plus DeepLesion patients; each split is a file like configs/data/split_joint*
    splits = {"joint": {k: MAYO_SPLIT[k] + DL_SPLIT[k] for k in MAYO_SPLIT},
              "joint_dl": {"train": DL_SPLIT["train"], "val": DL_SPLIT["val"],
                           "test": MAYO_SPLIT["test"] + DL_SPLIT["test"]},
              "joint_mayo": {**MAYO_SPLIT, "test": MAYO_SPLIT["test"] + DL_SPLIT["test"]}}
    for name, sp in splits.items():
        save_yaml({"split": {"name": f"split_{name}", **sp}}, tmp / f"split_{name}.yaml")

    def joint(split, extra=(), model="learn_orig"):
        files = ["configs/data/joint.yaml", str(tmp / f"split_{split}.yaml"), f"configs/model/{model}.yaml"]
        return load_config(files, _paths(tmp / "joint", tmp) + [
            f"data.dicom_root={tmp / 'dicom'}", f"data.deeplesion_root={dl_root}",
            f"data.info_csv={dl_root}/DL_info.csv", "data.max_slices_per_patient=4",
            f"data.norm.mu_ref={mu_ref!r}", f"data.patients=[{', '.join(MAYO + DL)}]",
            *COMMON, *SMALL[model], *extra]).replace(geometry=small_geom)

    data_prep.preprocess(joint("joint"))
    for c in (mayo, joint("joint")):
        operators.build_operators(c, [small_geom], sizes=[32, 64])
        simulation.simulate_all(c, [small_geom], batch=2)
    evaluation.freeze_configs(mayo, "pytest")
    evaluation.freeze_configs(joint("joint"), "pytest")
    return {"mayo": mayo, "joint": joint, "tmp": tmp}


def _fit(cfg):
    t = Trainer(cfg)
    s = t.fit()
    assert s["status"] == "done" and np.isfinite(s["best_val_psnr"])
    return t, s


@pytest.fixture(scope="module")
def runs(world):
    """Runs shared by the tests below (each run trains once)."""
    j = world["joint"]
    out = {"mayo_gd": _fit(world["mayo"]),
           "dl_gd": _fit(j("joint_dl", ["exp=p1_dl"])),
           "joint_gd": _fit(j("joint", ["exp=p2_joint"])),
           "mayo_regformer": _fit(j("joint_mayo", ["exp=p0_mayo"], model="regformer")),
           "joint_regformer": _fit(j("joint", ["exp=p2_joint"], model="regformer"))}
    for key, (t, _) in out.items():
        if key != "mayo_gd":
            assert evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2) is not None
    return out


def test_joint_store_holds_both_datasets_on_the_mayo_scale(world):
    mayo, j = world["mayo"], world["joint"]("joint")
    a, b = ImageStore(mayo.data.images_h5), ImageStore(j.data.images_h5)
    assert b.mu_ref == a.mu_ref and load_json(j.data.stats_json)["mode"] == "fixed"
    assert set(b.patients()) == set(MAYO + DL) and b.n_slices("DL000001") == 4 and b.n_slices("L067") == 3
    # Mayo images in the joint store match the Mayo store exactly
    for pid in ("L067", "L333", "L506"):
        assert np.array_equal(a.read(pid, "img256"), b.read(pid, "img256"))
    assert len(dataset_for(j, "train")) == 7 * 3 + 4 * 4 and len(dataset_for(j, "test")) == 2 * 3 + 2 * 4
    assert len(dataset_for(world["joint"]("joint_dl"), "train")) == 4 * 4
    assert data_prep.patient_source(j, "DL000003")[0] == "deeplesion"
    assert data_prep.patient_source(j, "L067")[0] == "dicom"
    with pytest.raises(ConfigError):
        load_config([], ["data.norm.mode=fixed"])                                # mu_ref missing
    with pytest.raises(ValueError, match="deeplesion_root"):
        data_prep.patient_source(load_config(["configs/data/joint.yaml"], ["data.deeplesion_root=null"]), "DL000003")


def test_mayo_run_evaluated_on_joint_store(world, runs):
    """A run trained on the Mayo store evaluated on the joint store: Mayo patients reproduce the old results, plus
    DeepLesion."""
    run = runs["mayo_gd"][0].run
    own = evaluation.evaluate_run(run, "test", device="cpu", batch_size=2)
    dcfg = world["joint"]("joint_mayo")
    cross = evaluation.evaluate_run(run, "test", device="cpu", batch_size=2, data_cfg=dcfg)
    assert set(cross.pid) == {"L310", "L506", "DL000007", "DL000008"} and (cross.train_split == "split_7_1_2").all()
    both = own.merge(cross, on=["pid", "idx"], suffixes=("_own", "_joint"))
    assert len(both) == 6 and np.allclose(both.psnr_own, both.psnr_joint, atol=1e-4)
    assert results_store(dcfg).find(exp="mayo_x", split="test")                  # written to the joint store's results
    assert evaluation.evaluate_run(run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is None    # already present
    # check validation PSNR on the joint store (same validation patient L333)
    assert recheck.val_psnr(run, "cpu", dcfg) == pytest.approx(run.summary()["best_val_psnr"], abs=1e-4)
    # runs moved to another machine are found by experiment in the joint store's runs
    import shutil

    shutil.copytree(run.dir, dcfg.paths.runs / "mayo_x" / run.run_id)
    df = recheck.recheck(dcfg, ["mayo_x"], tol_db=1e-3, on_config_data=True)
    assert len(df) == 1 and df.ok.all()


def test_regformer_runs_through_the_same_pipeline(world, runs):
    """RegFormer (a model outside LEARN) trains, saves, reloads and evaluates through the LEARN services."""
    t, s = runs["mayo_regformer"]
    assert s["arch"] == "regformer" and s["update"] == "fbp" and s["metric"] is None
    assert t.run.run_id.startswith("p0_mayo_regformer_") and s["n_params_regularizer"] == s["n_params"] - 2
    m = pd.read_csv(t.run.metrics_csv)
    assert "weights" in m.columns and "alphas" not in m.columns and "val_cg_iters" not in m.columns
    df = results_store(world["joint"]("joint")).load_slices(split="test", exp="p0_mayo")
    assert len(df) == 14 and set(df["update"]) == {"fbp"} and set(df.train_split) == {"split_joint_mayo"}
    assert np.isfinite(df.psnr).all() and (df.regularizer == "regformer").all()
    # the reloaded best checkpoint reproduces the recorded validation PSNR (the internal FBP is rebuilt from the
    # projector)
    assert recheck.val_psnr(t.run, "cpu") == pytest.approx(s["best_val_psnr"], abs=1e-4)


def test_lbfgs_retrain_from_trained_runs(world, runs):
    """L-BFGS attached to a trained run through train.init_from, for both the LEARN framework and RegFormer: the
    regularizer and beta are loaded, alpha restarts at 1."""
    import torch

    j = world["joint"]
    # source: the first listed experiment with a matching run; a source with another seed is accepted when it is the
    # only one
    cfg = j("joint", ["exp=lbfgs_x", "model.update=lbfgs", "train.init_from=khong_co,p2_joint"])
    src = find_init_run(cfg)
    assert src.run_id == runs["joint_gd"][0].run_id
    assert find_init_run(j("joint", ["seed=3", "model.update=lbfgs", "train.init_from=p2_joint"])).run_id == src.run_id
    t = Trainer(cfg)
    assert t.setup()
    ref, own = torch.load(src.checkpoint("best"), weights_only=False)["model"], t.model.state_dict()
    assert all(torch.equal(own[k], ref[k]) for k in own if k.startswith("reg.") or k == "b")
    assert "a" in ref and "a" not in own and torch.allclose(t.model.alphas, torch.ones(2))    # alpha reset to 1
    s = t.fit()
    assert s["status"] == "done" and s["update"] == "lbfgs" and s["metric"] is None and s["init_from"] == src.run_id
    assert {"alphas", "betas"} <= set(pd.read_csv(t.run.metrics_csv).columns)
    assert [e["run_id"] for e in t.run.events() if e.get("event") == "init_from"] == [src.run_id]
    with pytest.raises(ConfigError, match="init_from"):
        find_init_run(j("joint", ["train.init_from=khong_co"]))
    # RegFormer: regularization blocks from the FBP run, the FBP weight dropped, a new alpha step
    cfg_r = j("joint_mayo", ["exp=lbfgs_x", "model.update=lbfgs", "train.init_from=p0_mayo,mayo_x"], model="regformer")
    t2 = Trainer(cfg_r)
    assert t2.setup()
    src_r = runs["mayo_regformer"][0].run
    ref, own = torch.load(src_r.checkpoint("best"), weights_only=False)["model"], t2.model.state_dict()
    assert all(torch.equal(own[k], ref[k]) for k in own if k.startswith("stages."))
    assert "w" in ref and "w" not in own and "a_qn" in own
    s2 = t2.fit()
    assert s2["status"] == "done" and s2["update"] == "lbfgs" and s2["init_from"] == src_r.run_id
    for tr in (t, t2):
        df = evaluation.evaluate_run(tr.run, "test", device="cpu", batch_size=2)
        assert np.isfinite(df.psnr).all() and set(df["update"]) == {"lbfgs"}


def test_one_grid_command_trains_sources_then_dependents(world):
    """As on the server: ONE queue for both the first-order and the L-BFGS grids, run as real child processes. L-BFGS
    runs queued FIRST still wait for their source run to be DONE and load exactly its checkpoint."""
    from lact.services import scheduling as experiments

    tmp = world["tmp"]
    c = world["joint"]("joint_dl")
    save_yaml({"geometry": {"name": "t120", "span_deg": 120.0, "step_deg": 2.0, "n_det": 96, "det_spacing_mm": 10.0}},
              tmp / "t120.yaml")
    common = {"configs": ["configs/paths.yaml", "configs/data/mayo_3mm.yaml", str(tmp / "split_joint_dl.yaml"),
                          "configs/train/default.yaml"],
              "lr_table": {"learn_orig": 1.0e-4},
              "grid": {"model": ["configs/model/learn_orig.yaml"], "geometry": [str(tmp / "t120.yaml")], "seed": [0]}}
    save_yaml({"exp": "grid_base", **common}, tmp / "grid_base.yaml")
    save_yaml({"exp": "grid_lbfgs", **common, "overrides": ["model.update=lbfgs", "train.init_from=grid_base"]},
              tmp / "grid_lbfgs.yaml")
    sets = _paths(tmp / "joint", tmp) + [
        f"data.dicom_root={c.data.dicom_root}", f"data.deeplesion_root={c.data.deeplesion_root}",
        f"data.info_csv={c.data.info_csv}", "data.max_slices_per_patient=4", f"data.norm.mu_ref={c.data.norm.mu_ref!r}",
        f"data.patients=[{', '.join(MAYO + DL)}]", *COMMON, *SMALL["learn_orig"], "device=cpu"]
    jobs = [j for spec in ("grid_lbfgs", "grid_base")                               # dependent runs queued first
            for j in experiments.expand(str(tmp / f"{spec}.yaml"), ["configs/data/joint.yaml"], sets, require_lr=True)]
    assert [j.run_id.split("_learn_")[0] for j in jobs] == ["grid_lbfgs", "grid_base"] and all(j.pending for j in jobs)
    assert experiments.run_jobs(jobs, gpus=["GPU-pytest"], per_gpu=2) == [0, 0]
    base, lb = (run_store_for(load_config(j.files, j.overrides, use_defaults=False)) for j in jobs[::-1])
    assert base.status == lb.status == "done"
    assert lb.summary()["init_from"] == base.run_id and lb.summary()["update"] == "lbfgs"
    assert (lb.dir / "config.yaml").stat().st_mtime >= (base.dir / "DONE").stat().st_mtime       # ran AFTER its source
    # missing source: the dependent run never starts and the command returns an error code instead of hanging
    save_yaml({"exp": "grid_orphan", **common, "overrides": ["model.update=lbfgs", "train.init_from=khong_co"]},
              tmp / "grid_orphan.yaml")
    orphan = experiments.expand(str(tmp / "grid_orphan.yaml"), ["configs/data/joint.yaml"], sets, require_lr=True)
    codes = experiments.run_jobs(orphan, gpus=["GPU-pytest"], per_gpu=2)
    assert codes and all(code != 0 for code in codes)
    cfg = load_config(orphan[0].files, orphan[0].overrides, use_defaults=False)
    assert run_store_for(cfg).status == "todo"


@pytest.mark.parametrize(("model", "tag", "stage_cols"), [
    ("fbpconvnet", "post", set()), ("lpd", "pd", set()),
    ("learn_orig", "gd", {"alphas", "betas"})])
def test_compared_methods_run_through_the_same_pipeline(world, model, tag, stage_cols):
    """Compared methods train, save, reload and evaluate through the services of the other runs."""
    j = world["joint"]
    t, s = _fit(j("joint_mayo", ["exp=cmp_x"], model=model))
    assert s["arch"] == model and s["update"] == tag and s["metric"] is None
    assert t.run.run_id.startswith(f"cmp_x_{model}_") and 0 < s["n_params_regularizer"] <= s["n_params"]
    cols = set(pd.read_csv(t.run.metrics_csv).columns)
    assert stage_cols <= cols and "val_cg_iters" not in cols
    assert not ({"alphas", "betas", "weights"} - stage_cols) & cols
    dcfg = j("joint")
    assert evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is not None
    df = results_store(dcfg).load_slices(split="test", exp="cmp_x")
    df = df[df.arch == model]
    assert len(df) == 14 and set(df["update"]) == {tag} and np.isfinite(df.psnr).all()
    assert recheck.val_psnr(t.run, "cpu") == pytest.approx(s["best_val_psnr"], abs=1e-4)


def test_dpma_trains_in_two_runs_like_the_original(world, small_geom):
    """DPMA through the services: a stage-1 run trains M1; a stage-2 run loads that M1 (frozen) and trains M2;
    evaluation adds the refinement iterations, while the validation PSNR recorded during training is that of the
    learned part."""
    j = world["joint"]
    base = j("joint_mayo", model="dpma")
    operators.build_operators(base, [small_geom.replace(name="t240", span_deg=240.0)], sizes=[32])
    t1, s1 = _fit(j("joint_mayo", ["exp=dpma1_x", "model.regularizer.stage=1"], model="dpma"))
    assert s1["arch"] == "dpma" and s1["update"] == "dual" and t1.model.m2 is None
    cfg2 = j("joint_mayo", ["exp=dpma_x", "train.init_from=dpma1_x"], model="dpma")
    assert find_init_run(cfg2).run_id == t1.run.run_id
    t2, s2 = _fit(cfg2)
    # only M2 learns, same size as M1
    assert s2["init_from"] == t1.run.run_id and s2["n_params"] == s1["n_params"]
    best1 = torch.load(t1.run.checkpoint("best"), map_location="cpu", weights_only=False)["model"]
    own = t2.model.state_dict()
    assert all(torch.equal(own[k], v) for k, v in best1.items()) and any(k.startswith("m2.") for k in own)
    with pytest.raises(ConfigError):            # without a stage-1 run, stage 2 cannot start
        Trainer(j("joint_mayo", ["exp=dpma_y", "train.init_from=khong_co"], model="dpma")).fit()
    assert recheck.val_psnr(t2.run, "cpu") == pytest.approx(s2["best_val_psnr"], abs=1e-4)
    dcfg = j("joint")
    assert evaluation.evaluate_run(t2.run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is not None
    df = results_store(dcfg).load_slices(split="test", exp="dpma_x")
    assert len(df) == 14 and set(df["update"]) == {"dual"} and np.isfinite(df.psnr).all()
