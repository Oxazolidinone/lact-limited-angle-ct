import pytest

from lact.config import (
    Config,
    ConfigError,
    ExperimentSpec,
    SplitConfig,
    compose,
    config_hash,
    diff_configs,
    load_config,
    load_config_file,
    parse_value,
    save_yaml,
)


def test_compose_base_and_overrides():
    c = compose(["configs/train/sweep_lr.yaml", "configs/model/learn_vit_pm2m.yaml"], ["train.lr=3e-4", "seed=2"])
    assert c["train"]["epochs"] == 20                     # sweep_lr overrides default
    assert c["train"]["grad_clip"] == 1.0                 # inherited from default
    assert c["model"]["regularizer"]["dim"] == 128        # pm2m overrides learn_vit
    assert c["model"]["regularizer"]["patch"] == 8
    assert c["model"]["K"] == 14                          # from _common
    assert isinstance(c["train"]["lr"], float) and abs(c["train"]["lr"] - 3e-4) < 1e-12
    assert c["seed"] == 2


def test_parse_value_numbers():
    assert parse_value("1e-4") == 1e-4
    assert parse_value("[3e-5, 1e-4]") == [3e-5, 1e-4]
    assert parse_value("true") is True
    assert parse_value("L067") == "L067"


def test_hash_and_diff():
    a = {"x": 1, "y": {"z": 2}}
    b = {"x": 1, "y": {"z": 3}}
    assert config_hash(a) != config_hash(b)
    assert diff_configs(a, b) == {"y.z": (2, 3)}


def test_typed_config_and_roundtrip(tmp_path):
    cfg = load_config(["configs/model/learn_convnext.yaml", "configs/geometry/fan_a120.yaml"], ["seed=3"])
    assert isinstance(cfg, Config)
    assert cfg.seed == 3 and cfg.require_model().regularizer.name == "convnext_ed"
    assert cfg.require_geometry().n_views == 480
    save_yaml(cfg.to_dict(), tmp_path / "c.yaml")
    back = load_config_file(tmp_path / "c.yaml")
    assert back.to_dict() == cfg.to_dict()


def test_unknown_key_and_missing_sections():
    with pytest.raises(ConfigError):
        load_config(overrides=["train.lr_typo=1"])
    cfg = load_config()
    with pytest.raises(ConfigError):
        cfg.require_model()
    with pytest.raises(ConfigError):
        cfg.require_geometry()


def test_split_disjoint():
    with pytest.raises(ConfigError):
        SplitConfig(train=["A"], val=["A"], test=[])


def test_experiment_spec_validation(tmp_path):
    spec = ExperimentSpec.load("configs/experiment/e3_main.yaml")
    assert spec.exp == "e3_main" and spec.lr_from == "e2_sweep" and len(spec.grid["model"]) == 15
    bad = tmp_path / "bad.yaml"
    bad.write_text("exp: x\ngrid:\n  optimizer: [adam]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ExperimentSpec.load(str(bad))
