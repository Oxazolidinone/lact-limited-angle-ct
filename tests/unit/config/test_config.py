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
    c = compose(["configs/train/bmvc.yaml", "configs/model/learn_densenet.yaml"], ["train.lr=3e-4", "seed=2"])
    assert c["train"]["optimizer"] == "adamw"             # bmvc overrides default
    assert c["train"]["epochs"] == 50                     # inherited from default
    assert c["model"]["regularizer"]["name"] == "densenet_ed"
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
    cfg = load_config(["configs/model/learn_densenet.yaml", "configs/geometry/fan_a120.yaml"], ["seed=3"])
    assert isinstance(cfg, Config)
    assert cfg.seed == 3 and cfg.require_model().regularizer.name == "densenet_ed"
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
    spec = ExperimentSpec.load("configs/experiment/cmp_mayo.yaml")
    assert spec.exp == "cmp_mayo" and spec.lr_from is None and len(spec.grid["model"]) == 3
    assert spec.lr_table == {"fbpconvnet": 1e-4, "lpd": 1e-4, "learn_orig": 1e-4}
    bad = tmp_path / "bad.yaml"
    bad.write_text("exp: x\ngrid:\n  optimizer: [adam]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ExperimentSpec.load(str(bad))
