import pytest
import torch

from lact.config import load_config
from lact.core.models import build_regularizer, build_unrolled, count_params
from lact.core.models.blocks import RegNetBlock
from lact.core.models.regularizers import regnet_stages

EXPECTED = {
    "learn_cnn": 60_097,
    "learn_unet": 2_323_489, "learn_unet_regnet": 2_323_497,
    "learn_resnet": 4_086_081, "learn_resnet_regnet": 4_228_759,
    "learn_resnext": 2_952_513, "learn_regnet": 3_141_269,                    # learn_regnet = RegNetY-200MF
    "learn_densenet": 2_695_553, "learn_densenet_regnet": 2_876_659,
    "learn_efficientnet": 2_512_257, "learn_efficientnet_regnet": 2_494_595,
    "learn_convnext": 1_979_201, "learn_convnext_regnet": 2_057_271,
    "learn_vit": 3_673_921, "learn_vit_regnet": 3_514_397,
    "learn_resnet_pm2m": 2_102_337, "learn_regnet_pm2m": 2_038_097, "learn_vit_pm2m": 1_866_497,
}
MAIN = [k for k in EXPECTED if "pm2m" not in k]


def _reg(name: str, img_size: int = 256):
    r = load_config([f"configs/model/{name}.yaml"]).require_model().regularizer
    params = dict(r.params, **({"img_size": img_size} if r.name in ("vit", "vit_regnet") else {}))
    return build_regularizer(r.name, **params)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_param_counts(name):
    assert count_params(_reg(name)) == EXPECTED[name]


def test_matched_group_within_10_percent():
    target = EXPECTED["learn_convnext"]
    for n in ("learn_resnet_pm2m", "learn_regnet_pm2m", "learn_vit_pm2m"):
        assert abs(EXPECTED[n] - target) / target <= 0.10


@pytest.mark.parametrize("name", MAIN)
def test_io_and_zero_init(name):
    r = _reg(name, img_size=64)
    x = torch.randn(2, 1, 64, 64)
    y = r(x)
    assert y.shape == x.shape
    assert torch.count_nonzero(y) == 0          # zero-initialised last layer => R = 0 at start
    y.sum().backward()                          # backpropagation works


def test_vit_takes_reconstruction_size():
    m = build_unrolled("vit", {"patch": 8, "dim": 64, "depth": 1, "heads": 2}, img_size=32, K=2)
    assert m.regularizer(0)(torch.zeros(1, 1, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(("params", "widths", "depths"), [
    ((24, 36.44, 2.49, 13, 8), [24, 56, 152, 368], [1, 1, 4, 7]),        # RegNetX-200MF
    ((24, 24.48, 2.54, 22, 16), [32, 64, 160, 384], [1, 2, 7, 12]),      # RegNetX-400MF
    ((48, 36.97, 2.24, 16, 24), [48, 96, 240, 528], [1, 3, 5, 7]),       # RegNetX-600MF
    ((48, 27.89, 2.09, 16, 8), [48, 104, 208, 440], [1, 3, 6, 6]),       # RegNetY-400MF
])
def test_regnet_stages_match_published_models(params, widths, depths):
    """The quantised linear rule must reproduce the configurations of the RegNet paper (Radosavovic et al., 2020)."""
    assert regnet_stages(*params) == (widths, depths)


def test_regnet_block_grouping_and_shortcut():
    blk = RegNetBlock(24, 56, stride=2, group_width=8)
    assert blk.conv2.groups == 7 and blk.proj is not None and blk.se is None
    assert blk(torch.randn(1, 24, 16, 16)).shape == (1, 56, 8, 8)
    same = RegNetBlock(56, 56, group_width=8, se_ratio=0.25)                # Y block
    assert same.proj is None and same.se is not None and same.se.fc1.out_channels == 14
    with pytest.raises(ValueError):
        RegNetBlock(24, 60, group_width=8)       # 60 is not divisible by the group width


def test_regnet_needs_four_stages_and_accepts_explicit_widths():
    with pytest.raises(ValueError, match="exactly 4 levels"):
        build_regularizer("regnet_ed", regnet={"depth": 2})      # yields only 2 stages
    m = build_regularizer("regnet_ed", channels=[16, 32, 64, 128], depths=[1, 1, 2, 2], group_width=8)
    assert [len(level) for level in m.enc] == [1, 0, 1, 1]      # the stride-2 block is the first block of stages 2-4
    assert m(torch.zeros(1, 1, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(("twin", "base"), [
    ("learn_unet_regnet", "learn_unet"), ("learn_resnet_regnet", "learn_resnet"), ("learn_regnet", "learn_resnext"),
    ("learn_densenet_regnet", "learn_densenet"), ("learn_efficientnet_regnet", "learn_efficientnet"),
    ("learn_convnext_regnet", "learn_convnext"), ("learn_vit_regnet", "learn_vit")])
def test_regnet_layout_twin_matches_family_size(twin, base):
    """The RegNet-layout variant of each family has a parameter count within 10% of the uniform layout."""
    assert abs(EXPECTED[twin] - EXPECTED[base]) / EXPECTED[base] <= 0.10


def test_regnet_layout_names_require_regnet_rule():
    with pytest.raises(ValueError, match="needs a regnet entry"):
        build_regularizer("resnet_regnet", channels=[32, 64, 128, 256], depths=[2, 2, 2, 2])
    m = build_regularizer("resnet_regnet", regnet={"w0": 10.8, "wa": 16.4})          # wm, depth of RegNetX-200MF
    assert [len(level) for level in m.enc] == [1, 1, 4, 7] and m.stem.out_channels == 8
    with pytest.raises(ValueError, match="invalid keys"):
        build_regularizer("unet_regnet", regnet={"width": 8})


def test_regnet_default_is_regnety_200mf_and_twins_carry_se():
    """The thesis RegNet is RegNetY-200MF: every X block has SE; the RegNet variants of the families do too."""
    reg = _reg("learn_regnet")
    n_blocks = sum(1 for m in reg.modules() if isinstance(m, RegNetBlock))
    assert n_blocks == 16 and all(m.se is not None for m in reg.modules() if isinstance(m, RegNetBlock))
    assert count_params(build_regularizer("regnet_ed", se_ratio=0.0)) == 2_649_465        # without SE = RegNetX-200MF
    for name in ("learn_unet_regnet", "learn_resnet_regnet", "learn_densenet_regnet", "learn_convnext_regnet"):
        twin, base = _reg(name), _reg(name.replace("_regnet", ""))
        n_se = lambda net: sum(1 for m in net.modules() if type(m).__name__ == "SqueezeExcite")
        assert n_se(twin) == 16 and n_se(base) == 0
