import pytest
import torch

from lact.config import load_config
from lact.core.models import build_regularizer, build_unrolled, count_params
from lact.core.models.regularizers import regnet_stages

EXPECTED = {"learn_densenet": 2_695_553, "learn_orig": 60_097}      # learn_orig: one of its 30 unshared stages


def _reg(name: str):
    r = load_config([f"configs/model/{name}.yaml"]).require_model().regularizer
    return build_regularizer(r.name, **r.params)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_param_counts(name):
    assert count_params(_reg(name)) == EXPECTED[name]


@pytest.mark.parametrize("name", list(EXPECTED))
def test_shapes_and_backward(name):
    r = _reg(name)
    x = torch.randn(2, 1, 64, 64)
    y = r(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert all(p.grad is not None for p in r.parameters() if p.requires_grad)


def test_densenet_starts_as_zero_regularizer():
    """The last layer of LEARN-DenseNet is zero-initialised, so R = 0 and the first stages are plain data steps."""
    assert torch.count_nonzero(_reg("learn_densenet")(torch.randn(1, 1, 64, 64))) == 0


def test_unrolled_densenet_builds_from_its_configuration():
    m = load_config(["configs/model/learn_densenet.yaml"]).require_model()
    net = build_unrolled(m.regularizer.name, m.regularizer.params, img_size=64, K=m.K, share_theta=m.share_theta)
    assert net.K == 14 and net.share_theta and count_params(net) == EXPECTED["learn_densenet"] + 2 * 14


def test_unknown_regularizer_lists_the_available_ones():
    with pytest.raises(KeyError, match="densenet_ed"):
        build_regularizer("resnet_ed")


@pytest.mark.parametrize(("params", "widths", "depths"), [
    ((24, 36.44, 2.49, 13, 8), [24, 56, 152, 368], [1, 1, 4, 7]),        # RegNetX-200MF
    ((24, 24.48, 2.54, 22, 16), [32, 64, 160, 384], [1, 2, 7, 12]),      # RegNetX-400MF
    ((48, 36.97, 2.24, 16, 24), [48, 96, 240, 528], [1, 3, 5, 7]),       # RegNetX-600MF
    ((48, 27.89, 2.09, 16, 8), [48, 104, 208, 440], [1, 3, 6, 6]),       # RegNetY-400MF
])
def test_regnet_stages_match_published_models(params, widths, depths):
    """The quantised linear rule of the stage layouts reproduces the RegNet paper (Radosavovic et al., 2020)."""
    assert regnet_stages(*params) == (widths, depths)
