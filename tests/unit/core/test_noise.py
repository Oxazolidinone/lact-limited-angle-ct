"""Measurement noise: Poisson statistics, reproducibility and configuration checks."""
from __future__ import annotations

import math

import pytest
import torch

from lact.config import ConfigError, load_config
from lact.core.imaging.noise import add_noise

MU_REF = 0.04


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def test_none_returns_input_unchanged():
    y = torch.rand(2, 1, 8, 16)
    assert add_noise(y, "none", MU_REF, _gen()) is y


@pytest.mark.parametrize(("p0", "i0"), [(1.0, 1e4), (3.0, 1e5)])
def test_poisson_matches_beer_lambert_statistics(p0, i0):
    """Measurement -ln(N / I0) with N ~ Poisson(I0 e^{-p}): mean ~ p, variance ~ e^{p} / I0."""
    y = torch.full((200_000,), p0 / MU_REF)            # y on the normalised scale: line integral = y * mu_ref
    p = add_noise(y, "poisson", MU_REF, _gen(), I0=i0) * MU_REF
    var = math.exp(p0) / i0
    assert abs(float(p.mean()) - p0) < 0.02 * math.sqrt(var) + var        # second-order bias ~ var / 2
    assert float(p.var()) == pytest.approx(var, rel=0.05)


def test_poisson_noise_grows_with_attenuation_and_dose_reduction():
    thin, thick = torch.full((50_000,), 1.0 / MU_REF), torch.full((50_000,), 5.0 / MU_REF)
    s = lambda y, i0: float((add_noise(y, "poisson", MU_REF, _gen(), I0=i0) - y).std())
    assert s(thick, 1e5) > 5 * s(thin, 1e5)             # rays through thick regions are noisier (e^{(5-1)/2} ~ 7.4x)
    assert s(thin, 1e5) == pytest.approx(math.sqrt(10) * s(thin, 1e6), rel=0.05)


def test_photon_starvation_stays_finite():
    y = torch.full((1000,), 40.0 / MU_REF)              # e^{-40} I0 ~ 0 photon
    out = add_noise(y, "poisson", MU_REF, _gen(), I0=1e5)
    assert torch.isfinite(out).all() and float(out.max()) <= math.log(1e5) / MU_REF + 1e-3


def test_same_seed_same_noise_and_gaussian_scale():
    y = torch.rand(4, 1, 16, 32) * 100
    a, b = add_noise(y, "poisson", MU_REF, _gen(7)), add_noise(y, "poisson", MU_REF, _gen(7))
    assert torch.equal(a, b) and not torch.equal(a, add_noise(y, "poisson", MU_REF, _gen(8)))
    g = add_noise(y, "gaussian", MU_REF, _gen(), sigma=0.01)
    assert float((g - y).std()) == pytest.approx(0.01 * float(y.abs().max()), rel=0.1)
    with pytest.raises(ValueError):
        add_noise(y, "salt", MU_REF, _gen())


def test_noise_config_is_validated():
    assert load_config().noise.type == "none"           # default: noise-free dataset
    with pytest.raises(ConfigError):
        load_config(overrides=["noise.type=poison"])
    with pytest.raises(ConfigError):
        load_config(overrides=["noise.type=poisson", "noise.I0=0"])
