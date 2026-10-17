"""Pipeline figures (lact.viz.plots / panels) with many models: the layout must widen instead of crowding one row."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lact.viz import panels, plots
from lact.viz.style import ARCH_ORDER, OTHER_METHODS, color_of, label_of

LEARNED = [r for r in ARCH_ORDER if r not in ("fbp", "tv") and r not in OTHER_METHODS]     # 15 LEARN-framework models


def _method_df(regs):
    return pd.DataFrame([{"regularizer": r, "geometry_test": g, "psnr_mean": 20.0 + 0.2 * i, "psnr_std": 0.1}
                         for g in ("fan_a120", "fan_a90") for i, r in enumerate(["fbp", "tv", *regs])])


def test_metric_by_arch_scales_with_models():
    few = plots.metric_by_arch(_method_df(LEARNED[:4]))
    many = plots.metric_by_arch(_method_df(LEARNED))
    assert many.get_figwidth() > 1.8 * few.get_figwidth()
    assert {t.get_rotation() for t in many.axes[0].get_xticklabels()} == {40.0}
    assert {t.get_rotation() for t in few.axes[0].get_xticklabels()} == {0.0}
    plt.close("all")


def test_pareto_many_uses_legend_only():
    df = pd.DataFrame({"regularizer": LEARNED[::-1], "n_params_m": np.linspace(0.1, 4, 15),
                       "psnr_mean": np.linspace(29, 31, 15), "psnr_std": 0.1})
    fig = plots.pareto_plot(df, x="n_params_m", logx=True)
    assert not fig.axes[0].texts                                         # no overlapping direct labels
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [label_of(r) for r in LEARNED]
    plt.close("all")


def test_grids_wrap(tmp_path):
    rng = np.random.default_rng(0)
    fig = plots.heatmaps({r: rng.random((16, 16)) for r in LEARNED})
    assert len({round(a.get_position().y0, 3) for a in fig.axes[:15]}) == 2      # 15 panels -> 2 rows
    ref = rng.random((32, 32)).astype(np.float32)
    imgs = {f"M{i}": ref + 0.01 * i for i in range(17)}
    wide = panels.recon_grid(imgs, ref, 0.02, 0.019, roi=(4, 4, 8, 8), out_path=tmp_path / "g.png")
    one = panels.recon_grid(dict(list(imgs.items())[:6]), ref, 0.02, 0.019)
    assert wide.get_figwidth() < 1.9 * 11 and wide.get_figheight() > 1.9 * one.get_figheight()
    assert sum(a.get_title().startswith("Tham chiếu") for a in wide.axes) == 2    # one reference image per band
    assert (tmp_path / "g.png").stat().st_size > 1000
    plt.close("all")


def test_methods_outside_learn_have_labels_and_come_last():
    """Methods outside the 15 LEARN models (RegFormer and the methods of the CG-GLORE comparison table): own label
    and colour, ordered after those 15 models."""
    others = ["regformer", "fbpconvnet", "lpd", "learn_orig", "dpma", "cg_glore"]
    assert label_of("regformer") == "RegFormer" and ARCH_ORDER[-6:] == others and len(LEARNED) == 15
    assert [label_of(r) for r in others[1:]] == ["FBPConvNet", "Learned Primal-Dual", "LEARN (bản gốc)", "DPMA",
                                                 "CG-GLORE"]
    colors = [color_of(r) for r in others]
    assert len(set(colors)) == 6
    assert not set(colors) & ({color_of(r) for r in LEARNED} | {color_of("fbp"), color_of("tv")})
