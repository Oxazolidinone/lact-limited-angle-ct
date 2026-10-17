"""Paper figure set (lact.viz.paper): style, statistics, LaTeX tables, and every plotting function on synthetic data."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from lact.viz import paper as pp
from lact.viz.paper import stats as S
from lact.viz.paper.figures import roi_by_structure

METHODS = ["fbp", "tv", "cnn3", "resnet_ed", "convnext_ed"]
GEOMS = ["fan_a90", "fan_a120"]
MU_REF, MU_W = 0.0379, 0.0192


@pytest.fixture(autouse=True)
def _style():
    pp.use("ieee", lang="en")
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def slices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {"fbp": 14, "tv": 22, "cnn3": 29, "resnet_ed": 31, "convnext_ed": 30.4}
    rows = []
    for g, off in zip(GEOMS, (0.0, 4.0)):
        for m in METHODS:
            for seed in ([0] if m in ("fbp", "tv") else [0, 1]):
                for i in range(20):
                    rows.append({"geometry": g, "method": m, "seed": seed, "pid": "P1", "idx": i,
                                 "psnr": base[m] + off + rng.normal(0, 0.5) + 0.05 * i,
                                 "ssim": 0.5 + base[m] / 80 + rng.normal(0, 0.01),
                                 "err_vis": rng.uniform(1, 3), "err_inv": rng.uniform(5, 9)})
    return pd.DataFrame(rows)


def _phantom(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    body = ((yy - n / 2) ** 2 / (0.4 * n) ** 2 + (xx - n / 2) ** 2 / (0.45 * n) ** 2) <= 1
    img = np.where(body, 1.0, 0.0) * 0.5
    img[n // 3:n // 3 + 6, n // 2:n // 2 + 6] = 0.9
    return img.astype(np.float32)


def _pdf_width_in(path) -> float:
    m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", path.read_bytes())
    assert m
    return float(m.group(1)) / 72


def test_presets_and_labels():
    p = pp.use("thesis", lang="vi")
    assert p.font == 10 and matplotlib.rcParams["font.size"] == 10
    assert pp.label("gt") == "Tham chiếu"
    assert pp.label("learn_resnet") == "LEARN-ResNet"           # alias
    pp.use("ieee", lang="en")
    assert pp.label("gt") == "Reference"
    assert pp.color("resnet_ed") == pp.color("learn_resnet_pm2m")  # same family -> same colour
    with pytest.raises(ValueError):
        pp.use("nature")


def test_stats():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = S.holm(p)
    assert np.all(adj >= p) and np.all(adj <= 1)
    assert adj[np.argsort(p)].tolist() == sorted(adj.tolist())
    x = np.random.default_rng(1).normal(3, 1, 200)
    lo, hi = S.bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert S.stars(1e-4) == "***" and S.stars(0.2) == "n.s."


def test_paired_vs(slices):
    pr = S.paired_vs(slices, "psnr", "resnet_ed", METHODS, "fan_a90")
    assert set(pr.method) == {"fbp", "tv", "cnn3", "convnext_ed"}
    assert (pr.set_index("method").loc["fbp", "delta_mean"] < 0)
    assert {"p_holm", "stars", "ci_lo", "ci_hi", "deltas"} <= set(pr.columns)


def test_summarize_uses_seed_std(slices):
    s = S.summarize(slices, "psnr")
    r = s[(s.geometry == "fan_a90") & (s.method == "cnn3")].iloc[0]
    assert r.n_seeds == 2 and r.n_slices == 20


def test_save_exact_width(tmp_path, slices):
    fig = pp.metric_strip(slices, "psnr", GEOMS, METHODS)
    paths = pp.save(fig, tmp_path, "strip", formats=("pdf", "png"))
    assert all(p.exists() for p in paths)
    assert _pdf_width_in(paths[0]) == pytest.approx(pp.PRESETS["ieee"].full, abs=0.01)


def test_all_figures_render(tmp_path, slices):
    ref = _phantom()
    imgs = {m: ref + np.random.default_rng(i).normal(0, 0.02, ref.shape).astype(np.float32)
            for i, m in enumerate(METHODS)}
    roi = roi_by_structure(ref, MU_REF, MU_W, size=12, margin=2)
    blk = {"label": "90°", "ref": ref, "images": imgs, "roi": roi, "metrics": {m: (30.0, 0.9) for m in METHODS}}
    figs = {"qual": pp.qualitative([blk, blk], METHODS, MU_REF, MU_W)}
    pr = pd.concat([S.paired_vs(slices, "psnr", "tv", METHODS, g) for g in GEOMS])
    figs["forest"] = pp.paired_forest(pr, "psnr", GEOMS, METHODS, ref="tv")
    figs["angle"] = pp.angle_lines(S.summarize(slices, "psnr"), METHODS, "PSNR (dB)")
    curve = pd.DataFrame({"epoch": np.arange(10), "val_psnr": np.linspace(25, 30, 10)})
    figs["curves"] = pp.curves({"cnn3": [curve, curve + 0.1], "resnet_ed": [curve + 1]})
    figs["lr"] = pp.lr_sweep({"cnn3": {1e-4: curve, 3e-4: curve + 1}}, best={"cnn3": 3e-4})
    n, J = 64, 8
    gates = (np.random.default_rng(0).uniform(size=(J, n, n)) > 0.4).astype(np.float32)
    src = np.stack([n / 2 + 40 * np.cos(np.linspace(3.5, 5.9, 30)), n / 2 - 40 * np.sin(np.linspace(3.5, 5.9, 30))], 1)
    figs["vis"] = pp.visibility({"fan_a90": gates.mean(0)}, {"fan_a90": gates}, {"fan_a90": src}, w="col")
    figs["split"] = pp.error_split(slices, METHODS, "fan_a90", n * n)
    figs["stage"] = pp.stage_params({"resnet_ed": {"alphas": [1.0] * 14, "betas": [0.9] * 14},
                                     "convnext_ed": {"alphas": [0.8] * 14, "betas": [0.7] * 14}})
    par = S.summarize(slices[slices.method.isin(["cnn3", "resnet_ed", "convnext_ed"])], "psnr")
    par = par[par.geometry == "fan_a90"].assign(params_m=[0.06, 2.0, 4.1])
    figs["pareto"] = pp.pareto(par, "params_m", "Parameters (M)", "PSNR (dB)", logx=True)
    for name, fig in figs.items():
        pp.save(fig, tmp_path, name, formats=("pdf",))
        assert (tmp_path / f"{name}.pdf").stat().st_size > 1000


def test_roi_inside_body():
    ref = _phantom(96)
    r0, c0, h, w = roi_by_structure(ref, MU_REF, MU_W, size=16, margin=4)
    assert r0 >= 0 and r0 + h <= 96 and c0 >= 0 and c0 + w <= 96
    hu = (ref[r0:r0 + h, c0:c0 + w] * MU_REF / MU_W - 1) * 1000
    assert (hu > -500).all()                                    # fully inside the body


def test_latex_table(slices):
    summ = {m: S.summarize(slices, m) for m in ("psnr", "ssim")}
    tex = pp.results_table(summ, GEOMS, METHODS, sig={("fan_a90", "resnet_ed"): True},
                           groups={"Non-learned": ["fbp", "tv"], "Learned": ["cnn3", "resnet_ed", "convnext_ed"]})
    assert tex.count(r"\textbf{") == 4 and tex.count(r"\underline{") == 4    # 2 geometries x 2 metrics
    assert r"$^\dagger$" in tex and r"90$^\circ$" in tex and r"\toprule" in tex
    assert tex.strip().endswith(r"\end{table*}")


def test_many_methods_layouts(tmp_path, recwarn):
    """15 models + 2 baselines: figures must change layout (wrapped rows, legends) and still fit the page width."""
    from lact.viz.style import ARCH_ORDER, OTHER_METHODS

    many = ["fbp", "tv", *[r for r in ARCH_ORDER if r not in ("fbp", "tv") and r not in OTHER_METHODS]]
    learned = many[2:]
    assert len(learned) == 15
    rng = np.random.default_rng(0)
    sl = pd.DataFrame([{"geometry": g, "method": m, "seed": 0, "pid": "P1", "idx": i,
                        "psnr": 20 + 0.3 * k + rng.normal(0, 0.3), "err_vis": 2.0, "err_inv": 6.0}
                       for g in GEOMS for k, m in enumerate(many) for i in range(8)])
    ref = _phantom()
    blk = {"label": "90°", "ref": ref, "images": {m: ref for m in many},
           "roi": roi_by_structure(ref, MU_REF, MU_W, size=12, margin=2), "metrics": {m: (30.0, 0.9) for m in many}}
    curve = pd.DataFrame({"epoch": np.arange(10), "val_psnr": np.linspace(25, 30, 10)})
    summ = S.summarize(sl, "psnr")
    par = summ[(summ.geometry == "fan_a90") & summ.method.isin(learned)]
    par = par.assign(params_m=np.linspace(0.06, 4.2, len(par)))
    for preset in ("thesis", "ieee"):
        pp.use(preset, lang="vi")
        one = pp.qualitative([blk], many[:6], MU_REF, MU_W)
        qual = pp.qualitative([blk], many, MU_REF, MU_W)
        rows = [len({round(a.get_position().y0, 3) for a in f.axes if a.images}) for f in (one, qual)]
        assert rows == [3, 6]                                             # two bands instead of one 18-column row
        titles = [a.get_title() for a in qual.axes if a.get_title()]
        assert titles.count(pp.label("gt")) == 2 and "EfficientNet-\nRegNet" in titles
        strip = pp.metric_strip(sl, "psnr", GEOMS, many)
        x0 = {round(a.get_position().x0, 3) for a in strip.axes}
        assert len(strip.axes) == 2 and len(x0) == 1                      # one row per scan range
        pareto = pp.pareto(par, "params_m", "Tham số (M)", "PSNR (dB)", logx=True)
        assert not pareto.axes[0].texts and len(pareto.legends[0].get_texts()) == 15
        lr = pp.lr_sweep({m: {1e-4: curve, 3e-4: curve + 1} for m in learned})
        assert len({round(a.get_position().y0, 3) for a in lr.axes}) == 3  # 3-row x 5-column grid
        stage = pp.stage_params({m: {"alphas": [1.0] * 14, "betas": [0.9] * 14} for m in learned})
        figs = {"qual": qual, "strip": strip, "pareto": pareto, "lr": lr, "stage": stage,
                "angle": pp.angle_lines(summ, many, "PSNR (dB)"),
                "curves": pp.curves({m: [curve + 0.1 * k] for k, m in enumerate(learned)}),
                "split": pp.error_split(sl, many, "fan_a90", 64 * 64)}
        for name, fig in figs.items():
            pp.save(fig, tmp_path, f"{preset}_{name}", formats=("png",))
    assert not [w for w in recwarn.list if "content is" in str(w.message)]   # no figure exceeds the width
