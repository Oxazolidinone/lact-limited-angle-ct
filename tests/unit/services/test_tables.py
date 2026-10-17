"""Report tables with additional backbones: ordering, the parameter-matched group, supplementary tables (new
backbones, RegNet layout). Expected column headers and labels are the Vietnamese report content."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lact.services.reporting.tables import (
    ReportSpec,
    registered_tests,
    supplementary_tests,
    table_cross_geometry,
    table_main,
    table_matched,
    table_stats,
)

REGS = {"learn_cnn": ("cnn3", 29.0), "learn_resnet": ("resnet_ed", 32.0),
        "learn_resnet_regnet": ("resnet_regnet", 32.6), "learn_resnext": ("resnext_ed", 30.5),
        "learn_regnet": ("regnet_ed", 31.5),
        "learn_convnext": ("convnext_ed", 31.0), "learn_vit": ("vit", 28.0)}
MATCHED = {"learn_resnet_pm2m": ("resnet_ed", 31.6), "learn_regnet_pm2m": ("regnet_ed", 31.2),
           "learn_vit_pm2m": ("vit", 27.8)}


def _rows(exp, arch, reg, g_train, g_test, base, rng, seeds=(0, 1, 2), n=12):
    out = []
    for seed in seeds:
        for pid in ("L310", "L506"):
            v = base + rng.normal(0, 0.1, n) + (0.0 if g_test == "fan_a120" else -5.0)
            out += [{"exp": exp, "arch": arch, "regularizer": reg, "geometry_train": g_train, "geometry_test": g_test,
                     "split": "test", "run_id": f"{exp}_{arch}_{g_train}_s{seed}", "seed": seed, "pid": pid, "idx": i,
                     "psnr": float(v[i]), "ssim": 0.9, "rmse": 0.02, "rmse_hu": 40.0, "n_params": 2_000_000}
                    for i in range(n)]
    return out


def _results() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in ("fan_a120", "fan_a90"):
        for name, base in (("fbp", 15.0), ("tv", 24.0)):
            rows += _rows("baseline", name, name, g, g, base, rng, seeds=(0,))
        for arch, (reg, base) in REGS.items():
            rows += _rows("e3_main", arch, reg, g, g, base, rng)
            other = "fan_a90" if g == "fan_a120" else "fan_a120"
            rows += _rows("e3_main", arch, reg, g, other, base - 3.0, rng)          # E6: cross-geometry inference
    for arch, (reg, base) in MATCHED.items():
        rows += _rows("e4_matched", arch, reg, "fan_a120", "fan_a120", base, rng)
    return pd.DataFrame(rows)


def test_main_table_lists_architectures_in_fixed_order():
    t = table_main(_results(), ReportSpec(n_boot=50))
    assert list(t["Phương pháp"]) == ["FBP", "TV", "LEARN–CNN", "LEARN–ResNet", "LEARN–ResNet–RegNet", "LEARN–ResNeXt",
                                      "LEARN–RegNet", "LEARN–ConvNeXt", "LEARN–ViT"]
    assert t.loc[4, "Seed"] == 3 and "±" in t.loc[4, "PSNR (dB) 120°"]


def test_matched_and_cross_geometry_tables_include_regnet():
    df, spec = _results(), ReportSpec(n_boot=50)
    m = table_matched(df, spec)
    # E4 scaled-down versions (ResNet, RegNet, ViT) + other architectures taken from E3; no full versions of those three
    assert list(m["Phương pháp"]) == ["LEARN–ResNet", "LEARN–ResNet–RegNet", "LEARN–ResNeXt", "LEARN–RegNet",
                                      "LEARN–ConvNeXt", "LEARN–ViT"]
    assert list(m["Nguồn"]) == ["E4 (thu nhỏ)", "E3", "E3", "E4 (thu nhỏ)", "E3", "E4 (thu nhỏ)"]
    assert {"Tham số (M)", "So với 2M"} <= set(m.columns)
    c = table_cross_geometry(df, spec)
    assert set(c["Phương pháp"]) >= {"LEARN–RegNet", "LEARN–ResNet–RegNet"}
    assert {"Test 120°", "Test 90°"} <= set(c.columns)


def test_supplementary_comparisons_stay_outside_registered_family():
    df, spec = _results(), ReportSpec(n_boot=200)
    reg = registered_tests(df, spec)
    assert [q for q, _, _ in reg] == ["RQ1"] * 4 + ["RQ2"] * 3 + ["RQ3"] * 3        # registered family unchanged
    assert not any("RegNet" in name or "ResNeXt" in name for _, name, _ in reg)
    sup = supplementary_tests(df, spec)
    # BS1: 2 new backbones (ResNeXt, RegNet) x 2 angles; BS2: 2 pairs (RegNet layout, uniform layout) x 2 angles; BS3:
    # ~2M group
    assert [q for q, _, _ in sup] == ["BS1"] * 4 + ["BS2"] * 4 + ["BS3"] * 3
    by = table_stats(sup).set_index("So sánh")
    assert by.loc["LEARN–RegNet vs LEARN–CNN, 120°", "Kết luận"] == "A tốt hơn"
    assert by.loc["LEARN–ResNet–RegNet vs LEARN–ResNet, 120°", "Kết luận"] == "A tốt hơn"   # RegNet helps ResNet
    assert by.loc["LEARN–RegNet vs LEARN–ResNeXt, 90°", "Kết luận"] == "A tốt hơn"
    assert by.loc["LEARN–RegNet vs LEARN–CNN, 120°", "BN có Δ > 0"] == "2/2"
    assert by.loc["LEARN–RegNet vs LEARN–CNN, 120°", "p Wilcoxon theo BN"] == "–"     # only 2 patients
    old_only = df[df.regularizer.isin(["fbp", "tv", "cnn3", "resnet_ed", "convnext_ed", "vit"])]
    assert supplementary_tests(old_only, spec) == [] and table_stats([]) is None    # new models not run yet


def test_seed_selection_reports_one_run_per_configuration():
    """Several trained seeds: reports use the seed with the best VALIDATION PSNR; single-seed setups are unchanged."""
    from lact.services.reporting.seeds import choose, chosen_run_ids, keep_chosen
    from lact.services.reporting.tables import table_seeds

    runs = pd.DataFrame([
        {"exp": "e3_main", "arch": "learn_resnet", "regularizer": "resnet_ed", "geometry": "fan_a120", "seed": s,
         "run_id": f"r{s}", "best_val_psnr": v} for s, v in ((0, 31.0), (1, 31.4), (2, 31.4))] + [
        {"exp": "e3_main", "arch": "learn_unet", "regularizer": "unet_ed", "geometry": "fan_a120", "seed": 0,
         "run_id": "u0", "best_val_psnr": 30.0}])
    best = choose(runs, "best_val")
    assert chosen_run_ids(best) == {"r1", "u0"} and list(best.n_seeds) == [3, 3, 3, 1]    # ties: the smaller seed
    assert chosen_run_ids(choose(runs, "first")) == {"r0", "u0"}
    assert chosen_run_ids(choose(runs, "all")) == {"r0", "r1", "r2", "u0"}
    with pytest.raises(ValueError):
        choose(runs, "best_test")
    df = pd.DataFrame({"exp": ["e3_main"] * 4 + ["baseline", "e2_sweep"],
                       "run_id": ["r0", "r1", "r2", "u0", "fbp", "r0"], "psnr": np.arange(6.0)})
    kept = keep_chosen(df, best, ["e3_main"])
    assert list(kept.run_id) == ["r1", "u0", "fbp", "r0"]                 # baselines and other experiments are kept
    assert len(keep_chosen(df, choose(runs, "all"), ["e3_main"])) == 6
    t = table_seeds(best)
    assert len(t) == 1 and t.loc[0, "Seed báo cáo"] == "1" and "s2: 31.400" in t.loc[0, "PSNR xác thực theo seed (dB)"]
    assert table_seeds(choose(runs[runs.arch == "learn_unet"], "best_val")) is None


def test_regnet_cost_table_pairs_each_family_with_its_regnet_version():
    """Table B13: original -> RegNet variant per family, quality and resources; missing measurements show "–"."""
    from lact.services.reporting.tables import table_regnet_cost

    df, spec = _results(), ReportSpec(n_boot=50)
    measured = (("resnet_ed", 4.09e6, 409.7, 30.0, 3882.0), ("resnet_regnet", 4.23e6, 165.7, 45.0, 2286.0),
                ("resnext_ed", 2.95e6, 432.4, np.nan, 10513.0), ("regnet_ed", 3.14e6, 134.1, 50.0, 6109.0))
    cost = pd.DataFrame([
        {"exp": "e3_main", "regularizer": reg, "geometry_test": "fan_a120", "n_params": n,
         "gflops_regularizer_total": gf, "median_ms": ms, "peak_infer_mem_mb": 900.0, "peak_train_mem_mb": mem,
         "train_hours": 5.0, "train_jobs_per_gpu": 2} for reg, n, gf, ms, mem in measured])
    t = table_regnet_cost(df, cost, spec)
    assert list(t["Họ"]) == ["LEARN–ResNet → LEARN–ResNet–RegNet", "LEARN–ResNeXt → LEARN–RegNet"]
    r = t.iloc[0]
    assert r["GFLOPs (K giai đoạn)"] == "409.7 → 165.7 (-60%)" and r["Bộ nhớ huấn luyện (MB)"] == "3882 → 2286 (-41%)"
    assert r["Suy luận (ms/lát)"] == "30.0 → 45.0 (+50%)" and r["Δ PSNR test (dB)"][0] in "+-"
    assert t.iloc[1]["Suy luận (ms/lát)"] == "–"
    assert table_regnet_cost(df, pd.DataFrame(), spec) is None

