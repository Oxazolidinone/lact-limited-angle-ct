"""Statistical tests behind the result tables: the registered comparisons of the analysis plan, the
supplementary ones, and their summary table."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ....core.metrics import compare, holm, interaction, verdict
from ....viz.style import NEW_BACKBONES, REGNET_TWINS, geom_label, label_of
from .format import ReportSpec
from .results import _same_geom


def _picker(df: pd.DataFrame):
    same = _same_geom(df)

    def pick(exp, reg, geom):
        return df[same & (df.exp == exp) & (df.regularizer == reg) & (df.geometry_test == geom)]

    return pick


def registered_tests(df: pd.DataFrame, spec: ReportSpec) -> list[tuple[str, str, dict]]:
    """Pre-registered comparisons (RQ1-RQ3): [(question, description, test result)]."""
    pick = _picker(df)
    tests = []
    for g in spec.geoms:
        for reg in ("resnet_ed", "convnext_ed"):
            a, b = pick(spec.main_exp, reg, g), pick(spec.main_exp, "cnn3", g)
            if len(a) and len(b):
                tests.append(("RQ1", f"{label_of(reg)} vs LEARN–CNN, {geom_label(g)}",
                              compare(a, b, n_boot=spec.n_boot)))
    if len(spec.geoms) >= 2:
        g1, g2 = spec.geoms[0], spec.geoms[1]
        for reg in ("cnn3", "resnet_ed", "convnext_ed"):
            a1, b1 = pick(spec.main_exp, reg, g1), pick(spec.main_exp, "vit", g1)
            a2, b2 = pick(spec.main_exp, reg, g2), pick(spec.main_exp, "vit", g2)
            if len(a1) and len(b1) and len(a2) and len(b2):
                tests.append(("RQ2", f"({label_of(reg)} − ViT): {geom_label(g1)} − {geom_label(g2)}",
                              interaction(a1, b1, a2, b2, n_boot=spec.n_boot)))
    mat = {"convnext_ed": pick(spec.main_exp, "convnext_ed", "fan_a120"),
           "resnet_ed": pick(spec.matched_exp, "resnet_ed", "fan_a120"),
           "vit": pick(spec.matched_exp, "vit", "fan_a120")}
    for ra, rb in (("convnext_ed", "resnet_ed"), ("convnext_ed", "vit"), ("resnet_ed", "vit")):
        if len(mat[ra]) and len(mat[rb]):
            tests.append(("RQ3", f"{label_of(ra)} vs {label_of(rb)} (~2M, 120°)",
                          compare(mat[ra], mat[rb], n_boot=spec.n_boot)))
    return tests


def supplementary_tests(df: pd.DataFrame, spec: ReportSpec) -> list[tuple[str, str, dict]]:
    """SUPPLEMENTARY comparisons for models added after the analysis plan was frozen.

    BS1: new backbones versus LEARN–CNN (as RQ1). BS2: RegNet-layout variant versus the uniform layout of the same
    family
    (does the RegNet integration help). BS3: LEARN–RegNet within the ~2M group. Kept apart from the registered family
    (table B8) with their own Holm correction, so the Holm p-values of RQ1-RQ3 do not change."""
    pick = _picker(df)
    tests = []
    for g in spec.geoms:
        b = pick(spec.main_exp, "cnn3", g)
        for reg in NEW_BACKBONES:
            a = pick(spec.main_exp, reg, g)
            if len(a) and len(b):
                tests.append(("BS1", f"{label_of(reg)} vs LEARN–CNN, {geom_label(g)}",
                              compare(a, b, n_boot=spec.n_boot)))
    for g in spec.geoms:
        for reg, base in REGNET_TWINS:
            a, b = pick(spec.main_exp, reg, g), pick(spec.main_exp, base, g)
            if len(a) and len(b):
                tests.append(("BS2", f"{label_of(reg)} vs {label_of(base)}, {geom_label(g)}",
                              compare(a, b, n_boot=spec.n_boot)))
    a = pick(spec.matched_exp, "regnet_ed", "fan_a120")
    mat = {"convnext_ed": pick(spec.main_exp, "convnext_ed", "fan_a120"),
           "resnet_ed": pick(spec.matched_exp, "resnet_ed", "fan_a120"),
           "vit": pick(spec.matched_exp, "vit", "fan_a120")}
    for other, b in mat.items():
        if len(a) and len(b):
            tests.append(("BS3", f"LEARN–RegNet vs {label_of(other)} (~2M, 120°)", compare(a, b, n_boot=spec.n_boot)))
    return tests


def _pct_improved(per_patient: dict) -> str:
    """Per-patient share of improved slices; with many patients, summarised by the median and range."""
    pct = {k: round(v["pct_improved"], 1) for k, v in per_patient.items()}
    if len(pct) <= 4:
        return json.dumps(pct)
    v = np.array(list(pct.values()))
    return f"trung vị {np.median(v):.1f} [{v.min():.1f}; {v.max():.1f}], {len(v)} BN"


def table_stats(tests: list[tuple[str, str, dict]]) -> pd.DataFrame | None:
    """"A tốt hơn"/"B tốt hơn" by the pre-registered rule (Δ, 95% CI, same sign for every patient, > seed SD); the
    Holm p column is reported alongside for reference."""
    if not tests:
        return None
    adj = holm([t[2].get("p_wilcoxon_max", np.nan) for t in tests])
    rows = []
    for (rq, name, r), p_adj in zip(tests, adj):
        inter = "interaction" in r
        rows.append({"Câu hỏi": rq, "So sánh": name,
                     "Δ PSNR (dB)": round(r["interaction"] if inter else r["delta"], 3),
                     "CI 95%": f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                     "p Wilcoxon (max theo BN)": "–" if inter else f"{r['p_wilcoxon_max']:.2e}",
                     "p Holm": "–" if inter else f"{p_adj:.2e}",
                     "% lát cải thiện": "–" if inter else _pct_improved(r["per_patient"]),
                     "BN có Δ > 0": "–" if inter else f"{r['n_patients_improved']}/{r['n_patients']}",
                     "p Wilcoxon theo BN": "–" if inter or not np.isfinite(r["p_wilcoxon_patients"])
                     else f"{r['p_wilcoxon_patients']:.2e}",
                     "Kết luận": ("CI không chứa 0" if r["ci_excludes_zero"] else "không rõ khác biệt") if inter
                     else verdict(r)})
    return pd.DataFrame(rows)
