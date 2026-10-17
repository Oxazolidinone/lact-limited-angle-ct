import numpy as np
import pandas as pd
import torch

from lact.core.metrics import compare, decide_improvement, holm, interaction
from lact.core.solvers import div2d, grad2d


def test_grad_div_adjoint():
    x = torch.randn(2, 1, 13, 17, dtype=torch.float64)
    p = torch.randn(2, 2, 13, 17, dtype=torch.float64)
    assert abs(float((grad2d(x) * p).sum() + (x * div2d(p)).sum())) < 1e-10


def _fake(delta, seeds=3, pids=("P1", "P2"), n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    base = {p: rng.normal(30, 1, n) for p in pids}
    for s in range(seeds):
        for p in pids:
            v = base[p] + delta + rng.normal(0, 0.05, n)
            rows += [{"pid": p, "idx": i, "seed": s, "psnr": float(v[i])} for i in range(n)]
    return pd.DataFrame(rows)


def test_compare_detects_improvement():
    r = compare(_fake(0.5, seed=1), _fake(0.0, seed=1), n_boot=500)
    assert abs(r["delta"] - 0.5) < 0.05 and r["ci_low"] > 0 and decide_improvement(r)
    r0 = compare(_fake(0.0, seed=2), _fake(0.0, seed=2), n_boot=500)
    assert r0["ci_low"] < 0 < r0["ci_high"] or abs(r0["delta"]) < 0.05


def test_patient_level_test_needs_enough_patients():
    two = compare(_fake(0.5, seed=1), _fake(0.0, seed=1), n_boot=200)
    assert two["n_patients"] == 2 and two["n_patients_improved"] == 2 and np.isnan(two["p_wilcoxon_patients"])
    pids = tuple(f"P{i}" for i in range(12))                 # as for DeepLesion: many test patients
    many = compare(_fake(0.5, pids=pids, n=12, seed=5), _fake(0.0, pids=pids, n=12, seed=5), n_boot=200)
    assert many["n_patients"] == 12 and many["n_patients_improved"] == 12 and many["p_wilcoxon_patients"] < 0.01
    none = compare(_fake(0.0, pids=pids, n=12, seed=6), _fake(0.0, pids=pids, n=12, seed=7), n_boot=200)
    assert none["p_wilcoxon_patients"] > 0.05 and not decide_improvement(none)


def test_interaction_and_holm():
    it = interaction(_fake(1.0, seed=3), _fake(0.0, seed=3), _fake(0.2, seed=4), _fake(0.0, seed=4), n_boot=300)
    assert abs(it["interaction"] - 0.8) < 0.1 and it["ci_excludes_zero"]
    adj = holm([0.01, 0.04, 0.03, float("nan")])
    assert adj[0] == 0.03 and np.isnan(adj[3]) and max(adj[:3]) <= 1.0
