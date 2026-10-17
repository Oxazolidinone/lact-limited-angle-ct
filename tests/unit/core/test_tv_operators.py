import numpy as np
import pandas as pd
import torch

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
