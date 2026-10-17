"""Per-stage analysis: PSNR of x_k and the magnitudes of the data-consistency and regularizer steps."""
from __future__ import annotations

import pandas as pd
import torch

from ..metrics.image import psnr, ssim


@torch.no_grad()
def stagewise_metrics(model, dataset, device, n_slices: int | None = 50, stride: int | None = None) -> pd.DataFrame:
    model.eval()
    n = len(dataset)
    idxs = list(range(0, n, stride or max(1, n // (n_slices or n))))[: (n_slices or n)]
    rows = []
    for i in idxs:
        b = dataset[i]
        y, x0, x = b["y"][None].to(device), b["x0"][None].to(device), b["x"][None].to(device)
        xs, terms = model(y, x0, return_stages=True, return_terms=True)
        for k, xk in enumerate(xs):
            r = {"slice": i, "pid": b["pid"], "idx": b["idx"], "k": k,
                 "psnr": float(psnr(xk, x)), "ssim": float(ssim(xk, x))}
            if k > 0:      # gd: alpha, beta; cg/admm: beta, lambda (+ rho, gamma) and the number of CG iterations
                r.update({name: v for name, v in terms[k - 1].items() if name != "k"})
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_stagewise(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("psnr", "ssim", "dc_norm", "reg_norm", "alpha", "beta", "lambda", "rho", "gamma", "cg_iters")
            if c in df.columns]
    g = df.groupby("k")[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out.reset_index()

