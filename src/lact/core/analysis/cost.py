"""Cost measurements: reconstruction time (FBP + K stages), GPU memory, parameter count, FLOPs."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_reconstruction(model, fbp, y: torch.Tensor, n_warmup: int = 10, n_iter: int = 100,
                             include_fbp: bool = True) -> dict:
    """y: one sinogram (1,1,V,D) on the target device. Batch size 1; the GPU is synchronised around each timing."""
    device = y.device
    model.eval()
    x0 = fbp(y)
    for _ in range(n_warmup):
        _ = model(y, fbp(y) if include_fbp else x0)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_iter):
        _sync(device)
        t0 = time.perf_counter()
        x_in = fbp(y) if include_fbp else x0
        _ = model(y, x_in)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {
        "median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "mean_ms": float(t.mean()),
        "std_ms": float(t.std()), "n_iter": n_iter, "include_fbp": include_fbp,
        "peak_infer_mem_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None),
    }


@torch.no_grad()
def benchmark_fbp(fbp, y: torch.Tensor, n_warmup: int = 5, n_iter: int = 50) -> dict:
    for _ in range(n_warmup):
        fbp(y)
    _sync(y.device)
    ts = []
    for _ in range(n_iter):
        _sync(y.device)
        t0 = time.perf_counter()
        fbp(y)
        _sync(y.device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"fbp_median_ms": float(np.median(ts)), "fbp_p95_ms": float(np.percentile(ts, 95))}


def time_data_loading(dataset, n: int = 50) -> dict:
    ts = []
    for i in range(min(n, len(dataset))):
        t0 = time.perf_counter()
        _ = dataset[i]
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"load_median_ms": float(np.median(ts)) if ts else float("nan")}


def regularizer_flops(reg: torch.nn.Module, img_size: int = 256, device="cpu") -> dict:
    """FLOPs of ONE call of R_theta. ViT attention, which FlopCounterMode may miss, is added by hand."""
    x = torch.zeros(1, 1, img_size, img_size, device=device)
    reg = reg.to(device).eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        reg(x)
    counted = int(fc.get_total_flops())
    extra = 0
    counters = [getattr(m, "attention_flops", None) for m in reg.modules()]
    att = sum(int(fn()) for fn in counters if callable(fn))
    if att:
        by_op = {str(k): v for k, v in fc.get_flop_counts().get("Global", {}).items()}
        sdpa_counted = sum(v for k, v in by_op.items() if "scaled_dot_product" in k or "flash" in k
                           or "efficient_attention" in k)
        extra = max(0, att - sdpa_counted)
    return {"gflops_per_call": (counted + extra) / 1e9, "gflops_counted": counted / 1e9,
            "gflops_attention_added": extra / 1e9}


def model_cost(model, img_size: int = 256, device="cpu") -> dict:
    """Parameter count and FLOPs for K stages (learned parts only; sparse projections and FBP are timed instead).

    LEARN framework: FLOPs of one regularizer call x K. Models whose stages differ (RegFormer) are counted over all K
    stages in one pass (``regularization_only``); the "per stage" column is the mean."""
    K = model.K
    if hasattr(model, "regularizer"):
        per = regularizer_flops(model.regularizer(0), img_size, device)["gflops_per_call"]
        total = per * K
    else:
        model = model.to(device).eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model.regularization_only(torch.zeros(1, 1, img_size, img_size, device=device))
        total = int(fc.get_total_flops()) / 1e9
        per = total / K
    return {"n_params": sum(p.numel() for p in model.parameters()),
            "n_params_regularizer": model.regularizer_params(),
            "gflops_regularizer_per_stage": per, "gflops_regularizer_total": total, "K": K}
