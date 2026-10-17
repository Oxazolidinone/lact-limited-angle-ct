"""Regularizer registry and model builders from primitive parameters (independent of the configuration).

Regularizers of the LEARN framework: ``densenet_ed`` (LEARN-DenseNet, a 4-level encoder-decoder of dense blocks)
and ``learn_orig`` / ``cnn3`` (the 3-layer CNN of the original LEARN). New ones are added with ``@register``.
"""
from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from .baselines.dpma import DPMA
from .baselines.fbpconvnet import FBPConvNet
from .baselines.lpd import LearnedPrimalDual
from .baselines.regformer import RegFormer
from .regularizers.cnn3 import CNN3
from .regularizers.densenet_ed import densenet_ed
from .unrolled import Unrolled, UnrolledLEARN

REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "cnn3": CNN3,
    "learn_orig": CNN3,     # LEARN as in the original paper (configs/model/learn_orig.yaml)
    "densenet_ed": densenet_ed,
}


def register(name: str):
    """Decorator that registers a new regularizer without touching the framework code."""
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def needs_img_size(name: str) -> bool:
    """Regularizers that take an ``img_size`` argument are built for the reconstruction grid size."""
    return "img_size" in inspect.signature(REGISTRY[name]).parameters


def build_regularizer(name: str, **params: Any) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"Regularizer '{name}' is not registered. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def build_unrolled(regularizer: str, params: dict[str, Any], img_size: int, K: int = 14, share_theta: bool = True,
                   alpha0: float = 1.0, beta0: float = 1.0, grad_checkpoint: bool = False,
                   projector=None, **update_kw: Any) -> UnrolledLEARN:
    """Build an UnrolledLEARN. ``update_kw``: the stage update rule (update, lbfgs_memory, lbfgs_damping,
    lbfgs_curv_min); empty = first-order LEARN."""
    params = dict(params)
    if regularizer in REGISTRY and needs_img_size(regularizer):
        params["img_size"] = int(img_size)
    return UnrolledLEARN(lambda: build_regularizer(regularizer, **params), K=K, share_theta=share_theta,
                         alpha0=alpha0, beta0=beta0, grad_checkpoint=grad_checkpoint, projector=projector,
                         **update_kw)


def build_regformer(params: dict[str, Any], K: int = 10, grad_checkpoint: bool = False, projector=None,
                    fbp_filter: str = "ram-lak", fbp_cutoff: float = 1.0, data_step: str = "fbp",
                    lbfgs_memory: int = 13, lbfgs_damping: float = 0.0, lbfgs_curv_min: float = 0.0) -> RegFormer:
    """Build RegFormer (``model.framework: regformer``); ``params`` = dim, window, depth, heads, mlp_ratio, drop_path.

    ``data_step = "fbp"`` keeps the original data step. ``"lbfgs"`` replaces it with L-BFGS on the data gradient; the
    regularizer blocks keep their names, so FBP-variant checkpoints still load; the alpha step is a new parameter."""
    return RegFormer(K=K, grad_checkpoint=grad_checkpoint, projector=projector, fbp_filter=fbp_filter,
                     fbp_cutoff=fbp_cutoff, data_step=data_step, lbfgs_memory=lbfgs_memory,
                     lbfgs_damping=lbfgs_damping, lbfgs_curv_min=lbfgs_curv_min, **params)


def _optional_methods() -> dict[str, type[Unrolled]]:
    """Compared methods whose module may be absent. CG-GLORE is rebuilt from code its authors have not published, so
    it is left out of the public repository; everything else works without it."""
    module = f"{__package__}.baselines.cg_glore"
    try:
        return {"cg_glore": importlib.import_module(module).CGGlore}
    except ModuleNotFoundError as e:
        if e.name != module:
            raise
        return {}


# Compared methods outside the LEARN framework and RegFormer (``model.framework``): name -> model class. Their
# hyper-parameters live in ``model.regularizer``, as for RegFormer.
COMPARED: dict[str, type[Unrolled]] = {**_optional_methods(), "fbpconvnet": FBPConvNet, "lpd": LearnedPrimalDual,
                                       "dpma": DPMA}
_NOT_DISTRIBUTED = {"cg_glore": "CG-GLORE is not distributed with this copy of the code"}
_USES_FBP = ("cg_glore", "dpma")            # build an FBP inside the model
_NO_STAGES = ("fbpconvnet", "dpma")         # no learned iteration count (K is always 1)


def build_compared(framework: str, params: dict[str, Any], K: int, grad_checkpoint: bool = False, projector=None,
                   fbp_filter: str = "ram-lak", fbp_cutoff: float = 1.0, target_projector=None) -> Unrolled:
    """Build a compared method. CG-GLORE and DPMA use an internal FBP (same filter as the FBP producing x^0);
    FBPConvNet and DPMA have no stages (K is always 1). ``target_projector``: DPMA's extended-range projector."""
    if framework not in COMPARED:
        why = _NOT_DISTRIBUTED.get(framework, "unknown method")
        raise KeyError(f"Method '{framework}' is not available ({why}). Available: {sorted(COMPARED)}")
    kw: dict[str, Any] = {**params, "grad_checkpoint": grad_checkpoint, "projector": projector}
    if framework in _USES_FBP:
        kw.update(fbp_filter=fbp_filter, fbp_cutoff=fbp_cutoff)
    if framework not in _NO_STAGES:
        kw["K"] = K
    if framework == "dpma":
        kw["target_projector"] = target_projector
    return COMPARED[framework](**kw)


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
