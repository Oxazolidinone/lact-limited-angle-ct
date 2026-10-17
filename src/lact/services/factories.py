"""Composition root: the ONLY place that connects the typed configuration with domain objects and adapters.

Other services call these functions instead of reading the configuration to build projectors, FBP, models, datasets.
"""
from __future__ import annotations

import torch

from ..config import Config, load_config_file
from ..core.imaging.augment import variants as aug_variants
from ..core.imaging.geometry import FanBeamGeometry
from ..core.models import Unrolled, build_compared, build_regformer, build_unrolled
from ..core.operators import FanBeamFBP, SparseFanBeam
from ..infra.data.cache import ArrayCache
from ..infra.data.dataset import LACTDataset
from ..infra.io.hdf5_store import ImageStore, SinoStore
from ..infra.operators.operator_store import load_projector
from ..infra.stores.results_store import ResultsStore
from ..infra.stores.run_store import RunStore
from .data.staging import local_input, operator_dir


def projector_for(cfg: Config, geom: FanBeamGeometry | None = None, size: int | None = None, device="cpu",
                  with_adjoint: bool = True) -> SparseFanBeam:
    geom = geom or cfg.require_geometry()
    size = size or cfg.data.img_size_rec
    return load_projector(geom, size, operator_dir(cfg, geom, size), device=device, with_adjoint=with_adjoint)


def fbp_for(cfg: Config, geom: FanBeamGeometry | None = None, device="cpu") -> FanBeamFBP:
    geom = geom or cfg.require_geometry()
    return FanBeamFBP(geom, cfg.data.img_size_rec, filt=cfg.baseline.fbp.filter, cutoff=cfg.baseline.fbp.cutoff,
                      device=device)


def model_for(cfg: Config, projector: SparseFanBeam | None = None) -> Unrolled:
    m = cfg.require_model()
    if m.framework == "regformer":        # the model's internal FBP uses the same filter as the FBP producing x^0
        return build_regformer(m.regularizer.params, K=m.K, grad_checkpoint=m.grad_checkpoint, projector=projector,
                               fbp_filter=cfg.baseline.fbp.filter, fbp_cutoff=cfg.baseline.fbp.cutoff,
                               data_step="lbfgs" if m.update == "lbfgs" else "fbp", lbfgs_memory=m.lbfgs_memory,
                               lbfgs_damping=m.lbfgs_damping, lbfgs_curv_min=m.lbfgs_curv_min)
    if m.framework != "learn":            # compared methods: FBPConvNet, Learned Primal-Dual, DPMA
        params, target = dict(m.regularizer.params), None
        if m.framework == "dpma":         # extended scan range: the run's geometry with another name and range
            geom = cfg.require_geometry()
            name, span = str(params.pop("target_name", "fan_a240")), float(params.pop("target_span_deg", 240.0))
            if name == "auto":            # per-geometry target (2DeteCT: the FIRST view is fixed, so centres differ)
                name = f"{geom.name}_to{span:g}"
            full = geom.replace(name=name, span_deg=span)
            target = projector_for(cfg, full, device=projector.device if projector is not None else "cpu",
                                   with_adjoint=False)
        return build_compared(m.framework, params, K=m.K, grad_checkpoint=m.grad_checkpoint,
                              projector=projector, fbp_filter=cfg.baseline.fbp.filter,
                              fbp_cutoff=cfg.baseline.fbp.cutoff, target_projector=target)
    return build_unrolled(m.regularizer.name, m.regularizer.params, img_size=cfg.data.img_size_rec, K=m.K,
                          share_theta=m.share_theta, alpha0=m.alpha0, beta0=m.beta0,
                          grad_checkpoint=m.grad_checkpoint, projector=projector, **m.update_kwargs())


def dataset_for(cfg: Config, split: str, geom_name: str | None = None, preload: bool | None = None,
                limit: int | None = None, augment: bool | None = None) -> LACTDataset:
    geom_name = geom_name or cfg.require_geometry().name
    use_aug = (split == "train") if augment is None else augment
    variants = tuple(aug_variants(cfg.data.augment)) if use_aug else ("orig",)
    return LACTDataset(local_input(cfg, cfg.data.images_h5), local_input(cfg, cfg.paths.sino_file(geom_name)),
                       cfg.split.patients(split), variants=variants,
                       preload=cfg.train.preload if preload is None else preload, limit=limit,
                       array_cache=array_cache(cfg))


def array_cache(cfg: Config) -> ArrayCache | None:
    """.npy array cache for the Dataset (only when cache.enabled and cache.arrays are set)."""
    if cfg.cache.enabled and cfg.cache.arrays and cfg.cache.dir is not None:
        return ArrayCache(cfg.cache.dir / "arrays")
    return None


def image_store(cfg: Config) -> ImageStore:
    return ImageStore(local_input(cfg, cfg.data.images_h5))


def sino_store(cfg: Config, geom_name: str) -> SinoStore:
    return SinoStore(local_input(cfg, cfg.paths.sino_file(geom_name)))


def results_store(cfg: Config) -> ResultsStore:
    return ResultsStore(cfg.paths.eval)


def load_trained(run: RunStore, which: str = "best", device="cpu") -> tuple[Unrolled, Config, dict]:
    """Rebuild a model from a run's config.yaml and load its ``which`` checkpoint (best | last)."""
    cfg = load_config_file(run.config_file)
    model = model_for(cfg)
    state = run.load_checkpoint(which, model, map_location=device)
    model.to(torch.device(device) if isinstance(device, str) else device).eval()
    return model, cfg, state
