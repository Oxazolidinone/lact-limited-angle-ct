"""Backend of the demo application (the UI lives in ``lact.cli.demo_app``).

"cache" mode reads saved predictions; live mode runs the model / TV (CPU ~3-4 s per slice).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import Config, load_geometry
from ..core.metrics import psnr, rmse, ssim
from ..core.operators import SparseFanBeam
from ..core.solvers import tv_pdhg
from ..infra.stores.run_store import RunStore
from .factories import image_store, load_trained, projector_for, results_store, sino_store


class DemoBackend:
    def __init__(self, cfg: Config, split: str = "test", device: str = "cpu", seed: int = 0):
        self.cfg, self.split, self.device, self.seed = cfg, split, device, seed
        self.results = results_store(cfg)
        self.images = image_store(cfg)
        self._projectors: dict[str, SparseFanBeam] = {}
        self._models: dict[str, torch.nn.Module] = {}

    # ------------------------------------------------------------------ catalogue
    def geometries(self) -> list[str]:
        return sorted({m["geometry_test"] for m in self.results.find(split=self.split).values() if m.get("has_pred")})

    def patients(self) -> list[str]:
        return [p for p in self.cfg.split.patients(self.split) if p in self.images.patients()]

    def n_slices(self, pid: str) -> int:
        return self.images.n_slices(pid)

    def methods(self, geom: str) -> dict[str, dict]:
        """{display label: metadata (with 'tag')} for results at ``geom`` (seed ``self.seed`` for learned models)."""
        from ..viz.style import label_of

        out = {}
        for tag, info in sorted(self.results.find(split=self.split, geometry_test=geom).items()):
            if info["geometry_train"] != geom or not info.get("has_pred"):
                continue
            if info["exp"] != "baseline" and int(info["seed"]) != self.seed:
                continue
            lab = label_of(info["regularizer"]) + ("" if info["exp"] == "baseline" else f" [{info['exp']}]")
            out[lab] = {**info, "tag": tag}
        return out

    # ------------------------------------------------------------------ reconstruction
    def _projector(self, geom: str) -> SparseFanBeam:
        if geom not in self._projectors:
            self._projectors[geom] = projector_for(self.cfg, load_geometry(geom), device=self.device)
        return self._projectors[geom]

    def _model(self, info: dict):
        key = info["run_id"]
        if key not in self._models:
            run = RunStore(self.cfg.paths.runs / info["exp"] / info["run_id"])
            self._models[key] = load_trained(run, "best", self.device)[0]
        return self._models[key]

    @torch.no_grad()
    def _live(self, info: dict, geom: str, y: np.ndarray, x0: np.ndarray) -> np.ndarray:
        if info["regularizer"] == "fbp":
            return x0
        yt = torch.from_numpy(y)[None, None].to(self.device)
        x0t = torch.from_numpy(x0)[None, None].to(self.device)
        P = self._projector(geom)
        if info["regularizer"] == "tv":
            prm = json.loads(info["checkpoint"])
            return tv_pdhg(P, yt, float(prm["lambda"]), int(prm["iters"]), x_init=x0t)[0, 0].cpu().numpy()
        return self._model(info).set_projector(P)(yt, x0t)[0, 0].cpu().numpy()

    def reconstruct(self, geom: str, pid: str, idx: int, method_label: str, live: bool = False) -> dict:
        """Image (normalised [0,1]) + metrics for one slice."""
        info = self.methods(geom)[method_label]
        ref = self.images.read(pid, "img256", idx).astype(np.float32)
        sino = sino_store(self.cfg, geom)
        y = sino.read(pid, "y", "orig", idx).astype(np.float32)
        x0 = sino.read(pid, "fbp", "orig", idx).astype(np.float32)
        pred = None if live else self.results.load_pred(info["tag"], pid, idx)
        if pred is None:
            pred = self._live(info, geom, y, x0)

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a))[None, None]

        mu_ref, mu_w = self.images.mu_ref, self.images.mu_water
        r = float(rmse(t(pred), t(ref)))
        mets = {"PSNR (dB)": float(psnr(t(pred), t(ref))), "SSIM": float(ssim(t(pred), t(ref))), "RMSE": r,
                "RMSE (HU)": r * 1000 * mu_ref / mu_w}
        return {"sino": y, "fbp": x0, "pred": pred, "ref": ref, "mu_ref": mu_ref, "mu_w": mu_w, "metrics": mets}
