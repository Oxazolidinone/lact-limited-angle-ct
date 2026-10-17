"""Model section: the regularizer, the framework (LEARN or a compared method) and the per-stage update rule."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import ConfigError


@dataclass
class RegularizerConfig:
    """Regularizer name in the registry plus its own parameters (architecture-specific, hence a dict)."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, where: str = "model.regularizer") -> RegularizerConfig:
        if "name" not in d:
            raise ConfigError(f"{where}: missing 'name'")
        return cls(name=str(d["name"]), params={k: v for k, v in d.items() if k != "name"})

    def to_dict(self) -> dict:
        return {"name": self.name, **self.params}


# Compared methods outside the LEARN framework -> update tag stored in the results (``update`` column): CG on CG-GLORE's
# FBP system, FBPConvNet's post-processing of the FBP image, learned primal-dual. Deliberately not "cg" so that the
# second-order tables never pick them up by mistake.
COMPARED_TAGS = {"cg_glore": "cg_fbp", "fbpconvnet": "post", "lpd": "pd", "dpma": "dual"}


@dataclass
class ModelConfig:
    regularizer: RegularizerConfig
    arch: str = ""
    # learn: LEARN framework (a registry regularizer + the update rule ``update`` below)
    # regformer: the RegFormer model (core/models/baselines/regformer.py); ``regularizer`` then only carries its
    # hyper-parameters
    # cg_glore | fbpconvnet | lpd | dpma: compared methods (core/models/baselines/), hyper-parameters also in
    # ``regularizer``
    framework: str = "learn"
    K: int = 14
    share_theta: bool = True
    alpha0: float = 1.0
    beta0: float = 1.0
    grad_checkpoint: bool = False
    # --- per-stage update rule (core/models/unrolled/learn.py)
    update: str = "gd"                # gd | cg | admm | lbfgs (quasi-Newton on the data gradient)
    metric: str = "identity"          # cg/admm: B = I (identity) | B = I + rho * D^T D (sobolev)
    lambda0: float = 100.0            # cg/admm: initial data weight lam_k
    rho0: float = 1.0                 # sobolev: initial rho_k
    gamma0: float = 1.0               # admm: initial dual step
    cg_iters: int = 10                # cg/admm: maximum CG iterations per stage
    cg_tol: float = 0.0               # cg/admm: stop once the residual drops by this factor (0 = run every iteration)
    lbfgs_memory: int = 13            # lbfgs: (s, y) pairs kept across stages; >= K - 1 keeps every stage
    lbfgs_damping: float = 0.0        # lbfgs: Powell eta (0 = off): H scales no direction by more than 1/eta
    lbfgs_curv_min: float = 0.0       # lbfgs: drop pairs with s'y < curv_min * s's (0 = off)

    def __post_init__(self):
        self.arch = self.arch or self.regularizer.name
        if self.framework not in ("learn", "regformer", *COMPARED_TAGS):
            raise ConfigError(f"model.framework must be learn, regformer or one of {sorted(COMPARED_TAGS)}, "
                              f"got {self.framework}")
        if self.framework in COMPARED_TAGS and self.update != "gd":
            raise ConfigError(f"model.framework = {self.framework} has its own update step; model.update is not "
                              "accepted")
        if self.framework == "regformer" and self.update not in ("gd", "lbfgs"):
            raise ConfigError("RegFormer accepts only model.update = gd (FBP of the residual) or lbfgs")
        if self.update not in ("gd", "cg", "admm", "lbfgs"):
            raise ConfigError(f"model.update must be gd, cg, admm or lbfgs, got {self.update}")
        if self.metric not in ("identity", "sobolev"):
            raise ConfigError(f"model.metric must be identity or sobolev, got {self.metric}")
        if self.update == "lbfgs" and self.metric != "identity":
            raise ConfigError("update = lbfgs does not use model.metric")
        if self.lbfgs_memory < 1:
            raise ConfigError("model.lbfgs_memory must be >= 1")
        if not 0.0 <= self.lbfgs_damping < 1.0 or self.lbfgs_curv_min < 0.0:
            raise ConfigError("need 0 <= model.lbfgs_damping < 1 and model.lbfgs_curv_min >= 0")
        if self.cg_iters < 1 or self.cg_tol < 0 or min(self.lambda0, self.rho0, self.gamma0) <= 0:
            raise ConfigError("model: need cg_iters >= 1, cg_tol >= 0 and lambda0, rho0, gamma0 > 0")

    @property
    def uses_cg(self) -> bool:
        return self.framework == "learn" and self.update in ("cg", "admm")

    @property
    def update_tag(self) -> str:
        """Update tag stored in the results: gd | cg | admm | lbfgs for the LEARN framework, or fbp / lbfgs for
        RegFormer (fbp = filtered backprojection of the residual, the default)."""
        if self.framework == "regformer":
            return "lbfgs" if self.update == "lbfgs" else "fbp"
        return COMPARED_TAGS.get(self.framework, self.update)

    def update_kwargs(self) -> dict[str, Any]:
        """Update-rule parameters, named exactly as the arguments of ``UnrolledLEARN``."""
        kw = {"update": self.update, "metric": self.metric, "lambda0": self.lambda0, "rho0": self.rho0,
              "gamma0": self.gamma0, "cg_iters": self.cg_iters, "cg_tol": self.cg_tol}
        if self.update == "lbfgs":
            kw["lbfgs_memory"] = self.lbfgs_memory
            kw["lbfgs_damping"], kw["lbfgs_curv_min"] = self.lbfgs_damping, self.lbfgs_curv_min
        return kw
