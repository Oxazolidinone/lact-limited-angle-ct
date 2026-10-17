"""Training section (optimiser, schedule, checkpoints, loss-of-work protection) and arithmetic precision."""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import ConfigError


@dataclass
class TrainConfig:
    optimizer: str = "adam"                     # adam | adamw | sgd
    momentum: float = 0.0                       # sgd only
    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0
    schedule: str = "cosine"
    warmup_epochs: float = 1.0
    lr_min: float = 1e-6
    epochs: int = 50
    batch_size: int = 1
    grad_clip: float | None = 1.0
    grad_clip_mode: str = "norm"                # norm: clip the global gradient norm | value: clip each element
    loss: str = "mse"
    select_by: str = "val_psnr"
    val_every: int = 1
    val_batch_size: int = 4
    num_workers: int = 0
    log_every: int = 50
    max_steps: int | None = None
    limit_train: int | None = None
    limit_val: int | None = None
    divergence_factor: float = 100.0
    resume: bool = True
    # Initialise from the best checkpoint of a FINISHED run with the same architecture and geometry in this experiment
    # (several names separated by commas: the first experiment with a matching run wins, e.g. "base_mayo,e3_main").
    # Parameters with matching names and shapes are loaded (regularizer, beta); parameters that exist only in the new
    # update rule (e.g. the L-BFGS alpha step) keep their initial values. Applies when a run starts, not on resume.
    init_from: str | None = None
    preload: bool = True
    tensorboard: bool = True
    # --- loss-of-work protection (Colab disconnects, machines stopping mid-run)
    checkpoint_every_min: float | None = 10.0   # save last.pt mid-epoch every N minutes (None = end of epoch only)
    checkpoint_every_steps: int | None = None   # or every N steps
    keep_prev_checkpoint: bool = True           # keep last_prev.pt in case last.pt is corrupted while being written
    oom_retry_checkpoint: bool = True           # GPU out of memory => enable gradient checkpointing and retry the step
    require_gpu: bool = False                   # stop when no GPU is present (avoids running on the CPU by mistake)

    def __post_init__(self):
        if self.optimizer.lower() not in ("adam", "adamw", "sgd"):
            raise ConfigError(f"train.optimizer must be adam, adamw or sgd, got {self.optimizer}")
        if self.schedule not in ("cosine", "constant", "exp"):
            raise ConfigError(f"train.schedule must be cosine, constant or exp, got {self.schedule}")
        if self.grad_clip_mode not in ("norm", "value"):
            raise ConfigError(f"train.grad_clip_mode must be norm or value, got {self.grad_clip_mode}")
        if self.schedule == "exp" and not 0 < self.lr_min <= self.lr:
            raise ConfigError("train.schedule = exp requires 0 < train.lr_min <= train.lr")
        if self.checkpoint_every_min is not None and self.checkpoint_every_min <= 0:
            raise ConfigError("train.checkpoint_every_min must be > 0 or null")
        if self.checkpoint_every_steps is not None and self.checkpoint_every_steps <= 0:
            raise ConfigError("train.checkpoint_every_steps must be > 0 or null")


@dataclass
class PrecisionConfig:
    tf32: bool = True
    amp: bool = False
    deterministic: bool = True
