"""Training use case: one run = one architecture x one geometry x one seed x one learning rate.

Protection against lost work (important on Colab):
- ``last.pt`` is saved at the end of every epoch AND mid-epoch every ``train.checkpoint_every_min`` minutes
  (or ``checkpoint_every_steps`` steps) together with the position in the epoch, so a resume continues at the exact
  slice.
- The previous file is kept as ``last_prev.pt``; if ``last.pt`` is corrupted (machine stopped while writing) it is used.
- First SIGTERM / stop request: save a checkpoint and stop cleanly (status ``interrupted``, no DONE marker).
- GPU out of memory (e.g. Colab assigned a smaller GPU): enable gradient checkpointing (same results) and retry the
  step; if memory is still short, save a checkpoint, write the OOM marker and stop.
- ``heartbeat.json``, ``sessions.jsonl`` and ``events.jsonl`` record progress, the machine/GPU of each session and
  incidents.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader, Subset

from ....config import Config, ConfigError
from ....config.loader import save_yaml
from ....core.metrics import psnr, rmse, ssim
from ....core.models import count_params, val_forward
from ....infra.data.dataset import collate
from ....infra.stores.run_store import RunStore
from ....infra.system.locks import exclusive_or_warn
from ....infra.system.runtime import (
    GracefulInterrupt,
    apply_precision,
    collect_env,
    get_logger,
    gpu_info,
    host_info,
    seed_everything,
    select_device,
)
from ...data.staging import training_inputs, verify_inputs
from ...factories import dataset_for, model_for, projector_for
from ..optim import build_optimizer, build_scheduler
from .guards import DivergenceGuard, EpochProgress, _is_oom, _jobs_per_gpu, _peak_mem_mb, _sync
from .runs import find_init_run, make_run_id, run_store_for


class Trainer:
    def __init__(self, cfg: Config, run: RunStore | None = None):
        self.run_id = cfg.run_id or make_run_id(cfg)
        self.cfg = cfg.replace(run_id=self.run_id)
        self.t = self.cfg.train
        self.run = run or run_store_for(self.cfg)
        self.log = get_logger("lact", self.run.log_file)
        self.ready = False
        self.stop: GracefulInterrupt | None = None
        self.oom_events = 0
        self.last_sec_per_step = float("nan")

    # ------------------------------------------------------------------ setup
    def setup(self) -> bool:
        """Build everything needed for training. Returns False when the run has already ended (DONE / DIVERGED /
        OOM)."""
        cfg, t = self.cfg, self.t
        self.run.dir.mkdir(parents=True, exist_ok=True)
        if self.run.finished:
            self.log.info("Skipping %s: status %s (delete the %s marker to run it again)", self.run_id, self.run.status,
                          self.run.status.upper())
            return False
        self._configure_runtime()
        self.geom = cfg.require_geometry()
        if cfg.verify_data:
            self._verify_data()
        self.P = projector_for(cfg, device=self.device)
        self.ds_train = dataset_for(cfg, "train", limit=t.limit_train)
        self.ds_val = dataset_for(cfg, "val", limit=t.limit_val, augment=False)
        self.bs = int(t.batch_size)
        self.steps_per_epoch = math.ceil(len(self.ds_train) / self.bs)
        self.epochs = int(t.epochs)
        self.total_steps = self.epochs * self.steps_per_epoch
        if t.max_steps:
            self.total_steps = min(self.total_steps, int(t.max_steps))

        self.model = model_for(cfg, self.P).to(self.device)
        self.opt = build_optimizer(self.model, t)
        self.sched = build_scheduler(self.opt, t, self.steps_per_epoch, self.total_steps)
        self.n_params = count_params(self.model)
        self.n_params_reg = self.model.regularizer_params()
        self._save_run_metadata()
        self.logger = self.run.logger(tensorboard=t.tensorboard)
        self._resume()
        if t.init_from and not self.resumed:
            self._init_from()
        # The multi-GPU scheduler sets CUDA_VISIBLE_DEVICES=<UUID> and the number of jobs sharing each GPU.
        self.run.log_session(start_step=self.step, start_epoch=self.progress.epoch,
                             start_batch=self.progress.batch, grad_checkpoint=self.model.grad_checkpoint,
                             torch=torch.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                             jobs_per_gpu=_jobs_per_gpu(), **host_info())
        self.log.info("Run %s | %s | %s %s | train %d slices, val %d slices | %d parameters (regularizer %d) | %s",
                      self.run_id, self.P, self.device, gpu_info().get("gpu") or "", len(self.ds_train),
                      len(self.ds_val), self.n_params, self.n_params_reg, cfg.require_model().regularizer.name)
        self.ready = True
        return True

    def _configure_runtime(self) -> None:
        prec = self.cfg.precision
        seed_everything(int(self.cfg.seed), prec.deterministic)
        apply_precision(prec.tf32)
        if prec.amp:
            self.log.warning("precision.amp is not supported (sparse matrix products need FP32); ignored.")
        self.device = select_device(self.cfg.device)
        if self.t.require_gpu and self.device.type != "cuda":
            raise RuntimeError("No GPU (train.require_gpu = true). On Colab: Runtime > Change runtime type "
                               "> select a GPU and run again; existing checkpoints are untouched.")

    def _verify_data(self) -> None:
        """SHA-256 of images, sinograms, matrices (the copies in use, cache copies included) against the manifest."""
        errs = verify_inputs(self.cfg, training_inputs(self.cfg, self.geom))
        if errs:
            raise RuntimeError("Data does not match the manifest:\n" + "\n".join(errs))

    def _save_run_metadata(self) -> None:
        derived = {"n_params": self.n_params, "n_params_regularizer": self.n_params_reg,
                   "n_train": len(self.ds_train), "n_val": len(self.ds_val),
                   "steps_per_epoch": self.steps_per_epoch, "total_steps": self.total_steps,
                   "opnorm": self.P.norm, "n_views": self.geom.n_views}
        save_yaml(self.cfg.replace(derived=derived).to_dict(), self.run.config_file)
        self.run.save_env(collect_env())

    def _resume(self) -> None:
        self.step, self.best, self.best_epoch = 0, -float("inf"), -1
        self.progress = EpochProgress()
        self.guard = DivergenceGuard(float(self.t.divergence_factor))
        self.elapsed_prev, self.peak_mem_prev = 0.0, 0.0
        self.resumed = False
        if not self.t.resume:
            return
        st = self.run.load_latest("last", self.model, self.opt, self.sched, map_location=self.device,
                                  restore_rng=True)
        if st is None:
            return
        self.resumed = True
        ex = st.get("extra", {})
        self.step = int(st["step"])
        self.best = st["best"] if st["best"] is not None else -float("inf")
        self.best_epoch = ex.get("best_epoch", -1)
        if ex.get("in_epoch"):
            self.progress = EpochProgress(int(st["epoch"]), int(ex.get("batch_in_epoch", 0)),
                                          float(ex.get("loss_sum", 0.0)), int(ex.get("loss_count", 0)))
        else:
            self.progress = EpochProgress(int(st["epoch"]) + 1)
        self.guard.ref_losses = list(ex.get("guard_ref", []))
        self.elapsed_prev = float(ex.get("elapsed_sec", 0.0))
        self.peak_mem_prev = float(ex.get("peak_mem_mb") or 0.0)
        self.oom_events = int(ex.get("oom_events", 0))
        if ex.get("grad_checkpoint"):
            self.model.grad_checkpoint = True
        self.log.info("Resuming from epoch %d, batch %d (step %d, best %.3f)", self.progress.epoch,
                      self.progress.batch, self.step, self.best)

    def _init_from(self) -> None:
        """Load parameters with matching names and shapes from the source run's best checkpoint (see
        ``train.init_from``)."""
        src = find_init_run(self.cfg)
        state = torch.load(src.checkpoint("best"), map_location=self.device, weights_only=False)["model"]
        own = self.model.state_dict()
        keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing_reg = [k for k in own if k.startswith(("reg.", "regs.", "stages.", "m1.")) and k not in keep]
        if missing_reg:
            raise ConfigError(f"{src.dir}: regularizer does not match the current model (missing {missing_reg[:3]}...)")
        self.model.load_state_dict(keep, strict=False)
        fresh = sorted(set(own) - set(keep))
        self.run.log_event("init_from", run_id=src.run_id, loaded=len(keep), fresh=fresh,
                           unused=sorted(set(state) - set(keep)))
        self.log.info("Initialised from %s: loaded %d parameters; fresh initialisation kept for %s", src.run_id,
                      len(keep), fresh or "-")

    # ------------------------------------------------------------------ state persistence
    def _elapsed(self) -> float:
        return self.elapsed_prev + (time.time() - self.t_session)

    def _peak(self) -> float:
        return max(self.peak_mem_prev, _peak_mem_mb(self.device) or 0.0)

    def _save_last(self, in_epoch: bool) -> None:
        p = self.progress
        extra = {"best_epoch": self.best_epoch, "in_epoch": in_epoch, "batch_in_epoch": p.batch,
                 "loss_sum": p.loss_sum, "loss_count": p.loss_count, "guard_ref": self.guard.ref_losses,
                 "elapsed_sec": self._elapsed(), "peak_mem_mb": self._peak(), "oom_events": self.oom_events,
                 "grad_checkpoint": self.model.grad_checkpoint}
        self.run.save_checkpoint("last", self.model, self.opt, self.sched, p.epoch, self.step, self.best,
                                 extra=extra, keep_prev=self.t.keep_prev_checkpoint)
        self._t_ckpt, self._step_ckpt = time.time(), self.step
        self._heartbeat("running")

    def _checkpoint_due(self) -> bool:
        t = self.t
        if t.checkpoint_every_steps and self.step - self._step_ckpt >= t.checkpoint_every_steps:
            return True
        every = t.checkpoint_every_min
        return bool(every) and time.time() - self._t_ckpt >= 60.0 * float(every or 0)

    def _heartbeat(self, status: str, **extra) -> None:
        remaining = max(0, self.total_steps - self.step)
        eta = remaining * self.last_sec_per_step / 3600 if math.isfinite(self.last_sec_per_step) else None
        self.run.write_heartbeat(status=status, run_id=self.run_id, epoch=self.progress.epoch,
                                 batch_in_epoch=self.progress.batch, steps_per_epoch=self.steps_per_epoch,
                                 step=self.step, total_steps=self.total_steps, sec_per_step=self.last_sec_per_step,
                                 eta_hours=eta, best_val_psnr=self.best if math.isfinite(self.best) else None,
                                 gpu=gpu_info().get("gpu"), grad_checkpoint=self.model.grad_checkpoint, **extra)

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        dl = DataLoader(self.ds_val, batch_size=int(self.t.val_batch_size), shuffle=False, collate_fn=collate,
                        num_workers=0)
        acc: dict[str, list[torch.Tensor]] = {"val_psnr": [], "val_ssim": [], "val_rmse": [], "val_dc": []}
        for b in dl:
            y, x0, x = b["y"].to(self.device), b["x0"].to(self.device), b["x"].to(self.device)
            out = val_forward(self.model)(y, x0)
            acc["val_psnr"].append(psnr(out, x).cpu())
            acc["val_ssim"].append(ssim(out, x).cpu())
            acc["val_rmse"].append(rmse(out, x).cpu())
            acc["val_dc"].append(self.P.data_residual(out, y).cpu())
        return {k: float(torch.cat(v).mean()) for k, v in acc.items()}

    # ------------------------------------------------------------------ training
    def _epoch_loader(self, epoch: int, start_batch: int = 0) -> DataLoader:
        """Slice order of an epoch fixed by (seed, epoch); ``start_batch`` skips what was already done before a
        resume."""
        g = torch.Generator().manual_seed(int(self.cfg.seed) * 100003 + epoch)
        perm = torch.randperm(len(self.ds_train), generator=g).tolist()[start_batch * self.bs:]
        return DataLoader(Subset(self.ds_train, perm), batch_size=self.bs, shuffle=False, collate_fn=collate,
                          num_workers=int(self.t.num_workers), drop_last=False)

    def _step_once(self, b: dict, clip: float) -> tuple[str, float, float, float]:
        dev = self.device
        y, x0, x = b["y"].to(dev), b["x0"].to(dev), b["x"].to(dev)
        own_loss = getattr(self.model, "training_loss", None)      # models with their own loss (e.g. DPMA) use it
        loss = own_loss(y, x0, x) if own_loss is not None else torch.mean((self.model(y, x0) - x) ** 2)
        lv = float(loss.detach())
        diverged, ref = self.guard.check(lv)
        if diverged:
            return "diverged", lv, float("nan"), ref
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.t.grad_clip_mode == "value":        # element-wise clipping (FBPConvNet); the norm is still logged
            gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf")))
            if math.isfinite(clip):
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip)
        else:
            gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
        self.opt.step()
        self.sched.step()
        return "ok", lv, gn, ref

    def _train_step(self, b: dict, clip: float) -> tuple[str, float, float, float]:
        """One update step. Returns (status, loss, |grad|, reference loss); status: ok | diverged | oom."""
        nan = float("nan")
        for attempt in range(2):
            err = None
            try:
                return self._step_once(b, clip)
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                err = str(e).splitlines()[0][:300]
            # leave the except block so the failed step's tensors are released before freeing memory
            self.opt.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.oom_events += 1
            can_retry = attempt == 0 and self.t.oom_retry_checkpoint and not self.model.grad_checkpoint
            self.run.log_event("oom", step=self.step, epoch=self.progress.epoch,
                               action="enable_grad_checkpoint" if can_retry else "stop", error=err, **gpu_info())
            if not can_retry:
                return "oom", nan, nan, nan
            self.log.warning("GPU out of memory at step %d: enabling gradient checkpointing and retrying this step",
                             self.step)
            self.model.grad_checkpoint = True
        return "oom", nan, nan, nan

    def _train_epoch(self) -> str:
        """Run (the rest of) the current epoch. Returns: ok | diverged | oom | interrupted."""
        dev, t, p = self.device, self.t, self.progress
        clip = float(t.grad_clip) if t.grad_clip else float("inf")
        self.model.train()
        t_log, n_since = time.perf_counter(), 0
        for b in self._epoch_loader(p.epoch, p.batch):
            status, lv, gn, ref = self._train_step(b, clip)
            if status == "diverged":
                self.log.error("DIVERGED at step %d (loss=%s, reference=%.3e)", self.step, lv, ref)
                return "diverged"
            if status == "oom":
                self.log.error("GPU out of memory at step %d even with gradient checkpointing", self.step)
                self._save_last(in_epoch=True)
                return "oom"
            self.step += 1
            n_since += 1
            p.batch += 1
            p.loss_sum += lv
            p.loss_count += 1
            if self.step % int(t.log_every) == 0 or self.step == 1:
                _sync(dev)
                self.last_sec_per_step = (time.perf_counter() - t_log) / max(1, n_since)
                t_log, n_since = time.perf_counter(), 0
                lr = self.opt.param_groups[0]["lr"]
                self.logger.log_step(self.step, loss=lv, lr=lr, grad_norm=gn, sec_per_step=self.last_sec_per_step,
                                     peak_mem_mb=_peak_mem_mb(dev) or 0.0)
                self.log.info("ep %d step %d/%d loss %.3e lr %.2e |g| %.2e %.3fs/step", p.epoch, self.step,
                              self.total_steps, lv, lr, gn, self.last_sec_per_step)
                self._heartbeat("running", loss=lv, lr=lr)
            if self.stop is not None and self.stop.requested:
                self._save_last(in_epoch=True)
                return "interrupted"
            if self.step >= self.total_steps:
                break
            if self._checkpoint_due():
                self._save_last(in_epoch=True)
        return "ok"

    def _end_epoch(self, ep_time: float) -> None:
        p = self.progress
        row = {"train_loss": p.mean_loss, "lr": self.opt.param_groups[0]["lr"], "epoch_sec": ep_time,
               # LEARN gd/lbfgs: alphas, betas; cg/admm: betas, lambdas (+ rhos, gammas); RegFormer: weights
               **{name: json.dumps([round(v, 5) for v in vals.detach().tolist()])
                  for name, vals in self.model.stage_params().items()}}
        is_last = (p.epoch == self.epochs - 1) or self.step >= self.total_steps
        if (p.epoch + 1) % int(self.t.val_every) == 0 or is_last:
            vm = self.validate()
            row.update(vm)
            if vm["val_psnr"] > self.best:
                self.best, self.best_epoch = vm["val_psnr"], p.epoch
                self.run.save_checkpoint("best", self.model, epoch=p.epoch, step=self.step, best=self.best,
                                         extra={"best_epoch": p.epoch, **vm}, with_rng=False)
            self.log.info("ep %d: train %.3e | val PSNR %.3f SSIM %.4f | best %.3f (ep %d) | %.0fs", p.epoch,
                          row["train_loss"], vm["val_psnr"], vm["val_ssim"], self.best, self.best_epoch, ep_time)
        self.logger.log_epoch(p.epoch, self.step, **row)
        self._save_last(in_epoch=False)

    def fit(self) -> dict:
        """Lock the run directory for the whole run: a second process training the same run (e.g. a job left behind by
        an old pipeline after an SSH drop) stops at once with FileExistsError instead of overwriting checkpoints."""
        self.run.dir.mkdir(parents=True, exist_ok=True)
        with exclusive_or_warn(self.run.dir / ".train.lock"):
            return self._fit()

    def _fit(self) -> dict:
        if not self.ready and not self.setup():
            return {"status": "skipped", "run_id": self.run_id}
        self.t_session = time.time()
        self._t_ckpt, self._step_ckpt = time.time(), self.step
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        status = "ok"
        with GracefulInterrupt() as stop:
            self.stop = stop
            while self.progress.epoch < self.epochs and self.step < self.total_steps:
                t_ep = time.time()
                status = self._train_epoch()
                if status != "ok":
                    break
                self._end_epoch(time.time() - t_ep)
                self.progress = EpochProgress(self.progress.epoch + 1)
            self.stop = None
        if status == "interrupted":
            return self._interrupted(stop.signal_name)
        return self._finish({"ok": "done", "diverged": "diverged", "oom": "oom"}[status])

    def _interrupted(self, signal_name: str | None) -> dict:
        self.run.log_event("interrupted", signal=signal_name, step=self.step, epoch=self.progress.epoch,
                           batch_in_epoch=self.progress.batch)
        self._heartbeat("interrupted")
        self.logger.close()
        self.log.warning("Checkpoint saved at step %d; run the same command again to continue.", self.step)
        return {"run_id": self.run_id, "status": "interrupted", "steps": self.step}

    def _finish(self, status: str) -> dict:
        m = self.cfg.require_model()
        gpus = sorted({str(s["gpu"]) for s in self.run.sessions() if s.get("gpu")})
        summary = {
            "run_id": self.run_id, "status": status, "best_val_psnr": self.best, "best_epoch": self.best_epoch,
            "steps": self.step, "train_seconds": round(self._elapsed(), 1), "peak_train_mem_mb": self._peak() or None,
            "n_params": self.n_params, "n_params_regularizer": self.n_params_reg, "arch": m.arch,
            "regularizer": m.regularizer.name, "geometry": self.geom.name, "seed": int(self.cfg.seed),
            "lr": float(self.t.lr), "exp": self.cfg.exp, "gpus": gpus, "sessions": len(self.run.sessions()),
            "grad_checkpoint": self.model.grad_checkpoint, "oom_events": self.oom_events,
            "update": m.update_tag, "metric": None,
            "init_from": next((e.get("run_id") for e in self.run.events() if e.get("event") == "init_from"), None),
            # > 1: the GPU was shared with other runs at times, so training time is not directly comparable
            "jobs_per_gpu": max([int(s.get("jobs_per_gpu") or 1) for s in self.run.sessions()] or [1]),
        }
        self.run.mark(summary)
        # keep last.pt of the last good epoch; save the failing state separately for inspection
        if status == "diverged":
            self.run.save_checkpoint("diverged", self.model, epoch=-1, step=self.step, best=self.best,
                                     with_rng=False)
        self._heartbeat(status)
        self.logger.close()
        return summary


def train(cfg: Config) -> dict:
    return Trainer(cfg).fit()
