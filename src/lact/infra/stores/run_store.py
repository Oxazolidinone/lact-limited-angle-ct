"""Store of one training run: the file layout of ``<runs>/<exp>/<run_id>/``, checkpoints, logs, status.

    config.yaml    merged configuration (with the derived section)
    env.json       environment (GPU, library versions, git)
    train.log      text log
    steps.csv      per-step log; metrics.csv: per-epoch log
    best.pt        checkpoint with the best validation PSNR; last.pt: end of the latest epoch (for resuming)
    summary.json   summary; DONE, DIVERGED or OOM: status markers
    heartbeat.json latest progress (step, epoch, loss, GPU, time) for remote monitoring
    sessions.jsonl one line per (resumed) session: machine, GPU, starting step
    events.jsonl   special events: GPU out of memory, interruption, fallback checkpoint loaded
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
from pathlib import Path

import torch

from ..io.files import load_json, save_json, write_text
from ..system.runtime import rng_state, set_rng_state

STATUSES = ("todo", "partial", "done", "diverged", "oom")
MARKERS = {"done": "DONE", "diverged": "DIVERGED", "oom": "OOM"}
log = logging.getLogger("lact")


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


class CSVWriter:
    """Append-only CSV writer; widens the header when new columns appear."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fields: list[str] | None = None
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.fields = next(csv.reader(f), None)

    def write(self, row: dict) -> None:
        if self.fields is None:
            self.fields = list(row.keys())
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, self.fields).writeheader()
        new = [k for k in row if k not in self.fields]
        if new:
            with open(self.path, encoding="utf-8") as f:
                old = list(csv.DictReader(f))
            self.fields += new
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, self.fields)
                w.writeheader()
                w.writerows(old)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, self.fields).writerow(row)


class RunLogger:
    """Per-step/epoch logging to CSV and TensorBoard (when installed)."""

    def __init__(self, run_dir: Path, tensorboard: bool = True):
        self.epochs = CSVWriter(run_dir / "metrics.csv")
        self.steps = CSVWriter(run_dir / "steps.csv")
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(run_dir / "tb"))
            except Exception:
                self.tb = None

    def log_step(self, step: int, **kv) -> None:
        self.steps.write({"step": step, **kv})
        if self.tb is not None:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"step/{k}", v, step)

    def log_epoch(self, epoch: int, step: int, **kv) -> None:
        self.epochs.write({"epoch": epoch, "step": step, **kv})
        if self.tb is not None:
            for k, v in kv.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(f"epoch/{k}", v, epoch)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()


class RunStore:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)

    # ------------------------------------------------------------------ paths
    @property
    def run_id(self) -> str:
        return self.dir.name

    @property
    def config_file(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def log_file(self) -> Path:
        return self.dir / "train.log"

    @property
    def metrics_csv(self) -> Path:
        return self.dir / "metrics.csv"

    @property
    def steps_csv(self) -> Path:
        return self.dir / "steps.csv"

    def checkpoint(self, name: str) -> Path:
        return self.dir / f"{name}.pt"

    # ------------------------------------------------------------------ status
    @property
    def status(self) -> str:
        for status, marker in MARKERS.items():
            if (self.dir / marker).exists():
                return status
        if self.checkpoint("last").exists() or self.checkpoint("last_prev").exists():
            return "partial"
        return "todo"

    @property
    def finished(self) -> bool:
        return self.status in ("done", "diverged", "oom")

    def summary(self) -> dict:
        p = self.dir / "summary.json"
        return load_json(p) if p.exists() else {}

    def mark(self, summary: dict) -> None:
        """Write summary.json and the DONE / DIVERGED / OOM marker from summary['status']."""
        save_json(summary, self.dir / "summary.json")
        marker = MARKERS.get(summary.get("status", ""), "DIVERGED")
        write_text(self.dir / marker, json.dumps(summary, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ monitoring
    @property
    def heartbeat_file(self) -> Path:
        return self.dir / "heartbeat.json"

    def write_heartbeat(self, **info) -> None:
        save_json({"time": _now(), **info}, self.heartbeat_file)

    def heartbeat(self) -> dict:
        return load_json(self.heartbeat_file) if self.heartbeat_file.exists() else {}

    def _append_jsonl(self, name: str, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": _now(), **record}, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_session(self, **info) -> None:
        self._append_jsonl("sessions.jsonl", info)

    def log_event(self, kind: str, **info) -> None:
        self._append_jsonl("events.jsonl", {"event": kind, **info})

    def _read_jsonl(self, name: str) -> list[dict]:
        p = self.dir / name
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # the last line can be incomplete if the machine stopped while writing
        return out

    def sessions(self) -> list[dict]:
        return self._read_jsonl("sessions.jsonl")

    def events(self) -> list[dict]:
        return self._read_jsonl("events.jsonl")

    # ------------------------------------------------------------------ metadata
    def save_env(self, env: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        save_json(env, self.dir / "env.json")
        write_text(self.dir / "git_hash.txt", str(env.get("git", "unknown")) + "\n")

    def logger(self, tensorboard: bool = True) -> RunLogger:
        self.dir.mkdir(parents=True, exist_ok=True)
        return RunLogger(self.dir, tensorboard)

    # ------------------------------------------------------------------ checkpoint
    def save_checkpoint(self, name: str, model, optimizer=None, scheduler=None, epoch: int = 0, step: int = 0,
                        best: float | None = None, extra: dict | None = None, with_rng: bool = True,
                        keep_prev: bool = False) -> None:
        """Atomic write (temporary file, then rename). ``keep_prev``: keep the previous file as ``<name>_prev.pt``."""
        path = self.checkpoint(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"model": model.state_dict(), "epoch": epoch, "step": step, "best": best, "extra": extra or {}}
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if with_rng:
            state["rng"] = rng_state()
        tmp = path.with_suffix(".pt.tmp")
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        if keep_prev and path.exists():
            os.replace(path, self.checkpoint(f"{name}_prev"))
        os.replace(tmp, path)

    def load_checkpoint(self, name: str, model=None, optimizer=None, scheduler=None, map_location="cpu",
                        restore_rng: bool = False) -> dict:
        state = torch.load(self.checkpoint(name), map_location=map_location, weights_only=False)
        if model is not None:
            model.load_state_dict(state["model"])
        if optimizer is not None and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if restore_rng and "rng" in state:
            set_rng_state(state["rng"])
        return state

    def load_latest(self, name: str, *args, **kwargs) -> dict | None:
        """Try the main checkpoint, then the fallback; stop if a file exists but none can be loaded."""
        last_error: Exception | None = None
        for cand in (name, f"{name}_prev"):
            if not self.checkpoint(cand).exists():
                continue
            try:
                state = self.load_checkpoint(cand, *args, **kwargs)
            except Exception as e:  # truncated file, decompression error, ...
                last_error = e
                log.warning("Cannot load %s: %s", self.checkpoint(cand).name, e)
                self.log_event("checkpoint_unreadable", file=self.checkpoint(cand).name, error=str(e))
                continue
            if cand != name:
                self.log_event("checkpoint_fallback", file=self.checkpoint(cand).name)
            return state
        if last_error is not None:
            raise RuntimeError(f"Cannot resume from a checkpoint in {self.dir}; stopping to keep the "
                               "earlier progress.") from last_error
        return None


def list_runs(runs_root: str | Path, exp: str) -> list[RunStore]:
    root = Path(runs_root) / exp
    return [RunStore(p) for p in sorted(root.glob("*")) if p.is_dir()] if root.exists() else []
