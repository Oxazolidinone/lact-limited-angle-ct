"""Runtime environment: seeds/determinism, device selection, logging, environment and git information."""
from __future__ import annotations

import logging
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import torch

from ...config.loader import REPO_ROOT


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        # warn_only: some operations (e.g. sparse matrix products on GPUs) have no deterministic implementation;
        # a warning is logged instead of stopping the program.
        torch.use_deterministic_algorithms(True, warn_only=True)


def apply_precision(tf32: bool) -> None:
    """Numerical mode of GPU convolutions and matrix products.

    TF32 (mantissa rounded to 10 bits for speed) exists only on Ampere and newer GPUs (RTX 30xx onwards); older GPUs
    always compute full FP32 whatever the flag, so ``tf32 = False`` is the only mode in which every GPU generation
    computes the same way. PyTorch enables TF32 for cuDNN and disables it for matrix products by default; both flags
    are set from the configuration in every command that runs a model (training, evaluation, timing) so they never
    compute in two different ways."""
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)


def worker_init_fn(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    # torch.load(map_location="cuda") moves the RNG state to the GPU as well, but the random generators
    # still require a CPU ByteTensor state.
    torch.set_rng_state(st["torch"].cpu())
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in st["cuda"]])


def git_hash(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.check_output(args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        h = out.decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        return h + ("-dirty" if dirty else "")
    except Exception:
        # copies without .git (e.g. unpacked on Colab): use the version code recorded at packaging time
        f = REPO_ROOT / "CODE_VERSION"
        return f.read_text(encoding="utf-8").strip() if f.exists() else "unknown"


def collect_env() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git": git_hash(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    for mod in ("numpy", "scipy", "h5py", "astra", "pydicom", "skimage"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def gpu_info(index: int = 0) -> dict:
    """GPU name, memory (GB), compute capability; ``gpu = None`` without CUDA."""
    if not torch.cuda.is_available():
        return {"gpu": None, "gpu_mem_gb": 0.0}
    p = torch.cuda.get_device_properties(index)
    return {"gpu": p.name, "gpu_mem_gb": round(p.total_memory / 2**30, 1), "capability": f"{p.major}.{p.minor}"}


def host_info() -> dict:
    return {"host": socket.gethostname(), "pid": os.getpid(), "colab": in_colab(), **gpu_info()}


class GracefulInterrupt:
    """Catch SIGTERM / SIGINT during training.

    The first signal only sets the ``requested`` flag: the loop saves a checkpoint and stops cleanly.
    A second SIGINT (pressing stop again) stops immediately with KeyboardInterrupt.
    Can only be installed from the main thread; elsewhere it does nothing.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)):
        self.signals = signals
        self.requested = False
        self.signal_name: str | None = None
        self._old: dict[int, object] = {}

    def _handler(self, signum, frame):
        if self.requested and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        logging.getLogger("lact").warning("Received %s: saving a checkpoint and stopping after the current step",
                                          self.signal_name)

    def __enter__(self) -> GracefulInterrupt:
        if threading.current_thread() is threading.main_thread():
            for sig in self.signals:
                self._old[sig] = signal.signal(sig, self._handler)
        return self

    def __exit__(self, *exc) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)  # type: ignore[arg-type]
        self._old.clear()


def select_device(pref: str = "auto") -> torch.device:
    """Resolve ``auto`` to cuda when available, otherwise cpu. MPS lacks support for CSR sparse matrix products."""
    pref = (pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "mps":
        logging.getLogger("lact").warning("MPS does not support CSR sparse matmul; falling back to cpu.")
        return torch.device("cpu")
    return torch.device(pref)


def get_logger(name: str = "lact", file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if file is not None:
        file = str(file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(file).resolve())
                   for h in logger.handlers):
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger
