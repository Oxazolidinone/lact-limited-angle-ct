"""Inter-process locks on one machine (Linux/Colab and macOS).

The operating system releases a lock when its process dies, so no stale lock files need cleaning after a crash.
Local disks only: this is not a distributed lock across Colab machines or containers.
"""
from __future__ import annotations

import errno
import fcntl
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("lact")
# File systems without flock support (some network/FUSE mounts) return these codes.
_UNSUPPORTED = {errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL}


@contextmanager
def file_lock(path: Path, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise FileExistsError(f"Another process is already running: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def exclusive_or_warn(path: Path) -> Iterator[bool]:
    """Non-blocking lock. Held elsewhere -> FileExistsError. Locking unsupported by the file system -> warn and run
    without the lock (yield False) instead of failing the whole training run."""
    try:
        cm = file_lock(path, blocking=False)
        cm.__enter__()
    except OSError as exc:
        if isinstance(exc, FileExistsError) or exc.errno not in _UNSUPPORTED:
            raise
        log.warning("Cannot lock %s (%s); no protection against duplicate runs.", path, exc)
        yield False
        return
    try:
        yield True
    finally:
        cm.__exit__(None, None, None)


@contextmanager
def wait_or_warn(path: Path) -> Iterator[bool]:
    """Blocking lock (for short read-modify-write steps such as the manifest). Unsupported -> warn and continue."""
    try:
        cm = file_lock(path, blocking=True)
        cm.__enter__()
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED:
            raise
        log.warning("Cannot lock %s (%s); writing without a lock.", path, exc)
        yield False
        return
    try:
        yield True
    finally:
        cm.__exit__(None, None, None)
