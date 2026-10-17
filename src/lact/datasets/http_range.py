"""HTTP Range helpers shared by the dataset downloaders: random-access reads of a remote file (2DeteCT zip files
on Zenodo) and resumable streaming downloads (Box)."""
from __future__ import annotations

import http.client
import io
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("lact")

USER_AGENT = {"User-Agent": "Mozilla/5.0 (lact downloader)"}


class RemoteFile:
    """Read bytes [a, b] of a URL with HTTP Range, retrying on network errors and rate limiting."""

    def __init__(self, url: str, retries: int = 12, timeout: float = 180.0):
        self.url, self.retries, self.timeout = url, retries, timeout
        self.size = int(self._open(urllib.request.Request(url, method="HEAD")).headers["Content-Length"])

    def _open(self, req):
        for k in range(self.retries):
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException) as e:
                wait = min(120.0, 5.0 * 2 ** k) if getattr(e, "code", None) == 429 else min(60.0, 3.0 * (k + 1))
                log.debug("network error (%s), retrying in %.0f s", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"Cannot read {self.url}")

    def get(self, a: int, b: int) -> bytes:
        b = min(b, self.size - 1)
        for k in range(self.retries):
            try:
                with self._open(urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})) as r:
                    if r.status != 206:
                        raise OSError(f"HTTP {r.status} (206 expected)")
                    data = r.read()
                if len(data) == b - a + 1:
                    return data
                raise OSError(f"truncated data: {len(data)} / {b - a + 1} bytes")
            # includes IncompleteRead on dropped links
            except (OSError, TimeoutError, http.client.HTTPException) as e:
                log.debug("reading %d-%d failed (%s), retrying", a, b, e)
                time.sleep(min(60.0, 3.0 * (k + 1)))
        raise RuntimeError(f"Cannot read bytes {a}-{b} of {self.url}")


class RangeReader(io.RawIOBase):
    """Seekable read-only file over a ``RemoteFile`` (e.g. for ``zipfile``)."""

    def __init__(self, remote: RemoteFile):
        self.r, self.pos = remote, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.r.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.r.size:
            return 0
        d = self.r.get(self.pos, min(self.r.size, self.pos + len(b)) - 1)
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def download_resumable(url: str, dst: str | os.PathLike, size: int, headers: dict | None = None, retries: int = 8,
                       chunk: int = 8 << 20, timeout: float = 60.0) -> Path:
    """Download ``url`` (``size`` bytes) to ``dst`` through ``dst.partial``, resuming an interrupted download with HTTP
    Range (restarting when the server ignores Range). Checks the final size."""
    headers = headers or {}
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size == size:
        log.info("Already have %s (%.0f MB)", dst.name, size / 2**20)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".partial")
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        if done > size:
            part.unlink()
            done = 0
        if done == size:
            break
        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "ab") as f:
                if done and r.status != 206:      # the server ignores Range: restart from scratch
                    f.truncate(0)
                    done = 0
                t0, last, start = time.time(), time.time(), done
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if time.time() - last > 15:
                        rate = (done - start) / max(1e-6, time.time() - t0)
                        log.info("  %s: %.0f/%.0f MB (%.1f MB/s)", dst.name, done / 2**20, size / 2**20,
                                 rate / 2**20)
                        last = time.time()
                f.flush()
                os.fsync(f.fileno())
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(60, 5 * attempt)
            log.warning("Network error while downloading %s (attempt %d/%d): %s; retrying in %ds", dst.name,
                        attempt, retries, e, wait)
            time.sleep(wait)
            continue
        if part.stat().st_size == size:
            break
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"Download of {dst.name} incomplete after {retries} attempts; run again to resume")
    os.replace(part, dst)
    log.info("Downloaded %s (%.0f MB)", dst, size / 2**20)
    return dst
