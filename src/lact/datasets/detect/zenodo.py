"""Download the chosen 2DeteCT slices from Zenodo, reading only the needed bytes of each zip file with HTTP Range."""
from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import os
import struct
import threading
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

from ..http_range import RangeReader, RemoteFile
from .raw import N_DET_RAW, slice_name

log = logging.getLogger("lact")

RAW_IDS = {1: 8014758, 2: 8014766, 3: 8014787, 4: 8014829, 5: 8014874}
REC_IDS = {1: 8017583, 2: 8017604, 3: 8017612, 4: 8017618, 5: 8017624}


def chunk_of(i: int) -> int:
    return (i - 1) // 1000 + 1


def zip_url(chunk: int, rec: bool) -> str:
    name = f"2DeteCT_slices{1000 * (chunk - 1) + 1}-{1000 * chunk}" + ("_RecSeg" if rec else "")
    return f"https://zenodo.org/records/{(REC_IDS if rec else RAW_IDS)[chunk]}/files/{name}.zip?download=1"


def zip_index(remote: RemoteFile, cache: Path, keep=lambda name: True) -> dict[str, list[int]]:
    """{name: [entry offset, compressed size, original size, compression method, name length]} of the ``keep``
    entries; cached as JSON."""
    if cache.exists():
        return json.loads(cache.read_text())
    zf = zipfile.ZipFile(io.BufferedReader(RangeReader(remote), buffer_size=4 << 20))
    idx = {zi.filename: [zi.header_offset, zi.compress_size, zi.file_size, zi.compress_type,
                         len(zi.filename.encode())] for zi in zf.infolist() if keep(zi.filename)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(idx))
    return idx


def read_member(remote: RemoteFile, entry: list[int], max_out: int | None = None) -> bytes:
    """Decompressed content of one entry; ``max_out``: only that many leading bytes (incremental decompression)."""
    off, csize, fsize, method, nlen = entry
    want = fsize if max_out is None else min(fsize, max_out)
    ratio = csize / max(fsize, 1)
    first = int(want * ratio * 1.05) + (64 << 10) if method == 8 else want
    head = remote.get(off, off + 30 + nlen + 64 + first - 1)         # entry header + first data bytes in one request
    n, m = struct.unpack("<HH", head[26:30])
    start = off + 30 + n + m
    if method == 0:
        data = head[30 + n + m:]
        if len(data) < want:
            data += remote.get(start + len(data), start + want - 1)
        return data[:want]
    if method != 8:
        raise ValueError(f"compression method {method} is not supported")
    d = zlib.decompressobj(-15)
    out = [d.decompress(head[30 + n + m:])]
    got, pos, end = len(out[0]), off + len(head), start + csize
    while got < want and pos < end:
        step = int((want - got) * ratio * 1.1) + (256 << 10)
        b = min(end, pos + step) - 1
        o = d.decompress(remote.get(pos, b))
        out.append(o)
        got += len(o)
        pos = b + 1
    data = b"".join(out)
    if len(data) < want:
        raise RuntimeError(f"short decompression: {len(data)} / {want} bytes")
    return data[:want]


def tiff_rows(data: bytes, n_rows: int | None = None) -> np.ndarray:
    """Uncompressed single-strip TIFF image (like the 2DeteCT files), readable even from a truncated file."""
    if data[:4] != b"II*\x00":
        raise ValueError("only little-endian TIFF is supported")
    ifd = struct.unpack("<I", data[4:8])[0]
    n_ent = struct.unpack("<H", data[ifd:ifd + 2])[0]
    tags = {}
    for k in range(n_ent):
        tag, typ, cnt, val = struct.unpack("<HHII", data[ifd + 2 + 12 * k: ifd + 14 + 12 * k])
        if typ == 3 and cnt == 1:
            val &= 0xFFFF
        if tag == 273 and cnt != 1:
            raise ValueError("TIFF with several strips: not supported")
        tags[tag] = val
    w, h, bits = tags[256], tags[257], tags.get(258, 16)
    fmt = tags.get(339, 1)                       # 1 = unsigned integer, 3 = floating point
    dtype = {(16, 1): np.uint16, (32, 3): np.float32, (8, 1): np.uint8}[(bits, fmt)]
    if tags.get(259, 1) != 1:
        raise ValueError("compressed TIFF: not supported")
    off = tags[273]
    item = np.dtype(dtype).itemsize
    rows = h if n_rows is None else min(h, n_rows)
    if len(data) < off + rows * w * item:
        raise ValueError(f"truncated TIFF data: need {off + rows * w * item}, have {len(data)} bytes")
    return np.frombuffer(data, dtype, rows * w, off).reshape(rows, w)


def download_slices(slices: list[int], raw_dir: str | os.PathLike, n_rows: int = 1200,
                    full: list[int] | None = None, workers: int = 32) -> Path:
    """Download the sinogram (first ``n_rows`` projections; ``full``: all 3601), dark, flat1, flat2 in mode 2 and the
    mode-2 1024² reference image of each slice into ``raw_dir/slice#####.npz``. Existing slices are skipped (re-run to
    resume)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_set = set(full or [])
    todo = sorted(set(slices) | full_set)
    todo = [i for i in todo if not (raw_dir / f"{slice_name(i)}.npz").exists()]
    log.info("2DeteCT: %d slices to download (%d present), %d leading projections per slice, %d threads",
             len(todo), len(set(slices) | full_set) - len(todo), n_rows, workers)
    remotes, idx, lock = {}, {}, threading.Lock()
    key_locks = {(c, rec): threading.Lock() for c in range(1, 6) for rec in (False, True)}

    def ready(chunk: int, rec: bool):
        key = (chunk, rec)
        with key_locks[key]:                     # one lock per zip file: central directories load in parallel
            if key not in idx:
                r = RemoteFile(zip_url(chunk, rec))
                keep = (lambda s: "/mode2/" in s and not s.endswith("/"))
                idx[key] = zip_index(r, raw_dir / "index" / f"chunk{chunk}{'_rec' if rec else ''}.json", keep)
                remotes[key] = r
        return remotes[key], idx[key]

    keys = sorted({(chunk_of(i), rec) for i in todo for rec in (False, True)})
    with cf.ThreadPoolExecutor(max(1, len(keys))) as ex:     # prefetch every needed central directory in parallel
        list(ex.map(lambda k: ready(*k), keys))
    log.info("2DeteCT: central directories of %d zip files loaded", len(keys))

    sino_bytes = 304 + N_DET_RAW * 2 * n_rows
    done, nbytes, t0 = [0], [0], time.time()

    def one(i: int) -> None:
        c = chunk_of(i)
        r, ix = ready(c, False)
        rr, rix = ready(c, True)
        base = f"{slice_name(i)}/mode2/"
        parts = {k: tiff_rows(read_member(r, ix[base + f"{k}.tif"]))[0] for k in ("dark", "flat1", "flat2")}
        sino = tiff_rows(read_member(r, ix[base + "sinogram.tif"], None if i in full_set else sino_bytes),
                         None if i in full_set else n_rows)
        rec = tiff_rows(read_member(rr, rix[base + "reconstruction.tif"]))
        tmp = raw_dir / f".{slice_name(i)}.tmp.npz"
        np.savez(tmp, sino=sino, rec=rec, slice=np.int32(i), **parts)
        os.replace(tmp, raw_dir / f"{slice_name(i)}.npz")
        with lock:
            done[0] += 1
            nbytes[0] += sino.nbytes + rec.nbytes
            if done[0] % 25 == 0 or done[0] == len(todo):
                dt = time.time() - t0
                log.info("  %d/%d slices, %.1f MB/s (decompressed), ~%.1f hours left", done[0], len(todo),
                         nbytes[0] / 1e6 / dt, dt / done[0] * (len(todo) - done[0]) / 3600)

    order = sorted(todo, key=lambda i: (i not in full_set, chunk_of(i), i))     # geometry-check slices first
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one, i): i for i in order}
        bad = []
        for f in cf.as_completed(futs):
            if f.exception() is not None:
                bad.append(futs[f])
                log.warning("slice %d failed: %s", futs[f], f.exception())
    if bad:
        raise RuntimeError(f"{len(bad)} slices could not be downloaded (run the command again): {sorted(bad)[:20]} ...")
    log.info("2DeteCT: done, data in %s", raw_dir)
    return raw_dir
