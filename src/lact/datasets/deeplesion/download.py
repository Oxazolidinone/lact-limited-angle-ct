"""Download DeepLesion from NIH's public Box folder: whole zip files (``download_deeplesion``) or only the drawn key
slices of every zip file (``download_key_slices``), checked against NIH's MD5 table."""
from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os
import re
import zipfile
from pathlib import Path

from ...infra.io.files import load_json, save_json
from ..box import BoxShare
from . import reader as dl
from .reader import INFO_CSV
from .split import read_key_manifest

log = logging.getLogger("lact")

DL_SHARE = "pealvgmu6kw3you44a8bc1blrluzf0hv"
DL_HOST = "nihcc.app.box.com"
DL_VANITY = "DeepLesion"
N_ZIPS = 56
MD5_FILE = "Images_png/MD5_checksums.txt"


def zip_name(k: int) -> str:
    if not 1 <= int(k) <= N_ZIPS:
        raise ValueError(f"Zip file number must be in 1..{N_ZIPS}, got {k}")
    return f"Images_png_{int(k):02d}.zip"


def _md5(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def expected_md5(table: str, name: str) -> str | None:
    """MD5 of ``name`` in the MD5_checksums.txt content (independent of the column order)."""
    for line in table.splitlines():
        if name in line:
            m = re.search(r"\b[0-9a-fA-F]{32}\b", line)
            if m:
                return m.group(0).lower()
    return None


def _marker(dest: Path, k: int) -> Path:
    return dest / f".extracted_{int(k):02d}.json"


def download_deeplesion(dest: str | os.PathLike, zips: list[int], extract: bool = True, keep_zip: bool = False,
                        force: bool = False) -> Path:
    """Download (resumable) and unpack the selected zip files; already unpacked files are skipped. Returns ``dest``."""
    dest = Path(dest)
    box = BoxShare(DL_SHARE, DL_HOST, vanity=DL_VANITY)
    box.download(box.find(INFO_CSV), dest / INFO_CSV)
    md5_table = box.download(box.find(MD5_FILE), dest / "zips" / "MD5_checksums.txt").read_text(errors="replace")
    for k in sorted({int(z) for z in zips}):
        name = zip_name(k)
        if _marker(dest, k).exists() and not force:
            log.info("%s already unpacked (%s)", name, load_json(_marker(dest, k)).get("extracted_at"))
            continue
        item = box.find(f"Images_png/{name}")
        zip_path = box.download(item, dest / "zips" / name)
        want = expected_md5(md5_table, name)
        if want is None:
            log.warning("No MD5 for %s in NIH's table; checking the size only", name)
        elif _md5(zip_path) != want:
            zip_path.unlink()
            raise RuntimeError(f"MD5 of {name} does not match NIH's table; the file was deleted, run again to "
                               "download it")
        if not extract:
            continue
        log.info("Unpacking %s -> %s", name, dest)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        save_json({"zip": name, "zip_bytes": item.size, "md5": want,
                   "extracted_at": _dt.datetime.now().isoformat(timespec="seconds")}, _marker(dest, k))
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
    if extract:
        series = dl.find_series(dest)
        if not series:
            raise RuntimeError(f"No series directory found in {dest} after unpacking")
        log.info("%d patients, %d series at %s", len(series), sum(len(v) for v in series.values()), dest)
    return dest


def download_key_slices(dest: str | os.PathLike, manifest: str | os.PathLike, zips: list[int] | None = None,
                        workers: int = 6, keep_zip: bool = False, force: bool = False) -> Path:
    """Fetch EXACTLY the slices of ``manifest`` from the ``Images_png_XX.zip`` files (all 56 by default, since the
    drawn slices are spread over the whole dataset) into ``<dest>/Images_png/<patient>_<study>_<series>/<slice>.png``.
    Each zip file (~4 GiB) is downloaded, checked against its MD5, stripped of the needed slices and deleted right away
    (unless ``keep_zip``), so 220 GiB of disk are not needed; ``workers`` files download in parallel. Re-running skips
    zip files already processed.

    NIH's ``Key_slices.zip`` cannot be used: its images are 8-bit RGB renderings, not 16-bit HU.
    ``dest`` must hold only drawn slices: do not point it at a directory with fully unpacked zip files."""
    from concurrent.futures import ThreadPoolExecutor

    dest = Path(dest)
    rows = read_key_manifest(manifest)
    want = {}
    for r in rows:
        series, sl = r["file_name"][:-len(".png")].rsplit("_", 1)
        want[f"{series}/{sl}.png"] = dest / "Images_png" / series / f"{sl}.png"
    todo = sorted({int(z) for z in (zips or range(1, N_ZIPS + 1))})
    done_mark = lambda k: dest / f".keyslices_{int(k):02d}.json"

    def missing() -> list[str]:
        return [m for m, f in want.items() if not f.is_file()]

    if not force and not missing():
        log.info("All %d key slices are already at %s", len(want), dest)
        return dest
    box = BoxShare(DL_SHARE, DL_HOST, vanity=DL_VANITY)
    box.download(box.find(INFO_CSV), dest / INFO_CSV)
    md5_table = box.download(box.find(MD5_FILE), dest / "zips" / "MD5_checksums.txt").read_text(errors="replace")
    items = {i.name: i for i in box.list_folder(box.find("Images_png").id)}

    def one(k: int) -> int:
        name = zip_name(k)
        if done_mark(k).exists() and not force:
            return int(load_json(done_mark(k)).get("n_slices", 0))
        zip_path = box.download(items[name], dest / "zips" / name)
        md5 = expected_md5(md5_table, name)
        if md5 is not None and _md5(zip_path) != md5:
            zip_path.unlink()
            raise RuntimeError(f"MD5 of {name} does not match NIH's table; the file was deleted, run again to "
                               "download it")
        n = 0
        with zipfile.ZipFile(zip_path) as z:
            for m in z.infolist():
                key = "/".join(m.filename.split("/")[-2:])
                if key in want:
                    want[key].parent.mkdir(parents=True, exist_ok=True)
                    want[key].write_bytes(z.read(m))
                    n += 1
        save_json({"zip": name, "n_slices": n, "md5": md5,
                   "done_at": _dt.datetime.now().isoformat(timespec="seconds")}, done_mark(k))
        if not keep_zip:
            zip_path.unlink(missing_ok=True)
        log.info("%s: %d key slices", name, n)
        return n

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        found = sum(pool.map(one, todo))
    left = missing()
    if left:
        raise RuntimeError(f"Still missing {len(left)}/{len(want)} slices of {manifest} (e.g. {left[:3]}); zip files "
                           f"checked: {todo[0]}..{todo[-1]}")
    log.info("%d patients, %d key slices at %s (this run drew %d slices)", len({r["patient"] for r in rows}),
             len(want), dest, found)
    return dest
