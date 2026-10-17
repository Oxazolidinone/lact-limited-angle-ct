"""Evaluation results store.

    <eval>/per_slice/<tag>.csv   per-slice metrics
    <eval>/preds/<tag>.h5        float16 predictions (/{pid}/pred, /{pid}/idx), attrs["info"] = JSON metadata
    <eval>/index.json            {tag: metadata}  -> lookup by attributes, never by parsing file names
    <eval>/test_access.log       log of every opening of the test split
tag = "<run_id>__<split>__<geometry_test>" is only a file name and must not be parsed back.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..io.files import load_json, save_json
from ..system.locks import wait_or_warn


def make_tag(info: dict) -> str:
    return f"{info['run_id']}__{info['split']}__{info['geometry_test']}"


class ResultsStore:
    def __init__(self, eval_dir: str | Path):
        self.dir = Path(eval_dir)
        self.index_file = self.dir / "index.json"

    # ------------------------------------------------------------------ ghi
    def csv_path(self, tag: str) -> Path:
        return self.dir / "per_slice" / f"{tag}.csv"

    def pred_path(self, tag: str) -> Path:
        return self.dir / "preds" / f"{tag}.h5"

    def exists(self, info: dict, require_pred: bool = False) -> bool:
        tag = make_tag(info)
        entry = self.index().get(tag)
        return bool(entry is not None and self.csv_path(tag).exists()
                    and (not (require_pred or entry.get("has_pred"))
                         or (entry.get("has_pred") and self.pred_path(tag).exists())))

    def save(self, info: dict, df: pd.DataFrame, preds: dict[str, dict[int, np.ndarray]] | None = None) -> str:
        tag = make_tag(info)
        p = self.csv_path(tag)
        p.parent.mkdir(parents=True, exist_ok=True)
        # the index entry marks completion; drop the old entry before replacing the result files.
        with self._index_lock():
            index = self.index()
            index.pop(tag, None)
            save_json(index, self.index_file)
        tmp = p.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, p)
        if preds is not None:
            hp = self.pred_path(tag)
            hp.parent.mkdir(parents=True, exist_ok=True)
            tmp = hp.with_suffix(".h5.tmp")
            with h5py.File(tmp, "w") as h5:
                h5.attrs["info"] = json.dumps(info, default=str)
                for pid, d in preds.items():
                    idx = sorted(d)
                    h5.create_dataset(f"{pid}/pred", data=np.stack([d[i] for i in idx]).astype(np.float16),
                                      compression="lzf")
                    h5.create_dataset(f"{pid}/idx", data=np.array(idx, np.int32))
            os.replace(tmp, hp)
        with self._index_lock():
            index = self.index()
            index[tag] = {**info, "has_pred": preds is not None}
            save_json(index, self.index_file)
        return tag

    def _index_lock(self):
        """Several evaluation commands (runs, baselines, E6) may write index.json concurrently when run in parallel."""
        return wait_or_warn(self.dir / ".index.json.lock")

    def log_test_access(self, forced: bool) -> None:
        f = self.dir / "test_access.log"
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}\t{'FORCED' if forced else 'ok'}\n")

    # ------------------------------------------------------------------ reading
    def index(self) -> dict[str, dict]:
        return load_json(self.index_file) if self.index_file.exists() else {}

    def find(self, **filters) -> dict[str, dict]:
        """Filter the index by metadata, e.g. ``split='test', geometry_test='fan_a120', exp='e3_main'``."""
        return {t: m for t, m in self.index().items() if all(m.get(k) == v for k, v in filters.items())}

    def load_slices(self, **filters) -> pd.DataFrame:
        tags = self.find(**filters)
        frames = [pd.read_csv(self.csv_path(t)) for t in tags if self.csv_path(t).exists()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def load_pred(self, tag: str, pid: str, idx: int) -> np.ndarray | None:
        f = self.pred_path(tag)
        if not f.exists():
            return None
        with h5py.File(f, "r") as h5:
            if pid not in h5:
                return None
            ids = list(h5[pid]["idx"][:])
            return h5[pid]["pred"][ids.index(idx)].astype(np.float32) if idx in ids else None

    def iter_preds(self, tag: str) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
        """(pid, slice indices, float32 predictions) per patient."""
        f = self.pred_path(tag)
        if not f.exists():
            return
        with h5py.File(f, "r") as h5:
            for pid in h5:
                yield pid, h5[pid]["idx"][:], h5[pid]["pred"][:].astype(np.float32)
