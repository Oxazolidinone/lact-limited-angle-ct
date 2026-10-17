"""Two real processes sharing the cache, no ASTRA/GPU needed."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

import lact
from lact.infra.data.cache import is_fresh, stage_file
from lact.infra.system.locks import file_lock
from lact.infra.system.processes import ProcessRunner


def test_stage_identity_includes_source_path(tmp_path):
    a, b = tmp_path / "a" / "same.bin", tmp_path / "b" / "same.bin"
    for path, content in ((a, b"aaa"), (b, b"bbb")):
        path.parent.mkdir()
        path.write_bytes(content)
        os.utime(path, ns=(1000000000, 1000000000))
    cached = stage_file(a, tmp_path / "cache")
    assert not is_fresh(b, cached)
    assert stage_file(b, tmp_path / "cache").read_bytes() == b"bbb"


def test_parallel_staging_and_dataset_build_once(tmp_path, monkeypatch):
    with h5py.File(tmp_path / "images.h5", "w") as f:
        f.attrs.update(mu_ref=1., mu_water_mm=.02)
        f["p/img256"] = np.ones((2, 4, 4), dtype=np.float32)
        f["p/z"] = [0., 1.]
    with h5py.File(tmp_path / "sino.h5", "w") as f:
        f["p/y"] = np.ones((2, 3, 8), dtype=np.float32)
        f["p/fbp"] = np.ones((2, 4, 4), dtype=np.float32)
    script = """
import pathlib, sys, time
import numpy as np
from lact.infra.data.cache import ArrayCache, stage_file
from lact.infra.data.dataset import LACTDataset
root, worker = pathlib.Path(sys.argv[1]), sys.argv[2]
(root / ('ready' + worker)).touch()
deadline = time.monotonic() + 15
while len(list(root.glob('ready*'))) < 2 and time.monotonic() < deadline:
    time.sleep(.02)
assert len(list(root.glob('ready*'))) == 2
images = stage_file(root / 'images.h5', root / 'cache', verify='sha256')
sino = stage_file(root / 'sino.h5', root / 'cache', verify='sha256')
read = LACTDataset._read_blocks
def counted(self, blocks):
    with (root / 'reads.txt').open('a') as f: f.write(worker + '\\n')
    time.sleep(.2)
    return read(self, blocks)
LACTDataset._read_blocks = counted
ds = LACTDataset(images, sino, ['p'], array_cache=ArrayCache(root / 'arrays'))
assert len(ds) == 2 and ds[0]['y'].shape == (1, 3, 8)
assert isinstance(ds.cache[('p', 'orig')]['y'], np.memmap)
"""
    commands = [(str(i), [sys.executable, "-c", script, str(tmp_path), str(i)]) for i in range(2)]
    # absolute PYTHONPATH: the children run in tmp_path, not in the repository.
    monkeypatch.setenv("PYTHONPATH", str(Path(lact.__file__).parents[1]))
    with ProcessRunner(tmp_path, max_hours=30 / 3600) as runner:
        assert runner.run(commands, workers=2) == {"0": 0, "1": 0}
    assert len((tmp_path / "reads.txt").read_text().splitlines()) == 1


def test_os_releases_lock_after_process_dies(tmp_path):
    path = tmp_path / "lock"
    script = "import fcntl, os, sys; f=open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); os._exit(0)"
    with ProcessRunner(tmp_path) as runner:
        assert runner.run([("lock", [sys.executable, "-c", script, str(path)])]) == {"lock": 0}
    with file_lock(path, blocking=False):
        assert path.exists()
