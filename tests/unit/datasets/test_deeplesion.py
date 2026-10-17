"""Second dataset (DeepLesion): 16-bit PNG reading, patient selection/splitting, preprocessing into images.h5."""
from __future__ import annotations

import numpy as np
import pytest

from lact.config import load_config
from lact.datasets.deeplesion import expected_md5, make_split, zip_name
from lact.datasets.deeplesion import reader as dl
from lact.infra.io.hdf5_store import ImageStore
from lact.services.data import prep
from lact.services.data.fake import make_fake_deeplesion


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    return make_fake_deeplesion(tmp_path_factory.mktemp("dl"), n_patients=12, n_slices=6, size=64)


def test_png_roundtrip_is_hu(tmp_path):
    hu = np.array([[-1024.0, 0.0], [40.0, 3071.0]], np.float32)
    dl.write_png_hu(tmp_path / "s" / "001.png", hu)
    assert np.array_equal(dl.read_png_hu(tmp_path / "s" / "001.png"), hu)


def test_series_are_grouped_by_patient_and_slices_sorted(fake_root):
    series = dl.find_series(fake_root)
    assert list(series) == [f"DL{p:06d}" for p in range(1, 13)]
    assert [d.name for d in series["DL000003"]] == ["000003_01_01", "000003_01_02"]
    assert [f.stem for f in dl.series_slices(series["DL000003"][1])] == ["020", "021", "022"]


def test_info_gives_official_patient_split_and_spacing(fake_root):
    info = dl.load_info(fake_root / "DL_info.csv")
    assert info["000001_01_01"]["spacing"] == [0.8, 0.8, 5.0] and not info["000001_01_01"]["noisy"]
    split = dl.official_split(info)
    assert [split[f"DL{p:06d}"] for p in (1, 8, 9, 10, 11, 12)] == ["train", "train", "val", "val", "test", "test"]
    assert dl.load_info(fake_root / "missing.csv") == {}


def test_read_patient_caps_slices_evenly_and_skips_noisy_or_odd_sized(fake_root):
    series, info = dl.find_series(fake_root), dl.load_info(fake_root / "DL_info.csv")
    hu, meta = dl.read_patient(series["DL000002"], info, None, size=64)
    assert hu.shape == (6, 64, 64) and hu.dtype == np.float32 and hu.min() >= -1001
    assert len({m["sop_uid"] for m in meta}) == 6 and [m["z"] for m in meta] == sorted(m["z"] for m in meta)
    hu4, meta4 = dl.read_patient(series["DL000002"], info, 4, size=64)
    assert hu4.shape[0] == 4 and [m["sop_uid"] for m in (meta4[0], meta4[-1])] == [meta[0]["sop_uid"],
                                                                                    meta[-1]["sop_uid"]]
    noisy = {**info, "000002_01_02": {**info["000002_01_02"], "noisy": True}}
    assert dl.read_patient(series["DL000002"], noisy, None, size=64)[0].shape[0] == 3
    with pytest.raises(RuntimeError):
        dl.read_patient(series["DL000002"], info, None, size=512)       # no 512 x 512 series


def test_pick_evenly():
    assert dl.pick_evenly(5, None) == [0, 1, 2, 3, 4] and dl.pick_evenly(3, 10) == [0, 1, 2]
    assert dl.pick_evenly(101, 5) == [0, 25, 50, 75, 100]


def test_split_follows_official_labels_and_is_loadable(fake_root, tmp_path):
    out = tmp_path / "split.yaml"
    res = make_split(fake_root, out, n_train=5, n_val=2, n_test=2, min_slices=4, size=64)
    assert res["split"] == {"train": [f"DL{p:06d}" for p in range(1, 6)], "val": ["DL000009", "DL000010"],
                            "test": ["DL000011", "DL000012"]}
    cfg = load_config(["configs/data/deeplesion.yaml", str(out)])
    assert cfg.split.train == res["split"]["train"] and cfg.data.patients[-1] == "DL000012"   # ids stay strings
    with pytest.raises(RuntimeError, match="Not enough patients"):
        make_split(fake_root, tmp_path / "x.yaml", n_train=5, n_val=2, n_test=3, min_slices=4, size=64)
    with pytest.raises(RuntimeError, match="Not enough patients"):
        make_split(fake_root, tmp_path / "y.yaml", n_train=1, n_val=1, n_test=1, min_slices=7, size=64)


def test_preprocess_deeplesion_writes_normalized_images(fake_root, tmp_path):
    split = tmp_path / "split.yaml"
    make_split(fake_root, split, n_train=3, n_val=1, n_test=1, min_slices=4, size=64)
    out = tmp_path / "out"
    cfg = load_config(["configs/data/deeplesion.yaml", str(split)], [
        f"data.dicom_root={fake_root}", f"data.info_csv={fake_root}/DL_info.csv", "data.max_slices_per_patient=4",
        "data.img_size_sim=64", "data.img_size_rec=32", f"paths.outputs={out}", f"paths.data={out}/data",
        f"data.images_h5={out}/data/images.h5", f"data.stats_json={out}/data/stats.json",
        f"data.slices_json={out}/data/slices.json"])
    prep.preprocess(cfg)
    store = ImageStore(cfg.data.images_h5)
    assert sorted(store.patients()) == sorted(cfg.split.train + cfg.split.val + cfg.split.test)
    assert all(store.n_slices(p) == 4 for p in store.patients())
    x = store.read(cfg.split.test[0], "img256", 0)
    assert x.shape == (32, 32) and x.min() >= 0.0 and x.max() <= 1.0 and x.max() > 0.2
    assert 0.0 < store.mu_ref < 0.1


def test_zip_names_and_md5_table():
    assert zip_name(1) == "Images_png_01.zip" and zip_name(56) == "Images_png_56.zip"
    with pytest.raises(ValueError):
        zip_name(57)
    table = "d41d8cd98f00b204e9800998ecf8427e  Images_png_01.zip\nImages_png_02.zip 0CC175B9C0F1B6A831C399E269772661\n"
    assert expected_md5(table, "Images_png_01.zip") == "d41d8cd98f00b204e9800998ecf8427e"
    assert expected_md5(table, "Images_png_02.zip") == "0cc175b9c0f1b6a831c399e269772661"
    assert expected_md5(table, "Images_png_03.zip") is None
