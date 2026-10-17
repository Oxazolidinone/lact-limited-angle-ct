"""lact data {download-*, inspect, preprocess, simulate, stage, fake}: data preparation (steps 00, 01, 04, 90)."""
from __future__ import annotations

import logging
from pathlib import Path

from ...config.loader import resolve_path
from ...datasets import deeplesion, detect, mayo
from ...services.data import fake as fake_data
from ...services.data import prep as data_prep
from ...services.data import simulation, staging
from ..common import GEOMS_MAIN, add_geometries, device, geometries, load_cfg

log = logging.getLogger("lact")
MANIFEST = f"configs/data/{deeplesion.KEY_MANIFEST}"


def cmd_inspect(a) -> int:
    df = data_prep.inspect_dicom(load_cfg(a))
    log.info("Total slices per split: %s", df.groupby("split")["n_slices"].sum().to_dict())
    return 0


def cmd_preprocess(a) -> int:
    data_prep.preprocess(load_cfg(a), overwrite=a.overwrite)
    log.info("Done. Manifest updated.")
    return 0


def cmd_simulate(a) -> int:
    cfg, dev = load_cfg(a), device(a)
    files = simulation.simulate_all(cfg, geometries(a.geometries), device=dev, batch=a.batch, overwrite=a.overwrite)
    log.info("Xong: %s", [str(f) for f in files])
    return 0


def cmd_fake(a) -> int:
    if a.format == "deeplesion":
        root = fake_data.make_fake_deeplesion(resolve_path(a.out), n_slices=a.n_slices, size=a.size, seed=a.seed)
    else:
        root = fake_data.make_fake_dicom(resolve_path(a.out), n_slices=a.n_slices, size=a.size, seed=a.seed)
    log.info("Synthetic data (%s) -> %s (NOT for results)", a.format, root)
    return 0


def _deeplesion_root(a):
    if a.dest:
        return resolve_path(a.dest)
    cfg = load_cfg(a)
    root = cfg.data.deeplesion_root if cfg.data.format == "mixed" else cfg.data.dicom_root
    if cfg.data.format not in ("deeplesion", "mixed") or root is None:
        raise SystemExit("--dest is required, or -c configs/data/deeplesion.yaml (or configs/variant/joint.yaml) to "
                         "get the DeepLesion directory")
    return root


def cmd_download_deeplesion(a) -> int:
    if a.key_slices:
        out = deeplesion.download_key_slices(_deeplesion_root(a), resolve_path(a.manifest), a.zips, a.workers,
                                             keep_zip=a.keep_zip, force=a.force)
        log.info("data.deeplesion_root should be: %s", out)
        return 0
    out = deeplesion.download_deeplesion(_deeplesion_root(a), a.zips or [1, 2], extract=not a.no_extract,
                                         keep_zip=a.keep_zip, force=a.force)
    log.info("data.dicom_root should be: %s", out)
    return 0


def cmd_split_deeplesion(a) -> int:
    if a.key_slices:
        info = resolve_path(a.info_csv) if a.info_csv else _deeplesion_root(a) / deeplesion.INFO_CSV
        rows = deeplesion.select_key_slices(info, a.n_train or 2000, a.n_val or 200, a.n_test or 300, a.seed)
        res = deeplesion.write_key_slice_configs(rows, resolve_path(a.manifest).parent, seed=a.seed)
        log.info("Slices per split: %s", res["n_slices"])
        return 0
    a.n_train, a.n_val, a.n_test = a.n_train or 45, a.n_val or 6, a.n_test or 15
    cfg = load_cfg(a)
    res = deeplesion.make_split(_deeplesion_root(a), resolve_path(a.out), a.n_train, a.n_val, a.n_test,
                                a.min_slices, size=cfg.data.img_size_sim)
    log.info("Usable slices per split: %s", {k: sum(res["usable_slices"][p] for p in v)
                                              for k, v in res["split"].items()})
    return 0


def cmd_download_mayo(a) -> int:
    if a.dest:
        dest = resolve_path(a.dest)
    else:
        root = load_cfg(a).data.dicom_root
        if root is None:
            raise SystemExit("--dest or data.dicom_root in the configuration is required")
        if root.name != "full_3mm":
            log.warning("data.dicom_root (%s) does not end with full_3mm; the data will be at %s/full_3mm",
                        root, root.parent)
        dest = root.parent
    out = mayo.download_mayo(dest, extract=not a.no_extract, keep_zip=not a.delete_zip, force=a.force)
    log.info("data.dicom_root should be: %s", out)
    return 0


LION_RECORDS_URL = ("https://raw.githubusercontent.com/CambridgeCIA/LION/main/LION/data_loaders/2deteCT/"
                    "default_data_records.csv")


def _detect_records(path) -> Path:
    """LION's slice -> sample table (to split as LION does); downloaded from LION's GitHub when missing."""
    path = resolve_path(path)
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s -> %s", LION_RECORDS_URL, path)
        urllib.request.urlretrieve(LION_RECORDS_URL, path)
    return path


def cmd_split_detect(a) -> int:
    d = load_cfg(a).data.detect
    rows = detect.select_slices(_detect_records(d.records_csv), a.n_train, a.n_val, a.n_test, a.seed)
    res = detect.write_split_configs(rows, resolve_path(a.out_dir), seed=a.seed)
    log.info("Slices per split: %s", res["n_slices"])
    return 0


def cmd_download_detect(a) -> int:
    d = load_cfg(a).data.detect
    slices = [r["slice"] for r in detect.read_slices(d.slices_csv)]
    detect.download_slices(slices, d.raw_dir, n_rows=a.rows or d.n_rows, full=a.full, workers=a.workers)
    return 0


def cmd_stage(a) -> int:
    files = staging.stage_all(load_cfg(a), a.geometries)
    log.info("Cache ready: %s", [str(f) for f in files])
    return 0


def register(sub, parent) -> None:
    p = sub.add_parser("data", help="Data: download and split datasets, images.h5, sinograms, cache")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    c = s.add_parser("inspect", parents=[parent], help="Inspect DICOM (slice count, thickness, kernel, HU range)")
    c.set_defaults(func=cmd_inspect)

    c = s.add_parser("preprocess", parents=[parent], help="DICOM -> normalised 512/256 images -> images.h5")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_preprocess)

    c = s.add_parser("simulate", parents=[parent], help="Simulate limited-angle sinograms + FBP for each geometry")
    add_geometries(c, GEOMS_MAIN)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_simulate)

    c = s.add_parser("download-mayo", parents=[parent],
                     help="Download the full-dose 3 mm images (10 patients) from AAPM's public Box, resuming after "
                          "network drops")
    c.add_argument("--dest", default=None, help="Target directory (default: the parent of data.dicom_root)")
    c.add_argument("--no-extract", action="store_true", help="Only download the zip file")
    c.add_argument("--delete-zip", action="store_true", help="Delete the zip file after unpacking")
    c.add_argument("--force", action="store_true", help="Download/unpack again even if present")
    c.set_defaults(func=cmd_download_mayo)

    c = s.add_parser("download-deeplesion", parents=[parent],
                     help="Download DeepLesion's Images_png_XX.zip files (public NIH Box, ~4 GiB per file)")
    c.add_argument("--zips", nargs="+", type=int, default=None,
                   help="Zip file numbers (1..56); default 1 2, all 56 files with --key-slices")
    c.add_argument("--dest", default=None, help="Target directory (default: data.dicom_root)")
    c.add_argument("--no-extract", action="store_true", help="Only download the zip files")
    c.add_argument("--keep-zip", action="store_true", help="Keep the zip files after unpacking")
    c.add_argument("--force", action="store_true", help="Download/unpack again even if present")
    c.add_argument("--key-slices", action="store_true",
                   help="Phase 2: scan the zip files and keep only the key slices of --manifest (each zip is deleted "
                        "once done)")
    c.add_argument("--workers", type=int, default=6, help="--key-slices: zip files downloaded in parallel")
    c.add_argument("--manifest", default=MANIFEST, help="List of drawn key slices (default: %(default)s)")
    c.set_defaults(func=cmd_download_deeplesion)

    c = s.add_parser("split-deeplesion", parents=[parent],
                     help="Pick DeepLesion patients for train/val/test with a fixed rule and write a YAML file")
    c.add_argument("--dest", default=None, help="Data directory (default: data.dicom_root)")
    c.add_argument("--out", default="configs/data/split_deeplesion.yaml")
    c.add_argument("--n-train", type=int, default=None, help="Default 45 patients; --key-slices: 2000 slices")
    c.add_argument("--n-val", type=int, default=None, help="Default 6 patients; --key-slices: 200 slices")
    c.add_argument("--n-test", type=int, default=None, help="Default 15 patients; --key-slices: 300 slices")
    c.add_argument("--key-slices", action="store_true",
                   help="Phase 2 (as QN-Mixer, CG-GLORE): draw KEY SLICES at random from the three official splits; "
                        "writes --manifest and the joint_patients.yaml, split_joint*.yaml files next to it")
    c.add_argument("--manifest", default=MANIFEST, help="List of drawn key slices (default: %(default)s)")
    c.add_argument("--info-csv", default=None, help="DL_info.csv (default: inside the DeepLesion data directory)")
    c.add_argument("--seed", type=int, default=0, help="Seed of the random draw (--key-slices)")
    c.add_argument("--min-slices", type=int, default=20, help="Drop patients with fewer usable slices than this")
    c.set_defaults(func=cmd_split_deeplesion)

    c = s.add_parser("split-2detect", parents=[parent],
                     help="2DeteCT: draw slices from LION's sample-level split, "
                          "write detect_slices.csv and split_detect.yaml")
    c.add_argument("--n-train", type=int, default=2000)
    c.add_argument("--n-val", type=int, default=200)
    c.add_argument("--n-test", type=int, default=None, help="Default: every LION test slice (470)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out-dir", default="configs/data")
    c.set_defaults(func=cmd_split_detect)

    c = s.add_parser("download-2detect", parents=[parent],
                     help="2DeteCT: download only mode 2 of the chosen slices from Zenodo "
                          "(partial zip reads), resumable")
    c.add_argument("--workers", type=int, default=32, help="Parallel connections")
    c.add_argument("--rows", type=int, default=None, help="Leading projections per slice (default data.detect.n_rows)")
    c.add_argument("--full", nargs="*", type=int, default=[],
                   help="Slices downloaded with all 3601 projections (to check the geometry on full-angle data)")
    c.set_defaults(func=cmd_download_detect)

    c = s.add_parser("stage", parents=[parent], help="Copy images, sinograms and matrices into the local cache "
                                                     "(cache.dir)")
    add_geometries(c, GEOMS_MAIN)
    c.set_defaults(func=cmd_stage)

    c = s.add_parser("fake", parents=[parent], help="Generate synthetic data for pipeline tests")
    c.add_argument("--format", default="dicom", choices=["dicom", "deeplesion"])
    c.add_argument("--out", default="data_raw/fake_mayo")
    c.add_argument("--n-slices", type=int, default=6)
    c.add_argument("--size", type=int, default=128)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_fake)
