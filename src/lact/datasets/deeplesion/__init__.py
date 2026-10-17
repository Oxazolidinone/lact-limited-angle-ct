"""Second dataset: DeepLesion (NIH Clinical Center) from NIH's public Box folder.

- ``download_deeplesion``: download the selected ``Images_png_XX.zip`` files (~4 GiB and ~80 patients each) and
  ``DL_info.csv``, verify MD5 and unpack into ``<dest>/Images_png/<patient>_<study>_<series>/<slice>.png``.
- ``make_split``: pick patients with a fixed rule and write the split file (YAML) shared by every run.
- ``select_key_slices`` / ``write_key_slice_configs`` / ``download_key_slices``: the phase-2 use of DeepLesion,
  following QN-Mixer and CG-GLORE: draw 2,000 / 200 / 300 KEY SLICES at random from the three official splits. The
  drawn slices are spread over the whole dataset, so all 56 zip files are scanned, keeping a few dozen slices of each.

Source: https://nihcc.app.box.com/v/DeepLesion. Cite Yan et al., "DeepLesion: automated mining of large-scale lesion
annotations and universal lesion detection with deep learning", J. Medical Imaging 5(3), 2018.
"""
from .download import (
    DL_HOST,
    DL_SHARE,
    MD5_FILE,
    N_ZIPS,
    download_deeplesion,
    download_key_slices,
    expected_md5,
    zip_name,
)
from .reader import (
    HU_OFFSET,
    INFO_CSV,
    OFFICIAL_SPLIT,
    find_series,
    is_patient_id,
    load_info,
    official_split,
    patient_id,
    patient_slices,
    pick_evenly,
    read_patient,
    read_png_hu,
    series_slices,
    write_png_hu,
)
from .split import KEY_MANIFEST, make_split, read_key_manifest, select_key_slices, write_key_slice_configs

__all__ = ["DL_HOST", "DL_SHARE", "HU_OFFSET", "INFO_CSV", "KEY_MANIFEST", "MD5_FILE", "N_ZIPS", "OFFICIAL_SPLIT",
           "download_deeplesion", "download_key_slices", "expected_md5", "find_series", "is_patient_id", "load_info",
           "make_split", "official_split", "patient_id", "patient_slices", "pick_evenly", "read_key_manifest",
           "read_patient", "read_png_hu", "select_key_slices", "series_slices", "write_key_slice_configs",
           "write_png_hu", "zip_name"]
