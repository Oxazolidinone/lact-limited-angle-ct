"""2DeteCT (Kiss et al., Scientific Data 10:576, 2023): measured 2-D fan-beam CT from CWI's FleX-ray scanner.

Set-up follows the limited-angle task of the LION benchmark (Kiss, Biguri et al., arXiv:2412.08350; code
CambridgeCIA/LION):
  - inputs and targets both use mode 2 (normal dose); targets are the iterative reference reconstructions from the
    full 360° data;
  - limited angle = the FIRST projections of the sinogram (900 for 90°, 1,200 for 120°, 0.1° steps);
  - split by SAMPLE as in LION (109 samples in order of appearance: 80 % / 10 % / last 10 %; sample 99 unused);
  - raw processing as in LION: detector shifted by 1 pixel for slices < 2830, flat = mean of flat1 and flat2.
As for DeepLesion: 2,000 training and 200 validation slices are drawn at random (seed 0); all test slices are kept.

Download: each zip file on Zenodo has its central directory at the end; the directory is read with HTTP Range and
only the needed bytes are fetched (mode 2 of the chosen slices), never the whole zip. The sinogram (uint16,
3601 x 1912, one contiguous strip ordered by projection) is decompressed only up to the projections needed. Each slice
is saved as ``slice#####.npz`` (raw counts, before the log) in ``raw_dir``.
"""
from .raw import (
    FOV_MM,
    N_DET_RAW,
    REC_VOXEL_MM,
    TRANSFORMS,
    first_angle_deg,
    load_slice,
    measured_sinogram,
    reference_mu,
    slice_name,
)
from .split import SLICES_CSV, lion_split, read_slices, select_slices, write_split_configs
from .zenodo import download_slices, read_member, tiff_rows, zip_index

__all__ = ["FOV_MM", "N_DET_RAW", "REC_VOXEL_MM", "SLICES_CSV", "TRANSFORMS", "download_slices", "first_angle_deg",
           "lion_split", "load_slice", "measured_sinogram", "read_member", "read_slices", "reference_mu",
           "select_slices", "slice_name", "tiff_rows", "write_split_configs", "zip_index"]
