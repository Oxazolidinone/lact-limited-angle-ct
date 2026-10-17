"""Infrastructure layer (adapters): file I/O and result stores, projection matrices (ASTRA), h5py and the runtime.

``astra`` is never imported at module level, so the rest runs without it installed. Raw-data readers and downloaders
of each dataset live in ``lact.datasets``.
"""
from .data.dataset import LACTDataset, collate
from .io.files import ensure_dir, load_json, save_json, sha256_file, write_text
from .io.hdf5_store import ImageStore, ImageWriter, SinoStore, SinoWriter
from .operators.operator_store import build_and_save, load_matrix, load_projector, operator_path
from .stores.manifest import update_manifest, verify_manifest
from .stores.results_store import ResultsStore, make_tag
from .stores.run_store import CSVWriter, RunLogger, RunStore, list_runs
from .system.runtime import collect_env, get_logger, git_hash, seed_everything, select_device

__all__ = [
    "CSVWriter",
    "ImageStore",
    "ImageWriter",
    "LACTDataset",
    "ResultsStore",
    "RunLogger",
    "RunStore",
    "SinoStore",
    "SinoWriter",
    "build_and_save",
    "collate",
    "collect_env",
    "ensure_dir",
    "get_logger",
    "git_hash",
    "list_runs",
    "load_json",
    "load_matrix",
    "load_projector",
    "make_tag",
    "operator_path",
    "save_json",
    "seed_everything",
    "select_device",
    "sha256_file",
    "update_manifest",
    "verify_manifest",
    "write_text",
]
