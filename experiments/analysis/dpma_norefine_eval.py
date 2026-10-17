"""Re-evaluate DPMA WITHOUT its 300 inference-time refinement iterations (refine_iters = 1: only the network output,
the part used for checkpoint selection on the validation split). Results are stored separately with run_id +
"_norefine" in each run's eval/, next to the original refined results (nothing is overwritten). Same data, metrics
and numerical mode as ``lact eval runs``.

    python experiments/analysis/dpma_norefine_eval.py outputs_detect/runs/dpma_dt/* outputs_joint/runs/dpma_mayo/* ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lact.config import load_config_file
from lact.infra.stores.results_store import ResultsStore
from lact.infra.stores.run_store import RunStore
from lact.infra.system.runtime import apply_precision, select_device
from lact.services.evaluation.evaluate import _run_info, check_test_access, evaluate_method
from lact.services.factories import dataset_for, load_trained, projector_for

SUFFIX = "_norefine"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Stage-2 DPMA run directories")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("lact")
    dev = select_device("auto")
    for d in a.runs:
        run = RunStore(Path(d))
        if run.status != "done":
            log.warning("Skipping %s: status %s", run.run_id, run.status)
            continue
        cfg = load_config_file(run.config_file)
        g = cfg.require_geometry()
        info = _run_info(cfg, run, g, a.split, "best")
        info["run_id"] += SUFFIX
        results = ResultsStore(cfg.paths.eval)
        if results.exists(info):
            log.info("Already have %s", info["run_id"])
            continue
        check_test_access(cfg, a.split)
        apply_precision(cfg.precision.tf32)
        model, _, state = load_trained(run, "best", dev)
        if not hasattr(model, "refine_iters"):
            raise SystemExit(f"{run.run_id}: not a DPMA model")
        model.refine_iters = 1
        P = projector_for(cfg, g, device=dev)
        model.set_projector(P).eval()
        info["epoch"] = int(state.get("epoch", -1))
        ds = dataset_for(cfg, a.split, g.name, preload=False, augment=False)
        df, _ = evaluate_method(model, ds, P, dev, info, a.batch, keep_preds=False)
        results.save(info, df, None)
        log.info("%s [%s, %s]: PSNR %.3f | SSIM %.4f (%d slices)", info["run_id"], a.split, g.name, df.psnr.mean(),
                 df.ssim.mean(), len(df))


if __name__ == "__main__":
    main()
