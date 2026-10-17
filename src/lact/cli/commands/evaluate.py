"""lact eval {freeze, runs, baselines, cross, tune-tv, cost, recheck, cg-check}: evaluation and cost measurement
(steps 20-23)."""
from __future__ import annotations

import argparse
import logging

from ...services.evaluation import baselines, recheck
from ...services.evaluation import evaluate as evaluation
from ..common import GEOMS_MAIN, add_geometries, device, geometries, load_cfg

log = logging.getLogger("lact")


def _eval_kw(a) -> dict:
    return {"save_pred": not a.no_pred, "batch_size": a.batch, "limit": a.limit, "force_test": a.force_test,
            "overwrite": a.overwrite}


def cmd_freeze(a) -> int:
    log.info("Created lock %s", evaluation.freeze_configs(load_cfg(a), a.note))
    return 0


def cmd_runs(a) -> int:
    if not (a.exp or a.runs):
        raise SystemExit("--exp or --runs is required")
    cfg = load_cfg(a)
    evaluation.evaluate_experiments(cfg, a.exp, a.runs, a.split, a.which, device(a), **_eval_kw(a),
                                    data_cfg=cfg if a.on_config_data else None)
    return 0


def cmd_baselines(a) -> int:
    evaluation.evaluate_baselines(load_cfg(a), a.names, geometries(a.geometries), a.split, device(a), **_eval_kw(a))
    return 0


def cmd_tune_tv(a) -> int:
    cfg, dev = load_cfg(a), device(a)
    for g in geometries(a.geometries):
        baselines.tune_tv(cfg, g, dev)
    return 0


def cmd_recheck(a) -> int:
    df = recheck.recheck(load_cfg(a), a.exp, a.per_geometry, a.tol, device(a), a.on_config_data)
    if df.empty:
        print("No DONE run to check.")
        return 0
    print(df.round(4).to_string(index=False))
    if not df.ok.all():
        log.error("Some runs do NOT match: data/matrices on this machine differ from those used to train them.")
        return 1
    print(f"Match: {len(df)} runs reproduce their recorded validation PSNR (tolerance {a.tol} dB).")
    return 0


def _eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("evaluation")
    g.add_argument("--split", default="val", choices=["val", "test"])
    g.add_argument("--batch", type=int, default=4)
    g.add_argument("--limit", type=int, default=None, help="Evaluate only the first N slices (debugging)")
    g.add_argument("--no-pred", action="store_true", help="Do not save predictions")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--force-test", action="store_true", help="Open the test split even if the configuration is not "
                                                             "frozen")
    return p


def register(sub, parent) -> None:
    p = sub.add_parser("eval", help="Evaluate models and baselines, lock the test split")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")
    ev = _eval_parser()

    c = s.add_parser("freeze", parents=[parent], help="Freeze the configuration (creates configs_frozen.lock) before "
                                                      "opening the test split")
    c.add_argument("--note", default="")
    c.set_defaults(func=cmd_freeze)

    c = s.add_parser("runs", parents=[parent, ev], help="Evaluate finished runs of experiments / directories")
    c.add_argument("--exp", nargs="*", default=[])
    c.add_argument("--runs", nargs="*", default=[], help="Specific run directories")
    c.add_argument("--which", default="best", choices=["best", "last"])
    c.add_argument("--on-config-data", action="store_true",
                   help="Cross-dataset: use the images, sinograms, split and results store of the -c configuration "
                        "instead of each run's (e.g. a Mayo-trained run on the joint Mayo + DeepLesion store)")
    c.set_defaults(func=cmd_runs)

    c = s.add_parser("baselines", parents=[parent, ev], help="Evaluate FBP / TV")
    c.add_argument("--names", nargs="+", default=list(baselines.BASELINES), choices=list(baselines.BASELINES))
    add_geometries(c, GEOMS_MAIN)
    c.set_defaults(func=cmd_baselines)

    c = s.add_parser("tune-tv", parents=[parent], help="Tune TV-PDHG (lambda, iterations) on the validation split")
    add_geometries(c, GEOMS_MAIN)
    c.set_defaults(func=cmd_tune_tv)

    c = s.add_parser("recheck", parents=[parent],
                     help="Check whether trained runs match the data/matrices on this machine (recorded validation "
                          "PSNR)")
    c.add_argument("--exp", nargs="+", default=list(recheck.MAIN_EXPS))
    c.add_argument("--per-geometry", type=int, default=2, help="Runs (different architectures) per geometry")
    c.add_argument("--tol", type=float, default=0.02, help="Allowed PSNR deviation (dB)")
    c.add_argument("--on-config-data", action="store_true",
                   help="Check on the data and validation split of the -c configuration instead of each run's")
    c.set_defaults(func=cmd_recheck)
