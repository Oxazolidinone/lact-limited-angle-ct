"""lact report {select-lr, tables, figures, analysis, phase2}: tables, figures and in-depth analyses (steps 30-32)."""
from __future__ import annotations

import argparse

from ...services.reporting import ReportSpec, make_figures, make_phase2_tables, make_tables, run_analysis, select_lr
from ...services.reporting.seeds import POLICIES
from ..common import device, load_cfg


def _spec(a) -> ReportSpec:
    return ReportSpec(split=a.split, main_exp=a.main_exp, matched_exp=a.matched_exp, a150_exp=a.a150_exp,
                      sweep_exp=a.sweep_exp, geoms=a.geoms, n_boot=getattr(a, "n_boot", 10000),
                      seed_policy=a.seeds)


def cmd_select_lr(a) -> int:
    select_lr(load_cfg(a), a.exp)
    return 0


def cmd_tables(a) -> int:
    make_tables(load_cfg(a), _spec(a))
    return 0


def cmd_figures(a) -> int:
    make_figures(load_cfg(a), _spec(a), seed=a.seed)
    return 0


def cmd_analysis(a) -> int:
    run_analysis(load_cfg(a), _spec(a), seed=a.seed, n_erf=a.n_erf, n_stage=a.n_stage, device=device(a))
    return 0


def cmd_phase2(a) -> int:
    for f in make_phase2_tables(load_cfg(a), a.split, a.geoms, a.exp or None):
        print(f)
    return 0


def _report_parser() -> argparse.ArgumentParser:
    d = ReportSpec()
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("reporting")
    g.add_argument("--split", default=d.split, choices=["val", "test"])
    g.add_argument("--main-exp", default=d.main_exp)
    g.add_argument("--matched-exp", default=d.matched_exp)
    g.add_argument("--a150-exp", default=d.a150_exp)
    g.add_argument("--sweep-exp", default=d.sweep_exp)
    g.add_argument("--geoms", nargs="+", default=d.geoms)
    g.add_argument("--seeds", default=d.seed_policy, choices=list(POLICIES),
                   help="Configurations with several seeds: best_val = seed with the best validation PSNR (default) | "
                        "first = smallest seed | all = mean ± standard deviation over all seeds")
    return p


def register(sub, parent) -> None:
    p = sub.add_parser("report", help="Tables B1-B10 and C1-C3, figures F1-F10, in-depth analyses")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")
    rp = _report_parser()

    c = s.add_parser("select-lr", parents=[parent], help="E2: choose the best lr -> <runs>/<exp>/best_lr.json")
    c.add_argument("--exp", default="e2_sweep")
    c.set_defaults(func=cmd_select_lr)

    c = s.add_parser("tables", parents=[parent, rp], help="Tables B1-B9 (CSV + Markdown + LaTeX)")
    c.add_argument("--n-boot", type=int, default=10000)
    c.set_defaults(func=cmd_tables)

    c = s.add_parser("figures", parents=[parent, rp], help="Figures F1-F6, F10 (PDF + PNG)")
    c.add_argument("--seed", type=int, default=0, help="Seed used for the qualitative figures")
    c.set_defaults(func=cmd_figures)

    c = s.add_parser("analysis", parents=[parent, rp], help="F7 frequency domain, F8 per stage, F9 ERF")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--n-erf", type=int, default=50)
    c.add_argument("--n-stage", type=int, default=50)
    c.set_defaults(func=cmd_analysis)

    c = s.add_parser("phase2", parents=[parent],
                     help="Phase 2: tables C1 (per test dataset), C2 (CG/ADMM versus first order), C3 (runs)")
    c.add_argument("--split", default="test", choices=["val", "test"])
    c.add_argument("--geoms", nargs="+", default=["fan_a90", "fan_a120"])
    c.add_argument("--exp", nargs="*", default=[], help="Only these experiments (default: every available result)")
    c.set_defaults(func=cmd_phase2)
