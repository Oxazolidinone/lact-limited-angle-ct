"""lact ops {build, check, density}: projection operators (step 02) and the E0 (03), E1 (05) checks."""
from __future__ import annotations

import json

from ...config.defaults import load_geometry
from ...services import operators
from ..common import GEOMS_CHECK, GEOMS_HALF, GEOMS_MAIN, add_geometries, device, geometries, load_cfg


def cmd_build(a) -> int:
    operators.build_operators(load_cfg(a), geometries(a.geometries), a.sizes, a.projector, a.overwrite)
    return 0


def cmd_check(a) -> int:
    res = operators.check_operators(load_cfg(a), geometries(a.geometries), figure=not a.no_figure)
    print(json.dumps(res, ensure_ascii=False))
    return 0 if all(res.values()) else 1


def cmd_density(a) -> int:
    pairs = [(load_geometry(f"fan_a{s}"), load_geometry(f"fan_a{s}_half")) for s in a.spans]
    piv = operators.angle_density(load_cfg(a), pairs, a.n_val, a.tv_lambda, a.tv_iters, device(a))
    print(piv.to_string(index=False))
    return 0 if bool(piv["pass"].all()) else 1


def register(sub, parent) -> None:
    p = sub.add_parser("ops", help="Projection operators: build matrices, E0/E1 checks")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    c = s.add_parser("build", parents=[parent], help="Build the sparse projection matrix A (ASTRA) and ||A||_2")
    add_geometries(c, GEOMS_MAIN + GEOMS_CHECK + GEOMS_HALF)
    c.add_argument("--sizes", nargs="+", type=int, default=None, help="Default: img_size_rec and img_size_sim")
    c.add_argument("--projector", default="line_fanflat", choices=["line_fanflat", "strip_fanflat"])
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_build)

    c = s.add_parser("check", parents=[parent], help="E0: adjoint, gradient, phantom FBP, sinogram edges")
    add_geometries(c, GEOMS_CHECK + GEOMS_MAIN)
    c.add_argument("--no-figure", action="store_true")
    c.set_defaults(func=cmd_check)

    c = s.add_parser("density", parents=[parent], help="E1: 0.25° versus 0.125° angular step (FBP and TV)")
    c.add_argument("--spans", nargs="+", type=int, default=[90, 120, 150])
    c.add_argument("--n-val", type=int, default=20)
    c.add_argument("--tv-lambda", type=float, default=1e-4)
    c.add_argument("--tv-iters", type=int, default=300)
    c.set_defaults(func=cmd_density)
