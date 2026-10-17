"""Single command-line entry point: ``lact <group> <command> [options]`` (or ``python -m lact ...``).

    lact data     download-mayo | download-deeplesion | split-deeplesion | download-2detect | split-2detect |
                  inspect | preprocess | simulate | stage | fake
    lact ops      build
    lact train    run | grid | status
    lact eval     freeze | runs | baselines | tune-tv | recheck
Every command accepts -c/--config, -s/--set, --no-defaults, --device. See ``lact <group> <command> -h``.
"""
from __future__ import annotations

import argparse
import sys

from ..config import ConfigError
from ..infra.system.runtime import get_logger
from .commands import data, evaluate, ops, train
from .common import config_parser

GROUPS = (data, ops, train, evaluate)
USER_ERRORS = (ConfigError, FileNotFoundError, FileExistsError, PermissionError)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lact", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="group", required=True, metavar="<group>")
    parent = config_parser()
    for g in GROUPS:
        g.register(sub, parent)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("lact")
    try:
        return int(args.func(args) or 0)
    except USER_ERRORS as e:
        log.error("%s: %s", type(e).__name__, e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
