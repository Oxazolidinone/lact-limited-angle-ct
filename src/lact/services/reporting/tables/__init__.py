"""Result tables B1-B10 (CSV + Markdown + LaTeX) from the evaluation store.

Each table is a ``table_*`` function returning a DataFrame (or None when data are missing) so it can be tested on its
own; ``make_tables`` only orchestrates and writes files. Column headers and cell labels are Vietnamese report content.
"""
from .cost import table_cost, table_regnet_cost
from .format import METRIC_FMT, ReportSpec, arch_rank, fmt_ms, sort_geoms, wide_table
from .make import load_costs, load_selected, make_tables, select_lr
from .results import (
    main_slices,
    table_a150,
    table_cross_geometry,
    table_e0,
    table_hyperparams,
    table_main,
    table_matched,
    table_per_patient,
    table_seed_spread,
    table_seeds,
    table_trend,
)
from .significance import registered_tests, supplementary_tests, table_stats

__all__ = ["METRIC_FMT", "ReportSpec", "arch_rank", "fmt_ms", "load_costs", "load_selected", "main_slices",
    "make_tables", "registered_tests", "select_lr", "sort_geoms", "supplementary_tests", "table_a150", "table_cost",
    "table_cross_geometry", "table_e0", "table_hyperparams", "table_main", "table_matched", "table_per_patient",
    "table_regnet_cost", "table_seed_spread", "table_seeds", "table_stats", "table_trend", "wide_table"]
