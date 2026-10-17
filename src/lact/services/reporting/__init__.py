"""Reporting: aggregation, tables B1-B9, figures F1-F10, in-depth analyses."""
from .aggregate import fmt_mean_std, per_method, per_method_patient, per_patient, per_run, select_best_lr
from .analysis import run_analysis
from .figures import make_figures
from .phase2 import make_phase2_tables
from .tables import ReportSpec, make_tables, select_lr, wide_table

__all__ = [
           "ReportSpec",
           "fmt_mean_std",
           "make_figures",
           "make_phase2_tables",
           "make_tables",
           "per_method",
           "per_method_patient",
           "per_patient",
           "per_run",
           "run_analysis",
           "select_best_lr",
           "select_lr",
           "wide_table",
]
