"""Experiment grids: expand configs/experiment/*.yaml into a list of training runs and run them.

Each run executes in a child process ``python -m lact train run ...`` to isolate GPU memory and so the command list
can be moved to SLURM (``lact train grid ... --emit runs.sh``).
"""
from .jobs import Job, emit_script, expand, filter_jobs, load_best_lr, renamed
from .planning import order_jobs, remaining_work, sweep_sec_per_step
from .runner import gpu_shares, run_jobs

__all__ = ["Job", "emit_script", "expand", "filter_jobs", "gpu_shares", "load_best_lr", "order_jobs", "remaining_work",
           "renamed", "run_jobs", "sweep_sec_per_step"]
