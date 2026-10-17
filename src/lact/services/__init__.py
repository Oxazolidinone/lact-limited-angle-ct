"""Application layer (use cases): orchestrates the domain (core) and the adapters (datasets, infra) from a typed
configuration.

Each sub-package matches one CLI command group:
    data/          prep, detect, simulation, fake, staging   -> lact data ...
    operators/     build (A)                                 -> lact ops build
    training/      trainer (one run), optim, monitor         -> lact train run / status
    scheduling/    experiment grids: jobs, runner, planning  -> lact train grid
    evaluation/    evaluate, baselines, recheck              -> lact eval ...
``factories`` is the single composition root that builds projectors / FBP / models / datasets from a ``Config``.
"""
