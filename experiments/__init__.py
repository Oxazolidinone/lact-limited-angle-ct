"""Thesis-specific analyses, figures and paper numbers built on top of the ``lact`` library.

    analysis/   one-off studies (visible/invisible split, two-stage reconstruction, DPMA without refinement,
                2DeteCT geometry check)
    figures/    collect plotting data from finished runs and draw the thesis figures
    paper/      tables and numbers of the paper (LaTeX)

Run from the repository root, by path (``python experiments/paper/make_numbers.py ...``) or as modules
(``python -m experiments.paper.make_numbers ...``). ``lact`` never imports this package (import-linter contract).
"""
