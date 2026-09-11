"""The common data shape every benchmark loader produces.

Family-agnostic on purpose: `hpobench_data.py` (HPOBench `TabularBenchmark`
-- an exact Cartesian grid) and `scattered_data.py` (LCBench/PD1/TaskSet
tabular -- a shared but non-gridded random design) both return this, and
everything downstream (`fit.py`, `fidelity_warp.py`, `surface.py`) only
branches on `is_gridded`, never on which loader produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskGrid:
    """One (family, task_id)'s response surface, on a design shared exactly
    across every task in the family (so correspondence between two tasks is
    free -- no interpolation/matching needed to compare "the same x").

    `x` is normalized to [0, 1]^d via the *declared* (log-space where
    applicable) search-space bounds -- never the empirical data range
    (`ARCHITECTURE.md` invariant #8). `y_mean_by_iter`/`y_var_by_iter` are
    over `n_seeds` replicate seeds at each cell where available; `n_seeds=1`
    and `y_var_by_iter` all-zero signals "no replicate-seed noise floor
    available for this family" (LCBench/PD1/TaskSet all lack per-config
    replicate seeds; `residual_to_noise_floor` is reported as NaN for these
    rather than a fabricated number).

    `is_gridded`: True if `x`'s design is an exact Cartesian product (one
    axis-aligned grid, HPOBench's `TabularBenchmark`) -- `grid_shape` is then
    the per-axis point counts and `surface.py` uses `interp.
    multilinear_interp`. False if it's a shared-but-scattered random design
    (LCBench/PD1/TaskSet tabular) -- `grid_shape` is then just `(n_configs,)`
    and `surface.py` uses `interp.kernel_interp` instead.
    """

    model: str
    task_id: str
    param_names: list[str]
    x: np.ndarray  # [n_configs, d], normalized to [0,1]^d
    ids: np.ndarray  # [n_configs], the shared design's own config identifiers --
    # HPOBench's exact grid never drops a cell, so `ids = arange(n)` there;
    # TaskSet drops configs whose curve diverged/is incomplete per task, so
    # two tasks under nominally "the same" design can retain *different*
    # subsets -- `ids` is what lets `fit.py` find the actual overlap before
    # doing any row-matched comparison (see the labbook entry this was
    # caught in: `spearmanr` crashing on mismatched array lengths).
    grid_shape: tuple[int, ...]  # gridded: e.g. (25, 25); scattered: (n_configs,)
    iters: np.ndarray  # [n_fidelities]
    y_mean_by_iter: np.ndarray  # [n_fidelities, n_configs], MEAN over n_seeds replicate seeds
    y_var_by_iter: np.ndarray  # [n_fidelities, n_configs], variance of a SINGLE seed at that cell
    n_seeds: int  # replicate seeds per cell -- Var(y_mean_by_iter) = y_var_by_iter / n_seeds
    is_gridded: bool = True
    minimize: bool = True  # the target is a loss: lower is better
