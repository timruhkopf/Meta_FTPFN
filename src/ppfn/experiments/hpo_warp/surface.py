"""Differentiable readout of a task's response surface -- the "B-side image"
`fit.py`/`fidelity_warp.py` warp A's coordinates into and read off.

Dispatches on `TaskGrid.is_gridded` (see `task_grid.py`) so the rest of the
fitting code never needs to know whether the underlying benchmark laid its
shared design out on a Cartesian grid (HPOBench `TabularBenchmark`,
`multilinear_interp`) or as a shared-but-scattered random sample (LCBench/
PD1/TaskSet tabular, `kernel_interp`) -- both give back a plain callable:
`build_config_reader` -> `Callable[[Tensor], Tensor]` on `[N, d]` config
queries; `build_full_reader` -> `Callable[[Tensor, float], Tensor]` taking
config queries and a separate log-fidelity scalar (see its own docstring for
why fidelity isn't just concatenated into the query)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ppfn.experiments.hpo_warp.interp import bandwidth_heuristic, kernel_interp, multilinear_interp
from ppfn.experiments.hpo_warp.task_grid import TaskGrid

Reader = Callable[[torch.Tensor], torch.Tensor]


def build_config_reader(grid: TaskGrid, iter_idx: int) -> Reader:
    """Reader over the config space alone, at one fixed fidelity slice --
    what `fit.py`'s config-space warp ladder is fit against."""
    d = grid.x.shape[1]
    y_at_iter = grid.y_mean_by_iter[iter_idx]

    if grid.is_gridded:
        axes = [torch.as_tensor(np.unique(grid.x[:, j]), dtype=torch.float32) for j in range(d)]
        values = torch.as_tensor(y_at_iter.reshape(grid.grid_shape), dtype=torch.float32)
        return lambda q: multilinear_interp(axes, values, q)

    x_ref = torch.as_tensor(grid.x, dtype=torch.float32)
    y_ref = torch.as_tensor(y_at_iter, dtype=torch.float32)
    bw = bandwidth_heuristic(grid.x)
    return lambda q: kernel_interp(x_ref, y_ref, q, bandwidth=bw)


FullReader = Callable[[torch.Tensor, float], torch.Tensor]


def build_full_reader(grid: TaskGrid) -> FullReader:
    """Reader over config space **and** log-fidelity -- what
    `fidelity_warp.py` reads B's surface through, so it can query at an
    arbitrary reparametrized fidelity, not just the fixed grid of fidelities
    B was actually evaluated at.

    Contract, both branches: `reader(config_query: [N,d] in [0,1]^d,
    log_iter: float) -> [N]` -- fidelity is a separate scalar argument, not
    concatenated into the query, specifically so the scattered branch can
    avoid ever building a single `n_configs*n_iters`-point kernel problem
    (see below for why that matters).

    First version concatenated config+log-fidelity into one `(d+1)`-D
    `kernel_interp` over all `n_configs*n_iters` cells jointly -- correct,
    but for LCBench (2000 configs x 51 epochs = 102000 reference points)
    `fit_fidelity_warp`'s 25-tau x 51-iter sweep turned into ~1275 kernel
    evaluations each costing O(batch x 102000), which never finished inside
    a 150s budget. Since fidelity_warp only ever queries *one* fixed
    log-fidelity across an entire batch at a time, that joint problem
    decomposes exactly: interpolate each reference config's own curve to
    that one fidelity first (a single cheap 1D `np.interp`-style step,
    vectorized over all configs at once), THEN kernel-interpolate over
    configs only -- the same O(batch x n_configs) cost as the plain
    config-only reader."""
    d = grid.x.shape[1]
    log_iters = np.log(grid.iters)

    if grid.is_gridded:
        axes = [torch.as_tensor(np.unique(grid.x[:, j]), dtype=torch.float32) for j in range(d)]
        axes.append(torch.as_tensor(log_iters, dtype=torch.float32))
        values = torch.as_tensor(
            grid.y_mean_by_iter.reshape(len(grid.iters), *grid.grid_shape), dtype=torch.float32
        ).permute(*range(1, d + 1), 0)

        def grid_reader(config_query: torch.Tensor, log_iter: float) -> torch.Tensor:
            fidelity_col = torch.full((config_query.shape[0], 1), float(log_iter), dtype=torch.float32)
            return multilinear_interp(axes, values, torch.cat([config_query, fidelity_col], dim=-1))

        return grid_reader

    x_ref = torch.as_tensor(grid.x, dtype=torch.float32)
    bw = bandwidth_heuristic(grid.x)  # config-space-only bandwidth, same as build_config_reader
    y_mean_by_iter = grid.y_mean_by_iter  # [n_iters, n_configs], log_iters sorted ascending to match

    def scattered_reader(config_query: torch.Tensor, log_iter: float) -> torch.Tensor:
        # Linear interp along the fidelity axis, vectorized over every
        # reference config at once (no Python-level loop over configs).
        idx = int(np.clip(np.searchsorted(log_iters, log_iter), 1, len(log_iters) - 1))
        lo_i, hi_i = log_iters[idx - 1], log_iters[idx]
        frac = 0.0 if hi_i == lo_i else (log_iter - lo_i) / (hi_i - lo_i)
        y_at_iter = y_mean_by_iter[idx - 1] * (1 - frac) + y_mean_by_iter[idx] * frac  # [n_configs]
        y_ref = torch.as_tensor(y_at_iter, dtype=torch.float32)
        return kernel_interp(x_ref, y_ref, config_query, bandwidth=bw)

    return scattered_reader
