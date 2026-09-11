"""Fidelity-axis warp: does a per-task-pair reparametrization of the
fidelity axis (here, HPOBench's `iter` -- SGD iterations, playing the role
epoch plays in LCBench) improve learning-curve alignment beyond what the
already-fitted config-space warp `T` and y-correction `h` explain?

Kept as its own, separate analysis from `fit.py`'s config-space warp, per
`docs/experiments/hpo-warp-complexity.md`: conflating the two axes would
leave it unclear which one is doing the work.

Only 5 fidelity values are available per task (HPOBench's successive-halving
grid, e.g. `[12, 37, 111, 333, 1000]`), which cannot support a flexible
per-axis spline the way the 25x25 config grid can -- so this fits a single
scalar "log-time dilation" `tau` around the mean log-fidelity:

    g_tau(log_iter) = mu + tau * (log_iter - mu),   mu = mean(log_iters)

`tau = 1` is the identity (no fidelity warp needed); `tau` far from 1 means
task A's fidelity axis needs to be stretched/compressed to align with B's
(e.g. "A needs relatively more iterations to reach the same relative
progress"). Severity is reported as `|log(tau)|`; `tau` is found by a coarse
grid search (cheap, robust, appropriate for a single scalar) rather than
gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ppfn.experiments.hpo_warp.fit import _apply_h
from ppfn.experiments.hpo_warp.interp import clamp01
from ppfn.experiments.hpo_warp.surface import build_full_reader
from ppfn.experiments.hpo_warp.task_grid import TaskGrid

TAU_GRID = np.geomspace(0.25, 4.0, 25)


@dataclass
class FidelityFitResult:
    model: str
    task_a: str
    task_b: str
    tau: float
    log_tau_severity: float
    r2_at_tau1: float
    r2_at_best_tau: float


def fit_fidelity_warp(
    grid_a: TaskGrid,
    grid_b: TaskGrid,
    h_lambda: float,
    h_scale: float,
    h_shift: float,
    elbow_warp: torch.nn.Module,
    test_idx: np.ndarray,
) -> FidelityFitResult:
    x_a = torch.as_tensor(grid_a.x[test_idx], dtype=torch.float32)
    with torch.no_grad():
        t_x = clamp01(elbow_warp(x_a))

    b_reader = build_full_reader(grid_b)
    log_iters_b = np.log(grid_b.iters)
    lo_b, hi_b = float(log_iters_b.min()), float(log_iters_b.max())

    log_iters_a = np.log(grid_a.iters)
    mu = float(log_iters_a.mean())
    y_a_test = grid_a.y_mean_by_iter[:, test_idx]  # [n_iters, n_test]

    def held_out_r2(tau: float) -> float:
        # Clamped to B's own observed fidelity range -- don't extrapolate
        # the dilation past what B was actually evaluated at.
        g_log_iter = np.clip(mu + tau * (log_iters_a - mu), lo_b, hi_b)
        preds = []
        for k in range(len(grid_a.iters)):
            with torch.no_grad():
                y_b_at = b_reader(t_x, float(g_log_iter[k]))
                pred = _apply_h(y_b_at, h_lambda, h_scale, h_shift).numpy()
            preds.append(pred)
        preds = np.stack(preds, axis=0)  # [n_iters, n_test]
        ss_res = np.sum((y_a_test - preds) ** 2)
        ss_tot = np.sum((y_a_test - y_a_test.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_at_1 = held_out_r2(1.0)
    r2_by_tau = {tau: held_out_r2(tau) for tau in TAU_GRID}
    best_tau = max(r2_by_tau, key=r2_by_tau.get)

    return FidelityFitResult(
        model=grid_a.model,
        task_a=grid_a.task_id,
        task_b=grid_b.task_id,
        tau=float(best_tau),
        log_tau_severity=float(abs(np.log(best_tau))),
        r2_at_tau1=float(r2_at_1),
        r2_at_best_tau=float(r2_by_tau[best_tau]),
    )
