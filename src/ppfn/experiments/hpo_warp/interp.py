"""Differentiable multilinear interpolation over a regular (possibly
non-uniformly-spaced) grid, in pure torch.

Used to treat B's response surface as a continuously-interpolated "image"
that the fitted warp's output is read out against (see `fit.py`'s module
docstring: this makes the whole y-correction + x-warp pipeline one
`loss.backward()`-able graph, with gradients flowing through the
interpolation itself rather than needing a finite-difference surrogate).

`d` is expected to be small (2-6, matching HPOBench's ML config spaces), so
the `2**d`-corner sum below is cheap; this does not scale to large `d`.
"""

from __future__ import annotations

import numpy as np
import torch


def multilinear_interp(axes: list[torch.Tensor], values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """axes[j]: [n_j] sorted 1D grid coordinates for axis j.
    values: tensor of shape (n_0, ..., n_{d-1}).
    x: [N, d] query points, clamped by the caller to each axis's range.
    Returns [N]. Differentiable w.r.t. `x` (piecewise-linear, non-smooth only
    at grid cell boundaries -- the standard, expected caveat)."""
    d = len(axes)
    n = x.shape[0]
    idx_lo, frac = [], []
    for j in range(d):
        ax = axes[j]
        idx = torch.searchsorted(ax.contiguous(), x[:, j].contiguous(), right=False)
        idx = idx.clamp(1, ax.numel() - 1)
        x0, x1 = ax[idx - 1], ax[idx]
        f = (x[:, j] - x0) / (x1 - x0).clamp_min(1e-12)
        idx_lo.append(idx - 1)
        frac.append(f)

    out = torch.zeros(n, dtype=x.dtype, device=x.device)
    for corner in range(2**d):
        weight = torch.ones(n, dtype=x.dtype, device=x.device)
        index = []
        for j in range(d):
            bit = (corner >> j) & 1
            weight = weight * (frac[j] if bit else (1.0 - frac[j]))
            index.append(idx_lo[j] + bit)
        out = out + weight * values[tuple(index)]
    return out


def clamp_to_axis(x: torch.Tensor, axes: list[torch.Tensor]) -> torch.Tensor:
    """Clamp each column of x to its axis's [min, max] range. Built via
    stack rather than in-place column assignment -- the latter trips
    autograd's in-place-modification check once `x` requires grad through a
    warp's parameters."""
    cols = [x[:, j].clamp(ax.min(), ax.max()) for j, ax in enumerate(axes)]
    return torch.stack(cols, dim=-1)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    """Clamp to the declared [0,1]^d box every config is normalized into
    (`ARCHITECTURE.md` invariant #8: declared bounds, not empirical data
    range) -- matches `ARCHITECTURE.md` §2.3(b)'s `clamp01` convention for
    transport estimates. Used instead of axis-specific `clamp_to_axis` for
    both grid and scattered surface readers, since a scattered random design
    rarely samples exactly to the declared edges and clamping to its
    empirical min/max would incorrectly reject genuinely in-bounds queries."""
    return x.clamp(0.0, 1.0)


def kernel_interp(x_ref: torch.Tensor, y_ref: torch.Tensor, query: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Nadaraya-Watson (Gaussian-kernel-weighted) interpolation over
    *scattered* reference points -- the counterpart to `multilinear_interp`
    for benchmarks whose shared design is a fixed random sample rather than
    a Cartesian grid (LCBench/PD1/TaskSet tabular: same config set across
    tasks, per `scattered_data.py`, but not laid out on a grid multilinear
    interpolation could exploit).

    x_ref: [M, d] reference configs (task B's). y_ref: [M]. query: [N, d].
    Returns [N]. Differentiable w.r.t. `query`. O(N*M) -- fine at the scale
    here (M up to ~2000-50000, N a training minibatch), not meant to scale
    beyond it."""
    diff = query.unsqueeze(1) - x_ref.unsqueeze(0)  # [N, M, d]
    sqdist = (diff * diff).sum(-1)  # [N, M]
    weight = torch.exp(-sqdist / (2.0 * bandwidth * bandwidth))
    weight_sum = weight.sum(-1).clamp_min(1e-12)
    return (weight * y_ref.unsqueeze(0)).sum(-1) / weight_sum


def bandwidth_heuristic(x_ref: np.ndarray, k: int = 15) -> float:
    """Median distance to the `k`-th nearest neighbor, over all reference
    points -- a k-NN-adaptive bandwidth for `kernel_interp`.

    A grid-spacing-style formula (`n^(-1/d)`, the first thing tried here)
    fails badly at LCBench/TaskSet's scale: with n=2000 points in d=7,
    `n^(1/7) ~= 3`, i.e. "3 points per axis if regular" -- wildly too coarse,
    giving a bandwidth wide enough to average over most of the cube and
    produce a nearly flat readout (caught empirically: the fitted warp
    ladder's held-out R^2 was *non-monotonic* in capacity, symptomatic of a
    loss surface too flat to have a useful gradient at all -- not a fitting
    bug, a bandwidth bug). k-NN distance instead measures local density
    directly and doesn't assume points fill the cube uniformly."""
    from scipy.spatial import cKDTree

    tree = cKDTree(x_ref)
    dist, _ = tree.query(x_ref, k=min(k + 1, x_ref.shape[0]))  # k+1: point itself is its own nearest neighbor
    return float(np.median(dist[:, -1]))
