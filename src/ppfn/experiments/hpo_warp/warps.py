"""Fittable warp families, ordered as a capacity ladder.

Same family choices as the synthetic prior (`docs/ROADMAP.md` §4.1,
`src/ppfn/prior/registration/warp.py`), but here they are *fit* by gradient
descent against real paired data instead of sampled from a prior:

    identity -> global affine -> per-axis monotone warp (P1) -> velocity field (P2)

`MonotoneAxisWarp` is the piecewise-linear member of the per-axis-monotone
family (nests the identity, analytically invertible, `k` bins is the
complexity knob) -- standing in for the rational-quadratic spline formally
specified in `ROADMAP.md` §4.1 for implementation simplicity. Swap in an RQ
spline later if the piecewise-linear family turns out to be the bottleneck;
it isn't expected to be, since both nest the identity and both are
determined by the same handful of knot values.

`VelocityField` is a direct torch port of `warp.py`'s numpy family
(stationary velocity field, RK4-integrated flow), with `centers`/`weights`
as fit parameters instead of prior draws, zero-initialized so training starts
at the identity -- matching `ARCHITECTURE.md`'s zero-init convention for
residual heads.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_warp(name: str, d: int) -> nn.Module:
    """Canonical name -> module constructor, so a fitted warp can be
    reconstructed from just `(name, d, state_dict)` -- see `artifacts.py`.
    Names: "affine", "spline_k<K>", "velocity_m<M>"."""
    if name == "affine":
        return AffineWarp(d)
    if name.startswith("spline_k"):
        return MonotoneAxisWarp(d, k=int(name[len("spline_k") :]))
    if name.startswith("velocity_m"):
        return VelocityField(d, m=int(name[len("velocity_m") :]))
    raise ValueError(f"unknown warp rung name: {name!r}")


class AffineWarp(nn.Module):
    """T(x) = I@x + b, zero/identity-initialized. The "global affine" rung."""

    def __init__(self, d: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + self.b

    def n_params(self) -> int:
        return self.A.numel() + self.b.numel()


class MonotoneAxisWarp(nn.Module):
    """Per-axis monotone piecewise-linear warp on [0,1]^d, nesting the
    identity. `k` equal-width bins per axis; softmax increments guarantee a
    strictly increasing map [0,1] -> [0,1] on each axis independently (P1:
    axis-aligned, no axis coupling)."""

    def __init__(self, d: int, k: int = 8):
        super().__init__()
        self.d, self.k = d, k
        # raw=0 -> softmax uniform -> piecewise-linear identity
        self.raw = nn.Parameter(torch.zeros(d, k))
        self.register_buffer("knot_x", torch.linspace(0.0, 1.0, k + 1))

    def _knot_y(self) -> torch.Tensor:
        incr = torch.softmax(self.raw, dim=-1)  # [d, k], rows sum to 1
        cum = torch.cumsum(incr, dim=-1)  # [d, k], last col = 1
        zero = torch.zeros(self.d, 1, device=cum.device, dtype=cum.dtype)
        return torch.cat([zero, cum], dim=-1)  # [d, k+1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        knot_y = self._knot_y()  # [d, k+1]
        x_clamped = x.clamp(0.0, 1.0)
        out = torch.empty_like(x)
        for j in range(self.d):
            idx = torch.searchsorted(self.knot_x, x_clamped[:, j].contiguous(), right=False)
            idx = idx.clamp(1, self.k)
            x0, x1 = self.knot_x[idx - 1], self.knot_x[idx]
            y0, y1 = knot_y[j, idx - 1], knot_y[j, idx]
            frac = (x_clamped[:, j] - x0) / (x1 - x0).clamp_min(1e-12)
            out[:, j] = y0 + frac * (y1 - y0)
        return out

    def n_params(self) -> int:
        return self.raw.numel()


class VelocityField(nn.Module):
    """v(u) = sum_m w_m * exp(-||u-c_m||^2 / 2 ell^2); T = flow(v, t: 0->1),
    RK4. `m` kernel centers is the complexity knob. Zero-initialized weights
    -> identity warp at the start of optimization, mirroring
    `ARCHITECTURE.md` §2.3(b)'s zero-init residual convention. Direct torch
    port of `src/ppfn/prior/registration/warp.py`'s `VelocityField`/
    `flow_rk4`, fit instead of sampled."""

    def __init__(self, d: int, m: int, n_steps: int = 5):
        super().__init__()
        self.d, self.m, self.n_steps = d, m, n_steps
        self.centers = nn.Parameter(torch.rand(m, d))
        self.weights = nn.Parameter(torch.zeros(m, d))
        self.log_ell = nn.Parameter(torch.log(torch.tensor(0.3)))

    def velocity(self, u: torch.Tensor) -> torch.Tensor:
        ell = torch.exp(self.log_ell)
        diff = u.unsqueeze(1) - self.centers.unsqueeze(0)  # [N,M,d]
        sqdist = (diff * diff).sum(-1)  # [N,M]
        kern = torch.exp(-sqdist / (2.0 * ell * ell))
        return kern @ self.weights  # [N,d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / self.n_steps
        u = x
        for _ in range(self.n_steps):
            k1 = self.velocity(u)
            k2 = self.velocity(u + 0.5 * dt * k1)
            k3 = self.velocity(u + 0.5 * dt * k2)
            k4 = self.velocity(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return u

    def n_params(self) -> int:
        return self.centers.numel() + self.weights.numel() + 1

    def bending_energy(self) -> float:
        """Closed-form RKHS norm of the velocity field, `w^T K w` with `K`
        the RBF Gram matrix over the fitted centers -- the field's own
        native smoothness penalty (curvature/"bending energy" of a
        thin-plate-spline-style deformation), rather than an ad hoc
        second-difference estimate. Well-defined here (unlike on
        `MonotoneAxisWarp`, whose second derivative is a train of Dirac
        deltas at the knots) because the field is a sum of smooth RBFs by
        construction."""
        with torch.no_grad():
            ell = torch.exp(self.log_ell)
            diff = self.centers.unsqueeze(1) - self.centers.unsqueeze(0)  # [M,M,d]
            sqdist = (diff * diff).sum(-1)
            kern = torch.exp(-sqdist / (2.0 * ell * ell))  # [M,M]
            energy = torch.einsum("mi,mn,ni->", self.weights, kern, self.weights)
        return float(energy)


def displacement_volume(warp: nn.Module, d: int, n_points: int = 3000) -> float:
    """Mean |T(x) - x| over `n_points` random points on [0,1]^d -- "how far
    did points move", independent of `logdet_jacobian_band`'s "how much did
    local volume distort". The two are deliberately not redundant: a pure
    translation has large displacement but a zero log-det band (constant
    Jacobian); a volume-preserving local swirl can have near-zero mean
    displacement but a large log-det band. `logdet_jacobian_band` alone is
    silent on the affine part of a warp -- it is exactly zero for any global
    affine map -- so this is the complementary number for "how big a move is
    this in absolute terms".

    Random points, not a `grid_n`^d regular grid: `warp.py`'s sampling-time
    version can afford a small regular grid because it only ever runs at
    d<=5, but LCBench/TaskSet reach d=7-8, where `grid_n=15` would mean
    `15**7 ~= 1.7e8` points. Cost here is `n_points` regardless of `d`."""
    points = np.random.default_rng(0).uniform(0.0, 1.0, size=(n_points, d))
    points_t = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        warped = warp(points_t).numpy()
    return float(np.mean(np.abs(warped - points)))


def logdet_jacobian_band(warp: nn.Module, d: int, n_points: int = 3000, eps: float = 1e-3) -> tuple[float, float]:
    """Realized severity of a fitted warp: the log|det J| band over
    `n_points` random points on [0,1]^d, by central finite differences --
    same statistic `warp.py`'s `logdet_jacobian_grid` computes for sampled
    fields (there, on a small regular grid, since it only ever runs at
    d<=5), applied here to a *fitted* one at d up to 7-8, where a regular
    grid's `grid_n**d` cost would be prohibitive (see `displacement_volume`).
    Returns (band, fold_fraction): `band` is max-min of log|det J| (0 for a
    pure translation, large for a strong local stretch); `fold_fraction` is
    the share of sampled points where det J <= 0 -- a fold, i.e. direct
    evidence the fitted map is not a diffeomorphism there (see
    `fit.diagnose_lack_of_fit`).
    """
    points = np.random.default_rng(0).uniform(0.0, 1.0, size=(n_points, d))
    grid_t = torch.as_tensor(points, dtype=torch.float32)

    jac = np.empty((points.shape[0], d, d))
    with torch.no_grad():
        for j in range(d):
            plus, minus = grid_t.clone(), grid_t.clone()
            plus[:, j] += eps
            minus[:, j] -= eps
            f_plus = warp(plus).numpy()
            f_minus = warp(minus).numpy()
            jac[:, :, j] = (f_plus - f_minus) / (2 * eps)
    sign, logdet = np.linalg.slogdet(jac)
    finite = np.isfinite(logdet)
    band = float(logdet[finite].max() - logdet[finite].min()) if finite.any() else float("nan")
    fold_fraction = float((sign <= 0).mean())
    return band, fold_fraction
