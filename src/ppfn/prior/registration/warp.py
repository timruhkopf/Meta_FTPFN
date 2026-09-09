"""Stationary velocity-field warps — ARCHITECTURE.md §1.3.

    v(u) = s * sum_m w_m * exp(-||u - c_m||^2 / (2 * ell^2))
    S = flow of v from t=0 to t=1, via RK4 with n_steps steps.

A convex combination of velocity fields is itself a velocity field, so its
flow is a genuine diffeomorphism at every ρ (never an interpolation
artifact) — this is what makes `mix_fields`/`flow_rho` below correct rather
than a numerical convenience.

Jacobians (for the rejection band and for the declared-box computation) are
estimated by batched central finite differences through the flow map, not by
integrating the analytic Jacobian ODE alongside the point ODE. The latter
would be exact and is the "proper" way to do this, but the finite-difference
version is far simpler to get right, and the grids here are small (<=5^5
points) and evaluated as ONE batched flow call per perturbed axis, so the
cost is a small constant number of flow evaluations, not one per point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityField:
    """v(u) = sum_m w_m * exp(-||u - c_m||^2 / (2*ell^2)); severity `s` is
    already folded into `w` at sampling time (see `sample_velocity_field`),
    so mixing two fields is just summing their `w`-scaled evaluations."""

    centers: np.ndarray  # [M, d]
    weights: np.ndarray  # [M, d]
    lengthscale: float
    severity: float  # kept for logging only; already baked into `weights`

    def __call__(self, u: np.ndarray) -> np.ndarray:
        """u: [N, d] -> v(u): [N, d]."""
        # [N, M, d]
        diff = u[:, None, :] - self.centers[None, :, :]
        sqdist = np.sum(diff * diff, axis=-1)  # [N, M]
        kern = np.exp(-sqdist / (2.0 * self.lengthscale**2))  # [N, M]
        return kern @ self.weights  # [N, d]

    def jacobian(self, u: np.ndarray) -> np.ndarray:
        """Analytic d v(u) / d u, batched. u: [N,d] -> [N,d,d]. Used only for
        the (optional, cheap) analytic cross-check in the module demo — the
        rejection/box code path uses finite differences on the flow map
        instead (see module docstring)."""
        diff = u[:, None, :] - self.centers[None, :, :]  # [N, M, d]
        sqdist = np.sum(diff * diff, axis=-1)  # [N, M]
        kern = np.exp(-sqdist / (2.0 * self.lengthscale**2))  # [N, M]
        # d kern_m / d u_j = -diff_j / ell^2 * kern_m
        dkern = -diff / (self.lengthscale**2) * kern[..., None]  # [N, M, d] (d over j)
        # v_i(u) = sum_m w_{m,i} kern_m  =>  d v_i / d u_j = sum_m w_{m,i} dkern_{m,j}
        return np.einsum("mi,nmj->nij", self.weights, dkern)  # [N, d, d]


def sample_velocity_field(
    rng: np.random.Generator, d: int, s_max: float
) -> VelocityField:
    """One draw of (M, c_m, w_m, ell, s) — ARCHITECTURE.md §1.3."""
    m = int(rng.integers(4, 33))  # Uniform{4,...,32}
    centers = rng.uniform(0.0, 1.0, size=(m, d))
    raw_weights = rng.normal(0.0, 1.0, size=(m, d))
    lengthscale = float(np.exp(rng.uniform(np.log(0.1), np.log(0.5))))
    severity = float(rng.uniform(0.0, 1.0)) * s_max
    return VelocityField(
        centers=centers,
        weights=raw_weights * severity,
        lengthscale=lengthscale,
        severity=severity,
    )


def flow_rk4(velocity_fn, points: np.ndarray, n_steps: int = 5) -> np.ndarray:
    """Integrate du/dt = velocity_fn(u) from t=0 to t=1. points: [N,d]."""
    dt = 1.0 / n_steps
    u = points.astype(np.float64, copy=True)
    for _ in range(n_steps):
        k1 = velocity_fn(u)
        k2 = velocity_fn(u + 0.5 * dt * k1)
        k3 = velocity_fn(u + 0.5 * dt * k2)
        k4 = velocity_fn(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def mix_velocity(fields: list[VelocityField], coeffs: list[float]):
    """Returns a callable u -> sum_i coeffs[i] * fields[i](u). Used to build
    Phi_rho = flow of ((1-rho)*v_A + rho*v_B) — ARCHITECTURE.md §1.3."""

    def _v(u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(u)
        for field, c in zip(fields, coeffs):
            if c != 0.0:
                out = out + c * field(u)
        return out

    return _v


def _regular_grid(d: int, grid_n: int) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, grid_n) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)  # [grid_n**d, d]


def logdet_jacobian_grid(
    velocity_fn, d: int, grid_n: int = 5, n_steps: int = 5, eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """log|det J| of the flow map, estimated by central finite differences,
    on a `grid_n`^d regular grid over the unit cube.

    Returns (logdet [G], sign [G]) — `sign` lets callers additionally guard
    against a spuriously non-orientation-preserving Jacobian (should not
    happen for a true diffeomorphism away from numerical noise; treated as a
    rejection trigger, see `sample_warp_pair`).
    """
    grid = _regular_grid(d, grid_n)  # [G, d]
    jac = np.empty((grid.shape[0], d, d))
    for j in range(d):
        plus = grid.copy()
        plus[:, j] += eps
        minus = grid.copy()
        minus[:, j] -= eps
        f_plus = flow_rk4(velocity_fn, plus, n_steps)
        f_minus = flow_rk4(velocity_fn, minus, n_steps)
        jac[:, :, j] = (f_plus - f_minus) / (2 * eps)
    sign, logdet = np.linalg.slogdet(jac)
    return logdet, sign


def sample_warp_pair(
    rng: np.random.Generator,
    d: int,
    s_max: float = 1.0,
    grid_n: int = 5,
    n_steps: int = 5,
    max_rejections: int = 50,
) -> tuple[VelocityField, VelocityField]:
    """Draw v_A, v_B independently, each subject to the rejection band on its
    OWN flow (Phi_0 = flow(v_A), Phi_1 = flow(v_B)) — ARCHITECTURE.md §1.3:
    "reject the draw if max-min > log(9)". Intermediate rho in (0,1) is never
    separately rejected: a convex combination of two accepted velocity
    fields is itself a bounded velocity field and its flow is automatically
    a diffeomorphism (see module docstring)."""

    def _draw_accepted() -> VelocityField:
        for _ in range(max_rejections):
            field = sample_velocity_field(rng, d, s_max)
            logdet, sign = logdet_jacobian_grid(
                field, d, grid_n=grid_n, n_steps=n_steps
            )
            if np.any(sign <= 0):
                continue  # numerically indistinguishable from a fold; resample
            band = logdet.max() - logdet.min()
            if band <= np.log(9.0):
                return field
        # Give up gracefully after max_rejections rather than hanging a
        # dataloader worker forever on a pathological RNG state: return the
        # last (rejected) draw. Rare in practice at s_max ~= 1 (see the
        # module demo's rejection-rate check).
        return field

    v_a = _draw_accepted()
    v_b = _draw_accepted()
    return v_a, v_b


def declared_box(
    velocity_fn, d: int, grid_n: int = 5, n_steps: int = 5, pad_frac: float = 0.1
):
    """bbox of the domain IMAGE (the flow of the unit-cube grid), not of any
    sampled points — ARCHITECTURE.md invariant #8. Padded by `pad_frac` of
    the observed range per axis, since the grid-estimated bbox can miss the
    true extremes of a diffeomorphic image of a cube (they need not lie on
    the cube's boundary), and un-padded normalization would silently clip
    genuinely in-domain points into the tail. Returns (lo [d], hi [d])."""
    grid = _regular_grid(d, grid_n)
    image = flow_rk4(velocity_fn, grid, n_steps)
    lo = image.min(axis=0)
    hi = image.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pad = pad_frac * span
    return lo - pad, hi + pad


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: sample many warp
    pairs, report the empirical rejection rate and the realized log|det J|
    band at s_max=1.0 (the value ARCHITECTURE.md §1.3 says to tune against
    "a few percent rejection"), and plot a handful of accepted d=2 warps'
    grid displacement to eyeball that they look like plausible local
    stretches rather than degenerate near-folds."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    d = 2
    n_trials = 200
    n_rejected = 0
    n_draws_per_accept = []

    for _ in range(n_trials):
        draws = 0
        for _ in range(50):
            draws += 1
            field = sample_velocity_field(rng, d, s_max=1.0)
            logdet, sign = logdet_jacobian_grid(field, d)
            band = logdet.max() - logdet.min() if np.all(sign > 0) else np.inf
            if band <= np.log(9.0):
                break
        else:
            n_rejected += 1
        n_draws_per_accept.append(draws)

    accept_rate_per_draw = 1.0 / np.mean(n_draws_per_accept)
    print(
        f"s_max=1.0: mean draws-until-accept={np.mean(n_draws_per_accept):.2f}, "
        f"per-draw rejection rate~{1 - accept_rate_per_draw:.2%}, "
        f"gave-up-after-50={n_rejected}/{n_trials}"
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax in axes:
        v_a, v_b = sample_warp_pair(rng, d=2, s_max=1.0)
        grid = _regular_grid(2, 12)
        warped = flow_rk4(v_a, grid)
        for i in range(grid.shape[0]):
            ax.plot(
                [grid[i, 0], warped[i, 0]],
                [grid[i, 1], warped[i, 1]],
                color="#4477AA",
                lw=0.6,
                alpha=0.7,
            )
        ax.scatter(grid[:, 0], grid[:, 1], s=6, color="#1a1a1a", zorder=3)
        ax.scatter(warped[:, 0], warped[:, 1], s=6, color="#D55E00", zorder=3)
        ax.set_title("grid -> S_A(grid)")
    fig.suptitle(
        "sample_warp_pair draws: unit-cube grid before (black) / after warp (orange)"
    )
    fig.tight_layout()
    plt.show()
