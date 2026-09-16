"""Top-level draw of one (A, E) pair -- ARCHITECTURE.md §1.1/§1.5.

`rho` is a REQUIRED argument, not sampled in here: the relative-warp
curriculum (ARCHITECTURE.md §4.2) depends on training progress, which this
module knows nothing about and shouldn't -- the caller (the dataset/training
loop) draws rho from the curriculum mixture and passes it in. This also
makes the ρ=0 invariants directly testable: `sample_pair(rng, rho=0.0)`.

The decoder-side context/query split (ARCHITECTURE.md §2: "a PFN over the
scarce cloud A plus its query points") is NOT done here -- it's a masking
decision over A's flat point set, made at batch-assembly time (varies per
training step, same as a standard PFN's single_eval_pos), not part of the
generative model. This module hands back the full labelled A-cloud and
E-cloud; splitting A into context/query happens in
`ppfn.prior.registration.dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ppfn.prior.registration.function_prior import sample_function_prior
from ppfn.prior.registration.normalize import normalize
from ppfn.prior.registration.region import sample_latent_A, sample_latent_B
from ppfn.prior.registration.warp import (
    declared_box,
    flow_rk4,
    mix_velocity,
    sample_warp_pair,
)

D_CHOICES = (1, 2, 3, 5)


@dataclass
class RegistrationPair:
    d: int
    n_a: int
    n_b: int
    rho: float

    x_a_norm: np.ndarray  # [n_a, d]  x-tilde^A
    y_a: np.ndarray  # [n_a]        y-tilde^A (standardized)
    x_e_norm: np.ndarray  # [n_b, d]  x-tilde^E
    y_e: np.ndarray  # [n_b]        y-tilde^E (standardized)

    a_inb_target: np.ndarray  # [n_a, d]  A_inB_target, normalized in box_E
    b_ina_target: np.ndarray  # [n_b, d]  B_inA_target, normalized in box_A

    pooled_context_x: np.ndarray  # [n_a + n_b, d]
    pooled_context_y: np.ndarray  # [n_a + n_b]

    meta: dict = field(default_factory=dict)


def sample_pair(
    rng: np.random.Generator,
    rho: float,
    d: int | None = None,
    s_max: float = 0.1,
    n_b_range: tuple[int, int] = (256, 1024),
    n_a_range: tuple[int, int] = (8, 256),
    warp_grid_n: int = 5,
) -> RegistrationPair:
    """One draw of (A, E) at the given relative-warp coefficient `rho`.
    `s_max=0.1` is the calibrated default -- empirically ~3-7% per-draw
    rejection at d in {1,2,3} under `warp.sample_warp_pair`'s log|det J|
    band, i.e. "a few percent" per ARCHITECTURE.md §1.3's tuning target (see
    the warp.py module demo / docs/labbook/ for the calibration sweep).

    `warp_grid_n`: override of `sample_warp_pair`/`declared_box`'s own
    `grid_n=5` default, threaded through here rather than left at their
    hardcoded default -- same override `ppfn.prior.lupi.sampler.sample_pair`
    already exposes (`warp_grid_n`, default 4 there), ported to this shared
    sampler rather than re-derived: the warp-rejection Jacobian check
    (`logdet_jacobian_grid`, via `declared_box`/`sample_warp_pair`) is
    ~80-97% of per-item cost REGARDLESS of n_a/n_b (see
    docs/labbook/2026-09-14-lupi-prior-warp-grid-speedup.md and
    docs/labbook/2026-09-15-id-token-baseline-oom-and-progress-bugs.md),
    and scales as `grid_n^d`, so 5->4 cuts that cost ~3x with a measured,
    acceptable loss of band-estimate accuracy (grid_n=3 was rejected there
    for underestimating the true log|det J| band by ~40%). Default kept at
    5 here -- unlike the LUPI-only module, this sampler is shared by
    arch_verification/step4_pathway/bridge_pfn/etc., so nothing changes for
    them unless a caller explicitly opts in via `configs/prior/registration.yaml`'s
    `warp_grid_n` field."""
    if d is None:
        d = int(rng.choice(D_CHOICES))
    assert 0.0 <= rho <= 1.0

    n_b = int(np.exp(rng.uniform(np.log(n_b_range[0]), np.log(n_b_range[1] + 1))))
    n_a = int(np.exp(rng.uniform(np.log(n_a_range[0]), np.log(n_a_range[1] + 1))))
    n_b = int(np.clip(n_b, *n_b_range))
    n_a = int(np.clip(n_a, *n_a_range))

    v_a, v_b = sample_warp_pair(rng, d, s_max=s_max, grid_n=warp_grid_n)

    z_a, region, volume_fraction = sample_latent_A(rng, d, n_a)
    z_b = sample_latent_B(rng, d, n_b)

    # Phi_0 = flow(v_a) alone -- the decoder cloud's own warp, and the
    # direction B_inA_target uses (ARCHITECTURE.md §1.3/§1.5).
    box_a = declared_box(v_a, d, grid_n=warp_grid_n)
    x_a_raw = flow_rk4(v_a, z_a)
    b_ina_target_raw = flow_rk4(v_a, z_b)

    # Phi_rho = flow of ((1-rho) v_a + rho v_b) -- the encoder cloud's warp
    # at this pair's relative-warp coefficient.
    phi_rho = mix_velocity([v_a, v_b], [1.0 - rho, rho])
    box_e = declared_box(phi_rho, d, grid_n=warp_grid_n)
    x_e_raw = flow_rk4(phi_rho, z_b)
    a_inb_target_raw = flow_rk4(phi_rho, z_a)

    x_a_norm = normalize(x_a_raw, box_a)
    x_e_norm = normalize(x_e_raw, box_e)
    a_inb_target = normalize(a_inb_target_raw, box_e)
    b_ina_target = normalize(b_ina_target_raw, box_a)

    # f ~ BNN prior on z (the shared latent frame -- ARCHITECTURE.md §1.1,
    # decisions.md D3). Evaluated at z_a and z_b together so sigma_obs's
    # "relative to std of f over the domain" (§1.4) is resolved against one
    # pooled probe rather than two independently-scaled noise levels.
    f = sample_function_prior(rng, d, probe_z=np.concatenate([z_a, z_b], axis=0))
    y_a_raw = f.sample_y(rng, z_a)
    y_e_raw = f.sample_y(rng, z_b)

    y_pool = np.concatenate([y_a_raw, y_e_raw])
    y_mean, y_std = float(y_pool.mean()), max(float(y_pool.std()), 1e-6)
    y_a = (y_a_raw - y_mean) / y_std
    y_e = (y_e_raw - y_mean) / y_std

    pooled_context_x = np.concatenate([x_a_norm, b_ina_target], axis=0)
    pooled_context_y = np.concatenate([y_a, y_e], axis=0)

    y_a_range = float(y_a_raw.max() - y_a_raw.min()) if n_a > 1 else 0.0
    y_e_range = float(y_e_raw.max() - y_e_raw.min()) if n_b > 1 else 1e-6
    y_range_fraction = y_a_range / max(y_e_range, 1e-6)

    meta = {
        "region_type": region.kind,
        "volume_fraction": volume_fraction,
        "y_range_fraction": y_range_fraction,
        "severity_a": v_a.severity,
        "severity_b": v_b.severity,
        "rho": rho,
    }

    return RegistrationPair(
        d=d,
        n_a=n_a,
        n_b=n_b,
        rho=rho,
        x_a_norm=x_a_norm,
        y_a=y_a,
        x_e_norm=x_e_norm,
        y_e=y_e,
        a_inb_target=a_inb_target,
        b_ina_target=b_ina_target,
        pooled_context_x=pooled_context_x,
        pooled_context_y=pooled_context_y,
        meta=meta,
    )


def sample_rho_curriculum(rng: np.random.Generator, progress: float) -> float:
    """ARCHITECTURE.md §4.2: rho = 0 w.p. 0.15 (permanent anchor), else
    Beta(a_k, b_k) with (a_k, b_k) shifting from Beta(1,5) at the start of
    training to Beta(1,1) (uniform) by 30% of training, held after.
    `progress`: fraction of total training steps completed, in [0, 1]."""
    if rng.random() < 0.15:
        return 0.0
    t = min(progress / 0.30, 1.0)
    a = 1.0  # both endpoints share a=1; only b moves (5 -> 1)
    b = 5.0 + t * (1.0 - 5.0)
    return float(rng.beta(a, b))


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: for d=1 and d=2, plot
    z / x_A / x_E / y for ~50 draws at a mix of rho values, in separate A/B
    subplots, plus a rho=0 draw to eyeball that both clouds coincide."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, d in enumerate((1, 2)):
        ax_a, ax_e, ax_y = axes[row]
        for _ in range(50):
            rho = float(rng.uniform(0.0, 1.0))
            pair = sample_pair(rng, rho=rho, d=d)
            if d == 1:
                ax_a.scatter(
                    pair.x_a_norm[:, 0],
                    np.zeros(pair.n_a),
                    s=6,
                    alpha=0.4,
                    color="#D55E00",
                )
                ax_e.scatter(
                    pair.x_e_norm[:, 0],
                    np.zeros(pair.n_b),
                    s=4,
                    alpha=0.2,
                    color="#4477AA",
                )
                ax_y.scatter(
                    pair.x_a_norm[:, 0], pair.y_a, s=6, alpha=0.4, color="#D55E00"
                )
            else:
                ax_a.scatter(
                    pair.x_a_norm[:, 0],
                    pair.x_a_norm[:, 1],
                    s=6,
                    alpha=0.4,
                    color="#D55E00",
                )
                ax_e.scatter(
                    pair.x_e_norm[:, 0],
                    pair.x_e_norm[:, 1],
                    s=4,
                    alpha=0.2,
                    color="#4477AA",
                )
                ax_y.scatter(
                    pair.x_a_norm[:, 0],
                    pair.x_a_norm[:, 1],
                    c=pair.y_a,
                    s=8,
                    alpha=0.6,
                    cmap="viridis",
                )
        ax_a.set_title(f"d={d}: x-tilde^A (context+query, normalized)")
        ax_e.set_title(f"d={d}: x-tilde^E (encoder cloud, normalized)")
        ax_y.set_title(f"d={d}: A colored/plotted by y-tilde^A")

    fig.suptitle("RegistrationPair -- 50 draws per row, rho ~ U[0,1]")
    fig.tight_layout()
    plt.show()

    # rho=0 anchor sanity: A_inB_target should equal x-tilde^A exactly.
    pair0 = sample_pair(rng, rho=0.0, d=2)
    print(
        "rho=0 check: max|A_inB_target - x_a_norm| =",
        float(np.abs(pair0.a_inb_target - pair0.x_a_norm).max()),
        "(expect ~0)",
    )
    print(
        "rho=0 check: max|pooled_context_x[n_a:] - x_e_norm| =",
        float(np.abs(pair0.pooled_context_x[pair0.n_a :] - pair0.x_e_norm).max()),
        "(expect ~0)",
    )
