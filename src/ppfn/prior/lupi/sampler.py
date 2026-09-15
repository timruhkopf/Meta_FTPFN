"""Top-level draw of one LUPI training pair -- build spec §6.1 (generative
process) + §6.2 (query mixture, "avoiding B-blindness"). Assembles pieces
that already exist (`ppfn.prior.registration.warp`/`.normalize`/`.region`/
`.function_prior`, all reused directly) with the two genuinely new pieces
this spec adds: the y-distortion `h` (`monotone.py`) and the acquisition-
biased A design (`acquisition.py`).

Unlike `ppfn.prior.registration.sampler.sample_pair`, the decoder-side
context/query split is NOT deferred to a `dataset.py`-level re-partition of
one flat cloud: spec §6.2's query mixture draws queries from sources (a
fresh uniform point, a perturbed B-design point, a perturbed A-context
point) that aren't all subsets of the context cloud, so query generation
needs the same (v_a, phi_rho, f, h, box_a, box_e) in scope as context
generation and happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ppfn.prior.lupi.acquisition import sample_beta, sample_latent_A_acquired
from ppfn.prior.lupi.ecdf import apply_ecdf, fit_ecdf
from ppfn.prior.lupi.monotone import sample_monotone_map
from ppfn.prior.registration.function_prior import sample_function_prior
from ppfn.prior.registration.normalize import normalize
from ppfn.prior.registration.region import sample_latent_B, sample_region
from ppfn.prior.registration.warp import declared_box, flow_rk4, mix_velocity, sample_warp_pair

D_CHOICES = (1, 2, 3, 5)


@dataclass
class LUPIPair:
    d: int
    rho: float
    beta: float

    x_b: np.ndarray  # [n_b, d]      B's own frame, normalized to [0,1]^d
    z_b: np.ndarray  # [n_b]         quantile-normalized value (F_hat_B(y_b_obs))
    x_b_inA: np.ndarray  # [n_b, d]  B transported into A's frame via A's OWN warp (no inversion --
    # B_inA = Phi_0(z_B), same trick _sample_query_z's near-B bucket already
    # uses). Shares z_b's values exactly (position differs, value doesn't) --
    # the "upper bound" pooled context [A_ctx ; B_inA] for
    # ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN.

    x_a_ctx: np.ndarray  # [n_a, d]
    z_a_ctx: np.ndarray  # [n_a]
    oracle_bpos_a_ctx: np.ndarray  # [n_a, d]  T(x_i^A), normalized in box_e -- oracle-mode position

    x_a_qry: np.ndarray  # [n_qry, d]
    z_a_qry: np.ndarray  # [n_qry]  target
    oracle_bpos_a_qry: np.ndarray  # [n_qry, d]
    qry_source: np.ndarray  # [n_qry] int: 0=uniform, 1=near-B, 2=near-A -- for monitors only

    meta: dict = field(default_factory=dict)


@dataclass
class LUPIPairInternals:
    """Everything needed to evaluate the TRUE (noiseless) generative
    process at ARBITRARY z-grid points after the fact -- for
    `notebooks/lupi_1d_visualization.ipynb`'s "true function" overlay, not
    used anywhere in training/loss. Only returned when `sample_pair(...,
    return_internals=True)`; the default (False) path is byte-identical to
    before this was added, so this costs nothing when unused.

    Grid in z-space, not x-space: z ~ Uniform([0,1]^d) is the shared latent
    the whole generative process is defined on, so a dense z-grid mapped
    through `to_a`/`to_b` gives dense, CORRECTLY-ORDERED x-space curves
    without ever needing to invert a warp (same trick `_sample_query_z`'s
    near-B bucket and `x_b_inA` already use)."""

    to_a: object  # Callable[[np.ndarray], np.ndarray] -- z [N,d] -> A-frame x [N,d]
    to_b: object  # Callable[[np.ndarray], np.ndarray] -- z [N,d] -> B-frame x [N,d]
    f: object  # FunctionPrior -- f(z) noiseless
    h: object  # MonotoneMap -- h(y)
    sorted_a: np.ndarray  # A's ECDF reference sample (fit_ecdf(y_a_ctx_obs))
    sorted_b: np.ndarray  # B's ECDF reference sample (fit_ecdf(y_b_clean))
    z_a_ctx: np.ndarray  # [n_a, d] -- the actual latents behind pair.x_a_ctx/z_a_ctx (A's small, acquisition-biased sample)
    z_b: np.ndarray  # [n_b, d] -- the actual latents behind pair.x_b/x_b_inA/z_b (B's large, uniform sample)

    def true_z_a(self, z: np.ndarray) -> np.ndarray:
        """z [N,d] -> noiseless quantile-normalized-on-A's-scale target
        [N] -- the "true function" curve, A's own value calibration."""
        return apply_ecdf(self.sorted_a, self.h(self.f(z)))

    def value_under_b_ecdf(self, z: np.ndarray) -> np.ndarray:
        """z [N,d] -> what the SAME raw value h(f(z)) would quantile-
        normalize to under B's OWN (larger, unbiased-by-acquisition) ECDF
        instead of A's -- for directly testing whether A's small,
        acquisition-biased sample assigns systematically different
        quantiles than B's large, uniform one would to the same points
        (docs/labbook/2026-09-15-lupi-reference-measure-mismatch.md)."""
        return apply_ecdf(self.sorted_b, self.h(self.f(z)))


def _sample_query_z(
    rng: np.random.Generator,
    d: int,
    n_qry: int,
    z_b: np.ndarray,
    z_a_ctx: np.ndarray,
    frac_uniform: float,
    frac_near_b: float,
    eps_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Spec §6.2's 3-way query mixture, expressed as z-space draws so the
    SAME (v_a, box_a, phi_rho, box_e, h, f) pipeline used for context tokens
    also produces queries -- see module docstring. Perturbing in z-space
    then warping (rather than perturbing the already-warped x directly) is a
    clean analogue of "T^{-1}(x_j^B) + eps": since x_j^B = Phi_rho(z_j^B),
    z_j^B + eps warped through A's OWN Phi_0 gives Phi_0(z_j^B + eps) ~=
    T^{-1}(x_j^B) + O(eps) without ever needing to invert Phi_rho. Returns
    (z_qry [n_qry, d], source [n_qry] int8: 0=uniform,1=near-B,2=near-A)."""
    n_unif = int(round(frac_uniform * n_qry))
    n_near_b = int(round(frac_near_b * n_qry))
    n_near_a = n_qry - n_unif - n_near_b

    z_unif = rng.uniform(0.0, 1.0, size=(n_unif, d))

    b_idx = rng.integers(0, z_b.shape[0], size=n_near_b)
    z_near_b = np.clip(z_b[b_idx] + rng.normal(0.0, eps_std, size=(n_near_b, d)), 0.0, 1.0)

    a_idx = rng.integers(0, z_a_ctx.shape[0], size=n_near_a)
    z_near_a = np.clip(z_a_ctx[a_idx] + rng.normal(0.0, eps_std, size=(n_near_a, d)), 0.0, 1.0)

    z_qry = np.concatenate([z_unif, z_near_b, z_near_a], axis=0)
    source = np.concatenate(
        [np.zeros(n_unif), np.ones(n_near_b), np.full(n_near_a, 2)]
    ).astype(np.int8)
    return z_qry, source


def sample_pair(
    rng: np.random.Generator,
    rho: float,
    d: int | None = None,
    s_max: float = 0.1,
    n_b_range: tuple[int, int] = (256, 1024),
    n_a_range: tuple[int, int] = (8, 256),
    n_qry_range: tuple[int, int] = (8, 128),
    frac_uniform: float = 0.4,
    frac_near_b: float = 0.4,
    query_eps_std: float = 0.03,
    warp_grid_n: int = 4,
    return_internals: bool = False,
) -> LUPIPair | tuple[LUPIPair, LUPIPairInternals]:
    """One draw at relative-warp coefficient `rho`. `frac_near_b`: set to 0.0
    to disable spec §6.2's anti-B-blindness mechanism as an ablation (the
    "optional" preferential-near-B query sampling); `frac_uniform` +
    `frac_near_b` must be <= 1, the remainder is spec's third bucket
    (near A's own context).

    `warp_grid_n`: LUPI-local override of `sample_warp_pair`/`declared_box`'s
    own `grid_n=5` default (never touches `ppfn.prior.registration`, which
    keeps its calibrated default). Profiling (2026-09-14,
    `docs/labbook/2026-09-14-lupi-prior-warp-grid-speedup.md`) found the
    log|det J| Jacobian-band rejection check inside `sample_warp_pair`
    dominates per-item cost (~84%, via a `grid_n^d` finite-difference grid --
    3125 points at d=5) and scales as `grid_n^d`: 5->4 cuts it ~3x for a
    measured, modest loosening of the accept/reject band estimate (kept over
    the more aggressive grid_n=3, which underestimated the true band by
    ~40% on average). `n_steps` (RK4 integration steps) is left at its
    default 5 -- untouched, not profiled as a bottleneck."""
    assert 0.0 <= frac_uniform + frac_near_b <= 1.0
    if d is None:
        d = int(rng.choice(D_CHOICES))
    assert 0.0 <= rho <= 1.0

    n_b = int(np.clip(np.exp(rng.uniform(np.log(n_b_range[0]), np.log(n_b_range[1] + 1))), *n_b_range))
    n_a = int(np.clip(np.exp(rng.uniform(np.log(n_a_range[0]), np.log(n_a_range[1] + 1))), *n_a_range))
    n_qry = int(np.clip(np.exp(rng.uniform(np.log(n_qry_range[0]), np.log(n_qry_range[1] + 1))), *n_qry_range))

    v_a, v_b = sample_warp_pair(rng, d, s_max=s_max, grid_n=warp_grid_n)
    box_a = declared_box(v_a, d, grid_n=warp_grid_n)
    phi_rho = mix_velocity([v_a, v_b], [1.0 - rho, rho])
    box_e = declared_box(phi_rho, d, grid_n=warp_grid_n)

    def to_a(z: np.ndarray) -> np.ndarray:
        return normalize(flow_rk4(v_a, z), box_a)

    def to_b(z: np.ndarray) -> np.ndarray:
        return normalize(flow_rk4(phi_rho, z), box_e)

    region, volume_fraction = sample_region(rng, d), None
    from ppfn.prior.registration.region import estimate_volume_fraction

    volume_fraction = estimate_volume_fraction(rng, region, d)

    # B: uniform over the full declared domain (spec §3.1 -- "no problem",
    # B's own sample already matches the reference measure, see ecdf.py).
    z_b = sample_latent_B(rng, d, n_b)
    x_b = to_b(z_b)
    x_b_inA = to_a(z_b)  # B transported into A's frame -- no inversion, see field docstring

    f = sample_function_prior(rng, d, probe_z=z_b)
    h = sample_monotone_map(rng)

    y_b_clean = f(z_b)  # noiseless -- the ECDF reference sample (ties preserved)
    y_b_obs = y_b_clean + rng.normal(0.0, f.sigma_obs, size=n_b)
    sorted_b = fit_ecdf(y_b_clean)
    z_b_quant = apply_ecdf(sorted_b, y_b_obs)

    # A's own noise scale, resolved against std(h(f(.))) over B's (full-domain)
    # probe -- same "relative to std(f) over the domain" convention
    # `function_prior.sample_function_prior` uses for f itself.
    hf_probe = h(y_b_clean)
    std_hf = max(float(hf_probe.std()), 1e-6)
    sigma_obs_a = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))) * std_hf

    beta = sample_beta(rng)
    z_a_ctx = sample_latent_A_acquired(rng, region, d, n_a, score_fn=f, beta=beta)
    x_a_ctx = to_a(z_a_ctx)
    y_a_ctx_clean = h(f(z_a_ctx))
    y_a_ctx_obs = y_a_ctx_clean + rng.normal(0.0, sigma_obs_a, size=n_a)
    sorted_a = fit_ecdf(y_a_ctx_obs)  # from the BIASED, NOISY sample -- spec §3.2(b)'s pathology
    z_a_ctx_quant = apply_ecdf(sorted_a, y_a_ctx_obs)
    oracle_bpos_a_ctx = to_b(z_a_ctx)

    z_qry, qry_source = _sample_query_z(
        rng, d, n_qry, z_b, z_a_ctx, frac_uniform, frac_near_b, query_eps_std
    )
    x_a_qry = to_a(z_qry)
    y_qry_clean = h(f(z_qry))
    y_qry_obs = y_qry_clean + rng.normal(0.0, sigma_obs_a, size=n_qry)
    z_a_qry_quant = apply_ecdf(sorted_a, y_qry_obs)
    oracle_bpos_a_qry = to_b(z_qry)

    meta = {
        "region_type": region.kind,
        "volume_fraction": volume_fraction,
        "beta": beta,
        "rho": rho,
    }

    pair = LUPIPair(
        d=d,
        rho=rho,
        beta=beta,
        x_b=x_b,
        z_b=z_b_quant,
        x_b_inA=x_b_inA,
        x_a_ctx=x_a_ctx,
        z_a_ctx=z_a_ctx_quant,
        oracle_bpos_a_ctx=oracle_bpos_a_ctx,
        x_a_qry=x_a_qry,
        z_a_qry=z_a_qry_quant,
        oracle_bpos_a_qry=oracle_bpos_a_qry,
        qry_source=qry_source,
        meta=meta,
    )
    if not return_internals:
        return pair
    internals = LUPIPairInternals(
        to_a=to_a, to_b=to_b, f=f, h=h, sorted_a=sorted_a, sorted_b=sorted_b,
        z_a_ctx=z_a_ctx, z_b=z_b,
    )
    return pair, internals


def sample_rho_curriculum(rng: np.random.Generator, progress: float) -> float:
    """Identical curriculum to `ppfn.prior.registration.sampler`'s own
    (CLAUDE.md invariant #4: the rho=0 anchor is permanent) -- duplicated
    rather than imported so this package has no torch-free-but-still-
    cross-package coupling on the registration prior's internals beyond the
    warp/normalize/region/function_prior modules it already reuses."""
    if rng.random() < 0.15:
        return 0.0
    t = min(progress / 0.30, 1.0)
    a = 1.0
    b = 5.0 + t * (1.0 - 5.0)
    return float(rng.beta(a, b))


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: rho=0 invariant check
    (oracle B-frame position must equal the A-frame position exactly, both
    clouds coincide) plus a d=1 plot of the three query-source buckets."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    pair0 = sample_pair(rng, rho=0.0, d=2)
    print(
        "rho=0 check: max|oracle_bpos_a_ctx - x_a_ctx| =",
        float(np.abs(pair0.oracle_bpos_a_ctx - pair0.x_a_ctx).max()),
        "(expect ~0)",
    )
    print(
        "rho=0 check: max|oracle_bpos_a_qry - x_a_qry| =",
        float(np.abs(pair0.oracle_bpos_a_qry - pair0.x_a_qry).max()),
        "(expect ~0)",
    )

    pair = sample_pair(rng, rho=0.6, d=1, n_qry_range=(300, 300))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#4477AA", 1: "#D55E00", 2: "#009E73"}
    labels = {0: "uniform", 1: "near-B", 2: "near-A-context"}
    for src in (0, 1, 2):
        m = pair.qry_source == src
        ax.scatter(pair.x_a_qry[m, 0], pair.z_a_qry[m], s=10, alpha=0.6, color=colors[src], label=labels[src])
    ax.scatter(pair.x_a_ctx[:, 0], pair.z_a_ctx, s=30, color="black", marker="x", label="A context")
    ax.set_xlabel("x (A's frame)")
    ax.set_ylabel("quantile-normalized value")
    ax.legend()
    ax.set_title(f"LUPIPair query mixture, d=1, rho=0.6, beta={pair.beta:.2f}")
    fig.tight_layout()
    plt.show()
