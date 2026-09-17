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
from ppfn.prior.lupi.ecdf import apply_ecdf, load_or_fit_f_ecdf, normalize_f_probe
from ppfn.prior.lupi.monotone import (
    MonotoneMap,
    calibrate_amplitude,
    sample_kumaraswamy_h,
    sample_monotone_map,
)
from ppfn.prior.registration.function_prior import sample_function_prior
from ppfn.prior.registration.normalize import normalize
from ppfn.prior.registration.region import sample_latent_B, sample_region
from ppfn.prior.registration.warp import declared_box, flow_rk4, mix_velocity, sample_warp_pair

D_CHOICES = (1, 2, 3, 5)


@dataclass
class LUPIPair:
    # NAMING NOTE (flagged 2026-09-16, at the user's direction -- not
    # renamed here, touches ~14 files including files owned by the
    # ppfn.model.lupi/monitor.lupi track): every field below named `z_*`
    # (z_b, z_a_ctx, z_a_qry, and LUPIBatch's enc_z/dec_ctx_z/dec_qry_z/
    # enc_z_inA downstream) is actually the OBSERVED VALUE -- what
    # CLAUDE.md's own convention table calls `y` (`y = f(z) + eps`), not the
    # latent domain coordinate `z` that table defines. `sample_pair`'s
    # internal `zlat_*` variables (the actual latents) were renamed away
    # from bare `z_*` for exactly this confusion on 2026-09-15, but that
    # rename never reached these public dataclass field names, which still
    # collide with the *other* meaning of `z`. A full fix is a field rename
    # across LUPIPair/LUPIBatch and every consumer (this file, dataset.py,
    # both loss files, id_token_pfn.py, lupi_id_token_pfn.py,
    # plain_pfn_bounds.py, lupi_bounds_pfn.py, calibration.py, the
    # notebooks, AND ppfn.model.lupi.model/ppfn.monitor.lupi/
    # ppfn.loss.lupi_loss/lupi_bounds_loss, which this session doesn't own)
    # -- needs doing, needs coordinating with whoever's on that other track
    # first since it'd break their code if done partially.
    d: int
    rho: float
    beta: float

    x_b: np.ndarray  # [n_b, d]      B's own frame, normalized to [0,1]^d
    z_b: np.ndarray  # [n_b]         raw observed value y_b_obs (unnormalized -- see sample_pair's normalization-shelving comment)
    x_b_inA: np.ndarray  # [n_b, d]  B transported into A's frame via A's OWN warp (no inversion --
    # B_inA = Phi_0(z_B), same trick _sample_query_z's near-B bucket already
    # uses).
    z_b_inA: np.ndarray  # [n_b]  B's OBSERVED value recalibrated into A's
    # frame via h: h(y_b_obs). Paired with x_b_inA, this is what a model
    # with FULLY solved registration (T *and* h) should see -- it sits on
    # A's own true curve (h(f(z)), up to B's own observation noise), NOT on
    # B's uncorrected scale. `z_b` above is deliberately kept separate:
    # it's B's raw value on B's OWN scale, the "T solved, h still not" half
    # of the two-quantity split -- corrected 2026-09-16 at the user's
    # direction after `z_b` was mistakenly used as the oracle's value
    # (position-only correction, still not on A's curve) in an earlier
    # version of this pipeline; see docs/labbook/.

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
    h: object  # MonotoneMap or KumaraswamyMap -- h(y)
    z_a_ctx: np.ndarray  # [n_a, d] -- the actual latents behind pair.x_a_ctx/z_a_ctx (A's small, acquisition-biased sample)
    z_b: np.ndarray  # [n_b, d] -- the actual latents behind pair.x_b/x_b_inA/z_b (B's large, uniform sample)
    # Only set when sample_pair(..., bounded01=True) produced this pair --
    # true_value_as_a/b need these to replicate the SAME self-normalize ->
    # global-ECDF -> h pipeline sample_pair itself used, or they'd silently
    # evaluate h on RAW (un-normalized) f(z), which is wrong under this mode
    # (KumaraswamyMap expects a [0,1]-scale input, not f's raw output).
    f_mean: float | None = None
    f_std: float | None = None
    f_ecdf_ref: np.ndarray | None = None

    def true_value_as_a(self, z: np.ndarray) -> np.ndarray:
        """z [N,d] -> noiseless value [N] AS A WOULD OBSERVE IT (h applied)
        -- the "true function" curve on A's own (h-distorted) scale. Under
        `bounded01`, replicates sample_pair's self-normalize -> global-ECDF
        -> h pipeline exactly; otherwise raw/unnormalized (h applied to f
        directly), matching sample_pair's own non-bounded01 path."""
        if self.f_ecdf_ref is not None:
            normed = (self.f(z) - self.f_mean) / self.f_std
            return self.h(apply_ecdf(self.f_ecdf_ref, normed))
        return self.h(self.f(z))

    def true_value_as_b(self, z: np.ndarray) -> np.ndarray:
        """z [N,d] -> noiseless value [N] AS B WOULD OBSERVE IT (no h) --
        the same underlying f, on B's own (undistorted) scale. Comparing
        this to `true_value_as_a` at the SAME z is exactly h's effect,
        isolated from any acquisition-bias or normalization confound."""
        if self.f_ecdf_ref is not None:
            normed = (self.f(z) - self.f_mean) / self.f_std
            return apply_ecdf(self.f_ecdf_ref, normed)
        return self.f(z)


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
    beta_override: float | None = None,
    force_h_identity: bool = False,
    h_gain_range: tuple[float, float] = (0.5, 2.0),
    bounded01: bool = False,
    h_severity: float | None = None,
    h_severity_range: tuple[float, float] = (0.0, 1.0),
    return_internals: bool = False,
) -> LUPIPair | tuple[LUPIPair, LUPIPairInternals]:
    """One draw at relative-warp coefficient `rho`. `frac_near_b`: set to 0.0
    to disable spec §6.2's anti-B-blindness mechanism as an ablation (the
    "optional" preferential-near-B query sampling); `frac_uniform` +
    `frac_near_b` must be <= 1, the remainder is spec's third bucket
    (near A's own context). `beta_override`: force a fixed acquisition
    aggressiveness (e.g. 0.0 for a clean, deterministic "no acquisition
    bias" ablation) instead of `sample_beta`'s own per-draw sampling --
    added 2026-09-15 since the only prior way to get an unbiased draw was
    `sample_beta`'s own 20% chance of landing on beta=0.

    `force_h_identity`: skip `sample_monotone_map` and use `h = identity`
    (a=1, b=0, c=0) instead -- added 2026-09-17 for the same reason
    `force_rho_zero` exists (`ppfn.prior.lupi.dataset.build_training_item`):
    isolating one of the two unknowns (`T` via `rho=0`, `h` via this flag)
    to verify a single component (e.g. the position-refinement step) in
    training against ground truth without the other unknown confounding the
    result -- see docs/labbook/2026-09-16-lupi-registration-mechanism-and-
    architecture-survey.md's own recommended incremental verification path.

    `h_gain_range`: bounds `h`'s CALIBRATED amplitude relative to `f`'s own
    realized scale on this draw (`ppfn.prior.lupi.monotone.calibrate_amplitude`,
    added 2026-09-17 -- see that function's docstring for the full root-cause
    writeup and `docs/labbook/2026-09-17-lupi-h-amplitude-calibration.md`).
    `sample_monotone_map`'s raw (a,b,c) compounding has no scale control at
    all (its own docstring: "not calibrated against any target statistic"),
    which made `FullSupportBarDistribution`'s one shared, globally-fit bin
    geometry (`ppfn.model.baselines.calibration.sample_calibration_borders`)
    badly mismatched for most individual draws (measured: ~4 of 64 bins
    covering a typical draw's own value range). `h`'s SHAPE diversity (the
    raw a,c,d sampling ranges) is untouched -- only the realized
    scale/location get bounded, still genuinely random per draw within
    `h_gain_range`'s ratio (default 4x) and an f-scale-anchored `N(0,1)`
    offset, respectively. IGNORED when `bounded01=True` (below) -- superseded
    by `KumaraswamyMap`/`h_severity` there.

    `bounded01` (2026-09-17): opt-in successor to the `h_gain_range` scheme
    above -- every reported value lands in `[0,1]` BY CONSTRUCTION (the way
    `ppfn.prior.bnn.bnn_prior_vec.BNNPrior`'s own output already does), not
    just a bounded-diversity real-valued range. See
    `ppfn.prior.lupi.ecdf.normalize_f_probe`/`fit_global_f_ecdf` and
    `ppfn.prior.lupi.monotone.KumaraswamyMap` for the mechanism and why it
    closes the granularity/step-artifact problem `calibrate_amplitude` only
    partially closed (root cause: `calibrate_amplitude` bounded `h`'s
    composed SCALE but preserved its LOCATION diversity, and a fixed/shared
    bar-distribution geometry needs BOTH controlled to give every draw good
    local resolution). Both `f` and `h`'s roles change under this flag:
    `f`'s raw output is self-normalized (via ITS OWN probe, `zlat_b`, no
    oracle/cross-draw information) then mapped through a global,
    prior-design-time-fixed reference; `h` becomes a `KumaraswamyMap`
    (`[0,1]->[0,1]` exactly, any `(a,b)`) instead of `MonotoneMap`.
    `force_h_identity=True` still works under `bounded01` (gives `a=b=1`,
    the Kumaraswamy identity). `h_gain_range` is ignored in this mode.

    `h_severity`/`h_severity_range` (only used when `bounded01=True`):
    `h`'s distortion severity, DELIBERATELY INDEPENDENT of `rho` (T's own
    severity) -- both are meant to be swept independently across their full
    range for controlled experiments (matching classical HPO's ability to
    observe the complete relatedness range, and this project's general
    "sweep axes independently" discipline, e.g.
    `oracle-and-baseline-ladder.md`'s Controls section for d/cardinality/
    severity). `h_severity`: force a fixed severity in `[0,1]` (0 = exact
    identity) instead of `h_severity_range`'s own per-draw sampling --
    mirrors `beta_override`'s pattern, for a deterministic severity sweep.
    `h_severity_range` (default `(0.0, 1.0)`): the range `h_severity` is
    drawn from per draw when `h_severity` itself is `None`.

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
    # B's own sample already matches the reference measure).
    #
    # `zlat_*` naming below (not `z_*`): these are LATENT domain coordinates
    # (inputs to to_a/to_b/f/h), completely different from the `z_*` VALUE
    # fields `LUPIPair` returns (`z_b`/`z_a_ctx`/`z_a_qry`, the observed,
    # possibly-h-distorted target). An earlier version of this function used
    # the bare name `z_a_ctx`/`z_b` for BOTH, shadowing the latent with the
    # value at the point where the value was computed -- confusing to read
    # and a real risk of grabbing the wrong one in a future edit. Renamed
    # 2026-09-15 at the user's request, no behavior change from the rename
    # alone.
    zlat_b = sample_latent_B(rng, d, n_b)
    x_b = to_b(zlat_b)
    x_b_inA = to_a(zlat_b)  # B transported into A's frame -- no inversion, see field docstring

    f = sample_function_prior(rng, d, probe_z=zlat_b)
    y_b_raw = f(zlat_b)  # noiseless, B's own scale -- h is NOT applied to B

    if bounded01:
        # 2026-09-17: everything lands in [0,1] BY CONSTRUCTION, the way
        # BNNPrior's own output already does -- see
        # `ppfn.prior.lupi.ecdf.normalize_f_probe`/`fit_global_f_ecdf` and
        # `ppfn.prior.lupi.monotone.KumaraswamyMap` for the full writeup.
        # `f_mean`/`f_std` are computed PURELY from THIS draw's own probe
        # (zlat_b, already uniform over the full domain) -- self-referential,
        # no cloud-role distinction, no oracle/deployment dependency.
        y_b_clean01, f_mean, f_std = normalize_f_probe(y_b_raw)
        f_ecdf_ref = load_or_fit_f_ecdf(d)
        y_b_clean01 = apply_ecdf(f_ecdf_ref, y_b_clean01)

        severity = 0.0 if force_h_identity else (
            h_severity if h_severity is not None else float(rng.uniform(*h_severity_range))
        )
        h = sample_kumaraswamy_h(rng, severity)

        # Small, ABSOLUTE noise directly in [0,1] units -- no "relative to
        # std(f)" dance needed anymore, since both clouds' clean values are
        # already unit-scaled by construction.
        obs_noise_b = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))
        y_b_obs = np.clip(y_b_clean01 + rng.normal(0.0, obs_noise_b, size=n_b), 1e-4, 1.0 - 1e-4)
        y_b_obs_inA = np.clip(h(y_b_obs), 1e-4, 1.0 - 1e-4)

        sigma_obs_a = float(np.exp(rng.uniform(np.log(0.005), np.log(0.05))))

        beta = sample_beta(rng) if beta_override is None else float(beta_override)
        zlat_a_ctx = sample_latent_A_acquired(rng, region, d, n_a, score_fn=f, beta=beta)
        x_a_ctx = to_a(zlat_a_ctx)
        y_a_ctx_norm = (f(zlat_a_ctx) - f_mean) / f_std
        y_a_ctx_clean = h(apply_ecdf(f_ecdf_ref, y_a_ctx_norm))
        y_a_ctx_obs = np.clip(y_a_ctx_clean + rng.normal(0.0, sigma_obs_a, size=n_a), 1e-4, 1.0 - 1e-4)
        oracle_bpos_a_ctx = to_b(zlat_a_ctx)

        zlat_qry, qry_source = _sample_query_z(
            rng, d, n_qry, zlat_b, zlat_a_ctx, frac_uniform, frac_near_b, query_eps_std
        )
        x_a_qry = to_a(zlat_qry)
        y_qry_norm = (f(zlat_qry) - f_mean) / f_std
        y_qry_clean = h(apply_ecdf(f_ecdf_ref, y_qry_norm))
        y_qry_obs = np.clip(y_qry_clean + rng.normal(0.0, sigma_obs_a, size=n_qry), 1e-4, 1.0 - 1e-4)
        oracle_bpos_a_qry = to_b(zlat_qry)
    else:
        y_b_clean = y_b_raw
        # h's amplitude is CALIBRATED against y_b_clean (B's own noiseless probe,
        # always available regardless of encoder/decoder role -- not privileged-
        # cloud leakage) so its composed scale/location stay a BOUNDED multiple
        # of f's own realized scale on THIS draw, rather than sample_monotone_map's
        # raw, uncontrolled compounding -- see calibrate_amplitude's docstring.
        if force_h_identity:
            h = MonotoneMap(a=1.0, b=0.0, c=0.0, d=1.0)
        else:
            h = calibrate_amplitude(sample_monotone_map(rng), y_b_clean, rng, gain_range=h_gain_range)

        y_b_obs = y_b_clean + rng.normal(0.0, f.sigma_obs, size=n_b)
        y_b_obs_inA = h(y_b_obs)  # B's observation recalibrated into A's scale -- the "fully registered" oracle value, see LUPIPair.z_b_inA's docstring

        # Value normalization SHELVED (2026-09-15, at the user's direction) --
        # both the original per-cloud ECDF quantile-normalization AND its
        # replacement (z-scoring against B's own (mean, std)) are gone. Two
        # reasons, not just one:
        #   1. Same as before: A's own ECDF couldn't self-diagnose its own
        #      acquisition bias (spec §3.2(b)'s pathology).
        #   2. Newly identified: standardizing against B's reference is itself
        #      a form of privileged leakage -- the STUDENT model only ever sees
        #      B's NOISY observations (enc_z), never B's true (y_mean, y_std),
        #      so handing the prior's own oracle statistics to both clouds
        #      quietly pre-solves part of the calibration problem the
        #      experiment exists to test. A real deployment wouldn't have that
        #      reference either.
        # `z_b`/`z_a_ctx`/`z_a_qry` (the LUPIPair fields, not the zlat_*
        # latents above) are now simply the raw observed values -- unnormalized,
        # scale set entirely by `f`'s and `h`'s own sampled parameters, nothing
        # divided out. This also DIRECTLY addresses the extreme (~20) z-scores
        # observed under the z-scoring version: those were partly an artifact
        # of dividing by a possibly-small estimated std, not just h's own
        # nonlinearity -- removing the division should narrow the realized
        # range, not widen it. Revisit normalization later if the model
        # struggles with cross-item scale variation (each draw's f/h are
        # independent, so raw scale genuinely differs draw to draw) -- the
        # fixed-bin bar-distribution head has no adaptive rescaling of its own.
        #
        # `bounded01=True` (2026-09-17) is the successor that DOES normalize
        # -- but self-referentially (each draw's own probe / a global,
        # prior-design-time reference), never against a privileged cloud's
        # true statistics. See the `bounded01` branch above.

        # A's own noise scale, resolved against std(h(f(.))) over B's (full-domain)
        # probe -- same "relative to std(f) over the domain" convention
        # `function_prior.sample_function_prior` uses for f itself.
        hf_probe = h(y_b_clean)
        std_hf = max(float(hf_probe.std()), 1e-6)
        sigma_obs_a = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))) * std_hf

        # beta_override: force a fixed acquisition aggressiveness instead of
        # sampling one -- e.g. 0.0 for a clean "no acquisition bias" ablation,
        # rather than relying on sample_beta's own 20% chance of beta=0. None
        # (default) preserves the existing sampled-beta behavior exactly.
        beta = sample_beta(rng) if beta_override is None else float(beta_override)
        zlat_a_ctx = sample_latent_A_acquired(rng, region, d, n_a, score_fn=f, beta=beta)
        x_a_ctx = to_a(zlat_a_ctx)
        y_a_ctx_clean = h(f(zlat_a_ctx))
        y_a_ctx_obs = y_a_ctx_clean + rng.normal(0.0, sigma_obs_a, size=n_a)
        oracle_bpos_a_ctx = to_b(zlat_a_ctx)

        zlat_qry, qry_source = _sample_query_z(
            rng, d, n_qry, zlat_b, zlat_a_ctx, frac_uniform, frac_near_b, query_eps_std
        )
        x_a_qry = to_a(zlat_qry)
        y_qry_clean = h(f(zlat_qry))
        y_qry_obs = y_qry_clean + rng.normal(0.0, sigma_obs_a, size=n_qry)
        oracle_bpos_a_qry = to_b(zlat_qry)

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
        z_b=y_b_obs,
        x_b_inA=x_b_inA,
        z_b_inA=y_b_obs_inA,
        x_a_ctx=x_a_ctx,
        z_a_ctx=y_a_ctx_obs,
        oracle_bpos_a_ctx=oracle_bpos_a_ctx,
        x_a_qry=x_a_qry,
        z_a_qry=y_qry_obs,
        oracle_bpos_a_qry=oracle_bpos_a_qry,
        qry_source=qry_source,
        meta=meta,
    )
    if not return_internals:
        return pair
    internals = LUPIPairInternals(
        to_a=to_a, to_b=to_b, f=f, h=h,
        z_a_ctx=zlat_a_ctx, z_b=zlat_b,
        f_mean=f_mean if bounded01 else None,
        f_std=f_std if bounded01 else None,
        f_ecdf_ref=f_ecdf_ref if bounded01 else None,
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
    ax.set_ylabel("standardized value (z-score)")
    ax.legend()
    ax.set_title(f"LUPIPair query mixture, d=1, rho=0.6, beta={pair.beta:.2f}")
    fig.tight_layout()
    plt.show()
