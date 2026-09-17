"""Per-pair quantile normalization -- build spec §3: map both tasks' values
through their own CDF (each task's own sample, option (b) of §3.2 for BOTH
tasks -- see module docstring below for why a shared choice is fine for B).

B (spec §3.1, "no problem"): B's design z_B ~ Uniform([0,1]^d)
(`ppfn.prior.registration.region.sample_latent_B`) already matches the
reference measure this project's other invariant #8 insists on (the
DECLARED domain, not an empirical/design-dependent one) -- B's own n_B
sample IS already an honest Monte-Carlo draw from the reference measure, so
no separate dense grid is needed to reach spec's "surrogate posterior means
on a uniform grid" (a real surrogate-fitting step this project doesn't need,
since the true f is available). CDF is built from the NOISELESS f(z_B)
(denoised, ties preserved, per spec/labbook's warning against fitting
quantiles from raw noisy observations), evaluated separately from the noisy
y_B actually used as B's token value.

A (spec §3.2(b), "the real problem"): NO such shortcut applies -- A's design
is acquisition-biased BY CONSTRUCTION (`acquisition.py`), so its own sample
is a biased draw from the reference measure. F_hat_A is built anyway, from
exactly the noisy, biased observations A actually has (spec: "compute F_hat_A
from those biased observations exactly as at deployment") -- the resulting
normalization error IS the pathology the model has to learn to correct
in-context, not something to paper over here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_F_ECDF_CACHE_DIR = Path(__file__).parent / "_ecdf_cache"


def fit_ecdf(values: np.ndarray) -> np.ndarray:
    """values: [N] -> sorted copy, used by `apply_ecdf` below. Kept as its
    own function (rather than inlining `np.sort`) so both call sites read
    the same way and so a future replacement (e.g. a smoothed/kernel CDF)
    has one place to change."""
    return np.sort(values)


def apply_ecdf(sorted_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """F_hat(values) under the empirical CDF of `sorted_ref`, via the
    Hazen/"rank over n+1" plotting-position convention (`(rank) / (n+1)`,
    rank = count of ref values <= v) -- keeps the output strictly inside
    (0, 1) even when `values` reproduces `sorted_ref`'s own extremes, which
    matters here because those become inputs to a BOUNDED (see CLAUDE.md-
    adjacent `BarDistribution` in `ppfn.model.pfn.bar_distribution`) [0,1]
    predictive head -- landing exactly on the boundary would fall into
    `map_to_bucket_idx`'s "== borders[0]" special case for spurious reasons
    rather than a genuine density statement."""
    n = sorted_ref.shape[0]
    rank = np.searchsorted(sorted_ref, values, side="right")
    return rank / (n + 1)


def normalize_f_probe(raw_probe_vals: np.ndarray) -> tuple[np.ndarray, float, float]:
    """z-score `raw_probe_vals` (one draw's `f(z)` over ITS OWN dense probe)
    by that SAME probe's own mean/std -- self-referential, computed purely
    from the just-sampled function's own realized behavior, no cloud-role
    distinction and no cross-draw information at all. This is the missing
    piece plain pooling-then-ECDF needed: measured directly (2026-09-17),
    `ppfn.prior.registration.function_prior.sample_function_prior`'s raw
    output has WILD per-draw location/scale diversity on its own (50 draws,
    d=1: per-draw mean ranged -3.96 to +6.05, per-draw std ranged 193x) --
    even worse than the old, already-rejected `h` family's diversity. A
    GLOBAL pooled reference fit from RAW (un-normalized) `f` draws inherits
    that heterogeneity and gives any individual fresh draw poor quantile
    spread (checked: one fresh draw's own quantiles collapsed to
    `[0.67, 0.70]` against such a reference -- the exact failure mode
    `calibrate_amplitude` fixed for `h`, reappearing one layer down in `f`
    itself). z-scoring EACH draw first, using nothing but that draw's own
    probe, removes exactly that heterogeneity before pooling -- verified:
    fresh draws against a reference built from z-scored pooled draws spread
    `[0.01, 0.85]`-`[0.19, 0.99]` (span 0.80-0.98) instead of collapsing.
    Returns `(z_scored_vals, probe_mean, probe_std)` -- the mean/std are
    returned so the SAME affine transform can be applied to OTHER points
    evaluated from the SAME function (e.g. A's context, a plotting grid),
    not just the probe itself."""
    mean = float(raw_probe_vals.mean())
    std = max(float(raw_probe_vals.std()), 1e-6)
    return (raw_probe_vals - mean) / std, mean, std


def fit_global_f_ecdf(
    d: int, n_draws: int = 50, samples_per_draw: int = 300, seed: int = 0, n_ref_samples: int = 2000
) -> np.ndarray:
    """Pools `n_draws` INDEPENDENT `FunctionPrior` draws' noiseless `f(z)`,
    EACH FIRST SELF-NORMALIZED via `normalize_f_probe` (see that function's
    docstring for why the self-normalization step is required, not
    optional), over a fresh `z ~ Uniform([0,1]^d)` probe each, decimates the
    pooled, sorted sample to `n_ref_samples` points -- the value-axis analog
    of `ppfn.prior.bnn.bnn_prior_vec.BNNPrior`'s own global `_ecdf_cache`
    (same construction: fit ONCE per prior-family config, reused identically
    by every future draw). This is a PRIOR-DESIGN constant, computed
    offline before any specific task is drawn -- NOT a per-task/per-draw
    oracle probe (that idea was tried and rejected, see
    `docs/labbook/2026-09-17-lupi-h-amplitude-calibration.md` and
    `notebooks/id_token_lupi_1d_comparison.ipynb`'s section on why a
    per-draw reference-CDF doesn't survive deployment) -- so it carries no
    deployment-time dependency on knowing any specific new task's true
    generator, same status as `pfn_variable_dim5_long`'s own ECDF cache
    already has.

    Composing this with `KumaraswamyMap` (`ppfn.prior.lupi.monotone`) is
    what gets LUPI's values onto `[0,1]` the way `BNNPrior`'s own output
    already is: self-normalize a fresh draw's `f` via ITS OWN probe
    (`normalize_f_probe`), then `apply_ecdf(fit_global_f_ecdf(d), ...)` on
    the result -- gives B's own frame a comparable, homogeneous `[0,1]`
    scale draw-to-draw, and a `KumaraswamyMap` composed on top keeps A's
    frame in `[0,1]` too, always, by construction."""
    from ppfn.prior.registration.function_prior import sample_function_prior

    rng = np.random.default_rng(seed)
    pooled = []
    for _ in range(n_draws):
        probe_z = rng.uniform(0.0, 1.0, size=(samples_per_draw, d))
        f = sample_function_prior(rng, d, probe_z=probe_z)
        normalized, _, _ = normalize_f_probe(f(probe_z))
        pooled.append(normalized)
    pooled_arr = np.concatenate(pooled)
    pooled_arr.sort()
    idx = np.linspace(0, len(pooled_arr) - 1, n_ref_samples).astype(int)
    return pooled_arr[idx]


def _f_ecdf_cache_path(d: int, n_draws: int, samples_per_draw: int, seed: int, n_ref_samples: int, cache_dir: Path) -> Path:
    digest = hashlib.sha1(f"{d}_{n_draws}_{samples_per_draw}_{seed}_{n_ref_samples}".encode()).hexdigest()[:16]
    return cache_dir / f"lupi_f_ecdf_d{d}_{digest}.npy"


def load_or_fit_f_ecdf(
    d: int,
    n_draws: int = 50,
    samples_per_draw: int = 300,
    seed: int = 0,
    n_ref_samples: int = 2000,
    cache_dir: str | Path | None = DEFAULT_F_ECDF_CACHE_DIR,
) -> np.ndarray:
    """Disk-cached wrapper around `fit_global_f_ecdf` -- same one-time-cost-
    per-config discipline as `ppfn.prior.bnn.bnn_prior_vec.BNNPrior`'s own
    `_load_or_fit_ecdf`. `cache_dir=None` skips the cache (always refits;
    useful for a notebook exploring different `(n_draws, seed)` settings
    without stale files accumulating)."""
    if cache_dir is None:
        return fit_global_f_ecdf(d, n_draws, samples_per_draw, seed, n_ref_samples)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _f_ecdf_cache_path(d, n_draws, samples_per_draw, seed, n_ref_samples, cache_dir)
    if path.exists():
        return np.load(path)
    ref = fit_global_f_ecdf(d, n_draws, samples_per_draw, seed, n_ref_samples)
    np.save(path, ref)
    return ref


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: fit an ECDF on a
    known distribution and confirm apply_ecdf recovers approximately uniform
    quantiles, plus the (0,1)-open-interval guarantee at the sample's own
    extremes."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, size=2000)
    sorted_ref = fit_ecdf(ref)

    probe = rng.normal(0.0, 1.0, size=5000)
    q = apply_ecdf(sorted_ref, probe)
    print("quantile mean (expect ~0.5):", q.mean())
    print("quantile std (expect ~0.289, Uniform(0,1)'s):", q.std())
    print("min/max quantile (expect strictly inside (0,1)):", q.min(), q.max())

    at_extremes = apply_ecdf(sorted_ref, np.array([sorted_ref.min(), sorted_ref.max()]))
    print("quantile AT the ref sample's own min/max (expect < 1, > 0):", at_extremes)

    print("\nfit_global_f_ecdf / load_or_fit_f_ecdf: cached global reference for f(z), d=1")
    import time

    ref_d1 = load_or_fit_f_ecdf(d=1, cache_dir=None)
    print(f"reference size: {ref_d1.shape}, range [{ref_d1.min():.3f}, {ref_d1.max():.3f}]")

    t0 = time.time()
    _ = load_or_fit_f_ecdf(d=1)  # first call: fits + caches to disk
    t_fit = time.time() - t0
    t0 = time.time()
    _ = load_or_fit_f_ecdf(d=1)  # second call: should hit the cache
    t_cached = time.time() - t0
    print(f"fit+cache: {t_fit:.2f}s, cached reload: {t_cached:.4f}s")

    from ppfn.prior.registration.function_prior import sample_function_prior

    rng = np.random.default_rng(99)
    print("\nSix FRESH draws (not in the reference), self-normalized then ECDF-transformed:")
    for trial in range(6):
        probe_z = rng.uniform(0.0, 1.0, size=(400, 1))
        f_new = sample_function_prior(rng, d=1, probe_z=probe_z)
        normalized, _, _ = normalize_f_probe(f_new(probe_z))
        q_new = apply_ecdf(ref_d1, normalized)
        print(f"  draw {trial}: min={q_new.min():.4f} max={q_new.max():.4f} mean={q_new.mean():.4f} "
              f"span={q_new.max() - q_new.min():.4f} (expect span >> 0, spread across most of (0,1))")
