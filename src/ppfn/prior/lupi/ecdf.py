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

import numpy as np


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
