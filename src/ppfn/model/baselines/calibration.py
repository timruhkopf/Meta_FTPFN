"""One-time calibration sample for `FullSupportBarDistribution`'s
quantile-fit bin borders (`ppfn.model.pfn.bar_distribution`).

`ppfn.prior.lupi`'s targets are raw and unnormalized (per-pair
normalization was shelved -- see `ppfn.prior.lupi.sampler`'s own comment),
so there's no fixed, known scale to hand a fixed-range bar distribution
(`ppfn.model.registration.heads.TailBarDistribution`'s own `[-4, 4]` body
turned out to be a poor match: most drawn pairs' informative range covers
only a handful of that head's 64 bins). Quantile-fit borders sidestep the
"pick a range" problem entirely by fitting to an actual sample of the
targets these baseline models are scored against -- A's own query/context
values (`z_a_qry`/`z_a_ctx`, i.e. `h(f(z))`), never B's raw `z_b` (nothing
in `ppfn.model.baselines` ever scores a loss against B's values).

Fixed seed -> reproducible borders across every call with the same
`(n_bins, n_draws, seed)`, so two checkpoints built from separate calls
(e.g. `IDTokenPFN` and `AAlonePlainPFN`, trained as separate Hydra runs)
still share one bar-distribution geometry and stay directly comparable.
"""

from __future__ import annotations

import numpy as np
import torch

from ppfn.model.pfn.bar_distribution import quantile_bin_borders, uniform_bin_borders
from ppfn.prior.lupi.sampler import D_CHOICES, sample_pair


def sample_calibration_borders(n_bins: int, n_draws: int = 80, seed: int = 0, bounded01: bool = False) -> torch.Tensor:
    """`bounded01=True` (2026-09-17, see `docs/labbook/2026-09-17-lupi-bounded01-prior.md`):
    under `sample_pair(..., bounded01=True)`, every reported value already
    lands in `[0,1]` BY CONSTRUCTION (self-normalized `f` + a global,
    prior-design-time-fixed reference + a `KumaraswamyMap` `h`) -- there is
    no per-draw scale/location diversity left for a data-driven quantile fit
    to correct for, so this just returns PLAIN uniform `[0,1]` bins
    (`uniform_bin_borders`), matching `ppfn.prior.bnn.bnn_prior_vec.BNNPrior`'s
    own convention exactly. Skips sampling any calibration draws at all --
    the bins are a closed-form function of `n_bins` alone in this mode."""
    if bounded01:
        return uniform_bin_borders(n_bins, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_draws):
        d = int(rng.choice(D_CHOICES))
        pair = sample_pair(
            rng, rho=float(rng.uniform(0.0, 1.0)), d=d,
            n_a_range=(8, 100), n_b_range=(8, 100), n_qry_range=(16, 16),
            # grid_n=1 is invalid for actual training/registration (the
            # rejection-sampling check it drives determines whether a warp's
            # transport target is trustworthy) but harmless here: this
            # function only ever reads z_a_ctx/z_a_qry (VALUES, from f/h
            # applied to latents), never a warped x-position, so the
            # expensive part of sample_pair (the O(grid_n^d) Jacobian-band
            # check, ~80-97% of its cost per docs/labbook/2026-09-14-...)
            # buys nothing here. Cuts calibration from minutes to seconds.
            warp_grid_n=1,
        )
        zs.append(pair.z_a_ctx)
        zs.append(pair.z_a_qry)
    ys = torch.from_numpy(np.concatenate(zs).astype(np.float32))
    return quantile_bin_borders(ys, n_bins)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: fit borders, confirm
    they're reproducible and dense where the calibration sample actually
    lives (not spread uniformly over some assumed range)."""
    borders_a = sample_calibration_borders(n_bins=64, n_draws=50, seed=0)
    borders_b = sample_calibration_borders(n_bins=64, n_draws=50, seed=0)
    print("borders shape:", tuple(borders_a.shape), " range:", borders_a[0].item(), "to", borders_a[-1].item())
    print("reproducible across calls with the same seed (expect 0.0):", (borders_a - borders_b).abs().max().item())
    widths = borders_a[1:] - borders_a[:-1]
    print("bucket widths -- min/median/max:", widths.min().item(), widths.median().item(), widths.max().item())
    print("(expect widths to vary a lot -- narrow where the sample is dense, wide out at the edges)")
