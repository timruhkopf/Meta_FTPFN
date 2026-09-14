"""Simulated acquisition for A's design -- build spec §3.2(b): "Generate A's
context via a simulated acquisition process, compute F_hat_A from those
biased observations exactly as at deployment." A deliberately cheap
stand-in for a real EI-on-GP loop (spec's own "reference acquisition"
suggestion), not a BO simulator: draw a candidate pool uniformly within A's
support region, score it with the (noiseless) latent function, and sample
`n_a` of them WITHOUT replacement, weighted toward low scores (this project
minimizes, `docs/labbook/2026-09-14-problem-setting-h-and-T-identifiability.md`).
`beta=0` reduces exactly to uniform-in-region sampling (no bias, matches
`ppfn.prior.registration.region.sample_latent_A`'s uniform component);
`beta` sampled per draw so the prior covers both "early BO" (small bias) and
"later BO" (aggressive clustering on the incumbent) regimes -- spec §3.2's
"two opposite biases" -- without simulating either one explicitly.

Reuses `ppfn.prior.registration.region`'s region machinery (Region,
sample_region, the private rejection sampler) rather than duplicating it --
the support-restriction invariant (CLAUDE.md invariant #7) is exactly the
same requirement here.
"""

from __future__ import annotations

import numpy as np

from ppfn.prior.registration.region import Region, _rejection_sample_uniform


def sample_beta(rng: np.random.Generator) -> float:
    """Acquisition aggressiveness. 20% chance of beta=0 (uniform design,
    keeps that regime in the training distribution per spec §6.1's
    "mixture of uniform ... and replayed/simulated BO trajectories");
    otherwise LogUniform[0.5, 8] -- 0.5 is mild (barely distinguishable from
    uniform over a modest pool), 8 concentrates almost all mass on the pool's
    best few points."""
    if rng.random() < 0.2:
        return 0.0
    return float(np.exp(rng.uniform(np.log(0.5), np.log(8.0))))


def sample_latent_A_acquired(
    rng: np.random.Generator,
    region: Region,
    d: int,
    n_a: int,
    score_fn,
    beta: float,
    pool_mult: int = 20,
) -> np.ndarray:
    """z_A ~ acquisition-biased sampling within `region` -- pool of
    `pool_mult * n_a` uniform candidates (>= n_a always, so `n_a` distinct
    points can always be drawn without replacement), ranked by `score_fn`
    (ascending = best first, since lower is better), sampled without
    replacement with weight `exp(-beta * rank / pool_size)`. Returns
    [n_a, d]."""
    pool_n = max(pool_mult * n_a, n_a)
    pool = _rejection_sample_uniform(rng, region, pool_n, d)
    scores = score_fn(pool)
    rank = np.argsort(np.argsort(scores))  # 0 = best (lowest score)

    if beta <= 0.0:
        weights = np.ones(pool_n)
    else:
        logits = -beta * rank / pool_n
        logits -= logits.max()  # numerically stable softmax
        weights = np.exp(logits)
    probs = weights / weights.sum()

    idx = rng.choice(pool_n, size=n_a, replace=False, p=probs)
    return pool[idx]


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: for a 1D bowl score
    function, plot the acquired sample's histogram at a few beta values --
    beta=0 should look uniform over the region, large beta should visibly
    cluster near the score function's minimum."""
    import matplotlib.pyplot as plt

    from ppfn.prior.registration.region import sample_region

    rng = np.random.default_rng(0)
    d = 1
    region = sample_region(rng, d)
    while region.kind != "full":  # force the simplest case for the demo
        region = sample_region(rng, d)

    def bowl(x: np.ndarray) -> np.ndarray:
        return ((x[:, 0] - 0.3) ** 2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    for ax, beta in zip(axes, [0.0, 1.0, 4.0, 8.0]):
        z = sample_latent_A_acquired(rng, region, d, n_a=300, score_fn=bowl, beta=beta)
        ax.hist(z[:, 0], bins=30, range=(0, 1), color="#D55E00", alpha=0.8)
        ax.axvline(0.3, color="#1a1a1a", lw=1, ls="--")
        ax.set_title(f"beta={beta}")
    fig.suptitle("sample_latent_A_acquired -- bias toward the bowl's minimum (x=0.3) as beta grows")
    fig.tight_layout()
    plt.show()
