"""Support restriction R_A and the mixture-of-designs sampler for z^A —
ARCHITECTURE.md §1.2. This is what produces "A covers a strict subregion of
B's support, with a non-uniform density inside it" without simulating an
optimizer.

Every region type is expressed as an indicator function over [0,1]^d and
sampled by rejection against a uniform-in-cube proposal. This is simple and
exactly correct for all four region shapes at once, at the cost of some
wasted proposals for small-volume regions -- acceptable here since regions
are drawn once per prior sample, not in a hot loop, and the volume floor
(subbox >= 0.15, ball radius >= 0.2) keeps the expected number of proposals
per accepted point bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Region:
    kind: str  # "full" | "subbox" | "ball" | "blobs"
    params: dict = field(default_factory=dict)

    def indicator(self, points: np.ndarray) -> np.ndarray:
        """points: [N, d] -> bool [N]."""
        if self.kind == "full":
            return np.ones(points.shape[0], dtype=bool)
        if self.kind == "subbox":
            lo, hi = self.params["lo"], self.params["hi"]
            return np.all((points >= lo) & (points <= hi), axis=-1)
        if self.kind == "ball":
            center, radius = self.params["center"], self.params["radius"]
            return np.linalg.norm(points - center, axis=-1) <= radius
        if self.kind == "blobs":
            inside = np.zeros(points.shape[0], dtype=bool)
            for center, radius in zip(self.params["centers"], self.params["radii"]):
                inside |= np.linalg.norm(points - center, axis=-1) <= radius
            return inside
        raise ValueError(f"unknown region kind {self.kind!r}")


def sample_region(rng: np.random.Generator, d: int) -> Region:
    """R_A ~ { full 0.40, subbox 0.30, ball 0.20, blobs 0.10 } — ARCHITECTURE.md §1.2."""
    u = rng.random()
    if u < 0.40:
        return Region("full", {})

    if u < 0.70:
        # axis-aligned sub-box, volume fraction ~ U[0.15, 0.7]. Isotropic side
        # length vf**(1/d) as a base, jittered per-axis (multiplicatively) and
        # renormalized so the achieved volume matches the target exactly --
        # this keeps sub-boxes from always being perfect cubes while still
        # hitting the sampled volume fraction on the nose.
        vf = rng.uniform(0.15, 0.7)
        base_side = vf ** (1.0 / d)
        jitter = np.exp(rng.uniform(-0.3, 0.3, size=d))
        sides = base_side * jitter
        sides *= (vf / np.prod(sides)) ** (1.0 / d)  # renormalize to hit vf exactly
        sides = np.clip(sides, 1e-3, 1.0)
        lo = rng.uniform(0.0, 1.0 - sides)
        hi = lo + sides
        return Region("subbox", {"lo": lo, "hi": hi, "target_volume_fraction": vf})

    if u < 0.90:
        radius = rng.uniform(0.2, 0.6)
        center = rng.uniform(radius * 0.3, 1.0 - radius * 0.3, size=d)
        return Region("ball", {"center": center, "radius": radius})

    # union of 2-3 blobs
    n_blobs = int(rng.integers(2, 4))
    radii = rng.uniform(0.15, 0.35, size=n_blobs)
    centers = rng.uniform(radii[:, None] * 0.3, 1.0 - radii[:, None] * 0.3)
    return Region("blobs", {"centers": centers, "radii": radii})


def _rejection_sample_uniform(
    rng: np.random.Generator, region: Region, n: int, d: int, max_rounds: int = 200
) -> np.ndarray:
    """Uniform samples within `region`, via rejection against a uniform-in-cube
    proposal. Batches proposals so small-volume regions don't cost a Python
    loop iteration per accepted point."""
    accepted = np.empty((0, d))
    batch = max(256, n * 8)
    for _ in range(max_rounds):
        cand = rng.uniform(0.0, 1.0, size=(batch, d))
        keep = cand[region.indicator(cand)]
        accepted = np.concatenate([accepted, keep], axis=0)
        if accepted.shape[0] >= n:
            return accepted[:n]
        batch *= 2
    raise RuntimeError(
        f"could not sample {n} points inside region {region.kind} after {max_rounds} rounds"
    )


def estimate_volume_fraction(
    rng: np.random.Generator, region: Region, d: int, n_mc: int = 20000
) -> float:
    """Monte Carlo estimate of |R_A| / |[0,1]^d| -- the *realized* fraction
    (ARCHITECTURE.md §1.2: "log the realized region type, volume fraction..."),
    not the nominal target used to parameterize subbox/ball draws."""
    if region.kind == "full":
        return 1.0
    cand = rng.uniform(0.0, 1.0, size=(n_mc, d))
    return float(region.indicator(cand).mean())


def _sample_mixture_in_region(
    rng: np.random.Generator, region: Region, n: int, d: int
) -> np.ndarray:
    """z ~ mixture of uniform and clustered (2-5 Gaussian blobs, bandwidth ~
    U[0.03, 0.15]) within `region` -- the sampling body of `sample_latent_A`,
    factored out so the demo below can also exercise it against a
    caller-chosen region kind instead of only the 0.40/0.30/0.20/0.10 draw.

    The mixture weight between the uniform and clustered components, and the
    number of blobs within {2,...,5}, are not pinned down by the spec beyond
    "a mixture" -- sampled fresh per call (p_cluster ~ U[0.3, 0.9], n_blobs ~
    {2,...,5}) so both extremes (near-uniform, heavily clustered) occur
    across the prior rather than picking one fixed weight.
    """
    p_cluster = rng.uniform(0.3, 0.9)
    n_blobs = int(rng.integers(2, 6))
    bandwidth = rng.uniform(0.03, 0.15)

    blob_centers = _rejection_sample_uniform(rng, region, n_blobs, d)

    z = np.empty((n, d))
    n_filled = 0
    while n_filled < n:
        remaining = n - n_filled
        is_clustered = rng.random(remaining) < p_cluster
        n_clust = int(is_clustered.sum())
        n_unif = remaining - n_clust

        if n_unif > 0:
            unif_pts = _rejection_sample_uniform(rng, region, n_unif, d)
        else:
            unif_pts = np.empty((0, d))

        if n_clust > 0:
            blob_idx = rng.integers(0, n_blobs, size=n_clust)
            raw = blob_centers[blob_idx] + rng.normal(0.0, bandwidth, size=(n_clust, d))
            in_cube = np.all((raw >= 0.0) & (raw <= 1.0), axis=-1)
            in_region = region.indicator(raw)
            clust_pts = raw[in_cube & in_region]
        else:
            clust_pts = np.empty((0, d))

        new_pts = np.concatenate([unif_pts, clust_pts], axis=0)
        take = min(new_pts.shape[0], remaining)
        z[n_filled : n_filled + take] = new_pts[:take]
        n_filled += take

    return z


def sample_latent_A(
    rng: np.random.Generator, d: int, n_a: int
) -> tuple[np.ndarray, Region, float]:
    """z^A ~ mixture of uniform and clustered points within R_A --
    ARCHITECTURE.md §1.2. Returns (z_A [n_a, d], region, realized_volume_fraction)."""
    region = sample_region(rng, d)
    z = _sample_mixture_in_region(rng, region, n_a, d)
    volume_fraction = estimate_volume_fraction(rng, region, d)
    return z, region, volume_fraction


def sample_latent_B(rng: np.random.Generator, d: int, n_b: int) -> np.ndarray:
    """z^B ~ p_z over [0,1]^d -- taken as plain uniform (the spec leaves p_z
    unspecified beyond "over [0,1]^d"; B is the abundant, undesigned cloud,
    so uniform is the natural default and keeps B's marginal geometry the
    honest baseline against which A's restriction is contrasted)."""
    return rng.uniform(0.0, 1.0, size=(n_b, d))


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: sample z_A/z_B for
    each region type at d=2 and scatter them, so the four shapes and the
    uniform/clustered mixture are eyeballable at a glance."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    d = 2
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    z_b = sample_latent_B(rng, d, 300)
    axes[0].scatter(z_b[:, 0], z_b[:, 1], s=8, alpha=0.6, color="#1a1a1a")
    axes[0].set_title("z^B (uniform, full domain)")

    # Force each region kind once for the demo rather than relying on the
    # 0.40/0.30/0.20/0.10 draw to hit all four in one run.
    forced_kinds = ["full", "subbox", "ball", "blobs"]
    for ax, kind in zip(axes[1:], forced_kinds):
        region = None
        for _ in range(2000):
            candidate = sample_region(rng, d)
            if candidate.kind == kind:
                region = candidate
                break
        assert region is not None, f"couldn't sample a {kind} region in 2000 tries"

        z = _sample_mixture_in_region(rng, region, 200, d)
        vf = estimate_volume_fraction(rng, region, d)
        ax.scatter(z[:, 0], z[:, 1], s=8, alpha=0.6, color="#D55E00")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{kind}, vol_frac~{vf:.2f}")

    fig.suptitle(
        "z^B (uniform) vs. z^A under each region type (uniform/clustered mixture)"
    )
    fig.tight_layout()
    plt.show()
