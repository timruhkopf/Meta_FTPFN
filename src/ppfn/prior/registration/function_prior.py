"""The function prior f, defined on the LATENT frame z -- ARCHITECTURE.md §1.4.

`y = f(z) + eps` is evaluated on z, never on a warped cloud's own
coordinates (see ARCHITECTURE.md §1.1's "Why f is defined on z" and
`docs/decisions.md` D3) -- this is what keeps both A's and E's observations
statistically symmetric (both are "a warped-input BNN of z"), which
role-swapping (ARCHITECTURE.md §4.3) depends on.

Plain NumPy MLP, sampled fresh per prior draw. No relation to
`ppfn.prior.bnn.bnn_prior.BNNPrior` (a different, ECDF-cache-based sampler
for an unrelated, single-cloud PFN prior) -- see this package's `__init__.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0.0, x),
    "gelu": lambda x: (
        0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    ),
}


@dataclass
class FunctionPrior:
    """A sampled MLP f: R^d -> R, plus the observation-noise scale sigma_obs."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activation: str
    sigma_obs: float  # absolute scale, already resolved against std(f) -- see `sample_function_prior`

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """z: [N, d] -> f(z): [N] (noiseless)."""
        act = _ACTIVATIONS[self.activation]
        h = z
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = act(h @ w + b)
        h = h @ self.weights[-1] + self.biases[-1]
        return h[:, 0]

    def sample_y(self, rng: np.random.Generator, z: np.ndarray) -> np.ndarray:
        """f(z) + eps, eps ~ N(0, sigma_obs^2) -- ARCHITECTURE.md §1.1."""
        return self(z) + rng.normal(0.0, self.sigma_obs, size=z.shape[0])


def sample_function_prior(
    rng: np.random.Generator, d: int, probe_z: np.ndarray
) -> FunctionPrior:
    """ARCHITECTURE.md §1.4: depth ~ Uniform{1,2,3}, width ~ LogUniform[16,128],
    act ~ Uniform{tanh,relu,gelu}, scale ~ LogUniform[0.5,2.0] (weight-std
    multiplier), sigma_obs ~ LogUniform[0.01,0.3] (relative to std(f) over
    the domain).

    `probe_z` [P, d]: points used to estimate std(f) so sigma_obs can be
    resolved from "relative to std of f over the domain" into an absolute
    scale (§1.4) -- pass the pool of z^A and z^B actually being generated
    for this pair, so no extra grid evaluation is needed.

    Weight init: W ~ N(0, (scale / sqrt(fan_in))^2), a standard variance-
    scaling init multiplied by the sampled `scale` -- the spec names `scale`
    as "weight std multiplier" but doesn't pin down a base scheme, and
    fan-in scaling is the natural default that keeps signal magnitude
    roughly stable across the sampled depth/width range.
    """
    depth = int(rng.integers(1, 4))  # Uniform{1,2,3} hidden layers
    width = int(np.exp(rng.uniform(np.log(16), np.log(128))))
    activation = rng.choice(["tanh", "relu", "gelu"])
    scale = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))

    dims = [d] + [width] * depth + [1]
    weights, biases = [], []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        std = scale / np.sqrt(fan_in)
        weights.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
        biases.append(rng.normal(0.0, std, size=(fan_out,)))

    prior = FunctionPrior(
        weights=weights, biases=biases, activation=str(activation), sigma_obs=0.0
    )
    f_probe = prior(probe_z)
    std_f = max(float(f_probe.std()), 1e-6)
    sigma_obs_relative = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
    prior.sigma_obs = sigma_obs_relative * std_f
    return prior


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: several independent
    d=1 draws overlaid, to eyeball the diversity of sampled ground-truth
    functions (depth/width/activation/scale all varying per draw)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(4)
    grid = np.linspace(0.0, 1.0, 300)[:, None]

    fig, ax = plt.subplots(figsize=(7, 5))
    for _ in range(8):
        prior = sample_function_prior(rng, d=1, probe_z=grid)
        y = prior(grid)
        ax.plot(
            grid[:, 0],
            y,
            alpha=0.8,
            lw=1.2,
            label=f"{prior.activation}, sigma_obs={prior.sigma_obs:.3f}",
        )
    ax.set_title("FunctionPrior -- 8 independent draws of f(z), d=1")
    ax.set_xlabel("z")
    ax.set_ylabel("f(z)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    plt.show()
