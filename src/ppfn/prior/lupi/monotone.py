"""The y-distortion h -- build spec §6.1: `f_A = h o f_B o T`. Applied to A's
raw values only; B is used in place, untouched (spec §2's decisive row).

Family: `h(y) = a*y + b + c*asinh(d*y)`, a>0, c*d>=0. This is monotone
increasing FOR ANY a>0, c>=0, d>=0 by construction (derivative
`a + c*d/sqrt(1+(d*y)^2)` is a sum of nonnegative terms with a>0, so it's
always > 0) -- no rejection sampling needed, unlike a generic monotone-spline
family. `asinh` is a deliberate choice: it's linear near 0 and saturates
(compresses range) for large |y|, which is qualitatively the same shape as
the fidelity-truncation effect the build spec's §0/§6.1 motivates this from
(low epoch budgets -> compressed performance range).

Note on why this matters at all despite quantile-normalization removing any
monotone map's effect in the noiseless/infinite-data limit (spec §3, §5.1):
with `h` disabled entirely, A's and B's raw y-values in this repo's other
registration prior (`ppfn.prior.registration.function_prior`) come from
literally the same shared BNN with a SHARED (pooled) standardization --
already comparable on raw scale, no genuine cross-task calibration required.
Applying an independently-scaled `h` to A only forces the model to actually
do the per-task quantile calibration the architecture is built around,
rather than free-riding on already-matched scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonotoneMap:
    a: float
    b: float
    c: float
    d: float

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.a * y + self.b + self.c * np.arcsinh(self.d * y)


def sample_monotone_map(rng: np.random.Generator) -> MonotoneMap:
    """a ~ LogUniform[0.3, 3] (linear slope), b ~ N(0, 1) (offset),
    c ~ LogUniform[0.1, 2] (nonlinear-term weight), d ~ LogUniform[0.3, 3]
    (nonlinear-term steepness) -- ranges chosen so the asinh term is
    comparable in magnitude to the linear term rather than negligible or
    dominant, not calibrated against any target statistic."""
    a = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    b = float(rng.normal(0.0, 1.0))
    c = float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
    d = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    return MonotoneMap(a=a, b=b, c=c, d=d)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: sample a handful of
    h draws and confirm monotonicity numerically over a fine grid (the
    construction guarantees it, but a direct check catches an implementation
    slip in `MonotoneMap.__call__` itself)."""
    rng = np.random.default_rng(0)
    grid = np.linspace(-3.0, 3.0, 2000)
    for i in range(8):
        h = sample_monotone_map(rng)
        vals = h(grid)
        n_violations = int((np.diff(vals) < 0).sum())
        print(f"h{i}: a={h.a:.2f} b={h.b:.2f} c={h.c:.2f} d={h.d:.2f} "
              f"monotonicity violations over 2000-pt grid (expect 0): {n_violations}")
