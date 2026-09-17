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
    dominant, not calibrated against any target statistic. SHAPE only --
    see `calibrate_amplitude` for why the raw draw from this function is
    never used directly anymore."""
    a = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    b = float(rng.normal(0.0, 1.0))
    c = float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
    d = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
    return MonotoneMap(a=a, b=b, c=c, d=d)


def calibrate_amplitude(
    h_raw: MonotoneMap,
    f_probe: np.ndarray,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (0.5, 2.0),
) -> MonotoneMap:
    """Rescales `h_raw` so its COMPOSED output over `f_probe` (typically
    `f(z_B)`, B's own noiseless probe -- always available regardless of
    encoder/decoder role, so this is prior-internal, not privileged-cloud
    leakage) has mean and std controlled to a BOUNDED multiple of
    `f_probe`'s own scale, instead of `sample_monotone_map`'s raw,
    uncontrolled compounding of `a~LogUniform[0.3,3]`, `b~N(0,1)`,
    `c~LogUniform[0.1,2]` (2026-09-17, diagnosed via
    `notebooks/id_token_lupi_1d_comparison.ipynb`'s binning investigation --
    see `docs/labbook/` for the write-up).

    Root cause this addresses: `FullSupportBarDistribution`'s bin borders
    are ONE fixed geometry shared by every draw (`sample_calibration_borders`
    quantile-fits them once, pooled across many draws) -- exactly like
    `ppfn.prior.bnn.bnn_prior_vec.BNNPrior`'s own global ECDF reference. That
    trick only gives good LOCAL resolution to an individual draw if the
    pooled population is scale-homogeneous. BNNPrior's `depth*log(crit)`
    targeting keeps every draw's raw output on a comparable, roughly
    zero-mean scale; `h`'s raw (a,b,c) sampling had no analogous control,
    so the pooled calibration sample was heavy-tailed (measured: calibration
    span 14.2 raw-z units, only ~4/64 bins covering a typical individual
    draw's own ~0.3-unit range -- see the notebook's section 4c).

    Does NOT collapse h's diversity to make the per-task calibration problem
    trivial -- SHAPE (the raw a,c,d ratios controlling how linear/curved h
    is) is untouched, and BOTH the realized scale (`gain`, still a real 4x
    range by default) and location (`loc`, still N(0,1) wide, just anchored
    to f_probe's own scale rather than floating free) remain genuinely
    random per draw. What's removed is only the UNBOUNDED, uncontrolled
    compounding that put some draws' calibrated range ~100x others' --
    the part with no experimental motivation (`sample_monotone_map`'s own
    docstring already flagged its ranges as "not calibrated against any
    target statistic").

    `rng`: same per-draw generator as everything else in `sample_pair` --
    NOT a fresh/independent one, so this consumes exactly 2 more draws
    (`gain`, `loc`) from the shared stream, same discipline as every other
    per-draw random choice in this prior."""
    probe = h_raw(f_probe)
    mean_raw = float(probe.mean())
    std_raw = max(float(probe.std()), 1e-6)
    f_std = max(float(f_probe.std()), 1e-6)

    gain = float(np.exp(rng.uniform(np.log(gain_range[0]), np.log(gain_range[1]))))
    loc = float(rng.normal(0.0, 1.0)) * f_std
    k = gain * f_std / std_raw

    # Re-center h_raw to zero mean over the probe (removes its own,
    # uncontrolled b's contribution to location), rescale its SPREAD to
    # `gain * f_std`, then re-add a NEW, f_std-proportional offset `loc`.
    # Still of the same a>0,c>=0,d>=0 monotone family (k>0 preserves the
    # sign constraints), so monotonicity is untouched.
    return MonotoneMap(a=k * h_raw.a, b=k * (h_raw.b - mean_raw) + loc, c=k * h_raw.c, d=h_raw.d)


@dataclass
class KumaraswamyMap:
    """h(u) = 1 - (1-u^a)^b, a,b > 0 -- the Kumaraswamy CDF (a cheap,
    closed-form cousin of the Beta CDF, no incomplete-beta function needed).
    Monotone increasing on [0,1] for ANY a,b > 0 (derivative
    `a*b*u^(a-1)*(1-u^a)^(b-1) >= 0`), and -- unlike `MonotoneMap` --
    EXACTLY maps [0,1] -> [0,1]: h(0)=0, h(1)=1 always, regardless of
    (a,b). a=b=1 is the identity map exactly.

    This is the fix for the binning/granularity problem `calibrate_amplitude`
    only partially closed (2026-09-17, see that function's docstring and
    `docs/labbook/2026-09-17-lupi-h-amplitude-calibration.md`): that fix
    bounded h's composed SCALE but deliberately preserved its LOCATION
    diversity, and a fixed/shared bar-distribution geometry can only give
    every draw good local resolution when locations don't move around --
    exactly BNNPrior's own situation (roughly zero-mean raw output by
    construction) but not LUPI's old h family's. Composed with a value
    that's ALREADY on [0,1] (see `ppfn.prior.lupi.ecdf.load_or_fit_f_ecdf`),
    this guarantees every draw's reported value stays in [0,1] too, by
    construction, no probe/calibration needed -- every draw gets full,
    equal bin resolution, the same way BNNPrior's own draws already do."""

    a: float
    b: float

    def __call__(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, 0.0, 1.0)
        return 1.0 - (1.0 - u**self.a) ** self.b


def sample_kumaraswamy_h(rng: np.random.Generator, severity: float, log_range: float = np.log(5.0)) -> KumaraswamyMap:
    """`severity` in [0,1]: 0 -> EXACT identity (a=b=1), 1 -> full range
    (a,b each LogUniform over `[1/5, 5]` by default). Deliberately a
    SEPARATE, independently-sweepable knob from `rho` -- see the labbook
    entry on why T-severity and h-severity are kept orthogonal (controlled
    testing: attributing a gap to registration-position-difficulty vs.
    value-calibration-difficulty requires being able to vary one while
    holding the other fixed, matching how `oracle-and-baseline-ladder.md`
    already insists on sweeping d/cardinality/severity independently).

    ALWAYS consumes exactly 2 draws from `rng` (`a`,`b`), regardless of
    `severity` -- at `severity=0`, `log_range` scales to exactly 0 so both
    `rng.uniform(-0,0)` calls deterministically return `0.0` (verified:
    numpy handles `low==high` cleanly, no exception, no special-cased
    early return needed) and `a=b=exp(0)=1` exactly, the SAME as an
    explicit identity special-case would give -- but keeping the draw
    budget constant regardless of severity means a severity SWEEP on a
    fixed base seed doesn't desync every OTHER random choice drawn later
    in the same `sample_pair` call (acquisition, noise, ...), unlike
    `force_h_identity`'s own early-return in `sample_pair` (a pre-existing,
    accepted asymmetry there, not introduced here)."""
    lr = severity * log_range
    a = float(np.exp(rng.uniform(-lr, lr)))
    b = float(np.exp(rng.uniform(-lr, lr)))
    return KumaraswamyMap(a=a, b=b)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: sample a handful of
    h draws and confirm monotonicity numerically over a fine grid (the
    construction guarantees it, but a direct check catches an implementation
    slip in `MonotoneMap.__call__` itself). Then check `calibrate_amplitude`
    actually does what it claims: composed mean/std land near `loc`/`gain*f_std`
    exactly (by construction), and are far less scattered across draws than
    the raw, uncalibrated `h(f_probe)` would be."""
    rng = np.random.default_rng(0)
    grid = np.linspace(-3.0, 3.0, 2000)
    for i in range(8):
        h = sample_monotone_map(rng)
        vals = h(grid)
        n_violations = int((np.diff(vals) < 0).sum())
        print(f"h{i}: a={h.a:.2f} b={h.b:.2f} c={h.c:.2f} d={h.d:.2f} "
              f"monotonicity violations over 2000-pt grid (expect 0): {n_violations}")

    print("\ncalibrate_amplitude: raw (uncontrolled) vs. calibrated composed std, 200 draws")
    rng = np.random.default_rng(1)
    f_probe = rng.normal(0.0, 1.3, size=500)  # stand-in for a typical f(z_B) probe, fixed scale across draws
    raw_stds, cal_stds, cal_means = [], [], []
    for _ in range(200):
        h_raw = sample_monotone_map(rng)
        raw_stds.append(float(h_raw(f_probe).std()))
        h_cal = calibrate_amplitude(h_raw, f_probe, rng)
        probe_cal = h_cal(f_probe)
        cal_stds.append(float(probe_cal.std()))
        cal_means.append(float(probe_cal.mean()))
    raw_stds, cal_stds, cal_means = np.array(raw_stds), np.array(cal_stds), np.array(cal_means)
    print(f"raw std(h(f_probe))       : min={raw_stds.min():.3f} median={np.median(raw_stds):.3f} max={raw_stds.max():.3f} "
          f"(max/min ratio: {raw_stds.max() / raw_stds.min():.0f}x)")
    print(f"calibrated std(h(f_probe)): min={cal_stds.min():.3f} median={np.median(cal_stds):.3f} max={cal_stds.max():.3f} "
          f"(max/min ratio: {cal_stds.max() / cal_stds.min():.1f}x, target range was 0.5-2.0x f_probe.std()={f_probe.std():.3f})")
    print(f"calibrated mean(h(f_probe)) spread: std={cal_means.std():.3f} (still genuinely diverse, just f_probe-scale-anchored)")

    print("\nKumaraswamyMap: monotonicity + exact [0,1] containment across a severity sweep")
    rng = np.random.default_rng(2)
    grid01 = np.linspace(0.0, 1.0, 2000)
    for severity in (0.0, 0.25, 0.5, 0.75, 1.0):
        hk = sample_kumaraswamy_h(rng, severity)
        vals = hk(grid01)
        n_violations = int((np.diff(vals) < 0).sum())
        print(f"severity={severity:.2f}: a={hk.a:.3f} b={hk.b:.3f} "
              f"range=[{vals.min():.4f},{vals.max():.4f}] monotonicity violations (expect 0): {n_violations}")
