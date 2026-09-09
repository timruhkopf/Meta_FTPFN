# Harmonics prior: domain-drift warp leaked the shift through B's raw x-coordinates

**commit:** pending
**Related:** `docs/milestones/M3-harmonics-prior.md`, `docs/milestones/M2-prior-contract-toy-visualizer.md`

## What was investigated

Whether the harmonics prior's A/B domain warp could be inferred by a model
"for free" from the raw x-coordinates alone, before looking at any y-value —
which would defeat the point of the prior (the model is supposed to infer the
drift from the (x, y) relationship).

## What was found

Both archived forks (`archive/src/ppfn/prior/harmonics/` and `harmonics_fix/`)
compute B's observed x as:

```python
X_obs = X - h + alpha * sin(2*pi*omega*X + beta)   # h ~ Uniform(-1.5, 1.5)
```

where `X` is drawn from a **fixed, config-level** canonical domain (e.g.
`(-5, 5)`), the same every instance. Since the domain center is a known
constant, `h` is recoverable directly:

```
h ≈ (min_x + max_x) / 2 - mean(X_B_obs)
```

Verified visually before touching any code — see the plot attached to this
conversation on 2026-08-27 (also reproducible via `git show` on the pre-fix
`archive/src/ppfn/prior/harmonics_fix/harmonic_mixture_prior.py`): B's
observed x-range visibly sits outside the marked canonical domain bounds,
shifted by roughly `h`, for every instance checked.

`harmonics_fix` (chosen over plain `harmonics` as the base for the port —
more rigorous design overall, see the milestone doc) inherited the identical
bug despite being otherwise much more careful.

## Fix

Replaced the translation-based `tau` with a domain-preserving Kumaraswamy
warp: rescale `x` to `[0,1]`, apply the Kumaraswamy CDF `1 - (1-u^a)^b`
(bijection of `[0,1]` onto itself for any `a, b > 0`), rescale back. B's
observed x now has exactly the same support as A's in every instance — the
drift only shows up in the (x, y) *shape*, never in where the x's sit.

### A wrong turn along the way, worth recording

The closed-form Kumaraswamy inverse alone was *not* numerically sufficient:
for `a` near the top of a "generic warper" range (e.g. up to 10, matching
`archive/src/ppfn/prior/warp.py::KumaraswamyInputWarper`'s own defaults),
`u^a` underflows float precision for `u` near the domain edges (`u=0.014,
a=9 => u^a ≈ 1e-17`), so many distinct `x` collapse to the same observed
`tau(x)` and the inverse silently returns the wrong `x` — real information
loss, not a rounding error, verified via `verify_prior.py` (max inverse error
`~2.1` on a 10-unit-wide domain, i.e. totally wrong, not "slightly off").
Fixed by (a) tightening the shape-parameter range to `[0.3, 3.5]` and (b)
polishing the closed-form guess with Newton iterations on `tau(x) - U = 0`
(12 iterations, converges from the already-close closed-form start to
`~1e-7`). This prior's whole design depends on `tau` being *exactly*
invertible (the oracle `B_in_A` context is only a valid upper bound if it
truly is a lossless transform of `B_obs`), so this wasn't optional polish.

## Verification

`python -m ppfn.prior.harmonics.verify_prior`:
- Inverse exactness: `max|dx| ~ 1e-7`, `max|dy| ~ 1e-15`.
- `X_B_obs` range matches the canonical domain exactly (was visibly offset
  before the fix).
- Oracle (`A + B_in_A`) dramatically outperforms `A` alone on related
  instances (MSE 0.0002 vs 3.42) and the ordering correctly flips on
  unrelated/trap instances (oracle MSE worse than `A` alone) — matches the
  design doc's stated invariant.
