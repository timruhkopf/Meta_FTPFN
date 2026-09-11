# Isotonic h retracted: it's an identifiability confound, not an optimization bug

commit: b968452

## What was investigated

Stepping back after the alternation fix (see
`2026-09-11-hpo-warp-sequential-h-then-t-was-biased.md`): does alternating
`h` and `T` actually solve the underlying problem, or just make it less bad?
User's framing, paraphrased: the goal was always "how much force on *both*
x and y is needed to map one task's surface onto the other's" — trying to
cleanly separate that force between `h` and `T` is a harder, different
question, and the synthetic case study kept showing the separation was
untrustworthy no matter how the fitting was iterated.

## Why alternation (previous entry) was a real improvement but not the fix

Alternation measurably helped (mean recovered severity rose from ~0.4 to
~1.8 over a few rounds on the synthetic ground-truth pair) — that finding
still stands, it isn't being retracted. But it was treating a symptom:
isotonic `h` is **any monotone function whatsoever** — dozens of effective
knots, fit fresh each round. No amount of re-fitting removes the underlying
capability that let it absorb `T`'s job in the first place; alternation
just gives the optimizer more chances to land somewhere better, with no
structural reason it reliably will. This is an **identifiability confound
between `h` and `T`**, not an optimization-convergence problem — the same
family of issue `docs/ROADMAP.md` §3.3 already names for "warped inputs"
vs. "different function" being hard to tell apart under a flexible enough
model.

## The fix: constrain h's functional family, informed by what HPO actually needs

Asked directly: what does real y-distortion between two HPO tasks' surfaces
actually look like? Enumerated with the user:

- **Global shift and scale (mandatory)** — different metrics have
  genuinely different ranges (cross-entropy range depends on class count),
  different Bayes-error/irreducible-loss floors, and sign conventions (loss
  vs. accuracy). This is affine, full stop, and has to stay representable
  exactly — not optional, not to be "constrained away."
- **At most one shape parameter** — bounded metrics (accuracy near a
  ceiling) compress nonlinearly near the boundary; heavy-tailed losses
  (occasional divergent configs — the exact phenomenon that broke TaskSet's
  fit, see the scattered-benchmarks entry) need a tail-compressing
  transform. Both are *global* properties of a metric, not something that
  varies from one region of hyperparameter space to another — i.e. exactly
  the kind of structure a single extra parameter can capture and a
  many-knot isotonic fit can't be prevented from over-capturing.
- **What's not needed**: arbitrary per-region monotone flexibility. No real
  y-distortion phenomenon identified needs more shape than one skew/
  boundedness parameter.

Landed on `h(y) = a * YeoJohnson(y; lambda) + b` (Yeo & Johnson, 2000; the
real-line generalization of Box-Cox, the same family HEBO uses for output
warping and already flagged, then under-prioritized, in an earlier
assessment of a "second opinion" on this project). Verified `YeoJohnson(y;
1) = y` for both positive and negative `y` before relying on it, so
`lambda=1` is exactly the affine floor — no shape distortion is one
precisely-reachable point in the family, not an approximation.

## Result

Fit by a 51-point grid scan over `lambda` with closed-form OLS for `(a,b)`
at each candidate (no gradient descent, no local-optima risk — the same
grid-scan-plus-closed-form shape `fidelity_warp.py`'s `tau` search already
used). On the synthetic ground-truth pair, a **single pass, no alternation**
now recovers:

| data seed | true severity | recovered severity | held-out R² |
|---|---|---|---|
| 7 | 2.78 | 3.34 | 0.998 |
| 11 | 1.87 | 1.23 | 0.994 |
| 23 | 2.97 | 2.89 | 0.978 |

All within the same order of magnitude as ground truth (worst case ~35%
off), against the isotonic version's 8x underestimate — and cheaper, since
the alternation rounds this replaces are no longer needed.

## What was removed and why it's not a backward-compatibility concern

`fit.py`'s `n_rounds`/alternation loop, `_fit_isotonic_h`, and
`sklearn.isotonic.IsotonicRegression` as a dependency of this module are
all deleted, not deprecated — per the earlier finding, alternating a
too-flexible `h` was masking the real problem, so keeping it as a fallback
option would just be keeping a worse method on hand. `FitResult`/
`FitArtifacts` now carry `h_lambda`/`h_scale`/`h_shift` (three floats)
instead of isotonic breakpoint arrays. `lr`/`svm`/`rf`/`lcbench` (~80%) were
swept under the isotonic method before any of this was found; not
retroactively relabeled, and not yet re-run — a real compute decision
(roughly 3x the original one-shot cost: `n_restarts=3` stays, the
alternation multiplier this removes was ~2x), flagged to the user rather
than decided here.
