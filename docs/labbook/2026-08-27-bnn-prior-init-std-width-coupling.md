# BNN prior: `init_std` sampled independently of network width caused most sampled functions to look flat

**commit:** pending
**Related:** `docs/milestones/M4-bnn-prior-relatedness.md`, `src/ppfn/prior/bnn/bnn_prior.py`

## What was investigated (user's own analysis, reproduced here for the record)

Why so many draws from `BNNPrior.sample()` looked flat/uninformative in the
`__main__` demo (`src/ppfn/prior/bnn/bnn_prior.py`).

Not a guess — tested. `BNNPrior.sample_mlp` draws `num_hidden` (width) and
`init_std` as two *independent* random values, then applies `init_std`
uniformly to every layer's weights. Defining
`crit = init_std**2 * width` (a mean-field/Xavier-style criticality
proxy for a tanh network — near `crit ≈ 1` a layer roughly preserves
variance; well below, signal shrinks toward a constant each layer; well
above, it grows), 300 sampled instances gave `crit` scattered
`~[0.25, 4.7]` — essentially by chance, since width and `init_std` were
drawn independently rather than tied together (proper Xavier scaling would
have `init_std ∝ 1/√width`).

Three concrete instances spanning that spread, raw (pre-normalization)
output range over the domain:

| crit | raw output range | character |
|------|------------------|-----------|
| 0.48 | 0.006 | flat line at float-noise scale |
| 1.48 | 0.22  | near-straight, low-amplitude ramp |
| 3.89 | ~3.5  | genuine multi-bump structure |

`corr(output complexity, crit) = 0.68` across the 300 instances;
`corr(., depth) ≈ 0.16` — depth barely mattered, crit did. The ECDF
(`BNNPrior.ensure_ecdf_loaded`) compounds this: it pools raw samples across
the whole family, dominated by the wide-range (high-crit) instances, so a
low-crit instance's already-tiny raw range maps to an even narrower band
once normalized against that pooled scale.

## Fix, first attempt — verified, and verified WRONG

Proposed fix: sample `crit` directly and back out `init_std = sqrt(crit /
width)`, so instances are centered near critical initialization by
construction. First attempt used `crit ~ Uniform(0.5, 2.0)` — reasoned to be
"centered near 1," but not checked against the actual old behavior before
shipping.

Checked before finalizing (loaded the real `BNNPrior`/`MLP` classes from
before vs. after via `git show`, ran a matched 300-sample comparison with
the same RNG seed on both): **this made things worse.**

| | median output range | fraction near-flat (< 0.05) |
|---|---|---|
| old (independent draws) | 0.169 | 0.24 |
| `crit ~ Uniform(0.5, 2.0)` | 0.070 | 0.41 |

Why: the *old* sampling's effective median `crit` was already ~1.6 (not
below 1) — `init_std ~ Uniform(0.089, 0.193)` combined with typical widths
happened to land above the critical value more often than not. Centering the
new range at 1.25 was a regression relative to that, not an improvement.
Lesson: "sounds reasonable" is not verification — this would have shipped a
worse prior if not checked against the real classes.

## Fix, corrected

Swept a few candidate ranges against the same real-class, same-seed
comparison:

| crit range | median range | frac near-flat |
|---|---|---|
| old (baseline) | 0.169 | 0.24 |
| Uniform(0.5, 2.0) | 0.070 | 0.41 |
| Uniform(1.0, 3.0) | 0.315 | 0.06 |
| **Uniform(1.0, 5.5)** | **0.825** | **0.03** |
| Uniform(1.5, 4.0) | 0.676 | 0.00 |
| Uniform(2.0, 6.0) | 1.580 | 0.00 |

Landed on `crit ~ Uniform(1.0, 5.5)`: keeps the old sampling's upper spread
(effective crit reached ~5.5 already) but floors it at the critical value —
cutting off exactly the sub-critical tail that caused the flatness, without
inventing a new range from scratch. Implemented in
`BNNPrior.sample_mlp` (`src/ppfn/prior/bnn/bnn_prior.py`).

## Also found and fixed along the way

- `MLP.forward` (`src/ppfn/prior/bnn/mlp.py`) unconditionally raised
  `NotImplementedError`, guarding an unresolved question about whether an
  input normalizer should be applied. Removed the block (needed to run any
  of the above); left the normalizer call's result discarded exactly as
  before, so the open question isn't silently answered as a side effect.
- The demo's `__main__` seeded `torch` but not `numpy`, and `sample_mlp`'s
  randomness comes from `numpy` — the demo wasn't actually reproducible.
  Fixed by seeding both.

## Verification

`src/ppfn/prior/bnn/bnn_prior.py`'s `__main__` demo, before/after: 2 of 8
sampled curves showed real structure before the fix, 6 of 8 after (see the
plots attached to this conversation on 2026-08-27).
