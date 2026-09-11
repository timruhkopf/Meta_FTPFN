# One-shot sequential h-then-T fit understated severity by ~8x

commit: b968452

## What was investigated

User concern, reading the design doc: fitting `h` once at the identity
correspondence and then freezing it while `T` is fit could bias the result
toward attributing too much to `h` and too little to `T` — the sequencing
was chosen to dodge the ill-posedness of a fully joint `min_{T,h}` fit, but
"dodges ill-posedness" isn't the same claim as "recovers the right split
between the two."

## Empirical check, on the synthetic ground-truth case

Using the known-`T_true`/known-`h_true` synthetic pair (`notebooks/
hpo_warp_complexity_report.ipynb`'s case study; true severity = 2.78):
refit `h` using the *current* fitted `T`'s pushforward (instead of the raw
identity), then refit `T` again, repeated a few times.

| round | `h` fit against | recovered severity | held-out R² |
|---|---|---|---|
| 1 (old default: one-shot) | `y_B(x)` at identity | 0.36–1.41 (run-to-run noise) | 0.985–0.989 |
| 2 | `y_B(T_1(x))` | 1.62–1.65 | 0.990–0.994 |
| 3 | `y_B(T_2(x))` | 1.90–2.21 | 0.995–0.996 |

Confirmed: the one-shot fit systematically **understates severity**,
converging toward the true value (2.78) as alternation proceeds. Not a
small effect — up to ~8x off in the worst single run observed.

## Why alternation fixes it without reintroducing the ill-posedness

This is Alternating Conditional Expectations (Breiman & Friedman, 1985),
specialized to a monotone `h` and a diffeomorphic `T`: alternate refitting
`h` (isotonic regression, closed-form via PAVA, given `T`'s current
pushforward as the covariate) and refitting `T` (gradient descent, given
`h` fixed), rather than optimizing both simultaneously in one pass. The
degenerate solution a fully joint objective is vulnerable to (`T` collapses
the domain to a point, `h` maps that point wherever `A` needs, zero loss,
nothing learned) requires `T` and `h` to move *together* along a path that
keeps lowering the loss; alternation never lets that happen; because each
half-step is a complete, well-posed optimization of one object against the
other held *fixed*, not a joint gradient step.

## Fix

`fit_config_warp` (`fit.py`) now takes `n_rounds` (default 2, was
implicitly 1): after each round's full affine/spline/velocity ladder fit,
if another round remains, `h` is refit against the elbow-capacity `T`'s
pushforward on the matched configs, and the whole ladder is refit with the
new `h`. `y_only_r2` still always reports round 0's pure `T=identity` fit
(the honest no-warp baseline); `severity_logdet_band` and the rest of the
diagnostics report the *final* round. `n_rounds=1` reproduces the exact old
behavior, kept as an explicit option for comparison rather than the
recommended setting.

## What this means for the already-completed sweeps

`lr`, `svm`, `rf` (complete) and `lcbench` (77% complete when this was
found) were all run at the old, now-known-biased `n_rounds=1`. Severity and
shape-complexity numbers from those sweeps are likely **underestimates**;
`y_only_r2` and `residual_to_noise_floor` are less directly affected (they
characterize "how well is this explained," which moves much less between
rounds than "how is the credit split between `h` and `T`" does). Redoing
them at `n_rounds=2` roughly doubles the compute already spent (each round
re-runs the full ladder) — a real resource decision, not made unilaterally
here; flagged to the user rather than auto-relaunched.
