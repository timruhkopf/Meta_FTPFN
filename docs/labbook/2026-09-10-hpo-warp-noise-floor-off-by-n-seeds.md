# HPO-warp noise floor was too lenient by a factor of `n_seeds`

commit: b968452

## What was investigated

First real run of `src/ppfn/experiments/hpo_warp/` (design:
`docs/experiments/hpo-warp-complexity.md`) — fitting the config-space warp
ladder between HPOBench `TabularBenchmark(model="lr")` task pairs, on
ulysses. `residual_to_noise_floor` (best-fit held-out MSE divided by the
seed-replicate noise floor, meant to diagnose whether a diffeomorphism fully
explains a pair) is supposed to sit near 1 for a well-fit pair. Across the
first 151/812 pairs it came out with a median of **0.82**, and the elbow
capacity selection (`shape_complexity_m`) was suspiciously low (median 2,
the smallest rung in the sweep) for nearly every pair.

## First attempt (the running sweep) was wrong — say so plainly

`hpobench_data.load_task_grid` computes `y_var_by_iter` as the variance of
the **5 replicate seed observations** at each `(config, iter)` cell — the
variance of a *single* draw. `fit_config_warp` fits and evaluates against
`y_mean_by_iter`, the *mean* of those 5 seeds. The noise floor used to judge
"is this residual just noise" was `noise_floor = mean(y_var_by_iter)` —
the single-observation variance, not the variance of the mean being
compared against. Since `Var(mean of n) = Var(single) / n`, this made the
floor **5x too large** (`n_seeds=5`), so almost any fit looked like it had
"reached the noise floor" well before it actually had — which is exactly
why the elbow kept picking the smallest capacity in the sweep regardless of
whether more capacity actually helped.

Caught by checking the executed notebook's summary numbers before trusting
them, not by inspection of the code in isolation — the ratio being < 1 for
the *median* pair (not just occasionally, from sampling noise) was the
tell, since a correctly-calibrated floor should center near/above 1.

## Fix

- `TaskGrid` gained an `n_seeds` field (`hpobench_data.py`, read off
  `df["seed"].nunique()` rather than hardcoded, since it's a per-model
  constant from the source table, not assumed).
- `fit_config_warp`'s `noise_floor` is now `mean(y_var_by_iter) / n_seeds`.

Verified on the `lr` task pair `(3, 12)`: elbow moved from `M=2` (pre-fix)
to `M=8` (post-fix) for the same pair, and `residual_to_noise_floor` moved
from 0.51 to 0.72 — still not exactly 1 (expected: one random held-out
split has its own sampling variance in both the residual and the floor
estimate), but no longer systematically inflated.

## What this cost

The first 151 pairs were computed under the wrong floor and their elbow
selections (hence `severity_logdet_band`, `displacement_volume`,
`bending_energy`, `fold_fraction` — everything read off "the elbow-capacity
fit") are not just numerically off, they may have picked a **different**
elbow rung entirely. Discarded rather than patched: deleted
`data/hpo_warp_results/lr/` on ulysses and locally, redeployed the fixed
code, and relaunched the full 812-pair sweep from scratch. Cheap to redo
(~812 pairs at ~12s each / 14 parallel workers, well under the time already
spent chasing this), so a clean restart was the right call over trying to
salvage or reinterpret the first batch.
