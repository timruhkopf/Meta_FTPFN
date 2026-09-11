# Boundary dead-zone in warp fitting, fixed with a penalty term

commit: b968452

## What was investigated

User inspected the synthetic case study's true-vs-recovered warp
displacement plot and flagged two artifacts: weird/large displacement
signal near the plot boundaries, and the lower-right quadrant not
faithfully registering the true warp (the latter is covered separately,
see `2026-09-11-hpo-warp-gradient-identifiability-tradeoff.md`).

## What was found

The plotting cell calls the fitted velocity field directly on probe points
with **no** `clamp01`, while the training loop always clamps the warp's
output before reading B's surface at it. Once a raw output lands outside
`[0,1]^d`, `clamp01` saturates and the training loss becomes flat with
respect to further movement of the raw field out there — nothing penalizes
it for continuing to drift once it's already off the edge.

Quantified on the synthetic ground-truth pair: 6.6% of a 3000-point random
probe had raw warp output outside `[0,1]^2`. Near-boundary points (within
0.05 of an edge) showed about 2x the average displacement of interior
points (0.042 vs 0.019). This directly inflates `severity_logdet_band`,
which is a max-min range statistic and therefore sensitive to a small
fraction of outlier points.

## Fix

Added `BOUNDARY_PENALTY_WEIGHT = 1.0` to `fit.py` and modified
`_train_once` to compute the raw (unclamped) warp output and add
`((raw_t_x - clamp01(raw_t_x))**2).mean()` to the training loss, alongside
the existing fit loss. This gives the optimizer a reason to keep the raw
field in-bounds even though clamping would otherwise mask the incentive.

## Result

Verified standalone on the synthetic ground-truth pair (severity truth
2.779):

| | before | after |
|---|---|---|
| boundary-excursion fraction | 6.6% | 1.5% |
| held-out R² | 0.998 (comparable) | 0.9988 |
| recovered severity | 2.24 (~19% off) | 2.658 (~4% off, best result yet) |

Deployed to ulysses and the `lr`/`svm`/`rf` HPOBench sweeps were relaunched
under the fully-corrected method (Yeo-Johnson h + multi-restart + this
boundary penalty) rather than waiting for a separate follow-up pass.

## Correction (same day): the "~4% off" number above does not reproduce

Re-executing the actual notebook (not a standalone script) after adding the
quadrant plot gave `severity_logdet_band = 2.174` (~22% off) for the same
nominal case, not 2.658. Root cause: the throwaway verification script
above built `grid_a_warped` right after `grid_b`, consuming the shared
`np.random.default_rng(7)` two draws earlier than the notebook does (the
notebook also builds `grid_a_identical` in between, which consumes the RNG
for its own noise draw). Same "true severity" ground truth (2.779, same
warp field), but a **different noise realization** of `y_warped`'s observed
values — confirmed by rebuilding the RNG call sequence to match the
notebook exactly, which reproduces 2.174 deterministically.

So the standalone number wasn't wrong because of a bug in the fix, but
because it quietly fit different data than what's actually in the
notebook — a mistake in how the fix was verified, not in the fix itself.
Comparing like-for-like (full-pipeline notebook runs, before vs. after,
same RNG order):

| | before (isotonic→YJ+restarts, no boundary term) | after (+ boundary penalty) |
|---|---|---|
| recovered severity | 2.24 (~19% off) | 2.174 (~22% off) — essentially flat, slightly worse |
| residual_to_noise_floor (warped) | 4.1 | 3.68 — improved |

Net effect on this specific synthetic instance: the boundary penalty does
not reliably move `severity_logdet_band` closer to ground truth (severity
recovery is already known to be seed/noise-sensitive, see the restart-fix
entry), but it does improve fit quality (lower residual-to-noise-floor) and
still removes the actual dead-zone artifact it targeted (boundary excursion
fraction, verified independently of the RNG mismatch above). The fix is
kept — it addresses a real optimization pathology — but the "closer to
ground truth" framing above is retracted as unverified; "better-conditioned
fit, not necessarily better severity recovery on any single draw" is the
accurate claim.
