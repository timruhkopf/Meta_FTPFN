# Extending the HPO-warp pipeline to LCBench/TaskSet: four bugs, one deferred

commit: pending

## What was investigated

Extending `src/ppfn/experiments/hpo_warp/` beyond HPOBench (2-5D, exact
Cartesian grids) to LCBench-tabular (7D) and TaskSet-tabular (up to 8D),
both of which share their random-search design exactly across tasks
(verified empirically, matching HPOBench's exact-grid property) but lay it
out as a **scattered** point cloud, not a grid — so `multilinear_interp`
doesn't apply and a new kernel-based reader (`interp.kernel_interp`,
`surface.py`) was added alongside it, dispatched via a new
`TaskGrid.is_gridded` flag.

Four real bugs surfaced getting this working; a fifth issue (TaskSet's
heavy-tailed target scale) was found and deliberately deferred rather than
patched under time pressure.

## Bug 1: severity/displacement used a `grid_n**d` regular grid

`warps.py`'s `logdet_jacobian_band`/`displacement_volume` built a
`grid_n=15` regular meshgrid to evaluate the fitted warp on. Fine at
HPOBench's d<=5 (`15**5 ~= 760k` points, slow but survivable); at LCBench's
d=7 that's `15**7 ~= 1.7e8` points — the first LCBench fit attempt hung
rather than erroring, and was only diagnosed by timing a smaller isolated
piece and noticing the discrepancy. Fixed: both functions now sample
`n_points=3000` **random** points on `[0,1]^d` instead of a grid — cost is
`d`-independent, and a random sample is arguably a better-conditioned
estimate of the log|det J| distribution than a grid anyway (no grid-aligned
artifacts).

## Bug 2: `log(0)` from LCBench's epoch axis starting at 0

`fidelity_warp.py`/`surface.py` log-scale the fidelity axis. LCBench's
`epoch` column starts at 0 (the pre-training state), and `log(0) = -inf`
silently cascaded into NaN through the whole fidelity fit with no exception
raised — caught by NaN/warning spam in stderr, not a crash. Fixed in
`scattered_data.py`'s loader: fidelity values `<= 0` are dropped before
`TaskGrid` construction (not a meaningful comparison point anyway — nothing
has been learned at epoch 0).

## Bug 3: kernel bandwidth formula didn't survive higher `d` (the real find)

First `kernel_interp` bandwidth was `2 * n**(-1/d)` (grid-spacing-style).
For LCBench (n=2000, d=7): `2000**(1/7) ~= 3`, giving a bandwidth wide
enough to average over most of the unit cube. Symptom: the fitted warp
ladder's held-out R² was **non-monotonic and often worse than identity**
(`affine: 0.10` vs `identity: 0.24` on one pair) — not an obviously "wrong"
number, which is what made it worth writing up: it looked like a fitting
failure, but was actually a fully flat loss landscape (the oversmoothed
kernel readout barely changes with the query location, so there's nothing
for gradient descent to find). Fixed: bandwidth is now the median distance
to each point's 15th nearest neighbor (`scipy.spatial.cKDTree`), which
adapts to actual local density instead of assuming points fill the cube
uniformly. Verified: same pair's ladder became monotonic-ish and
capacity-sensitive after the fix (`affine: 0.38`, `velocity_m16: 0.46`,
both clearly above `identity: 0.24`).

## Bug 4: TaskSet configs can differ in *which* configs survive, not just count

TaskSet drops per-task configs whose curve is incomplete (divergence) —
confirmed two tasks under the nominally-identical `adam8p_wide_grid` design
retaining 1000 and 555 configs respectively. `fit_config_warp`'s identity
step (`spearmanr`, `pearsonr`, the isotonic `h` fit) assumed `grid_a.x[i]`
and `grid_b.x[i]` were the same config by construction — true for HPOBench
and LCBench (verified: never drop anything), false here, and it crashed
(`spearmanr` on mismatched array lengths) rather than silently misaligning,
which is the better failure mode but still needed fixing. Fixed:
`TaskGrid` gained an `ids` field (the design's own config identifiers);
`fit_config_warp` now intersects `grid_a.ids`/`grid_b.ids` and restricts
*only* the identity-comparison step to the overlap, computed with its own
independent train/test split. The x-warp ladder itself is unaffected by
this (it reads B through a continuous surface reader regardless of exact
correspondence), so this didn't need touching.

## Deferred, not fixed: TaskSet's target scale is heavy-tailed enough to break MSE fitting

After bugs 1-4 were fixed, TaskSet's `adam4p`/`adam8p` warp fits still came
out degenerate: **every rung** (affine through the max-capacity velocity
field) converged to the *same* near-zero R², all clearly worse than the
`identity` rung (which uses isotonic regression, not MSE). Root-caused:
`valid1_loss` for e.g. `FixedMAF_cifar10_3layer_bs64` has median ~2900 but a
99th percentile of ~9.8e8 — divergent configs blow up to values whose
*squared* error dominates the MSE loss completely regardless of batch
composition. The existing 1st/99th-percentile winsorization doesn't help:
this isn't a 1%-outlier problem, it's genuinely heavy-tailed (>1.6% of all
cells exceed 1e6, per task). Isotonic regression survives this fine
(rank-based, insensitive to the extreme tail's exact magnitude); the
MSE-based warp ladder does not.

**Not patched under time pressure** — the honest fix (working in
rank/quantile space for the warp-fitting loss too, or per-task robust
rescaling before any squared-error objective) is a real design decision
about the loss, not a one-line robustification, and shipping a rushed
version risked a sixth bug on top of four already found in one sitting.
TaskSet is excluded from the current sweep; LCBench (which has no such
scale pathology and was verified working correctly) covers the
"bigger search space + genuine fidelity axis" ask on its own for this round.
