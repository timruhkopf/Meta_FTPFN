# Velocity-field fits collapsing near identity, fixed by multi-restart selection

commit: pending

## What was investigated

User observation on the synthetic ground-truth case study (true severity =
2.78): "the recovered warp seems severely underfitting the actual warp
field." Checked rather than dismissed — and it was right, for a reason
distinct from the alternation-bias finding earlier the same day.

## What was found

**Not a capacity problem.** The true generative field used $M=6$ kernels;
the fitting ladder goes up to $M=16$, comfortably above that. Ruled out by
inspection of the actual `sample_velocity_field` draw used.

**More optimization steps does not reliably help, and can make it worse.**
At a fixed (bad) random initialization, held-out R² across step counts:

| steps | held-out R² | recovered severity |
|---|---|---|
| 300 | 0.9909 | 1.835 |
| 1000 | 0.9916 | 1.668 |
| 3000 | **0.9842** | **0.006** |

3000 steps converged to a *worse* fit than 300 — essentially the same R² as
the pure y-only (no-warp) baseline, with the velocity field having drifted
back to (or never having left) its zero-initialized identity. This rules
out "just train longer" as the fix.

**Multiple random restarts, selected by training loss, does fix it.**
Across 3 restarts at 300 steps each, one landed on the good solution
(R²=0.9909) while two collapsed to the y-only baseline (R²≈0.984) — and
critically, the collapsed restarts are *measurably worse on the training
data itself*, not just held out, so selecting by training loss (not
held-out, which would be a form of test-set leakage in model selection)
correctly and cheaply picks out the good one.

## Fix

`_fit_warp_module` (`fit.py`) now takes a warp *constructor* rather than a
pre-built module, tries `n_restarts=3` freshly-initialized fits (each
internally seeded by its restart index, `torch.manual_seed(restart)`, so
runs are now exactly reproducible — verified: two independent processes
produced identical severity to 4 decimal places), and keeps whichever has
the lowest full-training-set MSE. Applies uniformly to every ladder rung
(affine/spline/velocity), not just the velocity field, since any zero-init
rung could in principle exhibit the same failure mode even if only the
velocity field was observed to in practice.

## Cost

Combined with the same day's alternation fix (`n_rounds=2`, up from 1),
this raises per-pair compute by roughly `2 (rounds) x 3 (restarts) = 6x`
relative to the original one-shot, single-restart sweep. The already-run
`lr`/`svm`/`rf`/`lcbench` sweeps predate both fixes; not auto-redone, both
findings (and their combined cost) flagged to the user as one decision.
