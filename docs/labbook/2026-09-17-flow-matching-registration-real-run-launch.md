# First real run launched: FlowMatchingRegistrationPFN on ulysses

`commit: pending`

## What's running

`01-pretraining-lupi-flow-matching-registration`
(`configs/experiment/lupi_flow_matching_registration.yaml`, prepped in
[2026-09-17-prior-01-normalization-and-real-run-prep.md](2026-09-17-prior-01-normalization-and-real-run-prep.md),
launched once the `bounded01` prior dependency landed --
[2026-09-17-bounded01-ported-and-val-batch-bug.md](2026-09-17-bounded01-ported-and-val-batch-bug.md)).
Launched on ulysses, `nohup`+`disown` (no `atd` there, same constraint as
`scripts/uly_schedule_long_run_at.sh`), 100 epochs x 500 steps,
`batch_size=24`, `n_integration_steps=10`, `n_transport_samples=1`,
`sde_sigma=0.0` (deterministic ODE mean, first cut).

**Deployment note for next time**: `Meta_FTPFN-sde` had no worktree on
ulysses yet (only `Meta_FTPFN`/`lupi-iterative-registration` existed
there). Pushed this branch (commit `771873d`, via `gh`'s credential helper
-- plain HTTPS push has no cached credential in this environment),
`git worktree add` on ulysses, then `uv sync` failed the same way it does
locally (`external/ifbo_icml2024`'s editable dependency isn't a git
submodule, isn't tracked, and isn't present in a fresh worktree) -- fixed
by symlinking `Meta_FTPFN/external` into the new worktree before
`uv sync`, which then completed cleanly (real venv, not a borrowed
interpreter). GPU headroom checked before launch (12.4GB/24GB in use by
the peer session's two runs, `lupi-bounds`/`lupi-id-token`) and confirmed
still fine after (12.6GB, my job's dataloader workers only, training not
yet warmed up at check time).

Verified before committing GPU time: module diagnostic and the debug Hydra
config both re-run cleanly on ulysses itself (not just locally) --
identical output to the local runs in the previous two entries.

## What we hope to find

Three questions, in order of how directly this run can answer them:

1. **Does the flow-matching mechanism recover the registration at all?**
   Watch `flow/total` (the velocity-regression loss `v_φ` is trained
   against) fall steadily across training -- this is the one metric here
   with a genuine, interpretable floor (perfect registration -> velocity
   error 0 on the real, non-padded channels). A `flow/total` that plateaus
   high says `v_φ` isn't learning the transport at all, independent of
   anything downstream.

2. **Does the recovered registration actually help A's predictions, and how
   much of the ceiling does it reach?** This is CLAUDE.md's own "always
   report three gaps, never a bare NLL" discipline, mapped onto what this
   run can measure directly:
   - `loss/nll_teacher` (upper-1: predictor given the TRUE `B_inA`) is the
     ceiling this run's own architecture can reach if registration were
     solved exactly -- not the true oracle-2 pooled bound, but the tightest
     bound reachable by reusing `BoundsPFN`'s exact readout.
   - `loss/nll_student` (the deployable pathway: predictor given `v_φ`'s
     own SDE-integrated estimate) is what an actual deployment would see.
   - `loss/upper1_gap = nll_student - nll_teacher`, already logged every
     step, is the number to watch: it should SHRINK over training as `v_φ`
     gets better at registering, and it directly isolates "cost of not
     knowing the true registration" from every other cost in the pipeline
     (pooling, reading a differently-framed memory, etc. -- those are
     already paid by the teacher pathway too, so they cancel out of the
     gap).
   - The full three-gap ladder (`lower` -> `upper-1b` -> `upper-1` ->
     `model`) needs the companion `lupi_bounds` checkpoint (the peer
     session's own `01-pretraining-lupi-bounds` run, already in flight) for
     the `lower`/`upper-2`-adjacent rungs -- this run alone gives the
     `upper-1` -> `model` rung directly, the one this branch's own research
     question is actually about.

3. **Is the deterministic (`K=1`, `sigma=0`) first cut good enough to be
   worth extending to a genuine stochastic SDE (`K>1`, `sigma>0`) next?** If
   `upper1_gap` closes substantially (say, to a small fraction of its
   starting value) under a single deterministic trajectory, that's evidence
   the *mean* registration is already informative and the next real
   question is calibration/uncertainty quality (worth the `K>1` engineering
   cost). If `upper1_gap` stays large and flat despite `flow/total`
   dropping, that's evidence the predictor isn't successfully USING a
   good-enough registration -- a different bug/design gap than "the SDE
   hasn't learned the transport."

**What would make this run inconclusive rather than a clean answer either
way**: `ce_distil` dominating `loss/total` without `nll_student` itself
improving (the CE term pulls the student toward the teacher's distribution
shape but isn't itself evidence the student's OWN predictions got better;
`nll_student` and `upper1_gap` are the metrics that actually answer
question 2, not `loss/total`).

## Status

Launched `2026-09-17 14:37 UTC` (ulysses local log timestamp),
`run_name=flow-matching-registration`, MLflow experiment
`lupi-decoder-comparison` (same umbrella experiment the sibling
`lupi_bounds`/`lupi_id_token_baseline` runs report to, for direct
side-by-side comparison), checkpoint monitor `val/loss/nll_student`.

## Epoch 0 (first data point, ~530s/epoch -> ~14.7h projected for the full 100 epochs)

```
loss/total=3.8786  loss/nll_student=-0.0661  loss/nll_teacher=-0.0931
loss/ce_distil=4.0263  loss/upper1_gap=0.0269  flow/total=0.0116
flow/velocity_pos=0.0032  flow/velocity_val=0.0084  train/progress=0.0100
val/loss/nll_student=-0.0848  val/loss/nll_teacher=-0.1134  val/loss/upper1_gap=0.0287
```

Nothing conclusive yet (epoch 0 of 100), three things worth flagging while
watching the rest of the run:

- **Negative NLL is expected, not a bug**: under a continuous density (this
  `FullSupportBarDistribution`, bounded01's narrow `[0,1]` bins), `-log p(y)`
  goes negative wherever the density exceeds 1 -- unlike a discrete NLL,
  there's no floor at 0. Not investigated further, just noting it so a
  future reader doesn't mistake it for an error.
- **`flow/total` starting this low (0.0116) is very likely a curriculum
  artifact, not early mastery of hard registration**: `train/progress=0.01`
  means `sample_rho_curriculum` is still drawing from its easiest regime
  (low `rho`, close to the `rho=0` identity), so the target velocities
  themselves are small at this stage. The metric to actually judge "is
  `v_φ` learning" is its trend as `progress` climbs through the curriculum
  over the next several epochs, not its epoch-0 value in isolation.
- **`ce_distil` (4.03) dominates `loss/total`'s magnitude completely**,
  while both NLL terms are already near zero/negative. This is exactly the
  "inconclusive" failure mode flagged above before launch -- worth
  confirming over the next several epochs that `nll_student`/`upper1_gap`
  are moving because registration is improving, not just riding the CE
  term down while the student's own predictive quality stays flat.

First on-disk checkpoint not expected until epoch 5 (`min_save_epoch: 5`,
~44 min from launch at this pace) -- the comparison plot script
(`scripts/plot_bounds_vs_flow_matching_registration_1d.py`) can't run
against a real checkpoint before then.

commit: pending
