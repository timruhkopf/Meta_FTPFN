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
Confirmed alive and past the first CUDA/dataloader warm-up at launch+90s;
first full epoch (500 steps) not yet observed at the time of this entry --
follow-up entry once training curves are in.

commit: pending
