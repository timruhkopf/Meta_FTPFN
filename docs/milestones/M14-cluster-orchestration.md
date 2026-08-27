# M14 — Cluster orchestration (sweeps over seeds × priors/benchmarks)

## Status: sketch — blocked on M13's tracking backend being decided first

Don't build sweep scripts on top of a tracking backend that's still going to
change shape — get M13 settled, then this becomes mostly mechanical.

## Goal

Scripts that take Hydra overrides and dispatch a full sweep — seeds × priors,
eventually seeds × benchmarks — as a single `--multirun` job array, riding on
`configs/deployment/slurm.yaml` (`hydra-submitit-launcher`, already built and
verified in the earlier configs milestone).

## Deliverables (once unblocked)

1. A launch script wrapping something like:
   ```bash
   python src/train.py --multirun deployment=slurm \
       seed=1,2,3,4,5 prior=toy,harmonics,bnn \
       experiment_name=03-sweep-<name>
   ```
   that also resolves and exports `MLFLOW_EXPERIMENT_ID` once (from the login
   node, before `sbatch`) per the fast-path `MLflowCallback._setup_experiment()`
   already relies on — the script's job is to call that resolution once, not
   reimplement the race-avoidance logic that already exists.
2. The M13 aggregation step wired in as a `--dependency=afterany:$JOBID`
   follow-up (per `archive/slurm/eval_job.md`'s documented pattern), not a
   manually-triggered afterthought.
3. Update `archive/slurm/eval_job.md`'s guidance on `SIGUSR1`/`scancel`
   interaction with follow-up jobs (it flags this as untested — "make sure to
   test this behavior in a safe environment before deploying it in production")
   — this milestone is exactly that safe-environment test, not a place to carry
   the same caveat forward untested again.

## Acceptance criteria (once unblocked)

- [ ] A small real sweep (e.g. 3 seeds × 2 priors) completes end-to-end:
      submitted, dispatched via `submitit_slurm`, tracked without a lost or
      corrupted run (M13's guarantee), aggregated automatically afterward.
- [ ] A deliberately-timed-out task in the sweep still contributes a partial
      result (exercises M13's partial-run handling, not just the happy path).

## Non-goals

- Not designing the tracking backend here — that's M13, already a hard enough
  problem on its own.
- Not real-benchmark-specific dispatch logic (task-suite iteration, etc.) —
  that's M10's concern once it's unblocked; this milestone's sweep axes are
  seeds/priors, which already exist earlier in the roadmap.
