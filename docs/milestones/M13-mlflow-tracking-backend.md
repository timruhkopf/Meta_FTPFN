# M13 — MLflow tracking backend for concurrent, time-bound SLURM jobs

## Status: sketch, but the constraint is real and known now — worth designing
## for even before M11/M12 unblock, since it shapes how those get built

## The constraint (from you, 2026-08-27 — don't re-relitigate these)

- No persistent Postgres: you can run one as a SLURM job, but SLURM jobs on
  this cluster are wall-clock-bound (max 72h) and get killed at timeout,
  taking the DB down with anything still writing to it.
- SQLite is empirically not viable for this concurrency level — see below,
  already tested, not a re-open-and-reconsider item.
- An older MLflow version with filesystem-store tracking worked, but you've
  moved past that version and it's "fragile" and no longer supported going
  forward — don't reintroduce a pinned old MLflow just to keep that path alive.

## What's already been tried — in `archive/slurm/sqlite_attempt/`

This directory holds a genuinely rigorous prior attempt, not a guess:
`sqlite.sh` (an atomic `mkdir`-based leader-election lock so only one of a
50-task array initializes the DB with WAL mode, everyone else waits on a
ready-file), `train_worker.py` (a synchronized-start "flood" test that lines
up all array tasks to hit the DB at the same instant to force collisions), and
`local_concurrent.py` (a 10-process local stress test with the same
synchronized-start pattern, auditing how many runs finish `FINISHED` at the
end). This is the validation methodology to reuse for whatever replaces
SQLite here — don't judge a new approach as "probably fine," stress-test it the
same way before trusting it.

## The actual problem has two independent parts — don't conflate them

1. **Experiment-creation race**: many array tasks starting simultaneously all
   calling `get_experiment_by_name`/`create_experiment` at once. **Already
   solved** — `MLflowCallback._setup_experiment()` (`.claude/rules/mlflow.md`)
   checks `MLFLOW_EXPERIMENT_ID` in the environment first, meant to be resolved
   once by a leader/login-node process and exported before `sbatch`. Don't
   redesign this part.
2. **Concurrent run/metric-write race**: every array task logging metrics
   throughout its own (possibly hours-long) run, all hitting the same backend
   store at once. **This is the actual open problem** — SQLite's single-writer
   lock can't sustain it (empirically shown), and a shared FileStore over NFS
   has its own known correctness issues under true concurrent writers.

## Recommended direction: eliminate the concurrent writers, don't out-engineer them

Since each array task **owns** its own run for the task's entire lifetime, the
robust fix is architectural, not a better lock: each task logs to a
**private, node-local store** (no contention possible — it's the only writer),
and a **single, sequential aggregation step** — run after the array via SLURM
job dependencies, exactly the pattern you already documented in
`archive/slurm/eval_job.md` (`sbatch --dependency=afterok:$TRAIN_ID
aggregate.sh`) — imports each task's local run into the one shared MLflow
store afterward, one at a time. Zero concurrent writers to the shared store by
construction, using only primitives you already know work (local disk, SLURM
dependencies) rather than anything contingent on cluster policy.

Two things this needs that aren't decided yet:
- What "local" means for a task that gets killed by the 72h wall clock before
  finishing — the local run data up to that point needs to survive the kill
  (write to a path that isn't cleaned up automatically, e.g. persistent scratch
  rather than a job-scoped `$TMPDIR`) so the aggregation step can still import
  a partial run rather than losing it entirely.
- Whether the aggregation step imports via MLflow's own APIs (`mlflow.client`
  copying runs from one tracking URI to another) or a lower-level copy of the
  FileStore's run directory structure — check what MLflow 2.19.0 actually
  supports here before assuming either works.

## Alternative worth one conversation, not more design time here

Ask cluster support whether Ulysses (or your institution) offers *any*
persistent, IT-managed service outside the SLURM allocation model — a
database-as-a-service, a always-on VM you can request, etc. If yes, this whole
milestone simplifies to "point `MLFLOW_TRACKING_URI` at it." Worth a five-minute
email before investing in the aggregation-step design above; not worth
redesigning around an uncertain "maybe" in the meantime.

## Acceptance criteria

- [ ] A stress test structurally like `local_concurrent.py`/`train_worker.py`
      (same synchronized-flood methodology), run against whatever approach is
      chosen, at the same scale you tested SQLite at (~50 concurrent writers),
      with **zero** lost or corrupted runs at the end.
- [ ] A task killed mid-run (simulate the 72h timeout with a shorter signal in
      testing — `PPFNTrainer` already handles `SIGUSR1`/`SIGTERM` via
      `GracefulExit`, see `src/ppfn/utils/gracefull_exit.py`) still contributes
      its partial run to the aggregated store.
- [ ] Documented in `.claude/rules/mlflow.md` once decided, so this doesn't
      get re-litigated by a future session that hasn't seen this file.

## Non-goals

- Not re-trying bare SQLite or a bare shared-NFS FileStore under true
  concurrent writers — both are already empirically falsified for this
  workload.
- Not building a general-purpose distributed tracking system — the aggregation
  step only needs to handle this repo's own job-array shape.
