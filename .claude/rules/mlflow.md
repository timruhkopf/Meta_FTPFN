---
paths:
  - "src/ppfn/trainer/callbacks/mlflow_cb.py"
  - "configs/callbacks/**"
---

# MLflow conventions

`configs/callbacks/mlflow.yaml` (rebuilt 2026-08-27) now sets
`mlflow_tracking_uri: ${mlflow_tracking_uri}` — previously this field was
unset in config, so the experiment-level `mlflow_tracking_uri` (set in
`configs/experiment/*.yaml`) silently never reached the callback and it always
fell back to the `MLFLOW_TRACKING_URI` env var. Fixed, not a behavior change to
rely on being reverted.

## Current pattern (as implemented in `mlflow_cb.py`)

- `MLflowCallback` owns the run's lifecycle itself: `on_train_start` calls
  `mlflow.start_run(...)`, `log_on_train_end` calls `mlflow.end_run()`. There is no
  injected tracker/sink — the callback talks to the global `mlflow` module
  directly. Match this pattern for now rather than inventing a parallel one; see
  "Known debt" below if you're deliberately changing it.
- Experiment resolution goes through `_setup_experiment()`, which first checks
  `MLFLOW_EXPERIMENT_ID` in the environment (set by the SLURM leader node to avoid
  an NFS race on `mlflow.get_experiment_by_name`), and only falls back to
  `get_experiment_by_name`/`create_experiment` for local/interactive runs. Don't
  remove the env-var fast path — it exists because of a real race condition on the
  cluster's shared filesystem, not as a debug shortcut.
- Run names are dynamically generated from Hydra task overrides
  (`get_dynamic_run_name`), truncated to 97 chars, and are for human
  readability in the MLflow UI only — nothing in the codebase parses a run name
  back into structured data, and you shouldn't add code that does. Identity/
  filtering goes through tags and params instead:
  - `mlflow.source.git.commit` tag ← `githash()`.
  - Uncommitted diff (if any) logged as a `scripts/diff.patch` artifact — this is
    a *debug-run* convenience; `00-debug-*` runs are exempt from the clean-tree
    check (see `hydra.md`), so this is how you still recover what was actually run.
  - `mlflow.folder` tag ← `os.getcwd()` (the Hydra run dir).
  - Params ← flattened Hydra task overrides (`key=value` pairs from
    `HydraConfig.get().overrides.task`), plus `hydra_dir` and `mlflow_run_id`.
    This deliberately logs only the *overridden* keys, not the full resolved
    config tree — keep it that way; logging hundreds of full-config params per
    run makes the params table unusable and hits MLflow's per-run param limits
    on large sweeps.
- Metrics are logged once per epoch (`log_on_epoch_end`) with
  `step = eon * trainer.epochs + epoch` — if you add eon-aware logging elsewhere,
  reuse this exact step formula so metrics from the same run stay on one
  monotonic step axis instead of resetting per eon.

## Known debt (don't silently "fix" — this is a real tradeoff, not an oversight)

The callback instantiating and owning the MLflow run (rather than an entry point
creating a run and injecting a tracker into trainer+callbacks) means:
- A hypothetical `evaluate.py` with no trainer/callbacks can't reuse this tagging
  logic without either duplicating it or fabricating a fake trainer.
- There's no `NoOpTracker` equivalent — tests and smoke runs that build a
  `PPFNTrainer` with an `MLflowCallback` in `callbacks` will hit real MLflow calls
  unless the callback is simply omitted from the `callbacks` dict for that run
  (this is in fact how `tests/` avoids it — check before assuming a test needs
  MLflow mocking).

If asked to fix this, the shape is: move `mlflow.start_run`/`end_run` into
`train.py`'s `run()` as a context manager, and pass a thin tracker object into
`PPFNTrainer` and callbacks instead of importing `mlflow` directly inside
`MLflowCallback`. That's a real refactor (touches `trainer.py`'s callback
handling too) — don't do it opportunistically as part of an unrelated change.
