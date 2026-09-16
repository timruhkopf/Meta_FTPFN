#!/bin/bash

# Points at ~/PycharmProjects/Meta_FTPFN/mlruns -- NOT Meta_FTPFN_lupi.
# Counterintuitive but empirically confirmed 2026-09-16: the 01-pretraining-*
# baseline jobs are launched FROM the Meta_FTPFN_lupi worktree (cwd/PWD
# there, confirmed via /proc/<pid>/environ), but Hydra's `${root:}`
# resolver (ppfn.utils.paths.PROJECT_ROOT, a plain
# Path(__file__).resolve().parents[3] with no env var override present in
# the process's own environment) still evaluates to the OLD
# ~/PycharmProjects/Meta_FTPFN checkout for these particular long-running
# processes -- confirmed by reading the run's own logged
# `mlflow_tracking_uri` param and its `artifact_uri` in meta.yaml, both
# file:///home/ruhkopf/PycharmProjects/Meta_FTPFN/mlruns. A fresh `uv run
# python -c "import ppfn.utils.paths"` from Meta_FTPFN_lupi resolves
# correctly to Meta_FTPFN_lupi, so whatever's stale is specific to how
# these already-running processes were launched, not a general property of
# that worktree -- unresolved, not chased down further; checkpoints
# (CheckpointCallback, same ${root:}) will land in
# ~/PycharmProjects/Meta_FTPFN/models/ for this same reason.
ssh -tt -L 5000:127.0.0.1:5000 ulysses '
    cd ~/PycharmProjects/Meta_FTPFN &&
    MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui \
      --backend-store-uri file://$HOME/PycharmProjects/Meta_FTPFN/mlruns \
      --host 127.0.0.1 --port 5000
'


# kill:
# ssh ulysses 'ss -ltnp | grep 5000
# ps -fp <pid>
# kill the parent gunicorn process (the lowest PID, or match the one whose PPID the others share) with a plain kill
  #  <pid> — no need for -9, gunicorn shuts down its workers cleanly on SIGTERM.