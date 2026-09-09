#!/bin/bash
#SBATCH --job-name=mlflow_safe
#SBATCH --array=1-50
#SBATCH --output=logs/%a.out

DB_PATH="/bigwork/nhr/$USER/mlflow_project/mlflow.db"
LOCK_DIR="${DB_PATH}.init_lock"

# 1. Atomic Race Condition Protection
# Only the very first process to reach this line will succeed in 'mkdir'
if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Task $SLURM_ARRAY_TASK_ID: Initializing DB infrastructure..."
    # Create file and flip the bit to WAL mode persistently
    sqlite3 "$DB_PATH" "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"
    # Leave a marker so others know init is done
    touch "${DB_PATH}.ready"
else
    # Everyone else waits until the 'ready' file appears
    echo "Task $SLURM_ARRAY_TASK_ID: Waiting for Leader to finish init..."
    while [ ! -f "${DB_PATH}.ready" ]; do sleep 1; done
fi

# 2. Execute Training
export MLFLOW_TRACKING_URI="sqlite:///${DB_PATH}?timeout=60"
uv python train.py