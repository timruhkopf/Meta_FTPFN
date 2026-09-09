#!/bin/bash

# ==============================================================================
# SLURM RESOURCE CONFIGURATION
# ==============================================================================
#SBATCH --job-name=mlflow_train
#SBATCH --partition=amo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=B:TERM@120
##SBATCH --array=1-100%20

# ==============================================================================
# DIRECTORY & PROJECT SETUP
# ==============================================================================
REPO="Meta_FTPFN"
REPO_DIR="$BIGWORK/$REPO"
LOG_DIR="$REPO_DIR/logs"
FINAL_MLRUNS="$REPO_DIR/mlruns"

# Path setup
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/ifBO_main:$REPO_DIR/ifbo_icml2024:$PYTHONPATH"
cd "$REPO_DIR" || exit

# --- Environment Bootstrapping ---
if ! command -v uv &> /dev/null; then
    module load uv
fi
PYTHON_EXEC="uv run --frozen"

# ==============================================================================
# DYNAMIC RESOURCE & ARGUMENT DETECTION
# ==============================================================================
# 1. Detect Device
if [ -n "$SLURM_JOB_GPUS" ]; then
    echo ">>> GPU detected: $SLURM_JOB_GPUS"
    DEVICE_ARGS="device=cuda"
else
    echo ">>> WARNING: No GPU allocated, falling back to CPU"
    DEVICE_ARGS="device=cpu"
fi

# 2. Handle Job Arrays / Seeds
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    SEED_ARG="seed=$SLURM_ARRAY_TASK_ID"
fi

# 3. Registry Logging (CSV)
mkdir -p "$LOG_DIR"
mkdir -p "$FINAL_MLRUNS"
LOG_FILE="$LOG_DIR/job_registry.csv"

COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "not_a_repo")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [[ ! -f "$LOG_FILE" ]]; then
    echo "timestamp,commit,job_id,array_id,device,args" >> "$LOG_FILE"
fi

# ==============================================================================
# SIGNAL HANDLING & CLEANUP
# ==============================================================================
# Note: If using LOCAL_MLRUNS in the future, uncomment the sync logic below.
# cleanup() {
#    echo "Terminating: Syncing data..."
#    rsync -auq "$LOCAL_MLRUNS/" "$FINAL_MLRUNS/"
#    exit 0
# }
# trap 'cleanup' SIGTERM EXIT

# ==============================================================================
# EXECUTION
# ==============================================================================
export HYDRA_FULL_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Prepare arguments array for Hydra
FINAL_CMD_ARGS=(
    "$@"
    "$DEVICE_ARGS"
    "$SEED_ARG"
    "mlflow.tracking_uri=file://$FINAL_MLRUNS"
)

echo "----------------------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
echo "Commit: $COMMIT_HASH"
echo "Running: $PYTHON_EXEC $REPO_DIR/src/ppfn/train.py ${FINAL_CMD_ARGS[*]}"
echo "----------------------------------------------------------------"

# Log the submission to registry
echo "\"$TIMESTAMP\",\"$COMMIT_HASH\",\"$SLURM_JOB_ID\",\"$SLURM_ARRAY_TASK_ID\",\"$DEVICE_ARGS\",\"$*\"" >> "$LOG_FILE"

# Run Process
$PYTHON_EXEC "$REPO_DIR/src/ppfn/train.py" "${FINAL_CMD_ARGS[@]}" &

# Capture PID for trap handling
PY_PID=$!
wait $PY_PID