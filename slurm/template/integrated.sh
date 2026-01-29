#!/bin/bash
##SBATCH --job-name= # TODO set a job name here
#SBATCH --partition=amo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00 # TODO adjust time as needed
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --signal=B:TERM@120
###SBATCH --array=1-100%20 # TODO set array range as needed, limiting the concurrent jobs with %20


# (README) --------------------
# This Slurm script uv runs train.py with with mlflow and hydra.
# you can add hydra arguments to the sbatch call that will be passed to train.py
# e.g. sbatch mlflow_job.sh experiment_name=my_experiment dataset.dataset_name=my_dataset
# whether or not a gpu is allocated will be detected and the device argument passed to hydra accordingly.
# mlflow data is first written to a local tmp directory and synced back to a final destination on job completion or termination.
# This avoids excessive network file system writes during training. However, killing the job or reaching the time limit will still
# result in the data being synced back.
# Furthermore, we trap SIGUSR1 to allow manual syncing of mlflow data during job execution without terminating the job:
#  scancel --signal=USR1 <JOB_ID>
# notice: If your code is NOT programmed to handle USR1: The default behavior for most Linux applications is to terminate (crash) when receiving an unhandled USR1.
# lastly, a job registry log is kept in logs/job_registry.csv to know exactly what command was submitted with what arguments.

# Notice:
#SIGKILL: If you (or a sysadmin) run scancel without specifying a signal, Slurm often sends SIGKILL (Signal 9). SIGKILL cannot be trapped. The process dies instantly, and no syncing occurs.
#Recommendation: Always use scancel --signal=TERM if you want to stop a job early but keep your data.

# (USAGE NOTES) --------------------
# This is a template Slurm script to run MLflow experiments with Hydra configuration.
# optionally request GPUs: --gres=gpu:1  --partition=gpu --time=48:00:00 --job-name=<MANDATORY_NAME>

# alternatively use a cpu node
# --partition= --time=72:00:00 --job-name=<MANDATORY_NAME>

# post resource check :
#seff $SLURM_JOB_ID

# (EXAMPLE CALL) --------------------
#sbatch \
# --gres=gpu:1 \
# --partition=ai \
#  slurm/template/mlflow_job.sh \
#    experiment_name=bnn_interpolation_and_unrelated \
#    dataset.dataset_name=bnn_interpolation_and_unrelated  \
#    dataset.dataloader.store=True \
#    dataset.store_prior.prior.get_batch_fn._target_=ppfn.dataset.get_batch.bnn_output_interpolation.get_batch_mixed \
#    trainer.epochs=2000

# HYDRA_FULL_ERROR=1 uv run src/ppfn/train.py dataset.dataloader.store=True dataset.store_prior.prior.get_batch_fn._target_=ppfn.dataset.get_batch.bnn_output_interpolation.get_batch_mixed  trainer.epochs=2000 experiment_name=bnn_interpolation_and_unrelated dataset.dataset_name=bnn_interpolation_and_unrelated


REPO=Meta_FTPFN

# --- 1. Environment Setup (uv) ---
# Assuming .venv is in the repo root
export REPO_DIR="$BIGWORK/$REPO"
export PYTHONPATH="$REPO_DIR/src:$PYTHONPATH"
export PYTHONPATH="$REPO_DIR/ifBO_main:$PYTHONPATH"
export PYTHONPATH="$REPO_DIR/ifbo_icml2024:$PYTHONPATH"

cd "$REPO_DIR" || exit

# Check if uv is available, else load it
if ! command -v uv &> /dev/null; then
    module load uv  # Or path to your uv binary
fi

# Use 'uv run' to automatically handle the .venv
PYTHON_EXEC="uv run --frozen"

# --- 2. Dynamic Resource Detection ---
if [ -n "$SLURM_JOB_GPUS" ]; then
    echo "GPU detected: $SLURM_JOB_GPUS"
    DEVICE_ARGS="device=cuda"
    # Optional: ensure specific CUDA versions if needed
    # module load cuda/12.1
else
    echo "WARNING: No GPU allocated, falling back to CPU"
    DEVICE_ARGS="device=cpu"
fi

# --- 3. MLflow Local Scratch Setup ---
# Using a specific subdirectory for this job/array index to avoid collisions
#LOCAL_MLRUNS="/tmp/$USER/mlruns/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
FINAL_MLRUNS="$BIGWORK/$REPO/mlruns"

#mkdir -p "$LOCAL_MLRUNS"
mkdir -p  "$FINAL_MLRUNS"

# --- 4. Logging the Call (The "Registry") ---
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/job_registry.csv"
mkdir -p $LOG_DIR

COMMIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "not_a_repo")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Capture all arguments for the log
FULL_ARGS_STRING="$* $DEVICE_ARGS"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    FULL_ARGS_STRING="$FULL_ARGS_STRING seed=$SLURM_ARRAY_TASK_ID"
fi

#echo "FULL_ARGS_STRING: $FULL_ARGS_STRING"

# Ensure CSV header
if [[ ! -f "$LOG_FILE" ]]; then
    echo "timestamp,commit,job_id,array_id,device,args" >> "$LOG_FILE"
fi

# Append entry BEFORE execution (so you know what's running/queued)
echo "\"$TIMESTAMP\",\"$COMMIT_HASH\",\"$SLURM_JOB_ID\",\"$SLURM_ARRAY_TASK_ID\",\"$DEVICE_ARGS\",\"$FULL_ARGS_STRING\"" >> "$LOG_FILE"

# --- 5.0 Robust Cleanup & Sync ---
# Since we are using mlflow with a tmp file directory storage, we need to move the data back
# regardless of how the job ends (normal or killed due to time limit)
#cleanup() {
#    echo "Terminating: Syncing MLflow data from $LOCAL_MLRUNS to $FINAL_MLRUNS"
#    # rsync is safer than cp for interrupted transfers
#    rsync -auq "$LOCAL_MLRUNS/" "$FINAL_MLRUNS/"
#    rm -rf "$LOCAL_MLRUNS"
#    exit 0
#}
#
## Trap SIGTERM (Slurm walltime) and EXIT (Normal completion)
#trap 'cleanup' SIGTERM EXIT

# --- 5.1 OPTIONAL: Poke jobs to update status:
# poking is done using :
# scancel --signal=USR1 <JOB_ID>
# Check your logs and destination folder:
# Check logs/%j.out: It should say "Syncing MLflow data..."
#sync_only() {
#    echo "Manual sync triggered. Moving data to $FINAL_MLRUNS..."
#    rsync -auq "$LOCAL_MLRUNS/" "$FINAL_MLRUNS/"
#}
#trap 'sync_only' SIGUSR1

# --- 6. Execution ---
export HYDRA_FULL_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Starting Job: $SLURM_JOB_ID | Array Task: $SLURM_ARRAY_TASK_ID"

# DEBUG print:
# shellcheck disable=SC2145
echo "Executing CMD: train.py $@ $DEVICE_ARGS"
# Pass all script arguments ($@) to Hydra

FINAL_CMD_ARGS=(
    "$@"
    "$DEVICE_ARGS"
#    "seed=$SLURM_ARRAY_TASK_ID"
#    "hydra.run.dir=$LOCAL_MLRUNS/hydra_logs"
    "hydra.job.chdir=True"
    "mlflow.tracking_uri=file://$FINAL_MLRUNS"
)
echo "FINAL_CMD_ARGS: ${FINAL_CMD_ARGS[*]}"
$PYTHON_EXEC train.py "${FINAL_CMD_ARGS[@]}" &

    # move the hydra run dir to the local tmp as well to avoid writes to file system

# Capture the PID of the python process (the most recent background job: $!)
# This allows the trap to work properly
PY_PID=$!

# Wait for the background process by its PID
wait $PY_PID
