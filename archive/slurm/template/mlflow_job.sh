#!/bin/bash
#SBATCH --job-name=ml_scaling
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out
#SBATCH --signal=B:TERM@120  # Send SIGTERM 120s before walltime ends
#SBATCH --array=1-100 # for the number of seeds

# Define Paths
LOCAL_TMP="/tmp/$USER/mlruns_$SLURM_JOB_ID"
FINAL_DEST="/bigwork/mlruns"

mkdir -p "$LOCAL_TMP"
mkdir -p "$FINAL_DEST"

# CLEANUP FUNCTION: This runs on normal exit OR on SIGTERM (time limit)
cleanup() {
    echo "Signal caught or job finishing. Syncing files..."
    cp -r "$LOCAL_TMP/." "$FINAL_DEST/"
    rm -rf "$LOCAL_TMP"
    exit 0
}

# Trap the termination signal
trap 'cleanup' SIGTERM

# Run your code in the background (&) and wait for it
# The '&' is vital so the script can 'hear' the trap signal
HYDRA_FULL_ERROR=1; python train.py tmp="$LOCAL_TMP" seed=$SLURM_ARRAY_TASK_ID &
wait $!

# Final cleanup if job finishes before time limit
cleanup