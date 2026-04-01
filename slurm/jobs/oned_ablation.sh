#!/bin/bash
#SBATCH --job-name=gating_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=18:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

# Exit on any error and treat unset variables as an error
set -e
set -u

TRACKING_URI="file:////bigwork/nhwpruht/Meta_FTPFN/mlruns"
EXPERIMENT="gating_ablation"
# Define our logic branches based on the Array ID
case $SLURM_ARRAY_TASK_ID in
    0) MODE="softmin";      EXTRA_FLAGS="--global_gate --batch_size 6144 --steps 150000";     RUN_NAME="softmin" ;;
    1) MODE="distributional"; EXTRA_FLAGS="--global_gate --batch_size 6144 --steps 150000";   RUN_NAME="dist" ;;
    2) MODE="mean";         EXTRA_FLAGS="--global_gate --batch_size 6144 --steps 150000";      RUN_NAME="mean" ;;
esac

echo "Starting Run: $RUN_NAME with mode $MODE"
echo "Experiment: $EXPERIMENT"
echo "Tracking URI: $TRACKING_URI"

cd $BIGWORK/Meta_FTPFN
uv run $BIGWORK/Meta_FTPFN/src/ppfn/model/experimental/validation_manifold_adapter/training.py \
    --mlflow_tracking_uri "$TRACKING_URI" \
    --mlflow_experiment "$EXPERIMENT" \
    --mlflow_run_name "$RUN_NAME" \
    --pool_mode "$MODE" \
    $EXTRA_FLAGS \
    --batch_size 8192

echo "Completed Run: $RUN_NAME"