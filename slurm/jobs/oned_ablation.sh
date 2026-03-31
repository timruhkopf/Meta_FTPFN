#!/bin/bash
#SBATCH --job-name=gating_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/ablation_%A_%a.out

TRACKING_URI="file:////bigwork/nhwpruht/Meta_FTPFN/mlruns"
EXPERIMENT="gating-ablation"

# Define our logic branches based on the Array ID
case $SLURM_ARRAY_TASK_ID in
    0) MODE="mean";         EXTRA_FLAGS="--steps 200000";  RUN_NAME="pool_mean_ptwise_OFF" ;;
    1) MODE="softmax";      EXTRA_FLAGS="";               RUN_NAME="pool_softmax_ptwise_OFF" ;;
    2) MODE="softmin";      EXTRA_FLAGS="";               RUN_NAME="pool_softmin_ptwise_OFF" ;;
    3) MODE="distributional"; EXTRA_FLAGS="";             RUN_NAME="pool_dist_ptwise_OFF" ;;
    4) MODE="mean";         EXTRA_FLAGS="--pointwise";    RUN_NAME="pool_mean_ptwise_ON" ;;
    5) MODE="mean";         EXTRA_FLAGS="--pointwise --global_gate"; RUN_NAME="pool_mean_ptwise_global_ON" ;;
esac

echo "Starting Run: $RUN_NAME with mode $MODE"

python $BIGWORK/Meta_FTPFN/src/ppfn/model/experimental/validation_manifold_adapter/training.py \
    --mlflow_tracking_uri "$TRACKING_URI" \
    --mlflow_experiment "$EXPERIMENT" \
    --mlflow_run_name "$RUN_NAME" \
    --pool_mode "$MODE" \
    $EXTRA_FLAGS \
    --batch_size 8192