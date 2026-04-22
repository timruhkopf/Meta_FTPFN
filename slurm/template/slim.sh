#!/bin/bash
set -e
export MLFLOW_TRACKING_URI=file:////bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/mlruns

export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=60


export ROOT=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/
export DATADIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/data
export MODELDIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/models/

# 1. Extract the experiment_name from the Hydra arguments
EXP_NAME="Default_Experiment" # Fallback
for arg in "$@"; do
    if [[ $arg == experiment_name=* ]]; then
        EXP_NAME="${arg#*=}" # Strips out "experiment_name=" and keeps the value
    fi
done

echo "LEADER THREAD: Pre-creating MLflow Experiment '$EXP_NAME' to prevent race conditions..."

# 2. Pre-create the experiment using the extracted name
uv run python -c "
import mlflow
import os
uri = os.environ.get('MLFLOW_TRACKING_URI', 'file://$ROOT/mlruns')
mlflow.set_tracking_uri(uri)
mlflow.set_experiment('$EXP_NAME')
"

# 3. Launch the distributed multirun
# Make sure to run this via nohup or tmux so the login node doesn't time out!
uv run /bigwork/nhwpruht/Meta_FTPFN/src/train.py "$@"


#HYDRA_FULL_ERROR=1;bash ./slurm/template/slim.sh experiment=prototype  experiment_name=prototype +dispatch=slurm dataset.dataset_class.n_A=10,20,30,40,50,60 run_name=Baseline-n_A --multirun


#experiment=prototype  \
#experiment_name=prototype \
#+dispatch=slurm \
#dataset.dataset_class.n_A=10,20,30,40,50,60 \
#run_name=Baseline-n_A \
#--multirun