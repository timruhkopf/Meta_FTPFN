#!/bin/bash
set -e
export MLFLOW_TRACKING_URI=file:////bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/mlruns

export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=60


export ROOT=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/
export DATADIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/data
export MODELDIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/models/

EXP_NAME="Default_Experiment"
for arg in "$@"; do
    if [[ $arg == experiment_name=* ]]; then
        EXP_NAME="${arg#*=}"
    fi
done

echo "LEADER THREAD: Resolving MLflow Experiment ID for '$EXP_NAME'..."

# Capture the printed output of this python script into a bash variable
EXP_ID=$(uv run python -c "
import mlflow
import os
import sys

uri = os.environ.get('MLFLOW_TRACKING_URI', 'file://$BIGWORK/Meta_FTPFN/mlruns')
mlflow.set_tracking_uri(uri)

# Try to get existing, or create new
exp = mlflow.get_experiment_by_name('$EXP_NAME')
if exp:
    print(exp.experiment_id)
else:
    new_id = mlflow.create_experiment('$EXP_NAME')
    print(new_id)
")

echo "Success! Experiment '$EXP_NAME' is locked to ID: $EXP_ID"

# Export it so Submitit packages this environment variable for all workers
export MLFLOW_EXPERIMENT_ID=$EXP_ID

# Launch Hydra
nohup uv run /bigwork/nhwpruht/Meta_FTPFN/src/train.py "$@" > sweep_master.log 2>&1 &


#nohup bash ./slurm/template/slim.sh experiment=prototype  experiment_name=prototype +dispatch=slurm dataset.dataset_class.n_A=10,20,30,40,50,60 run_name=Baseline-n_A --multirun > sweep_master.log 2>&1 &


#experiment=prototype  \
#experiment_name=prototype \
#+dispatch=slurm \
#dataset.dataset_class.n_A=10,20,30,40,50,60 \
#run_name=Baseline-n_A \
#--multirun