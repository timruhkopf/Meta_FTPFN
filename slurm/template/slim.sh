#!/bin/bash
set -e
export MLFLOW_TRACKING_URI=file:////bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/mlruns

export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=60


export ROOT=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/
export DATADIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/data
export MODELDIR=/bigwork/nhwpruht/PycharmProjects/Meta_FTPFN/models/


uv run /bigwork/nhwpruht/Meta_FTPFN/src/train.py "$@"



#experiment=prototype  \
#experiment_name=prototype \
#+dispatch=slurm \
#dataset.dataset_class.n_A=10,20,30,40,50,60 \
#run_name=Baseline-n_A \
#--multirun