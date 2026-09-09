#!/bin/bash

# Exit on error
set -e

echo "Submitting Hydra jobs to SLURM via Submitit..."
cd $BIGWORK/Meta_FTPFN

source "$BIGWORK/Meta_FTPFN/.venv/bin/activate"

# Example 1: Launch an array of 3 jobs using specific values
# This submits one SLURM job array containing 3 tasks.
uv run python $BIGWORK/Meta_FTPFN/src/train.py --multirun seed=1,2,3 learning_rate=0.01

# Example 2: Launch an array combining multiple sweeps (Grid Search)
# This will launch an array of 6 jobs (3 seeds x 2 learning rates)
# python main.py -m seed=1,2,3 learning_rate=0.01,0.001

# Example 3: Sweeping over a range of seeds
# python main.py -m seed="range(1, 10)"

echo "Submission complete. Check 'squeue -u $USER' to view your jobs."