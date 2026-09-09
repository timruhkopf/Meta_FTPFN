#!/bin/bash

# We can always go from bottom to top to avoid running experiments, because the properties are contained in there!

# TODO: Introduce padding for A! making sure, that in MF we respect causality!


# (0)  Meta-task-learning abiliity --------------------------
# Toy1D Layer Verification

# (1) Same task experiments ---------------------------------
# (1.1) Lookup capability:
# Same Task, fix y0, ymax

# (1.2) Lookup under scale invariance:
# Same Task, y0, ymax resampling

# (1.3) Negative Transfer avoidance
# Same Task, y0, ymax resampling, share unrelated tasks

# (2) Manifold alignment & Task similarity ------------------
# BNN interpolation
# (2.1) manifold alignment, despite deformation:
# fix y0, ymax, but same task

# (2.2) Manifold alignment under scale invariance:
# y0, ymax resampling, but same task

# (2.3) Negative transfer avoidance
# y0, ymax resampling partially unrelated

# analyse the negative transfer detection rate!

# (3) Sim-to-real transfer ----------------------------------
# Given the checkpoint from 2.3, we can do hefty evaluations on the real data

# (3.1) Context size

# (3.2) number of tasks

# analyse the negative transfer detection per benchmark
#sbatch --job-name=bnn_align --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
# $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
#dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False experiment=slurm experiment_name=bnn_y0ymax device=cuda model=ppfn_mha_hp optimizer.lr=3e-4 +trainer.trainer_class.eons=2 run_name=bnn_mha_hp2 seed=1111 device=cuda
#

# LAYERS on BNN  - ----------------------------------
sbatch --job-name=bnn_align --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_align \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_align #\
#  trainer.trainer_class.use_amp=True


sbatch --job-name=bnn_mha --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_mha \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=10 \
  run_name=bnn_mha \
  trainer.trainer_class.use_amp=True

sbatch --job-name=bnn_mha_hp_gated --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_mha_hp_gated_gated \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_mha_hp_gated \
  trainer.trainer_class.use_amp=True

sbatch --job-name=bnn_nw --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_nw \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_nw \
  trainer.trainer_class.use_amp=True




# LAYERS on SAME ----------------------------------
sbatch --job-name=same_align --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=same_task dataset.dataset_name=same_task_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=same_y0ymax  device=cuda \
  model=ppfn_align \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=same_align \
  trainer.trainer_class.use_amp=True


sbatch --job-name=same_mha --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=same_task dataset.dataset_name=same_task_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=same_y0ymax  device=cuda \
  model=ppfn_mha \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=10 \
  run_name=same_mha \
  trainer.trainer_class.use_amp=True

sbatch --job-name=same_mha_hp --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=same_task dataset.dataset_name=same_task_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=same_y0ymax  device=cuda \
  model=ppfn_mha_hp_gated \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=same_mha_hp_gated \
  trainer.trainer_class.use_amp=True

sbatch --job-name=same_nw --partition=gpu --gres=gpu:a100:1 --cpus-per-task=4 --mem=32GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=same_task dataset.dataset_name=same_task_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=same_y0ymax  device=cuda \
  model=ppfn_nw \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=same_nw \
  trainer.trainer_class.use_amp=True