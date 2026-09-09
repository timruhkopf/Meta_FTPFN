#!/bin/bash
DATASET=bnn_interpolation
export $(grep -v '^#' .env_ulysses | xargs)
uv run ./src/train.py dataset.sample_prior=True dataset=$DATASET experiment_name=asdf dataset.store_prior.half_precision=True \
dataset.store_prior.get_batch_fn.transform.resample_y0_ymax=True \
dataset.dataset_name=bnn_interp_y0ymax_unrelated20 \
+dataset.store_prior.get_batch_fn.share_unrelated=0.2 \
dataset.dataset_class.storage_path=/home/ruhkopf/PycharmProjects/Meta_FTPFN/data/sepbigger0/ \
dataset.store_prior.local=False


DATASET=same_task
export $(grep -v '^#' .env_ulysses | xargs)
uv run ./src/train.py dataset.sample_prior=True dataset=$DATASET experiment_name=asdf dataset.store_prior.half_precision=True \
dataset.store_prior.get_batch_fn.transform.resample_y0_ymax=True \
dataset.dataset_name=same_task_y0ymax_unrelated20 \
+dataset.store_prior.get_batch_fn.share_unrelated=0.2 \
dataset.store_prior.n_chunks=300 \
dataset.dataset_class.storage_path=/home/ruhkopf/PycharmProjects/Meta_FTPFN/data/sepbigger0/ \
dataset.store_prior.local=False


#swatch -t R
#swatch() { watch -n 1 "squeue --me --format=\"%.10i %.9P %.20j %.2t %.10M %.10L %.R\" $*" }


# Run Training:

#sbatch --partition=gpu.test --gres=gpu:1 --time=00:15:00 --job-name=gpu.test \
sbatch --job-name=train_bnn_align --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=8GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_align \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_align


sbatch --job-name=train_bnn_mha --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=8GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_mha \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=10 \
  run_name=bnn_mha

sbatch --job-name=train_bnn_mha_hp --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=8GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_mha_hp \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_mha_hp

sbatch --job-name=train_bnn_nw --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=8GB --time=12:00:00 \
 $BIGWORK/Meta_FTPFN/slurm/template/integrated.sh \
 dataset=bnn_interpolation dataset.dataset_name=bnn_interp_y0ymax_unrelated20 dataset.sample_prior=False\
  experiment=slurm \
  experiment_name=bnn_y0ymax  device=cuda \
  model=ppfn_nw \
  optimizer.lr=3e-4 \
  +trainer.trainer_class.eons=2 \
  run_name=bnn_nw





# sbatch --partition=gpu --gres=gpu:1 --time=12:00:00 slurm/template/integrated.sh dataset=bnn_interpolation dataset.dataset_name=bnn_interpolation dataset.sample_prior=False device=cuda experiment=slurm experiment_name=trials run_name=bnn_0.2_unrelated

#sbatch --partition=gpu --gres=gpu:1 --time=12:00:00 slurm/template/integrated.sh dataset=same_task dataset.dataset_name=same_task_partially_unrelated dataset.sample_prior=False device=cuda experiment=slurm experiment_name=trials run_name=same_task_partially_unrelated


# copy data from cluster to local machine:
