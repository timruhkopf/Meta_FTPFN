#!/bin/bash
#SBATCH --job-name=tab_pfn_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # Number of GPUs you want to use
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=your_gpu_partition

# Important for DistributedPriorDataLoader
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355

# Use torchrun to handle the LOCAL_RANK and world size automatically
torchrun --nproc_per_node=4 train_script.py --path ./data --n_gpus 4


# requires the following python setup:
MyStr=$(cat << 'EOF'

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize your model
    model = YourTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])

    # Initialize the Distributed Loader
    # n_gpus should match your total world size
    dl = DistributedPriorDataLoader(load_path="./data", n_gpus=dist.get_world_size())

    for epoch in range(100):
        # The loader automatically handles rank-based offsets
        batch = dl.get_batch(device)

        output = model(batch.x, batch.y)
        loss = F.mse_loss(output, batch.target_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()


EOF
)


# original call on PFNs4HPO.main.py based on the README
#python main.py --epochs=800 --emsize=512 --nlayers=6 --num_borders 1000
# --batch_size=25 --subsample=1 --num_gpus 1 --prior hpo_lc_pfn_bopfn_broken
# --output_file bopfn_broken_1000curves_10params_2M.pt --seq_len 1000 --num_features 12
# --border_batch_size 1000 --load_path ${PATH_CHECKPOINT_DATASET_HERE}
#  --no-full_support --linspace_borders

