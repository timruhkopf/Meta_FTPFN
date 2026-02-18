import os
import pathlib
from typing import Callable

import numpy as np
import cloudpickle
import submitit
import torch

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch.distributed as dist


class StoredPriorDataset(torch.utils.data.Dataset):
    """Handles the physical structure of the data on disk."""

    def __init__(
        self,
        storage_path: str,
        get_batch_fn: Callable = None,
        # sample_on_init: bool = False,
        # sample_kwargs: dict = {},
        # submitit_kwargs: dict = {}
    ):
        super().__init__()
        self.storage_path = pathlib.Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.chunk_files = sorted(self.storage_path.glob("chunk_*.pkl"))
        self.get_batch_fn = get_batch_fn

        # FIXME: this is just a temporary fix to sample prior on initialization
        # if sample_on_init and len(self.chunk_files) == 0:
        #     logger.info("No chunk files found, sampling prior on initialization.")
        #     self.store_prior(**sample_kwargs, **submitit_kwargs)
        #     self.chunk_files = sorted(self.storage_path.glob("chunk_*.pkl"))
        # Cache for the currently active chunk to avoid redundant disk I/O
        self.current_chunk_id = -1
        self.cached_chunk_data = None

        if not self.chunk_files:
            self.items_per_chunk = 0
            self.total_size = 0
        else:
            # 1. Analyze the first chunk to determine capacity
            first_chunk = self.load_chunk(0)
            self.items_per_chunk = len(first_chunk)
            self.total_size = len(self.chunk_files) * self.items_per_chunk

    def load_chunk(self, chunk_id: int):
        # Optimization: Only load from disk if we aren't already holding this chunk
        if chunk_id == self.current_chunk_id:
            return self.cached_chunk_data

        chunk_file = self.storage_path / f"chunk_{chunk_id}.pkl"
        with open(chunk_file, "rb") as f:
            data = cloudpickle.load(f)

        self.current_chunk_id = chunk_id
        self.cached_chunk_data = data
        return data

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 2. Translate global index to chunk/batch coordinates
        # chunk_idx = idx // K
        # batch_idx = idx % K
        chunk_idx, batch_idx = divmod(idx, self.items_per_chunk)

        chunk_data = self.load_chunk(chunk_idx)
        return chunk_data[batch_idx]

    def store_prior(
        self,
        n_chunks,
        chunk_size,
        batch_size,
        seq_len,
        get_batch_fn,
        batch_kwargs={},
        eval_pos_sampler=None,
        local=False,
        **submitit_kwargs,
    ):
        """locally or via submitit."""

        def sample_chunk(
            path,  # must contain the partition folder
            chunk_id,
            chunk_size,
            batch_size,
            seq_len,
            get_batch_fn,
            eval_pos_sampler=None,
            get_batch_kwargs={},
        ):
            chunks = []
            for chunk in range(chunk_size // batch_size):
                # sample the train-test-split position for the entire batch
                if eval_pos_sampler is None:
                    # sample single eval pos log-uniformly ({1, ..., seq_len} log-uniformly - 1)
                    single_eval_pos = int(
                        np.floor(np.exp(np.random.uniform(0, np.log(seq_len + 1)))) - 1
                    )
                else:
                    single_eval_pos = eval_pos_sampler()
                assert single_eval_pos < seq_len

                batch = get_batch_fn(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    single_eval_pos=single_eval_pos,
                    **get_batch_kwargs,
                )
                chunks.append(batch)
                # todo on USR1 signal of process (--signal=B:TERM@120) break and dump the current
                # progress

            chunk_file = pathlib.Path(path) / f"chunk_{chunk_id}.pkl"
            with open(chunk_file, "wb") as file:
                cloudpickle.dump(chunks, file)

        # define the jobs
        chunk_tasks = [
            {
                "path": self.storage_path,
                "chunk_id": chunk_id,
                "chunk_size": chunk_size,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "get_batch_fn": get_batch_fn,
                "eval_pos_sampler": eval_pos_sampler,
                "get_batch_kwargs": batch_kwargs,
            }
            for chunk_id in range(n_chunks)
        ]

        if local:
            for task in chunk_tasks:
                sample_chunk(**task)
        else:
            executor = submitit.AutoExecutor(
                folder=self.storage_path / "tmp/submitit_logs"
            )
            executor.update_parameters(**submitit_kwargs)
            job_group = executor.map_array(
                sample_chunk, *zip(*[d.values() for d in chunk_tasks])
            )
            logger.info(f"Submitted {len(chunk_tasks)} jobs. Group ID: {job_group}")


# FIXME: Untested DDP dataloader preparation
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def prepare_dataloader(
    dataset: Dataset, batch_size: int, pin_memory: bool = True, num_workers=4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,  # Good practice for faster data transfer to GPU
        shuffle=False,  # MUST be False; the sampler handles shuffling
        sampler=DistributedSampler(
            dataset,
            shuffle=True,  # Shuffle happens within the sampler in DDP
            seed=42,
        ),
        num_workers=num_workers,  # Parallelize data loading
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from ppfn.dataset.get_batch.bnn_output_interpolation import get_batch_mixed

    import os
    import torch.distributed as dist

    def setup_ddp(rank, world_size):
        # These environment variables are usually set by torchrun
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def cleanup():
        dist.destroy_process_group()

    with TemporaryDirectory() as tmpdir:
        path = tmpdir
        chunk_id = 0
        chunk_size = 64
        batch_size = 32
        seq_len = 1000
        dataset = StoredPriorDataset(storage_path=tmpdir, get_batch_fn=get_batch_mixed)
        dataset.sample_chunk(
            path=path,
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            batch_size=batch_size,
            seq_len=seq_len,
            get_batch_fn=get_batch_mixed,
            num_features=12,
            single_eval_pos=1000,
        )

        dataset2 = StoredPriorDataset(
            storage_path=tmpdir,
        )
        dataset2[0]

        local_rank = int(0)  # os.environ["LOCAL_RANK"]
        # torch.cuda.set_device(local_rank)
        setup_ddp(local_rank, 1)
        dataloader = prepare_dataloader(dataset, batch_size=32)

        for batch in dataloader:
            # Process your batch here
            pass

        cleanup()
