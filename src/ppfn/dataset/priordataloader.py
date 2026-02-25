from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from typing import Callable
from pathlib import Path
import pathlib
import os

import numpy as np
import cloudpickle
import submitit
import torch

import logging

from ppfn.utils.mybatch import MyBatch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StoredPriorDataset(torch.utils.data.Dataset):
    """Handles the physical structure of the data on disk."""

    def __init__(
            self,
            storage_path: str,
            folder_name: str,
            get_batch_fn: Callable = None,
            shuffle=False,
            sample_on_init_kwargs: dict = None,

    ):
        super().__init__()
        self.storage_path = Path(storage_path) / folder_name
        self.name = folder_name
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.chunk_files: list[Path] = sorted(
            [f for f in self.storage_path.glob("chunk_*.pt")
             if f.is_file() and not f.name.startswith('.')]  # avoid rsync temp files during debugging
        )

        if len(self.chunk_files) == 0 and sample_on_init_kwargs is not None:
            # optional generation
            self.store_prior(
                local=True, # we don't want a trainer job to first spawn submitit jobs to then crash
                get_batch_fn=get_batch_fn,
                **sample_on_init_kwargs,
            )
            self.chunk_files: list[Path] = sorted(
                [f for f in self.storage_path.glob("chunk_*.pt")
                 if f.is_file() and not f.name.startswith('.')]  # avoid rsync temp files during debugging
            )
            # raise FileNotFoundError( f"No chunk files found in {self.storage_path}. Please run store_prior() to generate data.")


        if shuffle and len(self.chunk_files) > 0:
            self.chunk_files = np.random.permutation(self.chunk_files).tolist()

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


        # with open(chunk_file, "rb") as f:
        # data = torch.load(chunk_file, map_location='cpu')
        flag = True
        while flag:
            try:
                chunk_file = self.chunk_files[chunk_id]
                # map_location='cpu' is good, but let's ensure the file is actually readable
                data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                flag = False
            except Exception as e:
                logger.error(f"Failed to load chunk {self.chunk_files[chunk_id]}. File might be corrupted.")
                # pop the file
                self.chunk_files.pop(chunk_id)
                # raise RuntimeError(f"Corrupted file detected: {chunk_file}") from e
        # consider mmap for processes on the same machine to reduce memory overhead, but beware of potential issues with concurrent access

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
        batch = chunk_data[f"batch_{batch_idx}"]

        # TODO Padding
        # Since we are training on meta-tasks, with a target task and multiple related tasks, all of which may be at varying
        # x_train sizes -- especially the target task, that has usually less training data than the related tasks,
        # we need to pad the x and y of the target task to the maximum sequence length in the batch
        # FIXME: Notice, that we need to know in the train part, which tokens correspond to later fidelities PER HP (!)
        #  so that we can decide for each hp config, which tokens we want to remove by padding, ensuring, that we are
        #  not breaking sequence logic by padding in the middle of the sequence

        # upcast
        batch = {
            k: v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v
            for k, v in batch.items()
        }

        return MyBatch(
            **{k: v for k, v in batch.items()},
            target_y=batch["y"],
        )

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
            half_precision=False,
            single_eval_pos=None,
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
                half_precision=False,
                single_eval_pos=None,
        ):
            chunk_storage = {}
            for i, chunk in enumerate(range(chunk_size // batch_size)):

                # Train-test split position sampling for the batch.
                # Notice, that this must be the same for the entire batch due to the split MultiHeadAttention as implemented in the PFN
                if single_eval_pos is None:
                    if eval_pos_sampler is None:
                        # sample single eval pos log-uniformly ({1, ..., seq_len} log-uniformly - 1)
                        single_eval_pos = int(
                            np.floor(np.exp(np.random.uniform(0, np.log(seq_len + 1)))) - 1
                        )
                    else:
                        single_eval_pos = eval_pos_sampler()

                batch = get_batch_fn(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    single_eval_pos=single_eval_pos,
                    **get_batch_kwargs,
                )

                x, y, style = batch.x, batch.y, batch.style if hasattr(batch, "style") else None

                if half_precision:
                    x = x.half() if x.dtype == torch.float else x
                    y = y.half() if y.dtype == torch.float else y
                    style = style.half() if style is not None and style.dtype == torch.float else style

                chunk_storage[f"batch_{i}"] = {
                    "x": x,
                    "y": y,
                    # "target_y": batch.target_y,
                    "style": style,
                    # "mask": batch.src_key_padding_mask,
                    "single_eval_pos": batch.single_eval_pos
                }
                # todo on USR1 signal of process (--signal=B:TERM@120) break and dump the current
                # progress

            chunk_file = Path(path) / f"chunk_{chunk_id}.pt"
            torch.save(chunk_storage, chunk_file)
            # TODO src_key_padding_mask stored as torch.BoolTensor to save space or sample on demand?


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
                "single_eval_pos": single_eval_pos,
                "get_batch_kwargs": batch_kwargs,
                "half_precision": half_precision,
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
    from ppfn.dataset.get_batch.get_related_batch import get_batch

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
        dataset = StoredPriorDataset(storage_path=tmpdir, get_batch_fn=get_batch)
        dataset.store_prior(
            path=path,
            chunk_id=chunk_id,
            chunk_size=chunk_size,
            n_chunks=2,
            batch_size=batch_size,
            seq_len=seq_len,
            get_batch_fn=get_batch,
            local=True,
            half_precision=False,
            batch_kwargs={'num_features': 12}
        )

        # check that files are created and can be loaded
        assert torch.load(
            Path(tmpdir) / f"chunk_{chunk_id}.pt") is not None, "Chunk file was not created successfully."

        dataset2 = StoredPriorDataset(
            storage_path=tmpdir,
        )
        dataset2[1]

        local_rank = int(0)  # os.environ["LOCAL_RANK"]
        # torch.cuda.set_device(local_rank)
        setup_ddp(local_rank, 1)
        dataloader = prepare_dataloader(dataset, batch_size=32)

        for batch in dataloader:
            # Process your batch here
            pass

        cleanup()
