from pathlib import Path
from typing import Callable

import cloudpickle
from torch.utils.data import IterableDataset

from pfns4hpo.priors.utils import PriorDataLoader
from ppfn.dataset.get_batch.deprec.same_task import get_batch as same_task_get_batch, Prior
from ppfn.model.mymodel.ft_ppfn import MyBatch


class SyntheticTaskMetaTestDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        single_eval_pos: int,
        get_batch_fn: Callable,
        name="synthetic_task",
        device="cpu",
        **kwargs,
    ):
        self.data_path = Path(data_path) / name
        self.name = f"{name}_{single_eval_pos}"
        (self.data_path / "partition_0").mkdir(exist_ok=True, parents=True)

        self.device = device
        self.single_eval_pos = single_eval_pos
        self.get_batch_fn = get_batch_fn

        if not (self.data_path / "partition_0" / "chunk_0.pkl").exists():
            self.generate_and_store_data(**kwargs)

        self.__len__()

    def load_chunk(
        self,
        idx,
    ):
        file_path = self.data_path / "partition_0" / f"chunk_{idx}.pkl"

        if file_path.exists():
            with open(file_path, "rb") as f:
                loaded_chunk = cloudpickle.load(f)
            return loaded_chunk
        else:
            raise FileNotFoundError(f"Missing data: {file_path}")

    def generate_and_store_data(self, **kwargs):
        n_chunks = kwargs.pop("n_chunks", 1000) if "n_chunks" in kwargs else 1000
        pdl = PriorDataLoader(
            load_path=str(self.data_path),
            n_chunks=n_chunks,
            store=True,
            subsample=1,
            partition=None,
        )

        pdl.store_prior(
            prior=Prior(get_batch_fn=self.get_batch_fn),
            local=True,
            **kwargs,
        )

    def __len__(self):
        self.n_chunks = len(
            list(Path.glob(self.data_path / "partition_0", "chunk_*.pkl"))
        )
        self.n_batches = self.load_chunk(0)[0][1].x.shape[1]

        return self.n_chunks * self.n_batches

    def __iter__(self):
        for chunk_idx in range(self.n_chunks):
            chunk = self.load_chunk(chunk_idx)

            for _, batch in chunk:
                b = MyBatch(
                    x=batch.x,
                    y=batch.y,
                    target_y=batch.target_y,
                    single_eval_pos=self.single_eval_pos,
                    style=batch.style,
                )
                b = b.to(self.device)
                yield b


if __name__ == "__main__":
    same_task_dataset = SyntheticTaskMetaTestDataset(
        data_path="/home/ruhkopf/PycharmProjects/Meta_FTPFN/data/validation",
        single_eval_pos=64,
        n_chunks=16,
        batch_size=8,
        get_batch_fn=same_task_get_batch,
        name="same_task",
        seq_len=1000,
    )

    next(iter(same_task_dataset))

    from ppfn.dataset.get_batch.deprec.ftpfn import get_batch as unrelated_task_get_batch

    unrelated_task_dataset = SyntheticTaskMetaTestDataset(
        data_path="/home/ruhkopf/PycharmProjects/Meta_FTPFN/data/validation",
        single_eval_pos=64,
        n_chunks=4,
        batch_size=2,
        get_batch_fn=unrelated_task_get_batch,
        name="unrelated_task",
        seq_len=1000,
    )

    print()
