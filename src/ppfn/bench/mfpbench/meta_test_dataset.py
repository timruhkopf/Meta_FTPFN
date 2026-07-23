import torch
from torch.utils.data import IterableDataset
from pathlib import Path


from sklearn.model_selection import KFold

from ppfn.utils.mybatch import MyBatch


class MFPBenchMetaTestDataset(IterableDataset):
    target_first = True

    def __init__(
        self,
        data_path: str,
        benchmark_name: str,
        single_eval_pos: int,
        n_folds: int = 5,
        device="cpu",
    ):
        self.data_path = Path(data_path)

        self.task_ids = list(
            p.name for p in Path.glob(self.data_path / benchmark_name, "task_*")
        )
        available_repetitions = len(
            list(
                Path.glob(
                    self.data_path
                    / benchmark_name
                    / self.task_ids[0]
                    / f"sep_{single_eval_pos}",
                    "rep_*",
                )
            )
        )

        self.n_folds = n_folds
        self.device = device

        assert n_folds <= available_repetitions
        self.benchmark_name = benchmark_name
        self.name = f"{benchmark_name}_{single_eval_pos}"
        self.single_eval_pos = single_eval_pos

        self.task_ids = sorted(self.task_ids)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        self.folds = list(kf.split(self.task_ids))

    def __len__(self):
        return self.n_folds * len(self.task_ids)

    def load_chunk(
        self,
        task,
        rep_idx,
    ):
        file_path = (
            self.data_path
            / self.benchmark_name
            / f"{task}"
            / f"sep_{self.single_eval_pos}"
            / f"rep_{rep_idx}.pt"
        )

        if file_path.exists():
            # Shape: [ntasks_per_dataset, seq_len, num_features]
            return torch.load(file_path)
        else:
            raise FileNotFoundError(f"Missing data: {file_path}")

    def __iter__(self):
        """
        This implements your 'yield' logic correctly for PyTorch.
        """
        for fold_idx, (train_idx_list, test_idx_list) in enumerate(self.folds):
            # 1. LOAD ONCE: Cache the meta-train tasks for this fold
            # We use fold_idx as rep_idx for consistency
            train_chunks = [
                self.load_chunk(self.task_ids[i], fold_idx) for i in train_idx_list
            ]
            train_data = torch.cat(train_chunks, dim=1)  # [B_train, T, D]

            # 2. ITERATE: Stream the test tasks one by one
            for t_idx in test_idx_list:
                test_task_name = self.task_ids[t_idx]
                test_data = self.load_chunk(test_task_name, fold_idx)  # [B_test, T, D]

                # 3. COMBINE: [B_total, T, D] -> [T, B_total, D]
                combined = torch.cat([test_data, train_data], dim=1)
                # combined = combined.transpose(0, 1)

                yield MyBatch(
                    x=combined[..., :-1].to(self.device),
                    y=combined[..., -1].to(self.device),
                    target_y=combined[..., -1].to(self.device),
                    single_eval_pos=self.single_eval_pos,
                )


if __name__ == "__main__":
    dataset = MFPBenchMetaTestDataset(
        benchmark_name="lcbench_tabular",
        single_eval_pos=64,
        n_folds=2,
        data_path="/home/ruhkopf/VSCode/Meta_FTPFN/data/validation/",
    )

    data = next(iter(dataset))
    print(data.x.shape)

    print("Dataset loading completed.")
