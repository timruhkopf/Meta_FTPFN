from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from typing import Dict
import torch

from ppfn.dataset.mfpbench.meta_test_dataset import MetaTestDataset


class MetaTestBenchmarkCallback(AbstractCallback):
    """
    A callback to create a MetaTestDataset for evaluation during training.
    """

    def __init__(
            self,
            data_path: str,
            benchmark_name: str,
            single_eval_pos: int,
            n_folds: int = 5,
            frequency: int = 1,
            device: str = 'cpu'
    ):
        self.data_path = data_path
        self.benchmark_name = benchmark_name
        self.single_eval_pos = single_eval_pos
        self.n_folds = n_folds
        self.frequency = frequency
        self.device = device

    def get_dataset(self):
        return MetaTestDataset(
            data_path=self.data_path,
            benchmark_name=self.benchmark_name,
            single_eval_pos=self.single_eval_pos,
            n_folds=self.n_folds,
            device=self.device,
        )

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        if (epoch + 1) % self.frequency == 0:
            self.trainer.model.eval()

            evaluation_dataloader = torch.utils.data.DataLoader(
                self.get_dataset(),
                batch_size=1, # since the batch is jointly processed, it must be one!
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: x[0], # we need to collect only the single batch item
            )
            results = []
            with torch.no_grad():
                for batch in evaluation_dataloader:
                    losses, step_metrics = self.trainer._forward_pass(batch, self.single_eval_pos)
                    del step_metrics['loss']  # we don't log loss here (only aggregated stats)
                    results.append(step_metrics)

            aggregated_metrics = {}
            for key in results[0].keys():
                newkey = f'{self.benchmark_name}/{key}'
                aggregated_metrics[newkey] = sum(r[key] for r in results) / len(results)

            self.trainer.model.train()

            return aggregated_metrics




