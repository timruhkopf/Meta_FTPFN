from typing import Dict

import torch
from torch import amp

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback


class MetaTestCallback(AbstractCallback):
    """
    A callback to create a MetaTestDataset for evaluation during training.
    """

    def __init__(
        self,
        dataset,
        frequency: int = 1,
        device: str = "cpu",
        switch_to_eval: bool = True,
    ):
        self.dataset = dataset
        self.frequency = frequency
        self.device = device
        self.switch_to_eval = switch_to_eval
        self.name = dataset.name

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        if (epoch + 1) % self.frequency == 0:
            if self.switch_to_eval:
                # The real benchmarks need the model in eval due to the kfold design
                # and the iteration over the dedicated target task in the meta-test fold,
                # where the batch has a single target task
                self.trainer.model.eval()
                self.trainer.criterion.eval()  # hot-fix: separate stream_parser in objective (different from the model's) requires this
                aggregated_metrics = self._evaluate()
                self.trainer.criterion.train()
                self.trainer.model.train()
            else:
                # The synthetic benchmarks will work in pairs, where half the batch
                # is the target task of the pair and the other the half the respective
                # related task.
                aggregated_metrics = self._evaluate()

            return aggregated_metrics

    def _evaluate(self):

        evaluation_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,  # since the batch is jointly processed, it must be one!
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x[0],  # we need to collect only the single batch item
        )
        results = []

        with torch.no_grad():
            for batch in evaluation_dataloader:
                batch = batch.to(self.device)

                fwd_kwargs = {'single_eval_pos': batch.single_eval_pos}
                with amp.autocast(device_type="cuda", enabled=self.trainer.use_amp):
                    output = self.trainer.model(batch, **fwd_kwargs)

                    # loss calculation outside of autocast for stable training
                loss, step_metrics = self.trainer.criterion(output, batch=batch, **fwd_kwargs)

                results.append(step_metrics)

        aggregated_metrics = {}

        for key in results[0].keys():
            if key.startswith("nll/A") or key in ["nll/C", 'nll/similar_task', "nll/unrelated_task"]:
                # we skip A, because on a fixed dataset, this will always be constant!
                # similarilty, nll/C will be the same as nll/C-A, so we skip it to avoid redundancy in the logs.
                continue

            newkey = f"{key}:{self.name}"
            aggregated_metrics[newkey] = sum(r[key] for r in results) / len(results)

        return aggregated_metrics
