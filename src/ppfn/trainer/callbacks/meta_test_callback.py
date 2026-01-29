import warnings

from ppfn.trainer.callbacks.abstract_callback import AbstractCallback
from typing import Dict
import torch



class MetaTestCallback(AbstractCallback):
    """
    A callback to create a MetaTestDataset for evaluation during training.
    """

    def __init__(
            self,
            dataset,
            frequency: int = 1,
            device: str = 'cpu',
            switch_to_eval: bool = True,
    ):
        self.dataset = dataset
        self.frequency = frequency
        self.device = device
        self.switch_to_eval = switch_to_eval
        assert hasattr(self.dataset, 'name'), "Dataset must have a 'name' attribute for logging purposes."

        warnings.warn(
            'MetaTestCallback relies on TrainMetricsCallback implicitly to compute the '
            'metrics during evaluation. This is likely subject to change with a '
            'fixed architecture. Any callback with on_forward_end or on_loss_end '
            'methods will be called during this evaluation.',
            UserWarning)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], **kwargs):
        if (epoch + 1) % self.frequency == 0:

            if self.switch_to_eval:
                # The real benchmarks need the model in eval due to the kfold design
                # and the iteration over the dedicated target task in the meta-test fold,
                # where the batch has a single target task
                self.trainer.model.eval()
                aggregated_metrics = self._evaluate()
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
                # for convenience, we use the trainer's forward pass method
                # that includes the architectural modifications (if any) in
                # the callback.on_forward_end and callback.on_loss_end methods
                # from CrossFusionLossCallback
                # Currently the actual metrics are being computed by
                # TrainMetricsCallback.on_forwad_end, which we abuse here.
                # Future warning:
                losses, step_metrics = self.trainer._forward_pass(
                    batch, self.dataset.single_eval_pos
                )

                results.append(step_metrics)

        aggregated_metrics = {}
        for key in results[0].keys():
            newkey = f'{key}:{self.dataset.name}'
            aggregated_metrics[newkey] = sum(r[key] for r in results) / len(results)

        return aggregated_metrics



