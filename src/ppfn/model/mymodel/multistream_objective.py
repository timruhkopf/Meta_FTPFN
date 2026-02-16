import torch
import torch.nn as nn
from typing import Dict, Tuple

from pfns4hpo.utils import torch_nanmean


class MultiStreamObjective(nn.Module):
    """
    Encapsulates the 'Batch Trick' logic.
    It takes model outputs, computes the specific losses for streams A and C,
    and returns both the loss to optimize and the metrics to log.
    """

    def __init__(self, criterion: nn.Module, model=None, verbose=False):
        super().__init__()
        self.criterion = criterion
        self.verbose = verbose
        # FIXME: deprecaite this; it is needed only for batch parsing in forward (depending on
        #  train/eval state)
        self.model = model  # Optional reference to the model if needed for batch parsing

    def forward(self, output: torch.Tensor, targets: torch.Tensor, single_eval_pos, batch=None, src_key_padding_mask=None, **kwargs) -> \
            Tuple[
                torch.Tensor, Dict[str, float]]:
        # 1. Compute raw loss for all streams
        # Assuming output/targets are shaped correctly for the criterion

        if self.model is not None:
            # FIXME: depreciate this with the model reference
            parser = self.model.parse_batch
            b, _ = parser(batch, single_eval_pos, src_key_padding_mask=src_key_padding_mask)
            targets = b.y[single_eval_pos:, ...]

        # notice, that the padding is only on the train part; meaning we won't need to mask the raw loss,
        # because it is only concerned with the test part.
        raw_loss = self.criterion(output, targets)  # [T, B]

        B = output.shape[1]
        R = B // 3

        # 2. Separate Streams
        # Ensure we detach the part we don't want gradients flowing back into if necessary
        # (Though usually we just slice the loss tensor we want to optimize)
        loss_stream_A = raw_loss[:, :R, ...]  # Unconditional
        loss_stream_C = raw_loss[:, -R:, ...]  # Conditional / Workspace

        # 3. Define the Optimization Target
        # You only want to optimize Stream C
        optimization_loss, nan_share = torch_nanmean(
            loss_stream_C.mean(0), # T dim avg -- to see which examples failed
            axis=0, # final avg
            return_nanshare=True
        )

        # 4. Compute Metrics (The logic previously in TrainMetricsCallback)
        with torch.no_grad():
            nll_diff = (loss_stream_C - loss_stream_A).detach()

            metrics = {
                "nll/C-A": nll_diff.mean().item(),
                "nll/A": loss_stream_A.mean().item(),
                "nll/C": loss_stream_C.mean().item(),
            }

            # Handle style-based grouping if batch is provided
            if batch is not None and hasattr(batch, 'style') and batch.style is not None:
                style = batch.style[::2].squeeze()  # Extract style for stream A tasks
                metrics.update({
                    'nll/similar_task': nll_diff[:, style != 1].mean().item(),
                    'nll/unrelated_task': nll_diff[:, style == 1].mean().item()
                })

        if self.verbose:
            metrics['nan_share'] = nan_share.item()

        return optimization_loss, metrics
