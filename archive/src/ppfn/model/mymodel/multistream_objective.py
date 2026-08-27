import torch
import torch.nn as nn

from typing import Dict, Tuple

from pfns4hpo.utils import torch_nanmean

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from ppfn.model.mymodel.stream_parser import StreamParser


import logging
logger = logging.getLogger(__name__)


class MultiStreamObjective(nn.Module):
    """
    Encapsulates the 'Batch Trick' logic.
    Computes specific losses for streams A and C, returning optimization loss and metrics.
    """

    def __init__(
            self,
            criterion: nn.Module,
            verbose=False,
            lambda_sparsity=0.001,
            stream_parser=StreamParser()
    ):
        super().__init__()
        self.criterion = criterion
        self.verbose = verbose
        self.stream_parser = stream_parser
        self.lambda_sparsity = lambda_sparsity

    def forward(
            self,
            output: torch.Tensor,
            single_eval_pos,
            batch,
            src_key_padding_mask=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # 1. Extract and align streams using our parser
        streams = self.stream_parser.get_raw_streams(batch, src_key_padding_mask)
        o_streams = self.stream_parser.parse_output_streams(output, single_eval_pos)

        # 2. Slice the outputs AND targets to only include the test/query section
        # pfn already only returns the test/query section!
        A_out_test = o_streams["A"] #[single_eval_pos:, ...]
        C_out_test = o_streams["C"] #[single_eval_pos:, ...]

        # batch streams to get the targets! (y is on [1])
        A_target_test = streams["A"][1][single_eval_pos:, ...]

        # 3. Compute NLL
        loss_stream_A = self.criterion(A_out_test, A_target_test)  # [T_test, R]
        loss_stream_C = self.criterion(C_out_test, A_target_test)  # [T_test, R]

        # 4. Main optimization loss (mean over sequence dim 0, then batch dim)
        optimization_loss, nan_share = torch_nanmean(
            loss_stream_C.mean(0),
            axis=0,
            return_nanshare=True,
        )

        # 5. Fetch and add auxiliary sparsity loss
        optimization_loss = optimization_loss + self._get_auxiliary_loss(device=optimization_loss.device)

        # 6. Generate logging metrics
        metrics = self._compute_metrics(loss_stream_A, loss_stream_C, batch, nan_share)

        return optimization_loss, metrics

    def _get_auxiliary_loss(self, device):
        """Safely extracts gate losses from the MetaContext."""
        state_dict = vars(ForwardMetaContext._state)

        aux_losses = [
            val for k, val in state_dict.items()
            if k.startswith("gate_loss/")
        ]

        if not aux_losses:
            # Explicitly cast to the correct device to prevent CPU/GPU mismatch crashes
            return torch.tensor(0.0, device=device)

        return self.lambda_sparsity * torch.stack(aux_losses).mean()

    def _compute_metrics(self, loss_A, loss_C, batch, nan_share):
        """Handles purely detached metric calculations."""
        with torch.no_grad():
            nll_diff = (loss_C - loss_A).detach()

            metrics = {
                "nll/C-A": nll_diff.mean().item(),
                "nll/A": loss_A.mean().item(),
                "nll/C": loss_C.mean().item(),
            }

            # drop this in favor of actual callbacks with validation samples
            # # Handle style-based grouping
            # if batch is not None and getattr(batch, "style", None) is not None:
            #     # Caution: [::2] assumes train-time interleaving.
            #     # Ensure this matches the extracted R dimension!
            #     style = batch.style[::2].squeeze()
            #     metrics.update({
            #         "nll/similar_task": nll_diff[:, style != 0].mean().item(),
            #         "nll/unrelated_task": nll_diff[:, style == 0].mean().item(),
            #     })

            if self.verbose:
                metrics["nan_share"] = nan_share.item()

        return metrics
