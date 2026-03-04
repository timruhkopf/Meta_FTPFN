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
    It takes model outputs, computes the specific losses for streams A and C,
    and returns both the loss to optimize and the metrics to log.
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
        self.lambda_sparsity = lambda_sparsity  # Hyperparameter for the L1 penalty

    # def _find_cross_fusion_aux_loss(self):
    #     """Helper to recursively find CrossFusion modules and sum their aux losses."""



        # total_aux_loss = 0.0
        # # Iterate through all modules in the backbone
        # for module in self.model.modules():
        #     if isinstance(module, CrossFusionAdapter):
        #         total_aux_loss += module.get_aux_loss()
        # return total_aux_loss

    def forward(
            self,
            output: torch.Tensor,

            single_eval_pos,
            batch=None,
            src_key_padding_mask=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Compute raw loss for all streams
        # Assuming output/targets are shaped correctly for the criterion
        streams = self.stream_parser.get_raw_streams(batch, src_key_padding_mask)

        B = output.shape[1]
        R = B // 3

        A_output = output[:, :R, ...]  # Unconditional stream output
        C_output = output[:, -R:, ...]  # Conditional / Workspace stream output

        A_target = streams['A'][1][single_eval_pos:, ...]  # Unconditional stream targets

        # notice, that the padding is only on the train part; meaning we won't need to mask the raw loss,
        # because it is only concerned with the test part.
        loss_stream_A = self.criterion(A_output, A_target)  # [T, B] unconditional loss (reference)
        loss_stream_C = self.criterion(C_output, A_target)  # [T, B] conditional loss (what we optimize)


        optimization_loss, nan_share = torch_nanmean(
            loss_stream_C.mean(0),  # T dim avg -- to see which examples failed
            axis=0,  # final avg
            return_nanshare=True,
        )
        # FETCH AND ADD AUXILIARY SPARSITY LOSS
        state_dict = vars(ForwardMetaContext._state)

        aux_losses = [
            val for k, val in state_dict.items()
            if k.startswith("gate_loss/")
        ]
        if aux_losses:
            gatelosses = torch.stack(aux_losses).mean()
        else:
            gatelosses = torch.tensor(0.0)

        optimization_loss = optimization_loss + (self.lambda_sparsity * gatelosses)

        # # --- NEW: NLL Hinge Penalty for Unrelated Tasks ---
        # consistency_val = 0.0
        # if (
        #         batch is not None
        #         and hasattr(batch, "style")
        #         and batch.style is not None
        # ):
        #     style = batch.style[::2].squeeze()  # Extract style for stream A tasks
        #     unrelated_mask = (style == 0)
        #
        #     if unrelated_mask.any():
        #         # Extract the NLL losses for just the unrelated tasks in the batch
        #         # .detach() Stream A so we only penalize C without moving A's gradients
        #         nll_A_unrelated = loss_stream_A[:, unrelated_mask, ...].detach()
        #         nll_C_unrelated = loss_stream_C[:, unrelated_mask, ...]
        #
        #         # Hinge Penalty: Only penalize C if its NLL is HIGHER (worse) than A's NLL.
        #         # Because both are NLL, the numerical scale matches the main optimization_loss perfectly.
        #         consistency_loss = F.relu(nll_C_unrelated - nll_A_unrelated).mean()
        #
        #         optimization_loss = optimization_loss + consistency_loss
        #         consistency_val = consistency_loss.item()
        # # --------------------------------------------------

        # 4. Compute Metrics (The logic previously in TrainMetricsCallback)
        with torch.no_grad():
            nll_diff = (loss_stream_C - loss_stream_A).detach()

            metrics = {
                "nll/C-A": nll_diff.mean().item(),
                "nll/A": loss_stream_A.mean().item(),
                "nll/C": loss_stream_C.mean().item(),
            }

            # Handle style-based grouping if batch is provided
            if (
                    batch is not None
                    and hasattr(batch, "style")
                    and batch.style is not None
            ):
                style = batch.style[::2].squeeze()  # Extract style for stream A tasks
                metrics.update(
                    {
                        # note: similartasktransform defines style=1 for similar
                        # get_related_batch.get_batch defines unrelated as style=0, similar depending on the transform
                        "nll/similar_task": nll_diff[:, style != 0].mean().item(),
                        "nll/unrelated_task": nll_diff[:, style == 0].mean().item(),
                    }
                )

        if self.verbose:
            metrics["nan_share"] = nan_share.item()

        return optimization_loss, metrics
