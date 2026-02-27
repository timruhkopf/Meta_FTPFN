from ppfn.model.baselines.ft_pfn_padding import PaddableTransformerModel

from typing import Mapping, Optional
from ppfn.utils.load_ftpfn import load_frozen_model
from ppfn.model.mymodel.interleaved_model import HierarchicalPFN
from ppfn.utils.mybatch import MyBatch

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





class FT_PPFN(HierarchicalPFN):
    """
    Pre-conditioned Prior fitted Network (PPFN)

    The idea is that we take a frozen pre-trained model (e.g., a transformer)
    and make it batch-/ meta-aware by interleaving trainable cross-attention layers.
    This allows the model to learn the marginal conditional distribution p(x|\tau*, \tau_i)
    where \tau* is the target context and \tau_i are the support contexts in the batch.

    """

    def __init__(
        self,
        frozen_model: nn.Module,
        interleaved_layers: Mapping[str, nn.Module],
        force_same_query: bool = False,
    ):
        super().__init__(
            frozen_model=frozen_model,
            interleaved_layers=interleaved_layers,
        )
        self.force_same_query = force_same_query

        logger.info(
            f"FT_PPFN initialized with cross-fusion interleaved layers. "
            f"Number of parameters to train: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def parse_train_batch(
        self, batch: MyBatch, single_eval_pos, src_key_padding_mask=None
    ) -> tuple[MyBatch, Optional[torch.Tensor]]:
        """
        Modify the batch for cross-fusion training.
        In training mode, we expect triplets of tasks in the batch:
        (Target marginal (A), Related marginal (B), Target conditional (C)),
        where (A) & (B) are untainted predictions, and (C) is to be updated.
        """

        sep = single_eval_pos
        device = batch.x.device

        # add related tasks by rolling the batch
        target_marginal_x = batch.x[:, ::2, :]
        related_marginal_x = batch.x[:, 1::2, :]
        target_conditional_x = batch.x[:, ::2, :]

        target_marginal_y = batch.y[:, ::2]
        related_marginal_y = batch.y[:, 1::2]
        target_conditional_y = batch.y[:, ::2]

        if src_key_padding_mask is not None:
            # adjust the padding mask accordingly
            target_marginal_mask = src_key_padding_mask[::2, :]
            related_marginal_mask = src_key_padding_mask[1::2, :]
            target_conditional_mask = src_key_padding_mask[::2, :]

            src_key_padding_mask = torch.cat(
                [target_marginal_mask, related_marginal_mask, target_conditional_mask],
                dim=0,
            )

        # force the related tasks to have the same query positions for cross-fusion
        if self.force_same_query:
            related_marginal_x[sep:, ...] = batch.x[sep:, ::2, ...]
            related_marginal_y[sep:, ...] = batch.y[sep:, ::2, ...]

        # Join the three streams into a single batch
        x = torch.cat(
            [target_marginal_x, related_marginal_x, target_conditional_x], dim=1
        )
        y = torch.cat(
            [target_marginal_y, related_marginal_y, target_conditional_y], dim=1
        )

        return (
            MyBatch(
                x=x.to(device),
                y=y.to(device),
                target_y=y.to(device),
                single_eval_pos=sep,
            ),
            src_key_padding_mask.to(device)
            if src_key_padding_mask is not None
            else None,
        )

    def parse_eval_batch(
        self, batch: MyBatch, single_eval_pos, src_key_padding_mask=None
    ) -> tuple[MyBatch, Optional[torch.Tensor]]:
        """
        Modify the batch for cross-fusion evaluation.
        In evaluation mode, we expect:
        (Target marginal (A), Related marginals (B), Target conditionals (C)),
        where (A) & (B) are untainted predictions, and (C) is to be updated.

        Note that during evaluation, there is only one target task (A),
        and multiple related tasks (B), all in the same batch.
        """

        B = batch.x.shape[1]
        R = B - 1
        device = batch.x.device

        sep = single_eval_pos

        # prepare the three streams (X tensors)
        target_marginal_x = batch.x[:, :1, :].expand(-1, R, -1)
        related_marginal_x = batch.x[:, 1:, :]
        target_conditional_x = batch.x[:, :1, :].expand(-1, R, -1)

        # prepare the three streams (Y tensors)
        target_marginal_y = batch.y[:, :1].expand(-1, R)
        related_marginal_y = batch.y[:, 1:]
        target_conditional_y = batch.y[:, :1].expand(-1, R)

        if src_key_padding_mask is not None:
            # adjust the padding mask accordingly
            target_marginal_mask = src_key_padding_mask[:1, :].expand(R, -1)
            related_marginal_mask = src_key_padding_mask[1:, :]
            target_conditional_mask = src_key_padding_mask[:1, :].expand(R, -1)

            src_key_padding_mask = torch.cat(
                [target_marginal_mask, related_marginal_mask, target_conditional_mask],
                dim=0,
            )

        if self.force_same_query:
            # we only need to edit the related marginals' query, since target marginal
            # and conditional are the same by construction
            related_marginal_x[sep:, ...] = target_marginal_x[sep:, ...]
            related_marginal_y[sep:, ...] = target_marginal_y[sep:, ...]

        # Join the three streams into a single batch
        x = torch.cat(
            [target_marginal_x, related_marginal_x, target_conditional_x], dim=1
        )
        y = torch.cat(
            [target_marginal_y, related_marginal_y, target_conditional_y], dim=1
        )

        return (
            MyBatch(
                x=x.to(device),
                y=y.to(device),
                target_y=y.to(device),
                single_eval_pos=sep,
            ),
            src_key_padding_mask.to(device)
            if src_key_padding_mask is not None
            else None,
        )

    def forward(self, batch: MyBatch, **kwargs):
        """
        Forward pass with batch parsing for cross-fusion.
        """

        if "single_eval_pos" in kwargs.keys():
            # kwargs has precedence over batch attribute
            single_eval_pos = kwargs["single_eval_pos"]
        else:
            single_eval_pos = batch.single_eval_pos

        batch, src_key_padding_mask = self.parse_batch(
            batch,
            single_eval_pos,
            src_key_padding_mask=kwargs.get("src_key_padding_mask", None),
        )

        # Remove any caller-provided mask from kwargs to avoid accidentally
        # forwarding an unparsed src_key_padding_mask to frozen models that
        # don't support padding. We'll re-insert the parsed mask only when
        # the frozen model supports it.
        kwargs.pop("src_key_padding_mask", None)

        if isinstance(self.frozen_model, PaddableTransformerModel):
            # Insert the parsed/expanded mask that matches the batch joining
            kwargs.update(
                {
                    "src_key_padding_mask": src_key_padding_mask,
                    "single_eval_pos": single_eval_pos,
                }
            )
        else:
            if src_key_padding_mask is not None:
                logger.warning(
                    "Parsed a src_key_padding_mask for a frozen model that does not support padding. Ignoring the mask."
                )
            # Ensure single_eval_pos is still passed downstream and explicitly
            # clear any src_key_padding_mask to avoid accidental forwarding.
            kwargs.update({"single_eval_pos": single_eval_pos, "src_key_padding_mask": None})

        logger.debug(f"FT_PPFN.forward calling super with kwargs keys={list(kwargs.keys())}")
        if "src_key_padding_mask" in kwargs:
            mask = kwargs["src_key_padding_mask"]
            logger.debug(f"FT_PPFN.forward src_key_padding_mask shape={getattr(mask, 'shape', None)} type={type(mask)}")

        output = super().forward(batch, **kwargs)
        return output

    def parse_batch(
        self, batch: MyBatch, single_eval_pos, src_key_padding_mask=None
    ) -> tuple[MyBatch, Optional[torch.Tensor]]:
        """
        Override parse_batch to handle both train and eval parsing.
        """
        return (
            self.parse_train_batch(batch, single_eval_pos, src_key_padding_mask)
            if self.training
            else self.parse_eval_batch(batch, single_eval_pos, src_key_padding_mask)
        )

    @property
    def criterion(self):
        return self.frozen_model.criterion

    # def get_trainable_params(self, weight_decay):
    #     """
    #     Separates parameters into two groups: one with weight decay, one without.
    #
    #     Weight decay pushes all network weights towards zero to prevent overfitting. However, applying weight decay to 1D parameters—specifically biases and LayerNorm weights/biases—often degrades performance because these parameters control the shift and scale of activations, not the complexity of the feature interactions.
    #
    #     In PyTorch, the optimizer will blindly apply weight decay to everything unless you explicitly separate the parameters.
    #     """
    #     decay = []
    #     no_decay = []
    #
    #     for name, param in self.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #
    #         # Do not apply weight decay to LayerNorm weights, or any biases
    #         if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)
    #
    #     return [
    #         {'params': decay, 'weight_decay': weight_decay},
    #         {'params': no_decay, 'weight_decay': 0.0}
    #     ]


if __name__ == "__main__":
    # Example usage
    frozen_model = load_frozen_model()
    ppfn_model = FT_PPFN(frozen_model=frozen_model, interleaved_layers={})
    print(ppfn_model)
