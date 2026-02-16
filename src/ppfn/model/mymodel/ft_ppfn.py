from dataclasses import dataclass
from typing import Mapping, Optional
import os
from pathlib import Path

import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from ppfn.model.mymodel.interleaved_model import HierarchicalPFN

# TODO move to utils!
@dataclass
class MyBatch(Batch):

    def __add__(self, other) -> 'MyBatch':


        # Concatenate core tensors along the batch dimension (dim=1)
        # Assuming shape: [seq_len, batch_size, n_features]
        new_x = torch.cat([self.x, other.x], dim=1)
        new_y = torch.cat([self.y, other.y], dim=1)
        new_target_y = torch.cat([self.target_y, other.target_y], dim=1)

        if self.style is None or other.style is None:
            new_style = None
        else:
            new_style = torch.cat([self.style, other.style], dim=0)

        # Create the new instance
        return MyBatch(
            x=new_x,
            y=new_y,
            target_y=new_target_y,
            style=new_style
        )

    def to(self, device: torch.device) -> 'MyBatch':
        """Move all tensors in the batch to the specified device."""
        return MyBatch(
            x=self.x.to(device),
            y=self.y.to(device),
            target_y=self.target_y.to(device),
            style=self.style.to(device) if self.style is not None else None
        )

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

        logger.info(f'FT_PPFN initialized with cross-fusion interleaved layers. '
                    f'Number of parameters to train: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')

    def parse_train_batch(self, batch: MyBatch, single_eval_pos) -> MyBatch:
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
    
        # force the related tasks to have the same query positions for cross-fusion
        if self.force_same_query: 
            related_marginal_x[sep :, ...] = batch.x[sep :, ::2, ...]
            related_marginal_y[sep :, ...] = batch.y[sep :, ::2, ...]
        
        # Join the three streams into a single batch
        x = torch.cat([target_marginal_x, related_marginal_x, target_conditional_x], dim=1)
        y = torch.cat([target_marginal_y, related_marginal_y, target_conditional_y], dim=1)

        return MyBatch(
                x=x.to(device), y=y.to(device), target_y=y.to(device),
                single_eval_pos=sep,
        )

    def parse_eval_batch(self, batch: MyBatch, single_eval_pos) -> MyBatch:
        """
        Modify the batch for cross-fusion evaluation.
        In evaluation mode, we expect:
        (Target marginal (A), Related marginals (B), Target conditionals (C)),
        where (A) & (B) are untainted predictions, and (C) is to be updated.
        
        Note that during evaluation, there is only one target task (A),
        and multiple related tasks (B), all in the same batch.
        """
    
        B = batch.x.shape[1]
        R = (B - 1)
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

        if self.force_same_query:
            # we only need to edit the related marginals' query, since target marginal
            # and conditional are the same by construction
            related_marginal_x[sep :, ...] = target_marginal_x[sep :, ...]
            related_marginal_y[sep :, ...] = target_marginal_y[sep :, ...]

        # Join the three streams into a single batch
        x = torch.cat([ target_marginal_x, related_marginal_x, target_conditional_x],dim=1)
        y = torch.cat([ target_marginal_y, related_marginal_y, target_conditional_y], dim=1)

        return MyBatch(
                x=x.to(device), y=y.to(device), target_y=y.to(device),
                single_eval_pos=sep,
        )
    
    def forward(self, batch: MyBatch, **kwargs):
        """
        Forward pass with batch parsing for cross-fusion.
        """

        if 'single_eval_pos' in kwargs.keys():
            # kwargs has precedence over batch attribute
            single_eval_pos = kwargs['single_eval_pos']
            del kwargs['single_eval_pos']
        else:
            single_eval_pos = batch.single_eval_pos
        
        batch = self.parse_batch(batch, single_eval_pos)
        output = super().forward(batch, single_eval_pos, **kwargs)
        return output

    def parse_batch(self, batch: MyBatch, single_eval_pos) -> MyBatch:
        """
        Override parse_batch to handle both train and eval parsing.
        """
        return self.parse_train_batch(batch, single_eval_pos) if self.training \
            else self.parse_eval_batch(batch, single_eval_pos)

    @property
    def criterion(self):
        return self.frozen_model.criterion 


if __name__ == "__main__":
    # Example usage
    frozen_model = load_frozen_model()
    ppfn_model = FT_PPFN(frozen_model=frozen_model, interleaved_layers={})
    print(ppfn_model)