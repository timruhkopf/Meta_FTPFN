
from typing import Mapping
import os
from pathlib import Path

import torch
import torch.nn as nn


from ppfn.model.mymodel.interleaved_model import HierarchicalPFN

from pfns4hpo.priors import Batch 


def load_frozen_model() -> nn.Module:
    """Load frozen pre-trained PPFN model from ifBO."""
    import torch
    from dotenv import load_dotenv
    from ifbo.surrogate import FTPFN

    # Load from project root .env (4 levels up from this file)
    load_dotenv(dotenv_path=Path(__file__).parents[4] / ".env")

    model_path = os.getenv("MODELDIR", "models/") + "pfn_ckpt"
    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device=torch.device("cpu")
    ).model
    
    return frozen_model


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

    def parse_train_batch(self, batch: Batch) -> Batch:
        """
        Modify the batch for cross-fusion training.
        In training mode, we expect triplets of tasks in the batch:
        (Target marginal (A), Related marginal (B), Target conditional (C)), 
        where (A) & (B) are untainted predictions, and (C) is to be updated.
        """

        sep = batch.single_eval_pos
  
        # add related tasks by rolling the batch
        target_marginal_x = batch.x
        related_marginal_x = batch.x.roll(1, dims=1)
        target_conditional_x = batch.x

        target_marginal_y = batch.y
        related_marginal_y = batch.y.roll(1, dims=1)
        target_conditional_y = batch.y
    
        # force the related tasks to have the same query positions for cross-fusion
        if self.force_same_query: 
            related_marginal_x[sep :, ...] = batch.x[sep :, ...]
            related_marginal_y[sep :, ...] = batch.y[sep :, ...]
        
        # Join the three streams into a single batch
        x = torch.cat([target_marginal_x, related_marginal_x, target_conditional_x], dim=1)
        y = torch.cat([target_marginal_y, related_marginal_y, target_conditional_y], dim=1)

        return Batch(
                x=x, y=y, target_y=y,
                single_eval_pos=sep,
        )

    def parse_eval_batch(self, batch: Batch) -> Batch:
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

        sep = batch.single_eval_pos

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

        return Batch(
                x=x, y=y, target_y=y,
                single_eval_pos=sep,
        )
    
    def forward(self, batch, **kwargs):
        """
        Forward pass with batch parsing for cross-fusion.
        """
        
        batch = self.parse_train_batch(batch) if self.training else self.parse_eval_batch(batch)
        output = super().forward(batch, **kwargs)
        return output

    @property
    def criterion(self):
        return self.frozen_model.criterion 


if __name__ == "__main__":
    # Example usage
    frozen_model = load_frozen_model()
    ppfn_model = FT_PPFN(frozen_model=frozen_model, interleaved_layers={})
    print(ppfn_model)