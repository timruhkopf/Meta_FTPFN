
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
    ):
        super().__init__(
            frozen_model=frozen_model,
            interleaved_layers=interleaved_layers,
        )

    @staticmethod
    def parse_batch(training, batch: Batch) -> Batch:
        if training: 
            x = torch.cat([batch.x, batch.x.roll(1, dims=1), batch.x], dim=1)
            y = torch.cat([batch.y, batch.y.roll(1, dims=1), batch.y], dim=1)

        if not training: 
            # first batch item is target task marginal predictions
            # next batch items are related tasks' marginal predictions
            # last batch items are related tasks' conditional predictions to be updated
            B = batch.x.shape[1]
            R = (B - 1) 

            target_marginal = batch.x[:, :1, :].expand(-1, R, -1)
            related_marginal = batch.x[:, 1:, :]
            related_conditional = batch.x[:, :1, :].expand(-1, R, -1)
            x = torch.cat([
                        target_marginal,
                        related_marginal,
                        related_conditional,
                    ],dim=1)
            
            y = torch.cat([
                batch.y[:, :1].expand(-1, R, -1),
                batch.y[:, 1:],
                batch.y[:, :1].expand(-1, R, -1),
            ], dim=1)

          
            # FIXME: during eval we need to aggregate the conditional predictions
            # for now, we just return them as is 
            raise NotImplementedError("Aggregation of conditional predictions during eval not implemented yet.")
        
        return Batch(
                x=x, y=y, target_y=y,
                single_eval_pos=batch.single_eval_pos,
        )
    
    def forward(self, batch, **kwargs):
        # for the CrossFusion layers, we need to modify the batch: 
        # the first third is the unconditional marginal predictions of the target task
        # the second third is the unconditional marginal predictions of the related tasks
        # the last third is the conditional predictions of the related tasks to be updated
        
        batch = self.parse_batch(self.training, batch)
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