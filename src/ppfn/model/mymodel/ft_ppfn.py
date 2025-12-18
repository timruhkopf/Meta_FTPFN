
from typing import Mapping
import os
from pathlib import Path

import torch.nn as nn


from ppfn.model.mymodel.interleaved_model import HierarchicalPFN


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

    @property
    def criterion(self):
        return self.frozen_model.criterion 


if __name__ == "__main__":
    # Example usage
    frozen_model = load_frozen_model()
    ppfn_model = FT_PPFN(frozen_model=frozen_model, interleaved_layers={})
    print(ppfn_model)