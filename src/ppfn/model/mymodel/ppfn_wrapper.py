
from typing import Mapping
import os
import sys
from pathlib import Path

import torch.nn as nn


from ppfn.model.mymodel.interleavedmodel import InterleavedModel


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


class PPFN(InterleavedModel):
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
        pre_hook: bool = True,
    ):
        super().__init__(
            frozen_model=frozen_model,
            interleaved_layers=interleaved_layers,
            pre_hook=pre_hook,
        )

    @property
    def criterion(self):
        return self.frozen_model.criterion 


