import os
from pathlib import Path
from dotenv import load_dotenv

import torch

from ifbo.surrogate import FTPFN


def load_frozen_model() -> FTPFN:
    """Load frozen pre-trained PPFN model from ifBO."""

    # Load from project root .env (4 levels up from this file)
    load_dotenv(dotenv_path=Path(__file__).parents[4] / ".env")

    model_path = os.getenv("MODELDIR", "models/") + "pfn_ckpt"
    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device=torch.device("cpu")
    ).model

    return frozen_model
