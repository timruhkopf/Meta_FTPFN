
import os
from pathlib import Path

from dotenv import load_dotenv
from ifbo.surrogate import FTPFN

def ft_pfn():


    load_dotenv(dotenv_path=Path(__file__).parents[4] / ".env")

    model_path = os.getenv("MODELDIR") + "pfn_ckpt"
    frozen_model = FTPFN(
        target_path=Path(model_path), version="0.0.1", device="cpu"
    ).model

    criterion = frozen_model.criterion
    
    return frozen_model