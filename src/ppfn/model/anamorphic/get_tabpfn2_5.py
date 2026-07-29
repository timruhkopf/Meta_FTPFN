import torch
from tabpfn import TabPFNRegressor
from tabpfn.architectures.tabpfn_v2_5 import TabPFNV2_5


def get_raw_tabpfn_v2_5(device="cuda") -> TabPFNV2_5:
    """
    Instantiates the Scikit-Learn wrapper to handle downloading and config matching,
    then extracts and returns the raw PyTorch nn.Module.
    """
    # 1. Initialize the wrapper (downloads weights if missing)
    wrapper = get_tabpfn_model(version= ModelVersion.V2_5, fit_mode= "fit_without_cache")

    # 2. Extract the underlying PyTorch model.
    # Depending on the exact v2.5 pip release, the raw module is usually
    # accessible via .model or ._model.
    raw_model = wrapper.model[2] if isinstance(wrapper.model, tuple) else wrapper.model

    # Ensure it's in eval mode and on the correct device
    raw_model = raw_model.to(device)

    return raw_model