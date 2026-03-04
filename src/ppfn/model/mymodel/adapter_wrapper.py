from ppfn.model.mymodel.meta_context import ForwardMetaContext
import torch
import torch.nn as nn
import inspect


class AdapterWrapper(nn.Module):
    def __init__(self, original_layer, adapter_module):
        super().__init__()
        self.adapter = adapter_module
        self.original_layer = original_layer

        # analyse, which signature arguments the adapter module expects
        self.keys = set(inspect.signature(adapter_module.forward).parameters.keys()) - {"A", "B", "C", "sep"}

    def forward(self, x, *args, **kwargs):

        # Pull the out-of-band data required by the Nadaraya-Watson adapter
        single_eval_pos = ForwardMetaContext.get("single_eval_pos")

        adapter_kwargs = {k: ForwardMetaContext.get(k) for k in self.keys if ForwardMetaContext.get(k) is not None}

        R = x.shape[1] // 3  # Since the input is interleaved A/B/C, we can infer R from the shape

        # --- Extract Latent Streams ---
        A = x[:, :R, :].detach()
        B = x[:, R: 2 * R, :].detach()  # Undistorted target and related tasks
        C = x[:, 2 * R:, :]  # Target conditional (to be updated)

        if single_eval_pos is None:
            raise RuntimeError("AdapterWrapper called outside of MetaContext scope.")

        # 1. Run the adapter (Interleaved Layer)
        # It handles the 3-stream A/B/C manipulation internally
        A, B, C = self.adapter( A=A, B=B, C=C, sep=single_eval_pos, **adapter_kwargs)

        x_adapted = torch.cat([A, B, C], dim=1)

        # 2. Pass the adapted tensor to the frozen standard layer
        return self.original_layer(x_adapted, *args, **kwargs)