from ppfn.model.mymodel.layers.nw_adapter import NadarayaWatsonAdapter
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
        A, B, C = self.adapter(A=A, B=B, C=C, sep=single_eval_pos, **adapter_kwargs)

        x_adapted = torch.cat([A, B, C], dim=1)

        # 2. Pass the adapted tensor to the frozen standard layer
        return self.original_layer(x_adapted, *args, **kwargs)


class Unified1dValidationWrapper(nn.Module):
    def __init__(self, adapter_module, input_dim=2, d_model=64, seq_len=60):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, d_model)
        self.hp_proj = nn.Linear(max(1, input_dim - 1), d_model)
        self.adapter = adapter_module
        self.down_proj = nn.Linear(d_model, input_dim)
        self.seq_len = seq_len

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, A, B, C, B_belief_A, sep):
        """
        Accepts raw data streams (T, Batch, D).
        Handles projection and coordination internally.
        """
        # 1. Guarantee a clean telemetry state
        ForwardMetaContext.clear()

        if isinstance(self.adapter, NadarayaWatsonAdapter):
            A = torch.cat([A, B_belief_A], dim=0)
            B = torch.cat([B, B_belief_A], dim=0)
            C = torch.cat([C, B_belief_A], dim=0)

        # 2. Project all streams to latent space
        # We process them together for efficiency, then split
        x_raw = torch.cat([A, B, C], dim=1)  # (T, 3*Batch, D)
        h = self.up_proj(x_raw)

        # 3. Handle Hyperparameters (using x-coordinates from A_data)
        # We assume x-coords are shared across streams or taken from A
        hp_raw = x_raw[:, :, 0:1]
        hp = self.hp_proj(hp_raw)

        # 4. Split latent streams for the adapter
        R = A.shape[1]
        A_lat = h[:, :R, :]
        B_lat = h[:, R:2 * R, :]
        C_lat = h[:, 2 * R:, :]

        hp = (hp[:, :R, :] , hp[:, R:2 * R, :] , hp[:, 2 * R:, :])

        # 5. Apply Adapter
        A_out, B_out, C_out = self.adapter(
            A=A_lat,
            B=B_lat,
            C=C_lat,
            sep=sep,
            hp=hp
        )

        if isinstance(self.adapter, NadarayaWatsonAdapter):
            # we need to crop out the appended belief points before reassembly
            A_out, B_out, C_out = A_out[:self.seq_len], B_out[:self.seq_len], C_out[:self.seq_len]

        # 6. Reassemble and project down
        h_out = torch.cat([A_out, B_out, C_out], dim=1)
        out = self.down_proj(h_out)

        # 7. Retrieve telemetry
        stats = ForwardMetaContext.get_stats().copy()

        return out, stats
