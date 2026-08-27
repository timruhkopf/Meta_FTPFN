"""
ThinkingWarpTabPFN: Cross-Domain Feature Alignment via Memory Tokens

This module implements a dynamic domain-adaptation architecture for tabular data,
built on top of the TabPFN v2.5 framework. It aligns two datasets (Domain A and Domain B)
that share an identical schema but suffer from an underlying topological "warp"
(e.g., covariate shift, non-linear feature transformations, or concept drift).

Core Idea:
Since the two datasets have unequal sequence lengths and suffer from covariate shift (N_A != N_B), direct row-to-row
cross-attention is impossible. Instead, this architecture introduces learnable memory
tokens ("thinking rows") acting as a low-dimensional bridge to absorb, map, and
broadcast the geometric transformation required to pull Dataset B into Dataset A's domain.

Key Architectural & Design Decisions:

1. Contextualization via Shared Thinking Rows (The Absorption Phase)
   We append an identical set of learnable 'thinking rows' to both datasets.
   Passed through independent TabPFN self-attention blocks, these tokens absorb
   the structural signature, covariance, and sparsity of their respective domains.

2. The Bridge via All-to-All Attention (Resolving the Anchor Alignment Bug)
   To compute the inverse-warp translation, Domain B's thinking rows cross-attend
   to Domain A's thinking rows. Crucially, the thinking rows and features are flattened
   into a single sequence dimension (N_think * F). This bipartite matching allows
   the network to route information freely across both feature axes and topological anchors,
   learning the off-diagonal relationships of the domain warp.

3. The Broadcast via Directed Cross-Attention (Resolving Broadcast Dilution)
   To communicate the learned transformation back to Dataset B, we do not use
   standard self-attention. If we did, the massive inertia of the raw data rows
   would drown out the bridge tokens. Instead, we use an explicit directed cross-attention
   block where Dataset B's data rows (Queries) are forced to pull from the mapped
   thinking rows (Keys/Values), effectively dragging themselves into Domain A's coordinate space.

4. Robust Target Projection
   The transformed representations are projected into logits and evaluated against a
   `FullSupportBarDistribution` (NLL loss), allowing the model to smoothly approximate
   continuous distributions across the real number line without strict regression constraints.
"""

import torch
import torch.nn as nn
from typing import Any

# --- Assumed TabPFN Imports ---
from tabpfn.architectures.tabpfn_v2_5 import TabPFNBlock

import torch
import torch.nn as nn
from typing import Optional, Tuple

ENCODING_SIZE_MULTIPLIER = 2  # Original feature + NaN/Inf indicator


class TorchStandardScaler(nn.Module):
    """A PyTorch-native standard scaler matching TabPFN's internal scaler."""

    def __init__(self):
        super().__init__()

    def fit(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x shape typically [Ri, B, C] or similar. We compute mean/std over rows (dim=0)
        # For grouped features, it might be [Ri, B * G, F].
        # Mean/Std usually computed over the row dimension for each batch/feature independently.
        mean = torch.nanmean(x, dim=0, keepdim=True)
        std = torch.sqrt(
            # Using custom variance to safely ignore NaNs
            torch.nanmean((x - mean) ** 2, dim=0, keepdim=True)
        )
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return {"mean": mean, "std": std}

    def transform(self, x: torch.Tensor, fitted_cache: dict[str, torch.Tensor]) -> torch.Tensor:
        return (x - fitted_cache["mean"]) / fitted_cache["std"]


class TabPFNEncoder(nn.Module):
    """
    Standalone feature and target encoder extracted from TabPFNV2p5.
    Converts raw X and Y into the [Batch, Rows, Features, Cell_Dim] cell structure.
    """

    def __init__(self, emsize: int, features_per_group: int = 1, encoder_type: str = "linear"):
        super().__init__()
        self.emsize = emsize
        self.features_per_group = features_per_group

        # 1. Embedders
        self.feature_group_embedder = self._get_feature_group_embedder(encoder_type)
        self.target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, emsize)

        # 2. Scaler & Positional Embeddings
        self.standard_scaler = TorchStandardScaler()
        self.feature_positional_embedding_embeddings = nn.Linear(emsize // 4, emsize)

    def _get_feature_group_embedder(self, encoder_type: str) -> nn.Module:
        encoding_size = self.features_per_group * ENCODING_SIZE_MULTIPLIER
        if encoder_type == "mlp":
            hidden_dim = self.emsize * 2
            return nn.Sequential(
                nn.Linear(encoding_size, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, self.emsize, bias=False),
            )
        return nn.Linear(encoding_size, self.emsize, bias=False)

    def _generate_nan_and_inf_indicator(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.isnan(x) | torch.isinf(x)).to(x.dtype)

    def _impute_nan_and_inf_with_mean(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(x) | torch.isinf(x)
        if not mask.any():
            return x

        mean_val = torch.nanmean(x, dim=0, keepdim=True)
        # Fallback to 0 if an entire column is NaN
        mean_val = torch.nan_to_num(mean_val, nan=0.0)

        x_imputed = torch.where(mask, mean_val.expand_as(x), x)
        return x_imputed

    def _add_column_embeddings(self, x_BRGX: torch.Tensor) -> torch.Tensor:
        """
        Add a random positional embedding to each column.
        x_BRGX shape: [Batch, Rows, Groups (Features), Embed_Dim]
        """
        generator = torch.Generator(device=x_BRGX.device).manual_seed(42)
        num_cols = x_BRGX.shape[2]

        embs = torch.randn(
            (num_cols, self.emsize // 4),
            device=x_BRGX.device,
            dtype=x_BRGX.dtype,
            generator=generator,
        )
        embs = self.feature_positional_embedding_embeddings(embs)

        # Add to the groups (Features) dimension: [1, 1, Groups, Embed_Dim]
        x_BRGX += embs[None, None, :, :]
        return x_BRGX

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [Rows (T), Batch (B), Cols (C)]
            y: Tensor of shape [Rows (T), Batch (B), 1]

        Returns:
            cells: Tensor of shape [Batch, Rows, Features, Cell_Dim]
                   where Features = Cols + 1 (for the Y target)
        """
        T, B, C = x.shape

        # --- 1. Process X ---
        # Generate NaN/Inf indicators
        nan_inf_indicator_x = self._generate_nan_and_inf_indicator(x)

        # Impute missing
        x_imputed = self._impute_nan_and_inf_with_mean(x)

        # Standard Scale
        scaler_cache = self.standard_scaler.fit(x_imputed)
        x_scaled = self.standard_scaler.transform(x_imputed, scaler_cache)

        # Concatenate values with their indicators (dim=-1) -> [T, B, C * 2]
        x_concat = torch.cat([x_scaled, nan_inf_indicator_x], dim=-1)

        # Embed X
        # For simplicity in this standalone block, assuming C=1 or features_per_group=C
        # If C=1, encoding_size is 2.
        embedded_x_TBX = self.feature_group_embedder(x_concat)  # [T, B, emsize]

        # Reshape to explicitly define the feature (Group) dimension G
        # Assuming G = C (each column is its own group).
        # Reshape to [T, B, G, emsize]
        embedded_x_TBGX = embedded_x_TBX.view(T, B, C, self.emsize)

        # Permute to [Batch, Rows, Groups, emsize]
        embedded_x_BTGX = embedded_x_TBGX.permute(1, 0, 2, 3)

        # Add column positional embeddings
        embedded_x_BTGX = self._add_column_embeddings(embedded_x_BTGX)

        # --- 2. Process Y ---
        # Ensure Y has the feature dimension
        if y.dim() == 2:
            y = y.unsqueeze(-1)  # [T, B, 1]

        nan_inf_indicator_y = self._generate_nan_and_inf_indicator(y)
        y_imputed = self._impute_nan_and_inf_with_mean(y)

        y_concat = torch.cat([y_imputed, nan_inf_indicator_y], dim=-1)
        embedded_y_TBX = self.target_embedder(y_concat)  # [T, B, emsize]

        # Reshape to [Batch, Rows, 1, emsize]
        embedded_y_BT1X = embedded_y_TBX.permute(1, 0, 2).unsqueeze(2)

        # --- 3. Concatenate X and Y along the Feature Dimension ---
        # Result shape: [Batch, Rows, C + 1, emsize]
        cells = torch.cat((embedded_x_BTGX, embedded_y_BT1X), dim=2)

        return cells




class Config:
    """Mock config for hyperparams."""

    def __init__(self):
        self.cell_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.num_bins = 100
        self.min_y = -10.0
        self.max_y = 10.0
        # Borders required by FullSupportBarDistribution (num_bins + 1 borders)
        self.borders = torch.linspace(self.min_y, self.max_y, self.num_bins + 1)


import torch
import torch.nn as nn
from tabpfn.architectures.tabpfn_v2_5 import TabPFNBlock


class MLPBlock(nn.Module):
    """Transformer Feed-Forward Network to provide non-linear capacity after cross-attention."""

    def __init__(self, d_model, dim_feedforward=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.linear2(self.gelu(self.linear1(x)))
        return self.norm(res + x)


class ThinkingWarpTabPFN(nn.Module):
    def __init__(self, config, num_thinking_rows=8, num_features=2):
        super().__init__()
        self.cell_dim = config.cell_dim
        self.num_thinking_rows = num_thinking_rows
        self.num_features = num_features

        # 1. Encoder
        self.encoder = TabPFNEncoder(emsize=config.cell_dim, features_per_group=1)

        # 2. Shared Thinking Tokens (Identical for both streams to start)
        # Scaled initialization prevents early attention softmax saturation
        self.thinking_tokens = nn.Parameter(
            torch.randn(1, num_thinking_rows, num_features, self.cell_dim) * 0.02
        )

        # 3. Independent TabPFN Streams
        self.stream_A_blocks = nn.ModuleList([
            TabPFNBlock(
                emsize=config.cell_dim,
                nhead=config.num_heads,
                dim_feedforward=config.cell_dim
            ) for _ in range(config.num_layers)
        ])

        self.stream_B_blocks = nn.ModuleList([
            TabPFNBlock(
                emsize=config.cell_dim,
                nhead=config.num_heads,
                dim_feedforward=config.cell_dim
            ) for _ in range(config.num_layers)
        ])

        # 4. Phase 2 Bridge: All-to-All Cross Feature & Row Attention
        self.bridge_cross_attn = nn.MultiheadAttention(
            embed_dim=self.cell_dim, num_heads=config.num_heads, batch_first=True
        )
        self.bridge_norm = nn.LayerNorm(self.cell_dim)
        self.bridge_mlp = MLPBlock(self.cell_dim)  # Non-linear inverse warp engine

        # 5. Phase 3 Broadcast: Directed Cross-Attention
        self.broadcast_cross_attn = nn.MultiheadAttention(
            embed_dim=self.cell_dim, num_heads=config.num_heads, batch_first=True
        )
        self.broadcast_norm = nn.LayerNorm(self.cell_dim)
        self.broadcast_mlp = MLPBlock(self.cell_dim)

        # 6. Final Reasoning Block (Data rows integrate domain transformation)
        self.final_reasoning_block = TabPFNBlock(
            emsize=config.cell_dim,
            nhead=config.num_heads,
            dim_feedforward=config.cell_dim
        )

        # 7. Output Projection
        self.head = nn.Linear(self.cell_dim, config.num_bins)

    def forward(self, batch):
        train_dict = batch['train']

        # Dataloader inputs: [T, B, 1]
        X_A = train_dict['X_A']
        Y_A = train_dict['Y_A']
        X_B = train_dict['X_B']
        Y_B = train_dict['Y_B']

        # 1. Encode raw X and Y into [B, N, F, D] cell structures
        cells_A = self.encoder(X_A, Y_A)
        cells_B = self.encoder(X_B, Y_B)

        B_batch, N_A, F, D = cells_A.shape
        _, N_B, _, _ = cells_B.shape

        # --- PHASE 1: CONTEXTUALIZATION ---
        T_shared = self.thinking_tokens.expand(B_batch, -1, -1, -1)

        cells_A = torch.cat([cells_A, T_shared], dim=1)  # [B, N_A + N_think, F, D]
        cells_B = torch.cat([cells_B, T_shared], dim=1)  # [B, N_B + N_think, F, D]

        # Process independently using TabPFN list containers
        for block in self.stream_A_blocks:
            cells_A, _ = block([cells_A], single_eval_pos=None, save_peak_memory_factor=None)

        for block in self.stream_B_blocks:
            cells_B, _ = block([cells_B], single_eval_pos=None, save_peak_memory_factor=None)

        # --- PHASE 2: THE BRIDGE (All-to-All Attention + MLP) ---
        T_A = cells_A[:, N_A:, :, :]  # [B, N_think, F, D]
        T_B = cells_B[:, N_B:, :, :]

        seq_len_bridge = self.num_thinking_rows * F
        Q_bridge = T_B.reshape(B_batch, seq_len_bridge, D)
        K_bridge = V_bridge = T_A.reshape(B_batch, seq_len_bridge, D)

        attn_out_bridge, _ = self.bridge_cross_attn(query=Q_bridge, key=K_bridge, value=V_bridge)
        T_B_mapped = self.bridge_norm(Q_bridge + attn_out_bridge)
        T_B_mapped = self.bridge_mlp(T_B_mapped)  # Applies non-linear coordinate shift

        # --- PHASE 3: THE BROADCAST (Directed Cross-Attention) ---
        data_rows_B = cells_B[:, :N_B, :, :]  # [B, N_B, F, D]

        seq_len_data = N_B * F
        Q_data = data_rows_B.reshape(B_batch, seq_len_data, D)

        attn_out_broadcast, _ = self.broadcast_cross_attn(
            query=Q_data, key=T_B_mapped, value=T_B_mapped
        )

        data_rows_B_transformed = self.broadcast_norm(Q_data + attn_out_broadcast)
        data_rows_B_transformed = self.broadcast_mlp(data_rows_B_transformed)

        # --- PHASE 4: FINAL REASONING ---
        data_rows_B_transformed = data_rows_B_transformed.reshape(B_batch, N_B, F, D)
        data_rows_B_final, _ = self.final_reasoning_block(
            [data_rows_B_transformed], single_eval_pos=None, save_peak_memory_factor=None
        )

        # --- OUTPUT PROJECTION ---
        y_tokens = data_rows_B_final[:, :, -1, :]  # Grab target token [B, N_B, D]
        logits = self.head(y_tokens)  # [B, N_B, num_bins]

        # Returns [N_B, B, num_bins] to align with FullSupportBarDistribution
        return logits.permute(1, 0, 2)

if __name__ == "__main__":

    import torch.optim as optim
    from tqdm import tqdm


    # Imports based on your snippet
    from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    def train(epochs=200000):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Setup Config & Model
        config = Config()
        model = ThinkingWarpTabPFN(config, num_thinking_rows=8, num_features=2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # 2. Setup Bar Distribution Loss
        # We pass 'borders' as requested by FullSupportBarDistribution
        criterion = FullSupportBarDistribution(borders=config.borders.to(device))

        # 3. Setup Dataset Generator
        prior = HarmonicMixturePrior(noise_std=0.02, share_unrelated=0.0)
        dataset = InfiniteHarmonicsStream(
            prior, batch_size=128, n_A=20, n_B=50, n_test=128
        )

        # Create an iterator from the infinite stream
        data_iterator = iter(dataset)

        # 4. Training Loop

        pbar = tqdm(range(epochs),) # desc="Training Domain Bridge")

        for step in pbar:
            # Get next batch from the harmonic stream generator
            batch = next(data_iterator)

            # Move inputs to device (batch is nested dictionary)
            batch = {
                k: {k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()}
                for k, v in batch.items() if isinstance(v, dict)
            }

            optimizer.zero_grad()

            # Forward pass: yields logits of shape [T, B, num_bins]
            logits_B_in_A = model(batch)

            # Target is the known domain-transformed data: Y_B_in_A
            # Shape is [T, B, 1] -> squeeze to [T, B]
            target_Y = batch['train']['Y_B_in_A'].squeeze(-1)

            # Loss computation using FullSupportBarDistribution.forward()
            # forward(self, logits: T x B x num_bars, y: T x B) -> Returns NLL loss
            loss = criterion(logits=logits_B_in_A, y=target_Y)

            # Average the loss over the batch/sequence (if criterion returns element-wise)
            # Note: If FullSupportBarDistribution already returns scalar mean, skip this.
            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 50 == 0:
                pbar.set_postfix({'NLL Loss': f"{loss.item():.4f}"})

    train()

    #   8%|▊         | 16211/200000 [2:30:06<28:21:47,  1.80it/s, NLL Loss=1.4164]
    #  11%|█         | 21545/200000 [3:15:57<33:39:34,  1.47it/s, NLL Loss=1.2630]
    #  38%|███▊      | 38000/200000 [                    14.80it/s, NLL Loss=-0.50]
    #  84%|████████▍ | 168000/200000 [3:18:00<37:00,  14.33it/s, NLL Loss=-0.99]