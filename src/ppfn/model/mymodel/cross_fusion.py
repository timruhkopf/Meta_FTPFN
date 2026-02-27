from typing import Tuple

import torch
from torch import nn
from torch.nn import MultiheadAttention


class CrossFusion(nn.Module):
    def __init__(
            self, d_model, num_heads, dropout=0.0, use_prenorm=True, add_adapter=True, reuse_attention=False,
            C_as_Q=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_prenorm = use_prenorm
        self.reuse_attention = reuse_attention
        self.C_as_Q = C_as_Q

        # Default is batch_first=False, which expects (T, B, D)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout)
        if reuse_attention:
            self.test_attn = self.cross_attn
        else:
            self.test_attn = nn.MultiheadAttention(d_model, num_heads, dropout)

        if add_adapter:
            bottleneck_dim = max(d_model // 4, 1)  # Ensure at least dim 1
            self.adapter = nn.Sequential(
                # nn.Linear(d_model, bottleneck_dim),
                # nn.GELU(),
                # nn.Linear(bottleneck_dim, d_model)
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
        else:
            self.adapter = None

        if self.use_prenorm:
            self.norm_Q = nn.LayerNorm(d_model)
            self.norm_K = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.initialize_as_identity()
        self.single_eval_pos = None

    def initialize_as_identity(self):
        # 1. Initialize Post-Norm to identity if used
        if not self.use_prenorm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

        # 2. Zero out the final projection to ensure the residual block starts as an identity map
        if self.adapter is not None:
            nn.init.zeros_(self.adapter[0].weight)
            nn.init.zeros_(self.adapter[0].bias)
        else:
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
            if not self.reuse_attention:
                nn.init.zeros_(self.test_attn.out_proj.weight)
                nn.init.zeros_(self.test_attn.out_proj.bias)

    def validate_forward_args(self, x, *args, **kwargs) -> Tuple[int, int]:
        single_eval_pos = kwargs.get("single_eval_pos", self.single_eval_pos)
        assert single_eval_pos is not None, "single_eval_pos must be provided during training"

        B = x.shape[1]  # Batch dimension is index 1
        assert B % 3 == 0, "Batch size must be multiple of 3 (A, B, C task triplets)"
        return B, single_eval_pos

    def forward(self, x, *args, **kwargs):
        B_dim, sep = self.validate_forward_args(x, *args, **kwargs)
        R = B_dim // 3

        # 1. Extract raw streams across the Batch dimension (index 1)
        A = x[:, :R, :]
        B_stream = x[:, R: 2 * R, :]
        C = x[:, 2 * R:, :]

        # 2. Determine Query and Key/Value inputs for Attention
        Q_input = C if self.C_as_Q else A
        K_input = B_stream

        # 3. Apply Pre-Norm if enabled (safely normalizes Q without touching the residual C)
        if self.use_prenorm:
            Q_input = self.norm_Q(Q_input)
            K_input = self.norm_K(K_input)

        # 4. Train/Test split across Sequence dimension (index 0)
        Q_train, Q_test = Q_input[:sep, :, :], Q_input[sep:, :, :]
        K_train = K_input[:sep, :, :]

        # 5. Train/Test split for Residual (always un-normalized)
        C_train, C_test = C[:sep, :, :], C[sep:, :, :]

        # 6. Cross Attention (Returns (T, B, D))
        train_delta = self.cross_attn(Q_train, K_train, K_train)[0]
        test_delta = self.test_attn(Q_test, K_train, K_train)[0]

        # 7. Apply Adapter if present
        if self.adapter is not None:
            train_delta = self.adapter(train_delta)
            test_delta = self.adapter(test_delta)

        # 8. Residual Addition
        train_update = C_train + train_delta
        test_update = C_test + test_delta

        # 9. Apply Post-Norm if enabled (Standard formulation on the sum)
        if not self.use_prenorm:
            train_update = self.norm(train_update)
            test_update = self.norm(test_update)

        # 10. Recombine Sequence (dim=0) and Batch (dim=1)
        conditional = torch.cat([train_update, test_update], dim=0)
        output = torch.cat([x[:, : 2 * R, :], conditional], dim=1)

        return output