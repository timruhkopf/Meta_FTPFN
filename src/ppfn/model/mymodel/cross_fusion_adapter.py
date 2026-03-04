from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


class CrossFusionAdapter(nn.Module):
    def __init__(
            self, d_model, num_heads, dropout=0.0, use_prenorm=True,
            use_gate=True, add_linear=False, reuse_attention=False, C_as_Q=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_prenorm = use_prenorm
        self.reuse_attention = reuse_attention
        self.C_as_Q = C_as_Q
        self.use_gate = use_gate
        self.add_linear = add_linear

        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout)
        if reuse_attention:
            self.test_attn = self.cross_attn
        else:
            self.test_attn = nn.MultiheadAttention(d_model, num_heads, dropout)

        if add_linear:
            self.linear = nn.Linear(d_model, d_model)
        else:
            self.linear = None

        # 1. Explicit Gating Mechanism instead of a standard adapter
        if self.use_gate:
            # Takes concatenated [Query, Attention_Update] -> Outputs scalar gate
            self.gate_proj = nn.Linear(d_model * 2, 1)
        else:
            self.gate_proj = None

        if self.use_prenorm:
            self.norm_Q = nn.LayerNorm(d_model)
            self.norm_K = nn.LayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.initialize_as_identity()
        self.single_eval_pos = None

        # State variable to store the sparsity loss for the current forward pass
        self.aux_gate_loss = torch.tensor(0.0)

    def initialize_as_identity(self):
        if not self.use_prenorm:
            nn.init.constant_(self.norm.weight, 1)
            nn.init.constant_(self.norm.bias, 0)

        if self.linear is not None:
            nn.init.zeros_(self.linear.weight)

        if self.use_gate:
            # Initialize weights to 0
            nn.init.zeros_(self.gate_proj.weight)
            # Initialize bias to a negative value (e.g., -2.0 or -3.0).
            # Sigmoid(-2.0) = 0.11. This starts the gate "mostly closed", acting as a soft identity.
            nn.init.constant_(self.gate_proj.bias, -2.0)
        elif self.add_linear:
            # FIX: Zero both weights AND biases of the linear layer
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        else:
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)
            if not self.reuse_attention:
                nn.init.zeros_(self.test_attn.out_proj.weight)
                nn.init.zeros_(self.test_attn.out_proj.bias)


    def forward(self, A, B, C, sep, *args, **kwargs):

        # 1. Determine Query and Key/Value inputs for Attention
        Q_input = C if self.C_as_Q else A
        K_input = B

        # 2. Apply Pre-Norm if enabled (safely normalizes Q without touching the residual C)
        if self.use_prenorm:
            Q_input = self.norm_Q(Q_input)
            K_input = self.norm_K(K_input)

        # 3. Train/Test split across Sequence dimension (index 0)
        Q_train, Q_test = Q_input[:sep, :, :], Q_input[sep:, :, :]
        K_train = K_input[:sep, :, :]
        C_train, C_test = C[:sep, :, :], C[sep:, :, :]

        # 4. Cross Attention
        train_delta = self.cross_attn(Q_train, K_train, K_train)[0]
        test_delta = self.test_attn(Q_test, K_train, K_train)[0]

        if self.add_linear:
            train_delta = self.linear(train_delta)
            test_delta = self.linear(test_delta)

        # 5. Apply Explicit Gate and Compute Aux Loss
        self.aux_gate_loss = torch.tensor(0.0, device=A.device)
        if self.use_gate:
            # Compute gate scores: shape (T, R, 1)
            gate_train = torch.sigmoid(self.gate_proj(torch.cat([Q_train, train_delta], dim=-1)))
            gate_test = torch.sigmoid(self.gate_proj(torch.cat([Q_test, test_delta], dim=-1)))

            # Record L1 sparsity loss (average over sequence and batch)
            self.aux_gate_loss = gate_train.abs().mean() + gate_test.abs().mean()

            # Apply gate
            train_delta = train_delta * gate_train
            test_delta = test_delta * gate_test

        train_update = C_train + train_delta
        test_update = C_test + test_delta

        #  Apply Post-Norm if enabled (Standard formulation on the sum)
        if not self.use_prenorm:
            train_update = self.norm(train_update)
            test_update = self.norm(test_update)

        # Recombine train-test Sequence (dim=0) and Batch unaltered A, B with conditional C (dim=1)
        conditional = torch.cat([train_update, test_update], dim=0)

        return torch.cat([A, B, conditional], dim=1)

    # Optional: A helper to locate this module in a large model
    def get_aux_loss(self):
        return self.aux_gate_loss