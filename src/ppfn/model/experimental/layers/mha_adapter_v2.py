import torch
import torch.nn as nn
import torch.nn.functional as F
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class HistogramAttentionGate(nn.Module):
    def __init__(self, d_model, num_bins=10):
        super().__init__()
        self.num_bins = num_bins

        # Input is now num_bins + 2 (for Entropy and Max Weight)
        gate_input_dim = num_bins + 2

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Initialize bias to ~0.73
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.constant_(self.gate_mlp[2].bias, 1.0)
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight)

    def forward(self, attn_weights, C, C_update):
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 1. Histogram Calculation
        bin_indices = torch.clamp(
            (attn_weights * self.num_bins).long(),
            min=0, max=self.num_bins - 1
        )
        hist_one_hot = F.one_hot(bin_indices, num_classes=self.num_bins).float()
        hist_counts = hist_one_hot.sum(dim=-2)

        T_key = attn_weights.shape[-1]
        hist_normalized = hist_counts / T_key

        # 2. Entropy Calculation: H = -sum(p * log(p + eps))
        epsilon = 1e-9
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1)

        # 3. Max Weight Calculation
        max_weight = torch.max(attn_weights, dim=-1)[0]

        # 4. Detach and Assemble Gate Input
        hist_detached = hist_normalized.detach()
        entropy_detached = entropy.unsqueeze(-1).detach()
        max_weight_detached = max_weight.unsqueeze(-1).detach() # basically the sink value?

        # Shape: (Batch, T_query, num_bins + 2)
        gate_input = torch.cat([hist_detached, entropy_detached, max_weight_detached], dim=-1)

        # 5. Pass through MLP
        gate_input = gate_input.transpose(0, 1)  # Shape: (T_query, Batch, num_bins + 2)
        gate_values = self.gate_mlp(gate_input)  # Shape: (T_query, Batch, 1)

        # 6. Apply gated residual
        C_gated = C + (gate_values * C_update)

        return C_gated, gate_values


class SwiGLU(nn.Module):
    """Modern SwiGLU activation block to replace standard GELU FFN"""

    def __init__(self, d_model, hidden_dim_multiplier=2):
        super().__init__()
        # Multiplying by 2 to keep parameter count roughly equivalent to your old d_model * 4
        hidden_dim = d_model * hidden_dim_multiplier
        self.w12 = nn.Linear(d_model, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)




class MHA_SpatioRepresentationalAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_task_pe=True):
        super().__init__()
        self.d_model = d_model
        self.use_task_pe = use_task_pe
        self.address = "mha_spatio_adapter"

        if use_task_pe:
            self.task_embedding = nn.Embedding(2, d_model)

        self.q_proj = nn.Linear(d_model * 2, d_model)
        self.k_proj = nn.Linear(d_model * 2, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.hist_gate = HistogramAttentionGate(d_model, num_bins=10)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.ffn_drop = nn.Dropout(dropout)

        # NEW: ReZero Parameters for Identity Initialization
        self.mha_alpha = nn.Parameter(torch.zeros(1))
        self.ffn_alpha = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        # 1. No more zeroing out the out_proj or FFN weights!
        # We rely on the ReZero alphas for the step 0 identity map.

        if self.use_task_pe:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

    def forward(self, A, B, C, sep, hp, **kwargs):
        device = A.device
        hp_A, hp_B, hp_C = hp

        A_train, B_train = A[:sep], B[:sep]
        hp_A_train, hp_B_train = hp_A[:sep], hp_B[:sep]
        hp_C_active = hp_C

        if self.use_task_pe:
            emb_A = self.task_embedding(torch.tensor(0, device=device))
            emb_B = self.task_embedding(torch.tensor(1, device=device))

            A_train_pe = A_train + emb_A
            B_train_pe = B_train + emb_B
            C_pe = C + emb_A

            context_v = torch.cat([A_train_pe, B_train_pe], dim=0)

            C_concat = torch.cat([C_pe, hp_C_active], dim=-1)
            A_train_concat = torch.cat([A_train_pe, hp_A_train], dim=-1)
            B_train_concat = torch.cat([B_train_pe, hp_B_train], dim=-1)
            context_concat = torch.cat([A_train_concat, B_train_concat], dim=0)
        else:
            context_v = torch.cat([A_train, B_train], dim=0)
            C_concat = torch.cat([C, hp_C_active], dim=-1)
            context_concat = torch.cat([
                torch.cat([A_train, hp_A_train], dim=-1),
                torch.cat([B_train, hp_B_train], dim=-1)
            ], dim=0)

        q_fused = self.q_proj(C_concat)
        k_fused = self.k_proj(context_concat)

        # CROSS-ATTENTION
        attn_out, attn_weights = self.cross_attn(
            self.norm_q(q_fused),
            self.norm_k(k_fused),
            self.norm_v(context_v)
        )

        # Apply ReZero alpha to MHA update BEFORE the gate
        attn_out = attn_out * self.mha_alpha

        # The gate evaluates the raw weights, but applies to the scaled attn_out
        C, gate_values = self.hist_gate(attn_weights, C, attn_out)

        # FFN with ReZero alpha
        ffn_out = self.ffn_drop(self.ffn(self.norm_ffn(C)))
        C = C + (ffn_out * self.ffn_alpha)

        # Logging
        ForwardMetaContext.log_stats(
            layer_name=self.address,
            stats_dict=dict(
                gate_mean=gate_values.mean().item(),
                mha_alpha=self.mha_alpha.item(),
                ffn_alpha=self.ffn_alpha.item()
            )
        )

        return A, B, C