import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramAttentionGate(nn.Module):
    def __init__(self, d_model, num_bins=10):
        super().__init__()
        self.num_bins = num_bins

        # The gate evaluates the normalized histogram of attention weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_bins, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Initialize biases to favor an open gate (~0.73) initially
        # so gradients can flow to the attention mechanism early in training.
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.constant_(self.gate_mlp[2].bias, 1.0)
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight)

    def forward(self, attn_weights, C, C_update):
        """
        attn_weights: (Batch, T_query, T_key) - from nn.MultiheadAttention
        C: (T_query, Batch, d_model) - current state
        C_update: (T_query, Batch, d_model) - proposed update from MHA
        """
        # 1. Map attention weights [0, 1] to bin indices [0, num_bins - 1]
        # Example: weight 0.15 with 10 bins becomes index 1.
        bin_indices = torch.clamp((attn_weights * self.num_bins).long(), max=self.num_bins - 1)

        # 2. Convert to one-hot to count occurrences
        # Shape: (Batch, T_query, T_key, num_bins)
        hist_one_hot = F.one_hot(bin_indices, num_classes=self.num_bins).float()

        # 3. Sum over the Keys dimension (dim=-2) to get the counts per bin
        # Shape: (Batch, T_query, num_bins)
        hist_counts = hist_one_hot.sum(dim=-2)

        # 4. Normalize by the number of keys so the histogram sums to 1.0
        T_key = attn_weights.shape[-1]
        hist_normalized = hist_counts / T_key

        # 5. DETACH THE HISTOGRAM (Stop-Gradient)
        # This is the crucial step preventing gradient sabotage.
        gate_input = hist_normalized.detach()

        # 6. Pass through MLP. Transpose to match C's (Seq, Batch, Feature) shape.
        # Transposed shape: (T_query, Batch, num_bins)
        gate_input = gate_input.transpose(0, 1)

        # gate_values shape: (T_query, Batch, 1)
        gate_values = self.gate_mlp(gate_input)

        # 7. Apply the gated residual
        C_gated = C + (gate_values * C_update)

        return C_gated, gate_values


from ppfn.model.mymodel.meta_context import ForwardMetaContext  # Assuming this is your logging import


class MHA_SpatioRepresentationalAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_task_pe=True):
        super().__init__()
        self.d_model = d_model
        self.use_task_pe = use_task_pe
        self.address = "mha_spatio_adapter"  # For logging

        if use_task_pe:
            self.task_embedding = nn.Embedding(2, d_model)

        # Projections to fuse Representation + HP back to d_model
        self.q_proj = nn.Linear(d_model * 2, d_model)
        self.k_proj = nn.Linear(d_model * 2, d_model)

        # Cross-Attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # NEW: The Histogram Gate
        self.hist_gate = HistogramAttentionGate(d_model, num_bins=10)

        # FFN Modulator
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.init_as_identity()

    def init_as_identity(self):
        # 1. Zero out MHA output projection
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)

        # 2. Zero out the final linear layer in FFN (Index 3, NOT -1)
        final_linear_layer = self.ffn[3]
        nn.init.zeros_(final_linear_layer.weight)
        if final_linear_layer.bias is not None:
            nn.init.zeros_(final_linear_layer.bias)

        # 3. Initialize Task Embeddings (Small normal to break symmetry)
        if self.use_task_pe:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

        # 4. Initialize Q and K Spatio-Representational Fusions
        # Normal initialization here is safe because the out_proj catches and zeroes it.
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

        # NEW: Apply the detached Histogram Gate instead of raw addition
        # C_update is attn_out. The gate returns the gated C and the gate values.
        C, gate_values = self.hist_gate(attn_weights, C, attn_out)

        # Apply FFN (Uncommented assuming you want the FFN modulator active)
        C = C + self.ffn(self.norm_ffn(C))

        # Log the gate values to monitor the rejection rate!
        ForwardMetaContext.log_stats(
            layer_name=self.address,
            stats_dict=dict(
                gate_mean=gate_values.mean().item()
            )
        )

        return A, B, C