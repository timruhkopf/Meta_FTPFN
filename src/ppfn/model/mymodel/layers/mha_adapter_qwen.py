import torch
import torch.nn as nn
import torch.nn.functional as F
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class QwenGatedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, q_input_dim, k_input_dim, v_input_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 1. Standard Q, K, V Projections
        self.q_proj = nn.Linear(q_input_dim, d_model)
        self.k_proj = nn.Linear(k_input_dim, d_model)
        self.v_proj = nn.Linear(v_input_dim, d_model)

        # 2. Qwen3 Elementwise Gate Projection
        # Computed directly from the unprojected Query input
        self.element_gate_proj = nn.Linear(q_input_dim, d_model)

        # 3. Final Output Mixing
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

        self._init_qwen_weights()

    def _init_qwen_weights(self):
        # Soft-start Q/K to prevent Softmax saturation
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.zeros_(self.k_proj.bias)

        # Standard init for Values and Output
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

        # QWEN INITIALIZATION: Bias the gate to start slightly closed (-1.0).
        # This shields the network from the attention sink during early training,
        # allowing the ReZero prior to dominate until alignment is learned.
        nn.init.xavier_uniform_(self.element_gate_proj.weight)
        nn.init.constant_(self.element_gate_proj.bias, -1.0)

    def forward(self, hidden_states, key_states, value_states):
        """
        hidden_states: (T_q, Batch, Dim)
        key_states:    (T_kv, Batch, Dim)
        value_states:  (T_kv, Batch, Dim)
        """
        # Unpack assuming (Sequence, Batch, Dimension)
        q_len, bsz, _ = hidden_states.shape
        kv_len, _, _ = key_states.shape

        # 1. Project (output remains T, B, D)
        q = self.q_proj(hidden_states)
        k = self.k_proj(key_states)
        v = self.v_proj(value_states)

        # 2. Reshape for SDPA
        # From (T, B, d_model) -> (T, B, n_heads, head_dim) -> (Batch, n_heads, T, head_dim)
        q = q.view(q_len, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(kv_len, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(kv_len, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

        # 3. Compute Gate from RAW hidden_states
        gate_weights = torch.sigmoid(self.element_gate_proj(hidden_states))
        # Reshape gate to match SDPA output shape: (Batch, n_heads, T_q, head_dim)
        gate_weights = gate_weights.view(q_len, bsz, self.n_heads, self.head_dim).permute(1, 2, 0, 3)

        # 4. Execute SDPA
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )

        # 5. Apply the Sparse Gate (The Sink Killer)
        attn_output = attn_output * gate_weights

        # 6. Final Projection back to (T_q, Batch, d_model)
        # Permute from (Batch, n_heads, T_q, head_dim) -> (T_q, Batch, n_heads, head_dim)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(q_len, bsz, self.d_model)

        final_output = self.o_proj(attn_output)

        return final_output, gate_weights.mean()

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


class ThreeStreamAdapter(nn.Module):
    def __init__(self, d_model, n_heads, d_hp, dropout=0.1, use_task_pe=True):
        super().__init__()
        self.d_model = d_model
        self.use_task_pe = use_task_pe
        self.address = "three_stream_adapter"

        if use_task_pe:
            self.task_embedding = nn.Embedding(2, d_model)

        # Dimensionality routing: Q and K expect latents + coords. V expects only latents.
        q_k_dim = d_model + d_hp
        v_dim = d_model

        # Normalization layers applied before projection
        self.norm_q = nn.LayerNorm(q_k_dim)
        self.norm_k = nn.LayerNorm(q_k_dim)
        self.norm_v = nn.LayerNorm(v_dim)

        # Instantiate our factored Qwen engine
        self.gated_cross_attn = QwenGatedCrossAttention(
            d_model=d_model,
            n_heads=n_heads,
            q_input_dim=q_k_dim,
            k_input_dim=q_k_dim,
            v_input_dim=v_dim,
            dropout=dropout
        )

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model)
        self.ffn_drop = nn.Dropout(dropout)

        # ReZero Parameters
        # self.mha_alpha = nn.Parameter(torch.zeros(1))
        # self.ffn_alpha = nn.Parameter(torch.zeros(1))

        if self.use_task_pe:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def forward(self, A, B, C, sep, hp, **kwargs):
        device = A.device
        hp_A, hp_B, hp_C = hp

        # 1. Isolate Training Contexts
        A_train, B_train = A[:sep], B[:sep]
        hp_A_train, hp_B_train = hp_A[:sep], hp_B[:sep]

        # 2. Inject Task Embeddings
        if self.use_task_pe:
            emb_A = self.task_embedding(torch.tensor(0, device=device))
            emb_B = self.task_embedding(torch.tensor(1, device=device))

            A_train_pe = A_train + emb_A
            B_train_pe = B_train + emb_B
            C_pe = C + emb_A
        else:
            A_train_pe, B_train_pe, C_pe = A_train, B_train, C

        # 3. Construct the Information Streams
        # Values (Payload): Pure Latents + Task PE
        context_v = torch.cat([A_train_pe, B_train_pe], dim=0)

        # Queries (Workbench): Latents + PE + Coordinates
        query_states = torch.cat([C_pe, hp_C], dim=-1)

        # Keys (Routing): Latents + PE + Coordinates
        context_k = torch.cat([
            torch.cat([A_train_pe, hp_A_train], dim=-1),
            torch.cat([B_train_pe, hp_B_train], dim=-1)
        ], dim=0)

        # 4. Route through the Qwen Engine
        attn_out, mean_gate_value = self.gated_cross_attn(
            hidden_states=self.norm_q(query_states),
            key_states=self.norm_k(context_k),
            value_states=self.norm_v(context_v)
        )

        # 5. Apply ReZero and Residual Updates
        C = C + (attn_out ) # * self.mha_alpha)

        ffn_out = self.ffn_drop(self.ffn(self.norm_ffn(C)))
        C = C + (ffn_out) # * self.ffn_alpha)

        # 6. Telemetry
        ForwardMetaContext.log_stats(
            layer_name=self.address,
            stats_dict=dict(
                gate_mean=mean_gate_value.item(),
                mha_alpha=self.mha_alpha.item(),
                ffn_alpha=self.ffn_alpha.item()
            )
        )

        return A, B, C
