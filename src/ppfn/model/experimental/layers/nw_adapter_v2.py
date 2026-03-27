import torch
import torch.nn as nn
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class NadarayaWatsonAdapterV2(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            dropout=0.1,
            nw_dropout=0.0,
            reuse_attn=True,
            hp_only_attn=True,
            seq_len=1000,
            address=None
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.hp_only_attn = hp_only_attn
        self.address = address
        self.reuse_attn = reuse_attn

        # --- 1. The Corrector (Local Domain Alignment) ---
        self.norm_nw_q = nn.LayerNorm(d_model)
        self.norm_nw_k = nn.LayerNorm(d_model)
        self.norm_nw_v = nn.LayerNorm(d_model)

        self.mlp_err = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.nw_attn = nn.MultiheadAttention(d_model, n_heads, dropout=nw_dropout)

        # --- 2. The Extractor (Knowledge Transfer) ---
        self.norm_train_q = nn.LayerNorm(d_model)
        self.norm_test_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

        self.attn_train = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_test = self.attn_train if reuse_attn else nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # --- 3. The Gate (Negative Transfer Prevention) ---
        # Learned sigmoid gate to weigh the contribution of the conditional stream
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # --- 4. FFN Modulator ---
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.initialize_as_identity()

    def initialize_as_identity(self):
        """
        Ensures the adapter starts as an identity mapping: C_out = C_in + 0.
        """
        # Zero out Extractor projections
        nn.init.zeros_(self.attn_train.out_proj.weight)
        nn.init.zeros_(self.attn_train.out_proj.bias)
        if not self.reuse_attn:
            nn.init.zeros_(self.attn_test.out_proj.weight)
            nn.init.zeros_(self.attn_test.out_proj.bias)

        # Zero out FFN
        nn.init.zeros_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[3].bias)

        # Zero out Corrector (B_prime starts as B_active)
        nn.init.zeros_(self.nw_attn.out_proj.weight)
        nn.init.zeros_(self.nw_attn.out_proj.bias)

        # Initialize gate to roughly 0.5 (neutral)
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

    def forward(self, A, B, C, sep, hp, **kwargs):
        """
        A: Target stream
        B: Related stream
        C: Conditional stream (to be updated)
        hp: Hyperparameter coordinates (T, 3*Batch, d_model)
        """
        hp_A, hp_B, hp_C = hp

        # --- Split Streams ---
        A_train = A[:sep]
        B_train, B_test, B_belief_A = B[:sep], B[sep:self.seq_len], B[self.seq_len:]
        C_train, C_test, C_belief_A = C[:sep], C[sep:self.seq_len], C[self.seq_len:]

        hp_A_train = hp_A[:sep]
        hp_B_active = hp_B[:self.seq_len]
        hp_C_train = hp_C[:sep]
        hp_C_test = hp_C[sep:self.seq_len]

        # ==========================================
        # STAGE 1: THE CORRECTOR (Undistort Stream B)
        # ==========================================
        # Local distortion = (Target - B's prediction of Target)
        # We project this error into the latent space via mlp_err
        distortion_error = self.mlp_err(self.norm_nw_v(A_train - B_belief_A))

        # NW Smoothing: Route the error based on spatial closeness in HP space
        q_nw = self.norm_nw_q(hp_B_active)
        k_nw = self.norm_nw_k(hp_A_train)

        # Apply the correction to B's manifold
        corr_out, corr_weights = self.nw_attn(q_nw, k_nw, distortion_error)
        B_active = B[:self.seq_len]
        B_prime = B_active + corr_out

        # ==========================================
        # STAGE 2: THE EXTRACTOR (Query B_prime)
        # ==========================================
        # Route based on HPs, extract from corrected B values
        k_ext = self.norm_k(hp_B_active)
        v_ext = B_prime

        if self.hp_only_attn:
            q_train, q_test = self.norm_train_q(hp_C_train), self.norm_test_q(hp_C_test)
        else:
            q_train, q_test = self.norm_train_q(C_train), self.norm_test_q(C_test)

        c_train_update, train_weights = self.attn_train(q_train, k_ext, v_ext)
        c_test_update, test_weights = self.attn_test(q_test, k_ext, v_ext)

        # Align updates with sequence structure
        c_belief_zero_update = torch.zeros_like(C_belief_A)
        C_update = torch.cat([c_train_update, c_test_update, c_belief_zero_update], dim=0)

        # ==========================================
        # STAGE 3: GATED RESIDUAL & FFN
        # ==========================================
        # Concatenate original C and proposed update to compute gate
        # This allows the model to reject B's info if NLL is increasing
        gate_input = torch.cat([C, C_update], dim=-1)
        g = self.gate(gate_input)

        C = C + (g * C_update)
        C = C + self.ffn(self.norm_ffn(C))

        # --- Metadata for logging ---
        ForwardMetaContext.log_stats(
            layer_name=self.address,
            stats_dict=dict(
                corrector_att_scores=corr_weights,
                train_attn_scores=train_weights,
                test_attn_scores=test_weights,
                gate_mean=g.mean().item()  # Monitor how much B is being used
            )
        )

        return A, B, C