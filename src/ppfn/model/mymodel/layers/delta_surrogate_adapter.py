import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class LowDimAttention(nn.Module):

    def __init__(self, d_model, d_hp, d_k, nhead, dropout=0.1):
        super().__init__()
        assert d_k % nhead == 0
        assert d_model % nhead == 0

        self.nhead = nhead
        self.d_k = d_k
        self.head_dim_k = d_k // nhead
        self.head_dim_v = d_model // nhead
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_hp, d_k)
        self.k_proj = nn.Linear(d_hp, d_k)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def gather_attn_telemetry(self, q, k, layer, task_name):
        """
        Gathers attention telemetry for BMA blending.

        - Support Score ($uparrow$): High raw dot-product energy (I found relevant points).
        - Entropy ($downarrow$): Low entropy (I found specific relevant points).

        Why Unnormalized for Trust? When deciding between $B_0$ and $B_1$, the raw dot-product $Q K^T$ (before Softmax)
        tells you the absolute geometric proximity. A high raw score means I found a near-perfect match;
        a low raw score means I m averaging distant noise.

        # TODO other metrics to consider:
        - Alignment Error ($downarrow$): Small shared_domain_error (This task matches $A$s history).
        - Task Length ($uparrow$):
        """

        # 1. Calculate Raw Scores
        # (Batch, Heads, T_q, T_k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 2. Extract Support (Pre-Softmax LogSumExp is a good density proxy)
        # Higher density = more relevant tokens found in this task
        support = torch.logsumexp(attn_logits, dim=-1).mean(dim=1)  # (Batch, T_q)

        # 3. Calculate Softmax and Entropy
        attn_probs = F.softmax(attn_logits, dim=-1)
        # Entropy H = -sum(p * log(p))
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).mean(dim=1)

        # 4. Telemetry (Sidecar)
        # Store these for the final BMA blending logic
        ForwardMetaContext.set(f"{layer}/{task_name}/support", support)
        ForwardMetaContext.set(f"{layer}/{task_name}/entropy", entropy)

    def forward(self, q_raw, k_raw, v_raw, layer='l1', task_name='B'):
        # q_raw: (T_q, B, d_hp), k_raw: (T_k, B, d_hp), v_raw: (T_k, B, d_model)
        T_q, B, _ = q_raw.shape
        T_k = k_raw.shape[0]

        q = self.q_proj(q_raw).view(T_q, B, self.nhead, self.head_dim_k).permute(1, 2, 0, 3)
        k = self.k_proj(k_raw).view(T_k, B, self.nhead, self.head_dim_k).permute(1, 2, 0, 3)
        v = self.v_proj(v_raw).view(T_k, B, self.nhead, self.head_dim_v).permute(1, 2, 0, 3)

        self.gather_attn_telemetry(q, k, layer, task_name)

        dropout_rate = self.dropout_p if self.training else 0.0
        # Scaled dot product attention (FlashAttention compatible)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_rate)

        attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(T_q, B, -1)
        return self.out_proj(attn_out)


class DeltaSurrogateAdapter(nn.Module):

    def __init__(self, d_model, d_hp, d_k=64, n_heads=4, dropout=0.1, seq_len=1000, reuse_attn=False):
        """
        Basic idea of this adapter is, that we can use only the HP coordinates for value retrieval,
        making the attention lower dimensional and much more efficient. This way we can quickly identify
        the latent error patterns based on B's belief on A_train in the domain of B and the actual A_train.
        Propagating this error to the

        Note on reuse attn: since we are just asking for hp-based attention, what good does it do to have separate attn
        modules.
        """
        super().__init__()
        self.d_model = d_model
        self.d_hp = d_hp
        self.d_k = d_k
        self.seq_len = seq_len

        self.attn_shared = LowDimAttention(d_model, d_hp, d_k, n_heads, dropout)
        self.attn_extrapolate = LowDimAttention(d_model, d_hp, d_k, n_heads, dropout)
        if reuse_attn:
            self.attn_calibrate = self.attn_shared
        else:
            self.attn_calibrate = LowDimAttention(d_model, d_hp, d_k, n_heads, dropout)

        self.norm_shared_q = nn.LayerNorm(d_hp)
        self.norm_shared_k = nn.LayerNorm(d_hp)
        self.norm_shared_v = nn.LayerNorm(d_model)
        self.norm_ext_q = nn.LayerNorm(d_hp)
        self.norm_ext_k = nn.LayerNorm(d_hp)
        self.norm_ext_v = nn.LayerNorm(d_model)
        self.norm_cal_q = nn.LayerNorm(d_hp)
        self.norm_cal_k = nn.LayerNorm(d_hp)
        self.norm_cal_v = nn.LayerNorm(d_model)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.attn_shared.out_proj.weight, 0)
        nn.init.constant_(self.attn_extrapolate.out_proj.weight, 0)
        nn.init.constant_(self.attn_calibrate.out_proj.weight, 0)
        nn.init.constant_(self.ff_network[-1].weight, 0)

    def forward(self, A, B, C, sep, hp, **kwargs):

        hp_A, hp_B, hp_C = hp

        hp_A_train = hp_A[:sep, :, :]
        hp_B_train = hp_B[:sep, :, :]
        hp_C_test = hp_C[sep:, :, :]

        # --- Latent Slicing (Aligned with Coordinate Batch R) ---
        A_train = A[:sep, :, :]
        B_train = B[:sep, :, :]
        C_train = C[:sep, :, :]
        C_test = C[sep:, :, :]

        # Path 1: Shared Domain Calibration
        # Notice, how this will allow us to get attn scores telling us about the Venn diagram of A, B for each test token
        B_at_A = self.attn_shared(
            q_raw=self.norm_shared_q(hp_A_train),
            k_raw=self.norm_shared_k(hp_B_train),
            v_raw=self.norm_shared_v(B_train)
        )
        shared_domain_error = B_at_A - A_train

        # Path 2a: Extrapolation
        # FIXME: we can use B_belief_A_train from the PFN instead, saving an attention and error compounding / attn washing
        B_raw_at_C = self.attn_extrapolate(
            q_raw=self.norm_ext_q(hp_C_test),
            k_raw=self.norm_ext_k(hp_B_train),
            v_raw=self.norm_ext_v(B_train)
        )

        # Path 2b: Calibration (Using fixed argument names)
        error_at_C = self.attn_calibrate(
            q_raw=self.norm_cal_q(hp_C_test),
            k_raw=self.norm_cal_k(hp_A_train),
            v_raw=self.norm_cal_v(shared_domain_error)
        )

        C_test_update = B_raw_at_C - error_at_C

        # Modern Pre-Norm style
        C_test_update = self.dropout(C_test_update)
        C_test = C_test + C_test_update
        C_test = self.layer_norm_1(C_test)

        ff_out = self.dropout(self.ff_network(C_test))
        C_test = C_test + ff_out
        C_test_final = self.layer_norm_2(C_test)

        C_output = torch.cat([C_train, C_test_final], dim=0)

        return A, B, C_output


if __name__ == "__main__":
    T, sep, R = 20, 12, 8
    d_model, d_hp, d_k = 128, 12, 32

    A = torch.randn(T, R * 3, d_model)
    B = torch.randn(T, R * 3, d_model)
    C = torch.randn(T, R * 3, d_model)
    hp = torch.randn(T, R * 3, d_hp)

    model = DeltaSurrogateAdapter(d_model, d_hp, d_k)
    A_out, B_out, C_out = model(A, B, C, sep, hp)

    print(f"Success! Output C_out shape: {C_out.shape}")  # Should be (20, 8, 128)