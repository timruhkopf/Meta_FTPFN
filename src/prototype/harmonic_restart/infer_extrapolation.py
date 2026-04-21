import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch import Tensor
import math
import matplotlib.pyplot as plt


# ==========================================
# 1. MOCKS & HELPER FUNCTIONS
# ==========================================
def apply_adain(content, style, eps=1e-5):
    c_mean, c_std = content.mean(dim=0, keepdim=True), content.std(dim=0, keepdim=True)
    s_mean, s_std = style.mean(dim=0, keepdim=True), style.std(dim=0, keepdim=True)
    return (content - c_mean) / (c_std + eps) * (s_std + eps) + s_mean


class MockMetaContext:
    _store = {}

    @classmethod
    def set(cls, key, val): cls._store[key] = val

    @classmethod
    def get(cls, key): return cls._store.get(key)

    @classmethod
    def clear(cls): cls._store = {}


ForwardMetaContext = MockMetaContext


# ==========================================
# 2. PATCHED LAYER (Full-Domain Unwarping)
# ==========================================
class TriStreamLayerFixed(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, num_align_steps=3):
        super().__init__()
        self.num_align_steps = num_align_steps
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.align_attn = MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.valve_gain = nn.Parameter(torch.tensor(1.0))
        self.align_ffn = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model * 2)
        )
        self.valve_controller = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm_cross = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout_cross = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, A, B, C, sep):
        # 1. Self Attention (Simplified for test)
        normed_B = self.norm1(B)
        normed_C = self.norm1(C)
        # We let B self-attend fully so it maintains structural integrity across the domain
        B = B + self.dropout1(self.self_attn(normed_B, normed_B, normed_B)[0])

        train_C, test_C = normed_C[:sep], normed_C[sep:]
        out_C_train = self.self_attn(train_C, train_C, train_C)[0]
        out_C_test = self.self_attn(test_C, train_C, train_C)[0]
        C = C + self.dropout1(torch.cat([out_C_train, out_C_test], dim=0))

        # 2. Cross Attention (THE PATCHED GATED DEFORM BRANCH)
        normed_cross_C = self.norm_cross(C)
        normed_cross_B = self.norm_cross(B)

        # ---> FIX: KEEP FULL B SEQUENCE <---
        B_full_pred = normed_cross_B.clone()
        C_train = normed_cross_C[:sep, :, :]

        B_full_pred = apply_adain(content=B_full_pred, style=C_train)
        total_kl_loss = 0.0

        for step in range(self.num_align_steps):
            # A. Correspondence ONLY in the known subspace
            B_train = B_full_pred[:sep, :, :]
            align_context_train, _ = self.align_attn(query=B_train, key=C_train, value=C_train)

            # B. Global Context derived from subspace
            global_context = align_context_train.mean(dim=0, keepdim=True)

            # C. ---> FIX: EXTRAPOLATE CONTEXT TO FULL DOMAIN <---
            # For the blind spot, we substitute the missing local context with the global context
            align_context_extrapolated = global_context.expand(B_full_pred.shape[0] - sep, B_full_pred.shape[1], -1)
            align_context_full = torch.cat([align_context_train, align_context_extrapolated], dim=0)
            global_context_full = global_context.expand_as(B_full_pred)

            align_features = torch.cat([B_full_pred, align_context_full, global_context_full], dim=-1)
            B_delta_params = self.align_ffn(align_features)
            mu, log_var = torch.chunk(B_delta_params, chunks=2, dim=-1)

            if self.training:
                std = torch.exp(0.5 * log_var)
                B_delta = mu + torch.randn_like(std) * std
                step_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
                total_kl_loss += step_kl.mean()
            else:
                B_delta = mu

            # Shift the FULL sequence
            B_full_pred = B_full_pred + B_delta

        # Null Attention Valve (Uses Full B)
        mean_log_var = log_var.mean(dim=[0, 2])
        mean_energy = B_delta.pow(2).mean(dim=[0, 2])
        raw_gate = self.valve_controller(torch.stack([mean_log_var, mean_energy], dim=-1))
        gate_scale = torch.exp(raw_gate * self.valve_gain).unsqueeze(0)

        k_sink_base = normed_cross_C.mean(dim=0, keepdim=True)
        mu_B = B_full_pred.mean(dim=0, keepdim=True)
        dot_product = (k_sink_base * mu_B).sum(dim=-1, keepdim=True)
        norm_sq = (mu_B * mu_B).sum(dim=-1, keepdim=True) + 1e-8
        k_null = (k_sink_base - (dot_product / norm_sq) * mu_B)  * gate_scale
        v_null = torch.zeros_like(k_null)

        # Keys and Values are now the FULL domain + Sink
        K_train_with_sink = torch.cat([B_full_pred, k_null], dim=0)
        V_train_with_sink = torch.cat([B_full_pred, v_null], dim=0)

        # K_train_with_sink = B_full_pred
        # V_train_with_sink = B_full_pred
        cross_C, cross_attn_weights = self.cross_attn(
            query=normed_cross_C,
            key=K_train_with_sink,
            value=V_train_with_sink
        )
        C = C + self.dropout_cross(cross_C)

        # 3. FFN
        def ff_block(x):
            return self.linear2(self.dropout(F.relu(self.linear1(x))))

        B = B + self.dropout2(ff_block(self.norm2(B)))
        C = C + self.dropout2(ff_block(self.norm2(C)))

        ForwardMetaContext.set('kl_loss', total_kl_loss / max(1, self.num_align_steps))
        ForwardMetaContext.set('attn_weights', cross_attn_weights)

        return A, B, C


# ==========================================
# 3. BATCH GENERATION (FULL DOMAIN)
# ==========================================
def generate_full_domain_batch(batch_size, total_len=100, sep=50):
    # Normalized time 0 to 1, scaled to 2*PI
    t_base = torch.linspace(0, 1, total_len).unsqueeze(0).repeat(batch_size, 1)
    t_C = t_base * 2 * math.pi

    # B is observed over the FULL domain, but warped via x^k
    k = torch.empty(batch_size, 1).uniform_(0.5, 2.0)
    t_B = t_base.pow(k) * 2 * math.pi

    freq = torch.randint(1, 4, (batch_size, 1)).float()
    C_val_true = torch.sin(freq * t_C)
    B_val_true = torch.sin(freq * t_B)

    # C is constrained to the subspace (Blind spot after sep)
    C_val_in = C_val_true.clone()
    C_val_in[:, sep:] = 0.0

    return t_C, C_val_in, t_B, B_val_true, C_val_true, k


import math
import torch
import torch.nn as nn


class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        # Create a spread of frequencies.
        # You can tune the max frequency depending on your signal.
        freq_bands = torch.linspace(1.0, 10.0, dim // 2)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        # x is shape [Batch, Seq]
        # Expand to [Batch, Seq, Dim//2]
        x_expanded = x.unsqueeze(-1) * self.freq_bands * math.pi

        # Calculate sin and cos: Shape [Batch, Seq, Dim//2]
        sin_features = torch.sin(x_expanded)
        cos_features = torch.cos(x_expanded)

        # Concat to get [Batch, Seq, Dim]
        return torch.cat([sin_features, cos_features], dim=-1)


# ==========================================
# 4. WRAPPER & TRAINING LOOP
# ==========================================
# However, you do need to encode the continuous variable $x$. You cannot just pass raw $x$ as a
# linear scalar. Here are the two mathematical reasons why.1. The Dot-Product Failure (Translation
# Invariance)Cross-Attention calculates the similarity between a Query and a Key using a dot
# product: $Score = Q \cdot K^T$.If your feature is largely defined by a raw scalar $x$, look
# at how the dot product behaves for tokens that are exactly $0.1$ units apart:Attention between
# $x=0.1$ and $x=0.2$: 0.1 * 0.2 = 0.02Attention between $x=0.9$ and $x=1.0$: 0.9 * 1.0 = 0.90
# Even though the physical distance is identical ($\Delta x = 0.1$), the attention scores are
# drastically different simply because the absolute magnitude of $x$ is larger.By passing $x$
# through a combination of sines and cosines (Fourier Features), the dot product in the latent
# space becomes proportional to the relative distance between the coordinates, regardless of where
# they are on the axis. This guarantees translation invariance.2. Spectral Bias (The NeRF Discovery)
# In 2020, researchers working on Neural Radiance Fields (NeRFs) proved that standard neural
# networks suffer from "Spectral Bias." This means that if you feed a network a raw,
# low-dimensional coordinate like $(x, y)$, the network is mathematically biased toward learning
# smooth, low-frequency functions.It becomes nearly impossible for the network to reconstruct
# high-frequency data (like your rapidly oscillating sin(6 * pi * x) target) from a raw linear
# $x$ input.By encoding $x$ into multiple frequencies (e.g., $\sin(x), \cos(x), \sin(2x), \cos(2x)$),
# you give the network an orthogonal basis to easily reconstruct complex high-frequency signals.
# class ModelWrapper(nn.Module):
#     def __init__(self, dim=32):
#         super().__init__()
#         self.dim = dim
#
#         # 1. Lift the scalar signal value into the latent space
#         self.val_lift = nn.Linear(1, dim)
#
#         # 2. Encode the spatial coordinate into the latent space (uses full even dim)
#         self.pos_encoder = ContinuousPositionalEncoding(dim)
#
#         self.layer = TriStreamLayerFixed(d_model=dim)
#         self.proj = nn.Linear(dim, 1)
#
#     def forward(self, t_C, C_val, t_B, B_val, sep):
#         # A. Lift values (Shape: [Batch, Seq, Dim])
#         C_val_feat = self.val_lift(C_val.unsqueeze(-1))
#         B_val_feat = self.val_lift(B_val.unsqueeze(-1))
#
#         # B. Lift positions (Shape: [Batch, Seq, Dim])
#         C_pos_feat = self.pos_encoder(t_C)
#         B_pos_feat = self.pos_encoder(t_B)
#
#         # C. Add Token + Position (Standard Transformer Architecture)
#         C_feat = (C_val_feat + C_pos_feat).transpose(0, 1)
#         B_feat = (B_val_feat + B_pos_feat).transpose(0, 1)
#
#         # D. Forward Pass
#         _, _, C_out_f = self.layer(C_feat, B_feat, C_feat, sep)
#         return self.proj(C_out_f.transpose(0, 1)).squeeze(-1)


# For xdim >= 1 -------------------------------------
import torch
import torch.nn as nn
import math


class NDContinuousPositionalEncoding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, scale: float = 10.0):
        """
        Args:
            input_dim: The number of spatial dimensions of x (e.g., 2 for (x,y))
            embed_dim: The target latent dimension (your d_model, e.g., 32)
            scale: Controls the frequency bands. Higher = more sensitive to tiny local shifts.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even."

        # We create a random projection matrix B of shape [input_dim, embed_dim / 2]
        # We register it as a buffer so it is saved in the state_dict but NOT updated by backprop.
        B = torch.randn(input_dim, embed_dim // 2) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [Batch, Seq, input_dim]
        Returns:
            Tensor of shape [Batch, Seq, embed_dim]
        """
        # 1. Project the coordinates into frequency space
        # [Batch, Seq, input_dim] @ [input_dim, embed_dim // 2] -> [Batch, Seq, embed_dim // 2]
        x_proj = (2 * math.pi * x) @ self.B

        # 2. Apply sine and cosine
        sin_features = torch.sin(x_proj)
        cos_features = torch.cos(x_proj)

        # 3. Concatenate to get exactly embed_dim
        # Shape: [Batch, Seq, embed_dim]
        return torch.cat([sin_features, cos_features], dim=-1)


class ModelWrapper(nn.Module):
    def __init__(self, input_dim=3, dim=32):
        super().__init__()
        self.dim = dim

        # The values might still just be a 1D scalar (e.g., temperature at that 3D point)
        self.val_lift = nn.Linear(1, dim)

        # Now we tell the encoder to expect input_dim (e.g., 3) and output dim (e.g., 32)
        self.pos_encoder = NDContinuousPositionalEncoding(input_dim=input_dim, embed_dim=dim, scale=5.0)

        self.layer = TriStreamLayerFixed(d_model=dim)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x_C, C_val, x_B, B_val, sep):
        # x_C and x_B now have shape [Batch, Seq, input_dim]

        # A. Lift values [Batch, Seq, Dim]
        C_val_feat = self.val_lift(C_val.unsqueeze(-1))
        B_val_feat = self.val_lift(B_val.unsqueeze(-1))

        # B. Lift positions [Batch, Seq, Dim]
        C_pos_feat = self.pos_encoder(x_C)
        B_pos_feat = self.pos_encoder(x_B)

        # C. Add together
        C_feat = (C_val_feat + C_pos_feat).transpose(0, 1)
        B_feat = (B_val_feat + B_pos_feat).transpose(0, 1)

        # D. Forward Pass
        _, _, C_out_f = self.layer(C_feat, B_feat, C_feat, sep)
        return self.proj(C_out_f.transpose(0, 1)).squeeze(-1)


def train_and_visualize():
    total_len, sep = 100, 50
    model = ModelWrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Training Full-Domain Extrapolation...")
    for epoch in range(100000):
        model.train()
        optimizer.zero_grad()
        ForwardMetaContext.clear()

        t_C, C_in, t_B, B_in, C_true, _ = generate_full_domain_batch(32, total_len, sep)
        C_pred = model(t_C, C_in, t_B, B_in, sep)

        # recon_loss = F.mse_loss(C_pred[:, sep:], C_true[:, sep:])
        # Evaluate both the context preservation AND the extrapolation
        recon_loss = F.mse_loss(C_pred, C_true)
        loss = recon_loss + 0.005 * ForwardMetaContext.get('kl_loss')

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch + 1}, Recon Loss {recon_loss.item():.4f}, kl_loss: {ForwardMetaContext.get('kl_loss').item():.4f}")

        if (epoch + 1) % 5000 == 0:
            # --- VISUALIZATION ---
            model.eval()
            t_C, C_in, t_B, B_in, C_true, k = generate_full_domain_batch(1, total_len, sep)
            with torch.no_grad():
                C_pred = model(t_C, C_in, t_B, B_in, sep)

            attn = ForwardMetaContext.get('attn_weights')[0].numpy()

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax = axes[0]
            ax.plot(t_C[0].numpy(), C_true[0].numpy(), label='True Signal', color='black', alpha=0.3, lw=4)
            ax.plot(t_C[0].numpy(), B_in[0].numpy(), label=f'B (Full Domain, k={k.item():.2f})', color='red', ls='--')
            ax.plot(t_C[0].numpy(), C_pred[0].numpy(), label='C Predicted', color='blue')
            ax.axvspan(t_C[0][sep].item(), t_C[0][-1].item(), color='gray', alpha=0.1, label='C Test (Blind Spot)')
            ax.legend();
            ax.set_title("Full Domain Extrapolation via Parametric Unwarping")

            ax2 = axes[1]
            im = ax2.imshow(attn, aspect='auto', cmap='viridis', origin='upper')
            ax2.axvline(total_len - 0.5, color='red', linestyle='--', label='Sink Token')
            ax2.axhline(sep, color='white', linestyle='--', label='Train/Test Split')
            ax2.set_title("Cross-Attention Matrix")
            ax2.set_xlabel("Keys (Aligned B Index + Sink)")
            ax2.set_ylabel("Queries (C Index)")
            ax2.legend()
            plt.show()

            model.train()


train_and_visualize()
