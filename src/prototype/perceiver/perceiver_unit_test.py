import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from tabpfn.architectures.tabpfn_v2_5 import LowerPrecisionLayerNorm, _batched_scaled_dot_product_attention, Attention
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------
# 1. FIXED PERCEIVER MODULE (Standard Transformer Routing)
# ---------------------------------------------------------
class LowerPrecisionLayerNorm(nn.LayerNorm):
    pass  # Assuming this is just a standard LayerNorm in your codebase


class PerceiverDomainTransfer(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        # We focus on cross-feature attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, A_support, B_query):
        # A_support: (Batch, 4, Embed) -> The 4 points of the cube
        # B_query: (Batch, N, Embed)   -> The full cube or new points

        # Cross-Attention: Query B, Key/Value A
        # This asks: "How do I map my point B using the 4 reference points in A?"
        attn_out, _ = self.cross_attn(query=B_query, key=A_support, value=A_support)

        # Residual
        return self.ln(B_query + attn_out)
    # def __init__(self, embedding_size: int, num_heads: int, num_features: int, num_latents: int = 4):
    #     super().__init__()
    #     self.num_latents = num_latents
    #     self.CE_dim = num_features * embedding_size
    #
    #     self.latent_generator = nn.Linear(self.CE_dim, num_latents * self.CE_dim)
    #
    #     # Standard MHA (using PyTorch native for brevity in the sandbox)
    #     self.bottleneck_attn = nn.MultiheadAttention(embed_dim=self.CE_dim, num_heads=num_heads, batch_first=True)
    #     self.translate_attn = nn.MultiheadAttention(embed_dim=self.CE_dim, num_heads=num_heads, batch_first=True)
    #
    #     self.ln_A = nn.LayerNorm(self.CE_dim)
    #     self.ln_B1 = nn.LayerNorm(self.CE_dim)
    #     self.ln_B2 = nn.LayerNorm(self.CE_dim)  # NEW: LN for the MLP
    #
    #     # The MLP needs to be slightly wider to compute affine cross-terms
    #     self.mlp = nn.Sequential(
    #         nn.Linear(self.CE_dim, self.CE_dim * 2),
    #         nn.GELU(),
    #         nn.Linear(self.CE_dim * 2, self.CE_dim)
    #     )
    #
    # def encode_target(self, A_train_BRCE: torch.Tensor) -> torch.Tensor:
    #     B_batch, R_A, C, E = A_train_BRCE.shape
    #     A_flat = A_train_BRCE.reshape(B_batch, R_A, C * E)
    #     A_norm = self.ln_A(A_flat)
    #
    #     q_latents = self.latent_generator(A_norm.mean(dim=1)).view(B_batch, self.num_latents, self.CE_dim)
    #
    #     # Latents attend to A
    #     latents_out, _ = self.bottleneck_attn(query=q_latents, key=A_norm, value=A_norm)
    #     return latents_out
    #
    # def translate_source(self, B_train_BRCE: torch.Tensor, A_latents: torch.Tensor) -> torch.Tensor:
    #     B_batch, R_B, C, E = B_train_BRCE.shape
    #     B_flat = B_train_BRCE.reshape(B_batch, R_B, C * E)
    #
    #     # 1. Cross Attention
    #     B_norm = self.ln_B1(B_flat)
    #     attn_out, _ = self.translate_attn(query=B_norm, key=A_latents, value=A_latents)
    #
    #     # 2. First Residual: Mix B with A's context
    #     B_mixed = B_flat + attn_out
    #
    #     # 3. Second Residual + MLP: Compute the geometric transformation
    #     B_out = B_mixed + self.mlp(self.ln_B2(B_mixed))
    #
    #     return B_out.view(B_batch, R_B, C, E)
    #
    # def forward(self, A_train_BRCE: torch.Tensor, B_train_BRCE: torch.Tensor) -> torch.Tensor:
    #     latents = self.encode_target(A_train_BRCE)
    #     return self.translate_source(B_train_BRCE, latents)


# ---------------------------------------------------------
# 2. FIXED DATASET: Asymmetric Canonical Shape
# ---------------------------------------------------------
# We generate ONE random asymmetric point cloud to act as our "canonical" shape
# This gives the model recognizable landmarks to track rotation.
NUM_DENSE = 50
CANONICAL_B = torch.randn(NUM_DENSE, 2)

def generate_few_shot_cube_data(batch_size):
    # 1. Canonical Unit Cube (8 vertices)
    cube = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=torch.float32)
    # B_train_input is always the canonical, un-transformed shape
    B_train_input = cube.unsqueeze(0).repeat(batch_size, 1, 1)

    # 2. Sample Random Affine Transform
    R = torch.randn(batch_size, 3, 3) * 0.5
    T = torch.rand(batch_size, 1, 3) * 4 - 2

    # Ground Truth: The target shape we want B to become
    B_gt_transformed = torch.bmm(B_train_input, R) + T

    # 3. Create A_train: The 4-point basis landmarks from the TARGET domain
    basis_indices = [0, 1, 2, 3]
    A_train = B_gt_transformed[:, basis_indices, :]

    # Return: (Support, Query_Input, Target)
    return A_train, B_train_input, B_gt_transformed

# ---------------------------------------------------------
# 3. UNIT TESTER HARNESS
# ---------------------------------------------------------
class PerceiverUnitTester(nn.Module):
    def __init__(self, embedding_size=32):
        super().__init__()
        # We process each coordinate (1 feature) into embedding_size
        self.embedder = nn.Linear(1, embedding_size)

        # New embed_dim = Features * Embedding = 3 * 32 = 96
        self.perceiver = PerceiverDomainTransfer(embed_dim=3 * embedding_size)

        self.decoder = nn.Linear(embedding_size, 1)

    def forward(self, A_BRC, B_BRC):
        # A_BRC shape: (Batch, R, C=3)
        # 1. Embed individual coords: (Batch, R, C, E=32)
        A_emb = self.embedder(A_BRC.unsqueeze(-1))
        B_emb = self.embedder(B_BRC.unsqueeze(-1))

        # 2. Flatten Features into Embedding: (Batch, R, C*E=96)
        A_flat = A_emb.reshape(A_emb.shape[0], A_emb.shape[1], -1)
        B_flat = B_emb.reshape(B_emb.shape[0], B_emb.shape[1], -1)

        # 3. Perceiver now receives 3-D tensors (Batch, Sequence, Embed)
        B_translated_flat = self.perceiver(A_flat, B_flat)

        # 4. Reshape back to (Batch, R, C, E) for decoding
        B_translated_emb = B_translated_flat.view(B_flat.shape[0], B_flat.shape[1], 3, -1)

        return self.decoder(B_translated_emb).squeeze(-1)


model = PerceiverUnitTester()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100)

pbar = tqdm(range(10000))
for step in pbar:
    # Generate batch
    A_train, B_transformed, B_gt = generate_few_shot_cube_data(batch_size=64)

    # Forward
    B_pred = model(A_train, B_transformed)  # Or your updated Feature-wise model
    loss = F.mse_loss(B_pred, B_gt)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping is vital for geometric transforms
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Scheduler
    scheduler.step(loss)

    if step % 10 == 0:
        pbar.set_description(f"Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

import plotly.graph_objects as go


def plot_interactive_3d_alignment(A_train, B_gt, B_pred):
    # Move to CPU and convert to numpy
    A = A_train.detach().cpu().numpy()[0]
    B_target = B_gt.detach().cpu().numpy()[0]
    B_output = B_pred.detach().cpu().numpy()[0]

    fig = go.Figure()

    # 1. Target (Green)
    fig.add_trace(go.Scatter3d(x=B_target[:, 0], y=B_target[:, 1], z=B_target[:, 2],
                               mode='markers', name='Target (B_gt)',
                               marker=dict(size=5, color='green', opacity=0.3)))

    # 2. Support Set (Red X's)
    fig.add_trace(go.Scatter3d(x=A[:, 0], y=A[:, 1], z=A[:, 2],
                               mode='markers', name='Support Set (A)',
                               marker=dict(size=10, color='red', symbol='x')))

    # 3. Prediction (Blue)
    fig.add_trace(go.Scatter3d(x=B_output[:, 0], y=B_output[:, 1], z=B_output[:, 2],
                               mode='markers', name='Prediction (B_pred)',
                               marker=dict(size=5, color='blue', opacity=0.7)))

    fig.update_layout(
        title="Interactive 3D Affine Coordinate Alignment",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'  # Correct property name
        )
    )
    fig.show()


# Call the interactive plot
A, B_transformed, B_gt = generate_few_shot_cube_data(batch_size=1)
B_pred = model(A, B_transformed)
plot_interactive_3d_alignment(A, B_gt, B_pred)