import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import IterableDataset


from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream
# =============================================================================
# 2. TabPFN Attention & OT Modules (Unchanged mechanics, just condensed)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabpfn.architectures.tabpfn_v2_5 import TabPFNBlock


# ---------------------------------------------------------
# Placeholder for your actual TabPFN v2.5 import
# from src.tabpfn.architectures.tabpfn_v2_5 import TabPFNLayer
# ---------------------------------------------------------

def compute_gram(A):
    """
    Computes the Gram matrix strictly on the data tokens (excluding Thinking Rows).
    A expected shape: (Batch, Rows, Columns, Features) -> (B, R, C, D)
    """
    B, R, C, D = A.shape
    # Flatten Rows and Columns to compute the global feature covariance
    # Shape: (B, R*C, D)
    A_flat = A.view(B, R * C, D)

    # Gram matrix: (B, D, D). Normalize by sequence length.
    Gram = torch.bmm(A_flat.transpose(1, 2), A_flat) / (R * C)
    return Gram


class InContextDomainAutoencoder(nn.Module):
    def __init__(self, d_in,  d_model, n_heads=4, num_layers_enc=3, num_layers_dec=3, num_thinking_rows=10):
        super().__init__()
        self.num_thinking_rows = num_thinking_rows

        # 1. Learnable Thinking Rows (Prototypes)
        # We initialize separate thinking rows for the two domains.
        # Shape: (1, R_T, C, D)
        self.T_A_init = nn.Parameter(torch.randn(1, num_thinking_rows, 1, d_model))

        self.in_linear = nn.Linear(d_in, d_model)
        # 2. Encoder (Whitening / Disentangling)
        self.encoder = nn.ModuleList([
            TabPFNBlock(emsize=d_model, nhead=n_heads, dim_feedforward=64) for _ in range(num_layers_enc)
        ])

        # 3. Decoder (Recoloring / Style Transfer)
        self.decoder = nn.ModuleList([
            TabPFNBlock(emsize=d_model, nhead=n_heads, dim_feedforward=64) for _ in range(num_layers_dec)
        ])
        self.out_linear = nn.Linear(d_model, d_in)

    def encode(self, A_0, T_init):
        """
        Passes A_0 and T through the Encoder.
        A_0 shape: (B, R_A, C, D)
        """
        B, R_A, C, D = A_0.shape

        # Expand Thinking rows to match Batch and Columns
        # T shape: (B, R_T, C, D)
        T = T_init.expand(B, -1, C, -1)

        # Concatenate along the Row dimension
        # Z_0 shape: (B, R_T + R_A, C, D)
        Z = torch.cat([T, A_0], dim=1)

        # Pass through Encoder
        for layer in self.encoder:
            Z = layer(Z)

        # Split back into Thinking Rows and Data
        T_k = Z[:, :self.num_thinking_rows, :, :]
        A_k = Z[:, self.num_thinking_rows:, :, :]

        return T_k, A_k

    def decode(self, A_k, T_k_style):
        """
        Onloads a specific style T_k onto canonical data A_k.
        """
        # Concatenate canonical data with the target style
        Z = torch.cat([T_k_style, A_k], dim=1)

        # Pass through Decoder
        for layer in self.decoder:
            Z = layer(Z)

        # Return only the reconstructed/transported data
        A_2k = Z[:, self.num_thinking_rows:, :, :]
        return A_2k

    def forward(self, A_0, AinB_0):
        # ==========================================================
        # PHASE 1: ENCODE & WHITEN (A^0 -> A^k)
        # ==========================================================
        A_0 = self.in_linear(A_0)
        AinB_0 = self.in_linear(AinB_0)

        T_A_k, A_k = self.encode(A_0, self.T_A_init)
        T_AinB_k, AinB_k = self.encode(AinB_0, self.T_A_init)

        if self.training:
            # Bottleneck Gram Loss: A_k and AinB_k must be structurally identical.
            # This forces the domain differences entirely into T_A_k and T_AinB_k.
            gram_A_k = compute_gram(A_k)
            gram_AinB_k = compute_gram(AinB_k)
            loss_bottleneck = F.mse_loss(gram_A_k, gram_AinB_k)
        else:
            loss_bottleneck = None

        # ==========================================================
        # PHASE 2: DECODE & RECONSTRUCT (Cycle Consistency)
        # Onload A's style back onto A's canonical data
        # ==========================================================
        A_recon_2k = self.decode(A_k, T_A_k)

        if self.training:
            # Gram reconstruction loss vs original A_0
            gram_A_0 = compute_gram(A_0)
            gram_A_recon = compute_gram(A_recon_2k)
            loss_cycle = F.mse_loss(gram_A_recon, gram_A_0)
        else:
            loss_cycle = None

        # ==========================================================
        # PHASE 3: DECODE & TRANSFER (Domain Transport)
        # Onload AinB's style onto A's canonical data
        # ==========================================================
        A_trans_2k = self.decode(A_k, T_AinB_k)

        if self.training:
            # Gram transfer loss vs original target AinB_0
            gram_AinB_0 = compute_gram(AinB_0)
            gram_A_trans = compute_gram(A_trans_2k)
            loss_transfer = F.mse_loss(gram_A_trans, gram_AinB_0)

        A_trans_2k = self.out_linear(A_trans_2k)
        if self.training:
            A_recon_2k = self.out_linear(A_recon_2k)


        # Return outputs and losses
        losses = {
            'bottleneck': loss_bottleneck,
            'cycle': loss_cycle,
            'transfer': loss_transfer
        }

        return A_trans_2k, A_recon_2k, losses
# =============================================================================
# 3. Upgraded Visualization (Plotting the Harmonic Functions)
# =============================================================================
def visualize_harmonic_transport(model, dataset):
    model.eval()
    with torch.no_grad():
        # Draw a fresh batch, but use the dense TEST set for smooth plotting
        batch = dataset._sample_batch()

        X_A = batch['test']['X_A']
        Y_A = batch['test']['Y_A']
        X_AinB = batch['test']['X_A_in_B']
        Y_AinB = batch['test']['Y_A_in_B']

        # Run model (A -> B)
        _, _, A_trans, A_recon = model(X_A, X_AinB)

        # Extract first batch item
        x_a = X_A[0, :, 0].numpy()
        y_a = Y_A[0, :, 0].numpy()
        x_gt = X_AinB[0, :, 0].numpy()
        y_gt = Y_AinB[0, :, 0].numpy()

        x_trans = A_trans[0, :, 0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: The Domain Shift ---
    ax1.plot(x_a, y_a, 'k-', linewidth=2, label="Task A (Canonical Function)")
    ax1.plot(x_gt, y_gt, 'b--', linewidth=2, label="Task A_in_B (Warped Function)")

    # Show how the coordinates shifted horizontally for a few points
    for i in range(0, len(x_a), 8):
        ax1.annotate("", xy=(x_gt[i], y_gt[i]), xytext=(x_a[i], y_a[i]),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    ax1.set_title("The Covariate Shift (Ground Truth)")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y (Harmonic output)")
    ax1.legend()

    # --- Plot 2: The Learned Transport ---
    ax2.plot(x_gt, y_gt, 'b--', linewidth=2, label="Target Warped Domain (B)", alpha=0.5)

    # We plot the TRANSPORTED X coordinates against the TRUE Y coordinates of the warped space.
    # If transport is perfect, x_trans will equal x_gt, and the orange dots will perfectly trace the blue line.
    ax2.scatter(x_trans, y_gt, color='orange', s=20, label="X_A transported to B", zorder=3)

    ax2.set_title("In-Context Optimal Transport Result")
    ax2.set_xlabel("Transported X coordinate")
    ax2.set_ylabel("Y (Harmonic output)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Training Loop
# =============================================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InContextDomainAutoencoder(d_in=1, d_model=64, n_heads=4, num_thinking_rows=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    prior = HarmonicMixturePrior(noise_std=0.02)
    # Note: n_A=n_B=64 for training to ensure balanced Sinkhorn OT
    dataset = InfiniteHarmonicsStream(prior, batch_size=32, n_A=64, n_B=64, n_test=128)

    epochs = 10000

    print("Starting Meta-Training on Harmonic Stream...")

    for epoch in range(epochs):
        model.train()
        batch = dataset._sample_batch()

        # We train the OT strictly on A -> A_in_B (The LUPI path)
        X_A = batch['train']['X_A'].to(device)
        X_AinB = batch['train']['X_A_in_B'].to(device)

        optimizer.zero_grad()
        Gram_A, Gram_B, A_trans, A_recon = model(X_A, X_AinB)

        # Losses
        loss_gram = F.mse_loss(Gram_A / X_A.size(1), Gram_B / X_AinB.size(1))
        loss_transport = F.mse_loss(A_trans, X_AinB)  # Point-to-point supervision
        loss_cycle = F.mse_loss(A_recon, X_A)

        # Anneal Gram weight
        gram_weight = 10.0 * (0.99 ** epoch)
        loss = loss_transport + loss_cycle + (gram_weight * loss_gram)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Total: {loss.item():.4f} | "
                  f"OT: {loss_transport.item():.4f} | Gram: {loss_gram.item():.4f}")

    print("\nTraining complete. Generating functional visualization...")
    visualize_harmonic_transport(model.cpu(), dataset)


if __name__ == "__main__":
    train()