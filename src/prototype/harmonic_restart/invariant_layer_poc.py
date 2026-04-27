"""
This is a small experiment to showcase, that when we in prior know where A lives in B domain and vice versa,
then we can learn invariant rerpesentations wrt the transform (and its inverse) -- making us learn the diffeomorphism in context

potential issues:
* CE might penalize adjacent points to harshly, because it does not conisder that the classes are dependent.
* if Q or K don't have a permutation matrix (and the loss accordingly, then we might just shortwire by learning a positional
sequence encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, clear_output  # Useful if running in Jupyter/Colab



class SymmetryInContextModel(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, nhead=8, num_layers=4):
        super().__init__()
        # Encoder for (X, Y) pairs
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Self-Attention layers for invariant representation
        self.self_attn_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

        # Cross-Attention layer for the operator/alignment
        # Note: We'll use the raw MultiheadAttention to extract the weights easily
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

    def get_representation(self, x, y, mask=None):
        """Processes point clouds into latent representations."""
        # Stack (X, Y) to form point cloud features: [B, N, 2]
        # Transpose from dataset [N, B] -> [B, N]
        pts = torch.stack([x.T, y.T], dim=-1)

        # Replace NaNs (from padding) with 0 for the linear layer
        pts_clean = torch.nan_to_num(pts, nan=0.0)

        h = self.encoder(pts_clean)
        h_latent = self.self_attn_block(h, src_key_padding_mask=mask)
        return h_latent

    def forward(self, batch):
        train_data = batch['train']

        X_A, Y_A = train_data['X_A'], train_data['Y_A']
        X_A_in_B, Y_A_in_B = train_data['X_A_in_B'], train_data['Y_A_in_B']
        X_B_in_A, Y_B_in_A = train_data['X_B_in_A'], train_data['Y_B_in_A']
        X_B, Y_B = train_data['X_B'], train_data['Y_B']

        X_all = torch.cat([X_A, X_A_in_B, X_B_in_A, X_B], dim=1)
        Y_all = torch.cat([Y_A, Y_A_in_B, Y_B_in_A, Y_B], dim=1)

        mask_A = torch.isnan(X_A.T)
        mask_B = torch.zeros_like(X_B.T, dtype=torch.bool)
        mask_all = torch.cat([mask_A, mask_A, mask_B, mask_B], dim=0)

        h_all = self.get_representation(X_all, Y_all, mask=mask_all)
        h_A, h_A_in_B, h_B_in_A, h_B = torch.chunk(h_all, chunks=4, dim=0)

        Q = torch.cat([h_A, h_B_in_A], dim=1)
        K = torch.cat([h_A_in_B, h_B], dim=1)

        # --- THE FIX: Create and pass the Key Padding Mask ---
        # K is composed of [h_A_in_B, h_B].
        # We concatenate their respective masks to match the shape of K: [Batch, 100]
        cross_key_mask = torch.cat([mask_A, mask_B], dim=1)

        # Pass it to the MultiheadAttention
        _, attn_weights = self.cross_attn(Q, K, K, key_padding_mask=cross_key_mask)

        return h_A, h_A_in_B, h_B, h_B_in_A, attn_weights, mask_A

# def compute_loss(h_A, h_A_in_B, h_B, h_B_in_A, attn_weights, mask_A, lambda_inv=1.0):
#     # 1. Invariance Loss (Now strictly symmetric!)
#     valid_mask_A = ~mask_A
#
#     # MSE for valid A points
#     loss_inv_A = F.mse_loss(h_A[valid_mask_A], h_A_in_B[valid_mask_A])
#     # MSE for all B points (since B has no padding)
#     loss_inv_B = F.mse_loss(h_B, h_B_in_A)
#
#     loss_inv = loss_inv_A + loss_inv_B
#
#     # 2. Attention Loss (Cross Entropy on the diagonal)
#     diagonal_probs = torch.diagonal(attn_weights, dim1=-2, dim2=-1)
#
#     # Mask for the concatenated sequence [A, B_in_A]
#     batch_size = mask_A.shape[0]
#     seq_len_B = h_B.shape[1]
#     valid_mask_B = torch.ones((batch_size, seq_len_B), dtype=torch.bool, device=mask_A.device)
#
#     # Combine masks for the 100-length sequence
#     full_valid_mask = torch.cat([valid_mask_A, valid_mask_B], dim=1)
#
#     # Extract probabilities only for real, non-padded points
#     valid_probs = diagonal_probs[full_valid_mask]
#
#     # Negative log-likelihood
#     loss_attn = -torch.log(valid_probs + 1e-9).mean()
#
#     return loss_attn, loss_inv  # Returning separately so we can log them easily!


def compute_loss(h_A, h_A_in_B, h_B, h_B_in_A, attn_weights, mask_A, lambda_inv=1.0, lambda_col=1.0):
    """
        Computes the Symmetry-Aware Invariance and Optimal Transport Attention loss
        for Point Cloud Registration.

        This function enforces two geometric priors:
        1. Latent representations must be invariant to the symmetry group transformations.
        2. The cross-attention mechanism must learn a strict bijective (1-to-1) correspondence
           between domains, approximating an Optimal Transport plan.

        Theoretical Underpinnings & Buzzword Glossary:
        ---------------------------------------------
        * Invariant Representation (Latent Consistency): By penalizing the MSE between h_A
          and h_A_in_B, we force the self-attention encoder to project points onto a quotient
          space M/G (where G is the symmetry group). The features learn to ignore the "pose"
          or "distortion" and capture only intrinsic geometry.

        * Optimal Transport (OT): The mathematical framework of finding the most efficient
          way to move mass from one distribution (Cloud A) to another (Cloud B). Here, our
          "transport plan" is the Attention Matrix.

        * Bijection (1-to-1 Mapping): Because the physical distortions (affine/elastic) are
          invertible, a point in A maps to exactly one point in B, and vice-versa.

        * Doubly Stochastic Matrix (Permutation Matrix): A matrix where both rows and columns
          sum to 1.0. A strict 1-to-1 mapping requires the attention matrix to be a permutation
          matrix. Standard Softmax only guarantees rows sum to 1.0.

        * Sinkhorn Approximation: A classic OT algorithm that iteratively normalizes rows and
          columns to force a matrix to become Doubly Stochastic. By manually normalizing the
          columns and applying Dual CE, we emulate a Sinkhorn regularization step.

        * Dual Cross-Entropy (Symmetric Attention): We treat alignment as a classification task.
          - Row-wise CE (Forward Mapping): "Given point i in A, find its exact match in B."
          - Column-wise CE (Inverse Mapping): "Given point j in B, ensure it was exclusively
            claimed by point j in A."
          Together, they severely penalize "many-to-one" mode collapse.

        Args:
            h_A (torch.Tensor): Latent representation of point cloud A. Shape: [Batch, Seq_A, Dim].
            h_A_in_B (torch.Tensor): Latent representation of A warped into B's domain.
            h_B (torch.Tensor): Latent representation of point cloud B. Shape: [Batch, Seq_B, Dim].
            h_B_in_A (torch.Tensor): Latent representation of B warped into A's domain.
            attn_weights (torch.Tensor): Raw post-softmax attention weights from Cross-Attention.
                                         Shape: [Batch, Seq_A + Seq_B, Seq_A + Seq_B].
            mask_A (torch.Tensor): Boolean mask where True indicates padded (NaN) points in A.
                                   Shape: [Batch, Seq_A].
            lambda_inv (float): Weighting coefficient for the geometric invariance MSE loss.
            lambda_col (float): Weighting coefficient for the inverse mapping (column) CE loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - loss_attn: The scalar Dual Cross-Entropy loss enforcing the 1-to-1 alignment.
                - loss_inv: The scalar MSE loss enforcing symmetry-invariant feature extraction.
        """
    # 1. Invariance Loss (Symmetric)
    valid_mask_A = ~mask_A
    loss_inv_A = F.mse_loss(h_A[valid_mask_A], h_A_in_B[valid_mask_A])
    loss_inv_B = F.mse_loss(h_B, h_B_in_A)
    loss_inv = loss_inv_A + loss_inv_B

    # --- THE SUPER-MASK ---
    # Because Q = [A, B_in_A] and K = [A_in_B, B]
    # They actually share the exact same padding structure!
    batch_size = mask_A.shape[0]
    seq_len_B = h_B.shape[1]
    valid_mask_B = torch.ones((batch_size, seq_len_B), dtype=torch.bool, device=mask_A.device)

    # This mask works for both Rows (Queries) and Columns (Keys)
    full_valid_mask = torch.cat([valid_mask_A, valid_mask_B], dim=1)

    # 2. ROW-WISE Attention Loss (Forward Mapping)
    # attn_weights rows already sum to 1.0 via built-in Softmax
    row_diagonal = torch.diagonal(attn_weights, dim1=-2, dim2=-1)
    valid_row_probs = row_diagonal[full_valid_mask]
    loss_row_ce = -torch.log(valid_row_probs + 1e-9).mean()

    # 3. COLUMN-WISE Attention Loss (Inverse Mapping)
    # Normalize the columns so they sum to 1.0 (creating a valid probability distribution)
    # We sum across the Query dimension (dim=-2)
    col_probs = attn_weights / (attn_weights.sum(dim=-2, keepdim=True) + 1e-9)
    col_diagonal = torch.diagonal(col_probs, dim1=-2, dim2=-1)

    valid_col_probs = col_diagonal[full_valid_mask]
    loss_col_ce = -torch.log(valid_col_probs + 1e-9).mean()

    # Total Attention Loss is the sum of both directions
    loss_attn = loss_row_ce + (lambda_col * loss_col_ce)

    return loss_attn, loss_inv



# --- 1. Helper function to move the complex nested dict to GPU ---
def dict_to_device(d, device):
    """Recursively moves tensors in a dictionary to the specified device."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


# --- 2. Live Plotting Hook ---
def plot_progress(step, losses, attn_matrix, n_B, path=None):
    """Plots the training loss and the Attention Matrix vs Truth."""
    clear_output(wait=True)  # Clears previous plots in notebooks
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Loss Curve
    axes[0].plot(losses, label="Total Loss", color="blue", alpha=0.8)
    axes[0].set_title(f"Training Loss (Step {step})")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale('log')  # Log scale is often better for MSE
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Predicted Attention Matrix (Single Sample from Batch)
    # attn_matrix shape is [Batch, SeqLen, SeqLen]
    # We slice out the first sample in the batch
    pred_attn = attn_matrix[0].detach().cpu().numpy()
    im1 = axes[1].imshow(pred_attn, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Predicted Attention (Q vs K)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Target Identity Matrix
    seq_len = pred_attn.shape[0]
    target_attn = torch.eye(seq_len).numpy()
    im2 = axes[2].imshow(target_attn, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title("Target Truth (Identity)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Visual markers for the [A, B_in_A] concatenation boundaries
    for ax in [axes[1], axes[2]]:
        ax.axhline(y=n_B, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=n_B, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(path + f'/attn_step_{step}.png')
    plt.close(fig)  # Close the figure to free memory


# --- 3. The Main Training Loop ---
def train_model(model, dataset, num_steps=100000, plot_every=1000, device='cuda', path=None):
    model.to(device)

    # Transformer Best Practice: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Transformer Best Practice: OneCycleLR (handles warmup and cosine decay naturally)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=num_steps,
        pct_start=0.1  # 10% of training spent warming up
    )

    # Instantiate the data generator
    data_iterator = iter(dataset)

    loss_history = []

    # Use tqdm for the progress bar
    pbar = tqdm(range(num_steps), desc="Training Model")

    model.train()
    for step in pbar:
        # 1. Get batch and move to device
        batch = next(data_iterator)
        batch = dict_to_device(batch, device)
        train_data = batch['train']

        # 2. Forward Pass
        optimizer.zero_grad()
        h_A, h_A_in_B, h_B, h_B_in_A, attn_weights, mask_A = model(batch)

        # 3. Calculate Loss using our dedicated function
        loss_attn, loss_inv = compute_loss(h_A, h_A_in_B, h_B, h_B_in_A, attn_weights, mask_A)

        lambda_inv = 1.0
        total_loss = loss_attn + (lambda_inv * loss_inv)

        # 4. Backward Pass & Optimize
        total_loss.backward()

        # Transformer Best Practice: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # 5. Logging and Plotting
        loss_history.append(total_loss.item())
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'attn_loss': f"{loss_attn.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

        if (step + 1) % plot_every == 0:
            # Need n_B to draw the visual separator line in the plot
            n_B = dataset.n_B
            model.eval()  # Switch to eval for clean plotting
            with torch.no_grad():
                plot_progress(step + 1, loss_history, attn_weights, n_B, path)
            model.train()  # Back to train mode

    print("Training Complete!")
    return loss_history


# --- 4. Execution ---
if __name__ == '__main__':
    from prototype.harmonic_restart.harmonic_prior import GlobalSparseHarmonicsStream

    # Make sure to import SymmetryInContextModel from wherever you saved it!
    # from model_file import SymmetryInContextModel

    # 1. Setup the Prior
    problem = GlobalSparseHarmonicsStream(
        batch_size=1024, n_A=50, n_B=50, n_test=200, x_range=(-5, 5),
        num_components=4, noise_std=0.05, share_unrelated=0.0,
        scale=False, shift=True, warp=True
    )

    # 2. Initialize Model
    # Since inputs are [x, y], input_dim is 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SymmetryInContextModel(input_dim=2, embed_dim=128, nhead=8, num_layers=4)

    # 3. Train the beast
    losses = train_model(
        model=model,
        dataset=problem,
        num_steps=100000,  # Adjust based on how fast it converges
        plot_every=1000,  # Plot every 500 steps
        device=device,
        path = '/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/harmonics-warp'
    )