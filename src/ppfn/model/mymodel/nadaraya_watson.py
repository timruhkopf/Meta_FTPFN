import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple



class NadarayaWatsonAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, nw_dropout=0.0, reuse_attn=True):
        """
        A 3-stream meta-learning adapter that performs Non-Parametric Local Smoothing
        to align and extract knowledge from related tasks within a frozen Prior-Data Fitted Network (PFN).

        This architecture addresses the risk of negative transfer by treating frozen latent
        representations as a topological space where local domain distortions can be measured
        and corrected before cross-task extraction occurs.

        The forward pass operates on three parallel streams packed into a single batch dimension:
        - Stream A (Target Task): Acts as the ground-truth contextual anchor (Frozen).
        - Stream B (Related Task): The source of external knowledge, which may be distorted (Frozen).
        - Stream C (Modulated Target): The active stream being updated for the next (frozen) PFN layer.

        The adapter operates in two main stages:
        1. The Corrector (Nadaraya-Watson): Calculates the local distortion error between
           Stream A's training points and Stream B's "belief" of those points. It applies this
           error to B's domain using an NW-style attention kernel to undistort B into A's manifold.
        2. The Extractor: Stream C queries the newly undistorted B representations to extract
           relevant cross-task features safely, updating itself via a residual connection.

        Args:
            d_model (int): The embedding dimension of the PFN latent space.
            n_heads (int): The number of heads for the Multi-Head Attention mechanisms.
            dropout (float, optional): Standard dropout applied to the Extractor's attention
                matrices and the FFN modulator. Defaults to 0.1.
            nw_dropout (float, optional): Dropout applied specifically to the Nadaraya-Watson
                Corrector attention. It is highly recommended to leave this at 0.0. The NW step
                relies on exact local structural anchors to compute the geometric translation between
                domains. Randomly dropping out attention weights here can randomly sever those anchors,
                resulting in chaotic domain shifts and poisoned representations. Defaults to 0.0.
            reuse_attn (bool, optional): If True, shares the exact same Multi-Head Attention
                weights between the `attn_train` (C_train querying B_prime) and `attn_test`
                (C_test querying B_prime) operations in the Extractor. This mirrors the architectural
                design of the original PFN, which reuses attention for train/test splits on a
                single item to maintain unified feature extraction logic and saves learnable parameters.
                Defaults to True.
        """
        super().__init__()
        self.d_model = d_model


        # 1. The Corrector (Undistorting B into A's domain)
        self.norm_nw_q = nn.LayerNorm(d_model)
        self.norm_nw_k = nn.LayerNorm(d_model)
        self.norm_nw_v = nn.LayerNorm(d_model)

        # MLP to project the local distortion error before applying it
        self.mlp_err = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.nw_attn = nn.MultiheadAttention(d_model, n_heads, dropout=nw_dropout)

        # 2. The Extractor (Cross-Task Modulation) - Split into Train and Test
        self.norm_train_q = nn.LayerNorm(d_model)
        self.norm_test_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)  # Keys and Values come from B_prime

        self.attn_train = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        if reuse_attn:
            self.attn_test = self.attn_train  # Share the exact same weights
        else:
            self.attn_test = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 3. Standard FFN for the C Stream (Best practice after attention)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def validate_forward_args(self, x, *args, **kwargs) -> Tuple[int, int]:
        single_eval_pos = kwargs.get("single_eval_pos",None)
        assert single_eval_pos is not None, "single_eval_pos must be provided"
        B = x.shape[1]
        assert B % 3 == 0, "Batch size must be multiple of 3"
        return B, single_eval_pos

    def forward(self, x, *args, **kwargs):

        B_dim, sep = self.validate_forward_args(x, *args, **kwargs)
        R = B_dim // 3

        # 1. Extract raw streams across the Batch dimension (index 1)
        A = x[:, :R, :].detach()
        B = x[:, R: 2 * R, :].detach()
        C = x[:, 2 * R:, :]
        device = A.device

        # --- Split Streams ---
        A_train, A_test = A[:sep], A[sep:]
        B_train, B_test, B_belief_A = B[:sep], B[sep:2 * sep], B[2 * sep:] # TODO : calculate nll of the projected belief based on A train's true y values (dim: -1)
        C_train, C_test, C_belief_A = C[:sep], C[sep:2 * sep], C[2 * sep:]

        # ==========================================
        # STAGE 1: THE NW error propagation
        # ==========================================
        # Calculate local distortion and project B in the domain of A (=B')

        # Q: B_train, K: A_train, V: Projected Error
        # We capture the attention weights to track the distortion mapping
        # pre attn mlp:
        #  the latent space is highly non-linear. Subtracting $B$ from $A$ yields a naive Euclidean distance vector.
        #  In a curved manifold, moving along this naive vector might push the representation off the valid data manifold entirely
        error = self.mlp_err(self.norm_nw_v(A_train - B_belief_A))
        B_active = B[:2 * sep]
        corr_out, corr_weights = self.nw_attn(
            self.norm_nw_q(B_active),  # Query with all active B points
            self.norm_nw_k(A_train),  # Anchored by Train
            error # on B's belief over A_train
        )

        # Undistort B's domain completely
        B_prime = B_active + corr_out

        # ==========================================
        # STAGE 2: THE EXTRACTOR
        # ==========================================
        # Keys and Values for both Extractor MHAs are the aligned B_prime_train

         # Reusing the normalized tensor for V
        k = self.norm_k(B_prime)
        v = k # reusing norm results

        # Train-to-Train Attention
        c_train_update, train_weights = self.attn_train(
            self.norm_train_q(C_train), k, v
        )

        # Test-to-Train Attention
        c_test_update, test_weights = self.attn_test(
            self.norm_test_q(C_test), k, v
        )

        # Recombine C stream over the sequence dimension (T, B, D)
        c_dummy_update = torch.zeros_like(C_belief_A).to(device)
        C_update = torch.cat([c_train_update, c_test_update, c_dummy_update], dim=0)

        # Residual connection for Stream C
        C = C + C_update

        # ==========================================
        # STAGE 3: FFN Modulator
        # ==========================================
        C = C + self.ffn(self.norm_ffn(C))

        # ==========================================
        # STAGE 4: Batch Tensor Reconstruction
        # ==========================================
        batch = torch.cat([A, B, C], dim=1)

        # Return the modulated C stream and all MHA weights for monitoring/gating
        return batch, {
            "corrector": corr_weights,
            "train_attn_scores": train_weights,
            "test_attn_scores": test_weights
        }

class DualSignalFusion(nn.Module):
    def __init__(self, init_gamma=0.1):
        super().__init__()
        # gamma scales the structural penalty (entropy).
        # We init small so empirical NLL dominates the routing early in training.
        self.gamma = nn.Parameter(torch.tensor(init_gamma))

        # Temperature for the gating softmax.
        self.inv_tau = nn.Parameter(torch.tensor(1.0))

    def compute_entropy(self, attn_weights):
        """
        Computes the Shannon Entropy for the Corrector attention.
        attn_weights shape: (N, B, Seq_Q, Seq_K) or (N, B, Heads, Seq_Q, Seq_K)
        """
        # Robustness: strict epsilon clamp to prevent log(0) NaNs
        eps = 1e-9
        attn_weights = torch.clamp(attn_weights, min=eps)

        # H = -sum(p * log(p)) over the Key sequence dimension (the last dim)
        entropy = -torch.sum(attn_weights * torch.log(attn_weights), dim=-1)

        # Average over whatever dimensions are left between Batch and the end
        # to get a single scalar per related task, per batch item.
        # Output shape: (N, B)
        while len(entropy.shape) > 2:
            entropy = entropy.mean(dim=-1)

        return entropy

    def forward(
            self,
            uncond_prob,  # (T_test, B, n_bins) - Probabilities, NOT logits
            related_probs,  # (N, T_test, B, n_bins)
            uncond_nll_train,  # (B,) - The empirical loss of the unconditional model on A_train
            related_nll_train,  # (N, B) - The empirical loss of each related task on A_train
            corrector_attns,  # (N, B, eval_pos, eval_pos) - The local support footprint
            y_true_test  # (T_test, B) - Ground truth bin indices for the test sequence
    ):
        N, T_test, B, n_bins = related_probs.shape

        # ==========================================
        # 1. Compute Dual-Signal Energy
        # ==========================================
        # Structural Uncertainty (Local Support)
        # Shape: (N, B)
        entropy_scores = self.compute_entropy(corrector_attns)

        # Robustness: Softplus ensures gamma remains non-negative (entropy should strictly penalize)
        safe_gamma = F.softplus(self.gamma)

        # Related Task Energy: E_r = L_r + gamma * S_r
        # Shape: (N, B)
        related_energy = related_nll_train + (safe_gamma * entropy_scores)

        # Unconditional Baseline Energy: E_base = L_base
        # Reshape to (1, B) so it can be concatenated
        base_energy = uncond_nll_train.unsqueeze(0)

        # Combine all energies: index 0 is baseline, 1:N are related tasks
        # Shape: (N + 1, B)
        all_energies = torch.cat([base_energy, related_energy], dim=0)

        # ==========================================
        # 2. Compute Gating Weights (Lambda)
        # ==========================================
        # Robustness: Enforce positive temperature with a minimum bound
        tau = F.softplus(self.inv_tau) + 1e-4

        # Softmax over the task dimension (dim=0)
        # Lower energy -> Higher weight
        # Shape: (N + 1, B)
        lambdas = F.softmax(-all_energies / tau, dim=0)

        # ==========================================
        # 3. Bayesian Model Averaging (Linear Pooling)
        # ==========================================
        # Reshape lambdas so they broadcast over the sequence and bin dimensions
        # Base weight: (1, 1, B, 1) -> broadcasts to (T_test, B, n_bins)
        lambda_base = lambdas[0].view(1, B, 1)

        # Related weights: (N, 1, B, 1) -> broadcasts to (N, T_test, B, n_bins)
        lambda_related = lambdas[1:].view(N, 1, B, 1)

        # Linearly pool the actual probabilities
        fused_prob = (lambda_base * uncond_prob) + torch.sum(lambda_related * related_probs, dim=0)

        # Robustness: Minor numerical float errors can push sums to 1.00001 or 0.99999.
        # Normalize the final fused distribution to guarantee a valid PMF.
        fused_prob = fused_prob / fused_prob.sum(dim=-1, keepdim=True)

        # ==========================================
        # 4. Final Test Loss
        # ==========================================
        # NLL Loss requires log-probabilities.
        # Clamp before log to prevent log(0) NaNs from perfectly zeroed bins.
        fused_log_prob = torch.log(torch.clamp(fused_prob, min=1e-9))

        # Flatten for the standard PyTorch NLLLoss function
        flat_log_prob = fused_log_prob.view(-1, n_bins)
        flat_targets = y_true_test.view(-1)

        final_loss = F.nll_loss(flat_log_prob, flat_targets)

        return final_loss, fused_prob, lambdas


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np


    class HighDimAdapterWrapper(nn.Module):
        def __init__(self, input_dim=2, d_model=64, n_heads=4, dropout=0.0):
            super().__init__()
            # 1. Up-projection: 2D Spatial -> High-Dimensional Latent Space
            self.up_proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )

            # 2. The Core Meta-Learner (Operating in d_model space)
            self.adapter = NadarayaWatsonAdapter(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )

            # 3. Down-projection: Latent Space -> 2D Spatial
            self.down_proj = nn.Linear(d_model, input_dim)

        def forward(self, x, single_eval_pos):
            # x shape: (T, 3 * Batch, 2)
            h = self.up_proj(x)

            # The adapter now works its magic in 64D space
            h_out, attn_weights = self.adapter(h, single_eval_pos=single_eval_pos)

            # Project back to 2D for the loss calculation and plotting
            out = self.down_proj(h_out)
            return out, attn_weights


    # ==========================================
    # HELPER: 2D Spatial Geometry Fix
    # ==========================================
    # def bypass_layernorms(module):
    #     for name, child in module.named_children():
    #         if isinstance(child, nn.LayerNorm):
    #             setattr(module, name, nn.Identity())
    #         else:
    #             bypass_layernorms(child)


    # ==========================================
    # FULLY DYNAMIC DATA GENERATOR
    # ==========================================
    def generate_dynamic_batch(batch_size, T, sep, D=2):
        """
        Generates a batch where EVERY item has a unique Task A (random base function)
        AND a unique Task B (random distortion of that specific Task A).
        """
        # 1. Task A (Randomized Target Manifold)
        t_A = torch.linspace(0, 2 * np.pi, T).view(T, 1, 1).expand(T, batch_size, 1)

        # Sample random parameters for Task A's underlying function
        A_amp = torch.rand(1, batch_size, 1) * 1.0 + 0.5  # Amplitude (0.5 to 1.5)
        A_freq = torch.rand(1, batch_size, 1) * 0.8 + 0.6  # Frequency (0.6 to 1.4)
        A_phase = torch.rand(1, batch_size, 1) * 2 * np.pi  # Phase shift (0 to 2*pi)

        def task_A_func(x):
            return A_amp * torch.sin(A_freq * x + A_phase)

        A_data = torch.cat([t_A, task_A_func(t_A)], dim=-1)

        # 2. Task B (Randomized Distortion of Task A)
        t_B = torch.linspace(-0.5, 2 * np.pi + 0.5, T).view(T, 1, 1).expand(T, batch_size, 1)

        # Sample random parameters for Task B's distortion over A
        B_scale = torch.rand(1, batch_size, 1) * 1.5 + 0.5  # Amplitude stretch (0.5x to 2.0x)
        B_phase = torch.rand(1, batch_size, 1) * 2 * np.pi  # Domain phase shift (0 to 2*pi)
        B_trend = torch.rand(1, batch_size, 1) * 1.0 - 0.5  # Linear trend slope (-0.5 to +0.5)

        def task_B_func(x):
            # B is a distorted version of A's geometry
            return B_scale * task_A_func(x + B_phase) + B_trend * x

        B_data = torch.cat([t_B, task_B_func(t_B)], dim=-1)

        # 3. Stream C (Queries)
        C_data_init = torch.zeros(T, batch_size, D)
        C_data_init[:, :, 0] = t_A[:, :, 0]  # Ground truth X queries
        C_data_init[:, :, 1] = torch.randn(T, batch_size) * 0.1  # Corrupted Y values

        # 4. Belief Alignment
        t_A_train = t_A[:sep]
        B_belief_A_data = torch.cat([t_A_train, task_B_func(t_A_train)], dim=-1)

        # Pack the active B points + Belief
        B_data_corrected = torch.cat([B_data[:2 * sep], B_belief_A_data], dim=0)

        return A_data, B_data_corrected, C_data_init, B_belief_A_data


    # ==========================================
    # MAIN EXECUTION
    # ==========================================
    def run_dynamic_sanity_check():
        T = 60
        sep = 20
        D = 2
        epochs = 20000
        batch_size = 64  # We can handle larger batches now

        # Initialize the WRAPPER instead of the raw adapter
        # We give it d_model=64 and 4 attention heads for rich feature extraction
        adapter = HighDimAdapterWrapper(input_dim=D, d_model=64, n_heads=4, dropout=0.0)

        # Notice: NO bypass_layernorms() here!

        optimizer = optim.Adam(adapter.parameters(), lr=0.001)  # Slightly lower LR for deep network
        criterion = nn.MSELoss()

        print(f"Starting fully dynamic meta-training with High-Dim Latent Space...")
        for epoch in range(epochs):
            optimizer.zero_grad()

            A_data, B_data_corrected, C_data_init, _ = generate_dynamic_batch(batch_size, T, sep)

            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)

            # Forward pass through the wrapper
            out_batch, _ = adapter(x, single_eval_pos=sep)

            # The indexing remains exactly the same
            C_out = out_batch[:, 2 * batch_size:, :]

            loss = criterion(C_out[:sep * 2], A_data[:sep * 2])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1:04d} | Loss: {loss.item():.6f}")

        # ==========================================
        # INFERENCE ON A NOVEL FUNCTION & DISTORTION
        # ==========================================
        print("\nTesting on a novel Task A and novel Task B distortion...")
        with torch.no_grad():
            A_data, B_data_corrected, C_data_init, B_belief_A_data = generate_dynamic_batch(1, T, sep)

            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)
            final_batch, _ = adapter(x, single_eval_pos=sep)
            C_final = final_batch[:, 2:, :]

        A_plot = A_data[:sep * 2, 0, :].numpy()
        B_plot = B_data_corrected[:sep * 2, 0, :].numpy()
        C_plot = C_final[:sep * 2, 0, :].numpy()
        A_train = A_data[:sep, 0, :].numpy()
        B_belief_plot = B_belief_A_data[:, 0, :].numpy()

        plt.figure(figsize=(10, 6))

        plt.scatter(B_plot[:, 0], B_plot[:, 1], c='gray', alpha=0.5, label='Task B (Novel Distorted Domain)')
        plt.plot(A_plot[:, 0], A_plot[:, 1], 'g--', linewidth=2, label='Task A (Novel Ground Truth)')
        plt.scatter(A_train[:, 0], A_train[:, 1], c='green', s=100, marker='*', label='A Train (Anchors)')
        plt.scatter(B_belief_plot[:, 0], B_belief_plot[:, 1], c='red', s=40, marker='x', label='B Belief of A_train')
        plt.scatter(C_plot[:, 0], C_plot[:, 1], c='blue', s=30, label='Stream C (Adapter Output)')

        for i in range(sep):
            plt.plot([A_train[i, 0], B_belief_plot[i, 0]],
                     [A_train[i, 1], B_belief_plot[i, 1]],
                     'r:', alpha=0.4)

        plt.title('Sanity Check: Zero-Shot Alignment on Novel Functions')
        plt.legend()
        plt.grid(True)
        plt.show()


    run_dynamic_sanity_check()