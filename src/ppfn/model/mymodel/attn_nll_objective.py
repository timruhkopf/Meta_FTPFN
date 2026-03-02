import torch
from torch import nn
import torch.nn.functional as F

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
