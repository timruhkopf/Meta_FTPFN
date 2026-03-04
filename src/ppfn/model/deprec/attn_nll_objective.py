"""
This module implements a Dynamic Bayesian Arbiter for token-level blending of multiple conditioned Posterior Predictive
Distributions (PPDs).

* Scalability: The num_tasks dimension only appears in the stack operations.
The heavy lifting (the forward pass) is already done. The blending is just simple arithmetic on vectors of length
$T$.

* Confidence vs. Accuracy: By using F.relu(gain), we ensure that if Task $B_i$ is overconfident but historically
performs worse than the baseline on the $A_{train}$ intersection, its weight is killed regardless of how "sharp"
its attention is.

* The "Discovering" Mechanism: Because you train on an infinite prior, the model "discovers" that the
Smearing (Entropy) is a reliable signal for when the adapter has run out of support. It essentially learns to trust
the confidence factor because that is where the NLL on $A_{test}$ remains the most stable.

Narrative Addition for the Paper.
If you include this in your Method section, highlight that the adapter isn't just a "transformer layer"
—it's a Dynamic Bayesian Arbiter. It uses structural signals (Entropy) and empirical evidence (NLL gain) to perform
token-level model selection between related task manifolds and the unconditional prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicBayesianArbiter(nn.Module):
    """
    Performs token-level blending of multiple conditioned Posterior Predictive
    Distributions (PPDs) by weighting them against an unconditional baseline.

    The Arbiter uses two primary signals to guard against negative transfer:
    1. Global Evidence Gain: Measures if a conditioned stream improves NLL
       over the ground-truth intersection (A_train).
    2. Local Spatial Confidence: Uses the Shannon entropy of the adapter's
       attention scores to discount predictions in extrapolation zones.

    This mechanism ensures that the model defaults to the safe Unconditional
    Baseline (Stream A) whenever related tasks lack empirical support or
    geometric certainty.
    """

    def __init__(self, tau=1.0, eps=1e-9):
        super().__init__()
        self.tau = tau
        self.eps = eps

    def forward(self, log_probs_A, log_probs_C_list, attn_weights_list):
        """
        Args:
            log_probs_A (Tensor): (T, R, V) Log-probabilities of the unconditional baseline.
            log_probs_C_list (List[Tensor]): List of (T, R, V) log-probabilities from N task-conditioned streams.
            attn_weights_list (List[Tensor]): List of (T_query, K_keys) attention weights from the adapters.

        Returns:
            final_log_probs (Tensor): The blended PPD (T, R, V).
            debug_weights (Dict): Dictionary containing the blending weights for analysis.
        """
        T, R, V = log_probs_A.shape
        num_tasks = len(log_probs_C_list)

        if num_tasks == 0:
            return log_probs_A, {"baseline_weight": torch.ones(T, device=log_probs_A.device)}

        # 1. GLOBAL EVIDENCE GAIN (Empirical Validation)
        # We assume the first 'sep' indices in the T dimension represent the A_train intersection.
        # This calculates how much better each conditioned stream is compared to baseline A.
        ll_gains = []
        for lp_C in log_probs_C_list:
            # Evidence is measured over the intersection (train tokens)
            gain = torch.mean(lp_C) - torch.mean(log_probs_A)
            # ReLU prevents negative transfer; if it's worse than baseline, weight becomes 0.
            ll_gains.append(F.relu(gain))

        global_weights = torch.stack(ll_gains)  # (num_tasks)
        global_weights = F.softmax(global_weights / self.tau, dim=0)

        # 2. LOCAL SPATIAL CONFIDENCE (Geometric Support)
        # Higher attention entropy indicates 'smearing' (extrapolation),
        # which triggers a discount on the task's contribution.
        local_confidences = []
        for attn in attn_weights_list:
            # Calculate Shannon Entropy: H = -sum(p * log(p))
            entropy = -torch.sum(attn * torch.log(attn + self.eps), dim=-1)
            max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float))

            # Confidence ranges from 1.0 (sharp/interpolated) to 0.0 (uniform/extrapolated)
            confidence = 1.0 - (entropy / (max_entropy + self.eps))
            local_confidences.append(confidence)

        local_conf_stack = torch.stack(local_confidences, dim=0)  # (num_tasks, T)

        # 3. COMPUTE FINAL BLENDING WEIGHTS
        # The weight for Task i is its Global Evidence scaled by its Local Confidence.
        combined_task_weights = global_weights.unsqueeze(-1) * local_conf_stack  # (num_tasks, T)

        # The remainder of the probability mass is allocated to the Unconditional Baseline A.
        task_sum_weight = torch.sum(combined_task_weights, dim=0)  # (T)
        baseline_weight = torch.clamp(1.0 - task_sum_weight, min=0.0)  # (T)

        # 4. PERFORM PPD BLENDING (Linear combination in probability space)
        # Resulting mean and variance will naturally reflect the most 'supported' task.
        mixed_probs = torch.exp(log_probs_A) * baseline_weight.unsqueeze(-1).unsqueeze(-1)

        for i, lp_C in enumerate(log_probs_C_list):
            task_w = combined_task_weights[i].unsqueeze(-1).unsqueeze(-1)
            mixed_probs += torch.exp(lp_C) * task_w

        final_log_probs = torch.log(mixed_probs + self.eps)

        return final_log_probs, {
            "task_weights": combined_task_weights,
            "baseline_weight": baseline_weight,
            "ll_gains": global_weights
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
