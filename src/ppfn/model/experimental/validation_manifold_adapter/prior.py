import torch


class VectorizedComplexTaskGenerator:
    """
    The "Hidden Harmonic Mixture" Prior with Negative Transfer Injection.
    """
    def __init__(self, x_min=-5.0, x_max=5.0, noise_std=0.05, num_components=4):
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        self.num_components = num_components

    def sample_batch(self, batch_size, n_context=30, n_query=50, share_unrelated=0.2, device='cpu'):
        # 1. Randomize the "Hidden Blueprint" for A
        amps_A = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0.5, 2.0)
        freqs_A = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0.5, 3.0)
        phases_A = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0, 2 * torch.pi)

        # 2. Prepare Blueprint for B (Default to matching A)
        amps_B = amps_A.clone()
        freqs_B = freqs_A.clone()
        phases_B = phases_A.clone()

        # Overwrite a portion of B's blueprint with completely unrelated functions
        num_unrelated = int(batch_size * share_unrelated)
        if num_unrelated > 0:
            amps_B[:, :, -num_unrelated:, :] = torch.empty(self.num_components, 1, num_unrelated, 1, device=device).uniform_(0.5, 2.0)
            freqs_B[:, :, -num_unrelated:, :] = torch.empty(self.num_components, 1, num_unrelated, 1, device=device).uniform_(0.5, 3.0)
            phases_B[:, :, -num_unrelated:, :] = torch.empty(self.num_components, 1, num_unrelated, 1, device=device).uniform_(0, 2 * torch.pi)

        # 3. Transformations for Task A (Applied to Blueprint A)
        scale_A = torch.empty(1, batch_size, 1, device=device).uniform_(0.5, 1.5)
        v_shift_A = torch.empty(1, batch_size, 1, device=device).uniform_(-2.0, 2.0)
        h_shift_A = torch.empty(1, batch_size, 1, device=device).uniform_(-1.0, 1.0)

        # 4. Sample X coordinates
        x_cA = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
        x_cB = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
        x_qA = torch.linspace(self.x_min, self.x_max, n_query, device=device).view(-1, 1, 1).repeat(1, batch_size, 1)

        # Helper to evaluate based on explicit parameters
        def eval_function(x, amps, freqs, phases):
            x_expanded = x.unsqueeze(0)
            terms = amps * torch.sin(freqs * x_expanded + phases)
            return terms.sum(dim=0)

        # 5. Evaluate Y coordinates
        # Task B evaluates its own blueprint (which might be unrelated)
        y_cB = eval_function(x_cB, amps_B, freqs_B, phases_B) + torch.randn_like(x_cB) * self.noise_std

        # Task A evaluates Blueprint A with spatial and amplitude transformations
        y_cA_clean = scale_A * eval_function(x_cA - h_shift_A, amps_A, freqs_A, phases_A) + v_shift_A
        y_cA = y_cA_clean + torch.randn_like(x_cA) * self.noise_std

        # 6. Ground truth for plotting and loss
        y_qA_true = scale_A * eval_function(x_qA - h_shift_A, amps_A, freqs_A, phases_A) + v_shift_A
        y_qB_true = eval_function(x_qA, amps_B, freqs_B, phases_B)

        # 7. Add boolean mask for tracking which tasks are unrelated traps
        is_unrelated = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if num_unrelated > 0:
            is_unrelated[-num_unrelated:] = True

        return {
            "x_cA": x_cA, "y_cA": y_cA,
            "x_cB": x_cB, "y_cB": y_cB,
            "x_qA": x_qA, "y_qA_true": y_qA_true,
            "y_qB_true": y_qB_true,
            "is_unrelated": is_unrelated # Use this to split your evaluation loss!
        }

def create_padded_batch(generator, batch_size, n_A=4, n_B=30, n_query=50, device='cpu', share_unrelated=0.2):
    """Handles variable sequence lengths and appends queries to match T dim."""
    T_max_context = max(n_A, n_B)
    T_total = T_max_context + n_query

    batch = generator.sample_batch(batch_size, n_context=T_max_context, n_query=n_query, device=device, share_unrelated=share_unrelated)

    # 1. Create Base Context Masks (Shape: [Batch, T_max_context])
    seq_indices = torch.arange(T_max_context, device=device).unsqueeze(0).expand(batch_size, T_max_context)
    base_mask_A = seq_indices >= n_A
    base_mask_B = seq_indices >= n_B

    # Zero out padded payloads in the context phase
    batch["y_cA"] = batch["y_cA"].masked_fill(base_mask_A.transpose(0, 1).unsqueeze(-1), 0.0)
    batch["y_cB"] = batch["y_cB"].masked_fill(base_mask_B.transpose(0, 1).unsqueeze(-1), 0.0)

    # 2. Append Queries to A, B, and C
    # We append the query X coordinates to all three
    batch["x_cA"] = torch.cat([batch["x_cA"], batch["x_qA"]], dim=0)  # [T_total, B, 1]
    batch["x_cB"] = torch.cat([batch["x_cB"], batch["x_qA"]], dim=0)

    # We append dummy zeros for the Y coordinates of the queries in A and B
    dummy_y = torch.zeros_like(batch["x_qA"])
    batch["y_cA"] = torch.cat([batch["y_cA"], dummy_y], dim=0)
    batch["y_cB"] = torch.cat([batch["y_cB"], dummy_y], dim=0)

    # For C, the context part is dummy, and the query part is dummy
    # (since C is only for querying). We just make a full zero tensor.
    batch["x_cC"] = batch["x_cA"].clone()
    batch["y_cC"] = torch.zeros_like(batch["x_cA"])

    # 3. Update Masks for the appended Query points
    # A and B should NOT attend to the query dummy values during self-attention
    query_mask_true = torch.ones(batch_size, n_query, dtype=torch.bool, device=device)
    batch["mask_A"] = torch.cat([base_mask_A, query_mask_true], dim=1)  # [Batch, T_total]
    batch["mask_B"] = torch.cat([base_mask_B, query_mask_true], dim=1)

    # C should only attend to its query coordinates, not its dummy context coordinates
    context_mask_true = torch.ones(batch_size, T_max_context, dtype=torch.bool, device=device)
    query_mask_false = torch.zeros(batch_size, n_query, dtype=torch.bool, device=device)
    batch["mask_C"] = torch.cat([context_mask_true, query_mask_false], dim=1)

    batch["sep"] = T_max_context  # Store this so the model knows where queries start
    return batch

