"""





Hyperparameter evaluation:
* we have access to the coordinates of both A_train, B_train and C_test
 (which is what we can query for on A and B as well).

* we can always query for any hp coordinates (as estimate of y) and compare functional values

* Express B through A or the other way around?

* We always have C as a workbench context, preserving A and B as raw unaltered and unconditional contexts

Modeling approaches:
* Learn a relative encoding:
 self attention on A_train, B_train and then do some form of cross attention based on the relative encodings.

* learn a difference function based on the interpolation points between A and B, and then apply that difference to
  where we need to project B into A in order to transfer its information.

* Gating: the query should decide on SDPA, whether to cross-attend to B, and how much to update itself.

* Residual learning: C = C+update

* We can conceive warping of the input space between the two tasks - which implies that we learn how to "move the anchors"
 i.e. the hp coordinates of A to best align with B

* we can try to identify how much the model want to rely on the related task by looking at the attention weights to A vs B
  this can inform our final blending weights.


# TODO for Prior:
#  - [ ] Just have A and B  shifted versions of each other, have A truncated in x and B more dense, and then
#        cross attend to the dense B to extrapolate A's function beyond its sparse context points
#  - [ ] Then add complexity by adding scale to it as another distortion
#  - [ ] Then add complexity by adding a linear trend to it as another distortion
"""
from pathlib import Path

from torch import nn

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class MetaTaskGenerator:
    def __init__(self, x_min=-5.0, x_max=5.0, noise_std=0.05):
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std

    def generate_function_pair(self):
        """
        Generates a base function A and a systematically perturbed function B.
        Using a family of sine waves with linear trends for clear visualization.
        """
        # Randomize base parameters for Function A
        amp_A = np.random.uniform(0.5, 2.0)
        phase_A = np.random.uniform(0, 2 * np.pi)
        freq_A = np.random.uniform(0.5, 1.5)

        # Perturbations to create Function B
        phase_shift = np.random.uniform(-0.5, 0.5)
        amp_scale = np.random.uniform(0.8, 1.2)
        linear_trend = np.random.uniform(-0.2, 0.2)

        def f_A(x):
            return amp_A * torch.sin(freq_A * x + phase_A)

        def f_B(x):
            # Function B is a scaled, phase-shifted version of A, with an added trend
            return amp_scale * amp_A * torch.sin(freq_A * x + phase_A + phase_shift) + (linear_trend * x)

        return f_A, f_B

    def sample_data(self, f_A, f_B, n_context_A=4, n_context_B=30, n_query_A=50):
        """
        Samples data from f_A and f_B.
        Crucially, B has very few context points to force knowledge transfer.
        """
        # 1. Sample inputs (x) uniformly across the domain
        x_context_A = torch.empty(n_context_A, 1).uniform_(self.x_min, self.x_max)
        x_context_B = torch.empty(n_context_B, 1).uniform_(self.x_min, self.x_max)

        # Query points for B are laid out in a grid for evaluation/plotting
        x_query_A = torch.linspace(self.x_min, self.x_max, n_query_A).unsqueeze(1)

        # 2. Generate targets (y) and add observational noise
        y_context_A = f_A(x_context_A) + torch.randn_like(x_context_A) * self.noise_std
        y_context_B = f_B(x_context_B) + torch.randn_like(x_context_B) * self.noise_std

        # 3. Ground truth for query points (typically evaluated noiseless)
        y_query_A_true = f_A(x_query_A)

        return {
            "context_A": (x_context_A, y_context_A),
            "context_B": (x_context_B, y_context_B),
            "query_B": (x_query_A, y_query_A_true)
        }

class MLP(nn.Module):
    """A standard 2-layer MLP to add non-linear depth."""
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)



class NewLayer(nn.Module):
    """

    source for the a separated infomration flow, where we have k and v different from q in order to compute what to extract:
    https://arxiv.org/pdf/2107.14795 Perciever IO
    """
    def __init__(self, dmodel=128, use_gate=False):
        super().__init__()
        self.linear_AB1 = MLP(dmodel * 2, dmodel, dmodel*2)
        self.self_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4, )
        self.linear_AB2 = nn.Linear(dmodel * 2, dmodel)
        self.linear_C = nn.Linear(dmodel * 2, dmodel)
        self.cross_attention = nn.MultiheadAttention(embed_dim=2*dmodel, vdim=dmodel, num_heads=4,)

        self.C_test_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4,vdim=dmodel)

        self.out_proj = MLP(dmodel * 2, dmodel, dmodel*2)

        self.use_gate = use_gate
        if use_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(4 * dmodel, dmodel),
                nn.Sigmoid()
            )

        # Learnable dummy key and value (Shape: [1, 1, feature_dim])
        self.dummy_key = nn.Parameter(torch.randn(1, 1, 2 * dmodel))
        # Note: Adjust dummy_value dimension based on whether you concatenated hp into value earlier
        self.dummy_value = nn.Parameter(torch.randn(1, 1, dmodel))


        self.init_weights()

    def init_weights(self):
        """Neutral Initialization to fade-in the adapters layers """

    def forward(self, A, B, C, sep, hp, mask_A=None, mask_B=None, *args, **kwargs):
        device = A.device
        hp_A, hp_B, hp_C = hp

        # 1. Parse Context Components
        A_train, B_train = A[:sep], B[:sep]
        C_train, C_test = C[:sep], C[sep:]
        hp_A_train, hp_B_train = hp_A[:sep], hp_B[:sep]
        hp_C_train, hp_C_test = hp_C[:sep], hp_C[sep:]

        # 2.0 self attend for A and B to extract relative features
        #  e.g. "i am a point in a local maximum", "we all are trending linearly"
        ABC = torch.cat([
            torch.cat([hp_A_train, A_train], dim=-1),
            torch.cat([hp_B_train, B_train], dim=-1),
            torch.cat([hp_C_train, C_train], dim=-1) # if we do this prior to any cross attention this is redundant (C=A)
            
        ], dim=1)

        # down project to dmodel
        ABC = self.linear_AB1(ABC)

        # 2.1. Deal with padding in self attention
        # since A and B have different sequence lengths, we need to build an attention mask
        # to prevent attending to the padded points.
        mask_A_train = mask_A[:, :sep] if mask_A is not None else None
        mask_B_train = mask_B[:, :sep] if mask_B is not None else None


        if mask_A_train is not None and mask_B_train is not None:
            # C doesn't have a mask (it uses all its points), so it's all False
            mask_C_train = mask_A_train.clone()

            # Concat along batch dimension (dim=0) because ABC has 3*Batch size
            self_attn_mask = torch.cat([mask_A_train, mask_B_train, mask_C_train], dim=0)
        else:
            self_attn_mask = None

        # 2.2  A,B,c Shared self attention to extract relational features within A, B, C respectively.
        ABC, _ = self.self_attention(ABC, ABC, ABC, key_padding_mask=self_attn_mask)

        # Extract the feature descriptors for each task after self-attention
        a_dim, b_dim = A_train.shape[1], B_train.shape[1]
        A_feat, B_feat, C_feat = ABC[:, :a_dim], ABC[:, a_dim:a_dim+b_dim], ABC[:, a_dim+b_dim:]

        # 3. Expand C_feat to C_test_feat by hp attention across coordinates.
        C_test_feat, _ = self.C_test_attention(hp_C_test, hp_C_train, C_feat, key_padding_mask=mask_C_train)


        # 4. Cross Attention from C to A's and B's features
        # FIXME: THIS IS THE MOST CRITICAL ASPECT OF IT ALL!
        # throw in A_feat and B_feat into one context for C to cross attend to.
        # TODO Here we don't want the softmax constraint, because it will have a sum-to-one constraint across A and B,
        #  but we want the model to be able to choose to attend to B

        batch_size = A.shape[1]


        # Expand dummy tokens to match batch size: [1, Batch, D]
        d_key = self.dummy_key.expand(1, batch_size, -1)
        d_val = self.dummy_value.expand(1, batch_size, -1)

        query = torch.cat([
            torch.cat([hp_C_train, C_feat], dim=-1) ,
            torch.cat([hp_C_test, C_test_feat], dim=-1)
        ], dim=0)
        key = torch.cat([
            torch.cat([hp_A_train, A_feat], dim=-1),
            torch.cat([hp_B_train, B_feat], dim=-1),
            d_key  # <--- The Escape Valve
        ], dim=0)
        value = torch.cat([
            A_feat, B_feat,
            d_val # <--- The Escape Valve
        ], dim=0) # we only want the features to be the value
        # we cannot attend to the B features, because they are not grounded in the same domain as C
        # TODO ideally, we'd know how B looks like in C's domain (B_train'), then we could make the value the raw B_train' payload (without hp).

        # Build the cross-attention mask
        # C is looking at A and B, so the keys sequence length is 2*T_max.
        # Mask shape needs to be [Batch, 2*T_max]
        # Build the cross-attention mask (Shape: [Batch, 2*sep])
        if mask_A_train is not None and mask_B_train is not None:
            # Concat along the sequence dimension (dim=1) for the mask
            cross_attn_mask = torch.cat([mask_A_train, mask_B_train], dim=1)
        else:
            cross_attn_mask = None

        # We must also append a 'False' to the cross_attn_mask so the dummy token is never masked out
        if cross_attn_mask is not None:
            dummy_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            cross_attn_mask = torch.cat([cross_attn_mask, dummy_mask], dim=1)

        # Pass the mask to cross-attention
        C_cross, _ = self.cross_attention(query, key, value, key_padding_mask=cross_attn_mask)

        # 4. Residual update to C
        # C is originally (T, B, D). We project C_cross down to match.
        # Evaluate the query AGAINST the retrieved context
        # This is heavily inspired by Highway Networks and GRUs. It is local, point-by-point, and fully aware of the relationship between $C$ and $B$.
        # This allows the network to say: "I asked for a local maximum (Query), but the feature vector I got back from B looks like a steep drop (C_cross). This is useless to me. Close the gate."
        if self.use_gate:
            gate_input = torch.cat([query, C_cross], dim=-1)
            gate = self.gate_proj(gate_input)
        else:
            gate = 1.0

        # Gated Residual update
        C = C + gate * self.out_proj(C_cross)

        # C = C + self.out_proj(C_cross)

        return A, B, C




def create_padded_context(generator, batch_size, n_A=4, n_B=30, device='cpu'):
    """
    Samples data but forces A and B to the same sequence length (T_max) via padding,
    returning the padded data and the boolean masks for the attention layers.
    """
    T_max = max(n_A, n_B)

    # 1. Sample full length data for both to get the shapes right
    batch = generator.sample_batch(batch_size, n_context=T_max, n_query=50, device=device)

    # 2. Create boolean masks (Shape: [Batch, T_max])
    # True means "Mask this out / Ignore it"
    seq_indices = torch.arange(T_max, device=device).unsqueeze(0).expand(batch_size, T_max)

    mask_A = seq_indices >= n_A  # e.g., True for indices 4 through 29
    mask_B = seq_indices >= n_B  # e.g., All False if n_B is 30

    # Optional but good practice: Zero out the actual padded data payloads
    # (batch["y_cA"] is shape [T_max, B, 1], so we transpose mask to match)
    batch["y_cA"] = batch["y_cA"].masked_fill(mask_A.transpose(0, 1).unsqueeze(-1), 0.0)
    batch["y_cB"] = batch["y_cB"].masked_fill(mask_B.transpose(0, 1).unsqueeze(-1), 0.0)

    batch["mask_A"] = mask_A
    batch["mask_B"] = mask_B

    return batch

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. Data Generation & Masking
# ==========================================
# class VectorizedMetaTaskGenerator:
#     def __init__(self, x_min=-5.0, x_max=5.0, noise_std=0.05):
#         self.x_min, self.x_max, self.noise_std = x_min, x_max, noise_std
#
#     def sample_batch(self, batch_size, n_context=30, n_query=50, device='cpu'):
#         amp_A = torch.empty(1, batch_size, 1, device=device).uniform_(0.5, 2.0)
#         phase_A = torch.empty(1, batch_size, 1, device=device).uniform_(0, 2 * torch.pi)
#         freq_A = torch.empty(1, batch_size, 1, device=device).uniform_(0.5, 1.5)
#
#         phase_shift = torch.empty(1, batch_size, 1, device=device).uniform_(-0.5, 0.5)
#         amp_scale = torch.empty(1, batch_size, 1, device=device).uniform_(0.8, 1.2)
#         linear_trend = torch.empty(1, batch_size, 1, device=device).uniform_(-0.2, 0.2)
#
#         x_cA = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
#         x_cB = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
#         x_qA = torch.linspace(self.x_min, self.x_max, n_query, device=device).view(-1, 1, 1).repeat(1, batch_size,
#                                                                                                     1)
#
#         y_cA = amp_A * torch.sin(freq_A * x_cA + phase_A) + torch.randn_like(x_cA) * self.noise_std
#         y_cB = amp_scale * amp_A * torch.sin(freq_A * x_cB + phase_A + phase_shift) + (
#                     linear_trend * x_cB) + torch.randn_like(x_cB) * self.noise_std
#
#         y_qA_true = amp_A * torch.sin(freq_A * x_qA + phase_A)
#         y_qB_true = amp_scale * amp_A * torch.sin(freq_A * x_qA + phase_A + phase_shift) + (linear_trend * x_qA)
#
#         return {
#             "x_cA": x_cA, "y_cA": y_cA,
#             "x_cB": x_cB, "y_cB": y_cB,
#             "x_qA": x_qA, "y_qA_true": y_qA_true,
#             "y_qB_true": y_qB_true
#         }

class VectorizedComplexTaskGenerator:
    """
    The "Hidden Harmonic Mixture" Prior
    We will generate a highly complex, unpredictable "base shape" using a sum of several high-frequency sine
    waves with randomized parameters.

    According to the Nyquist-Shannon sampling theorem, 4 points are entirely insufficient to reconstruct a high-frequency
    mixture. If the model only looks at A, it will fail catastrophically.
    However, we will give Task B dense observations of this crazy base shape.
    Task A will simply be a shifted and scaled version of Task B.

    To succeed, the model must look at B to learn the high-frequency "fingerprint," and then use the 4 sparse points of A
    purely to figure out how to stretch and slide that fingerprint into place.
    """
    def __init__(self, x_min=-5.0, x_max=5.0, noise_std=0.05, num_components=4):
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        # Number of sine waves to mix. 4 is usually plenty to create an unpredictable shape.
        self.num_components = num_components

    def sample_batch(self, batch_size, n_context=30, n_query=50, device='cpu'):
        # 1. Randomize the "Hidden Blueprint" (Sum of Sines)
        # Shape: [Components, 1, Batch, 1] for broadcasting against [Sequence, Batch, 1]
        amps = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0.5, 2.0)
        # High frequencies make it impossible to guess from 4 points
        freqs = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0.5, 3.0)
        phases = torch.empty(self.num_components, 1, batch_size, 1, device=device).uniform_(0, 2 * torch.pi)

        # 2. Transformations for Task A
        # Task A is Task B, but scaled, shifted vertically, and shifted horizontally
        scale_A = torch.empty(1, batch_size, 1, device=device).uniform_(0.5, 1.5)
        v_shift_A = torch.empty(1, batch_size, 1, device=device).uniform_(-2.0, 2.0)
        h_shift_A = torch.empty(1, batch_size, 1, device=device).uniform_(-1.0, 1.0)

        # 3. Sample X coordinates
        x_cA = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
        x_cB = torch.empty(n_context, batch_size, 1, device=device).uniform_(self.x_min, self.x_max)
        x_qA = torch.linspace(self.x_min, self.x_max, n_query, device=device).view(-1, 1, 1).repeat(1, batch_size, 1)

        # Helper to evaluate the complex base function
        def eval_base_function(x):
            # x is [T, B, 1] -> unsqueeze to [1, T, B, 1]
            x_expanded = x.unsqueeze(0)
            # Broadcasting magic computes all components simultaneously
            terms = amps * torch.sin(freqs * x_expanded + phases)
            # Sum across the components dimension (dim=0)
            return terms.sum(dim=0)

        # 4. Evaluate Y coordinates
        # Task B is exactly the base blueprint
        y_cB = eval_base_function(x_cB) + torch.randn_like(x_cB) * self.noise_std

        # Task A slides the blueprint horizontally, scales it, and shifts it vertically
        y_cA_clean = scale_A * eval_base_function(x_cA - h_shift_A) + v_shift_A
        y_cA = y_cA_clean + torch.randn_like(x_cA) * self.noise_std

        # 5. Ground truth for plotting and loss
        y_qA_true = scale_A * eval_base_function(x_qA - h_shift_A) + v_shift_A
        y_qB_true = eval_base_function(x_qA)

        return {
            "x_cA": x_cA, "y_cA": y_cA,
            "x_cB": x_cB, "y_cB": y_cB,
            "x_qA": x_qA, "y_qA_true": y_qA_true,
            "y_qB_true": y_qB_true
        }


def create_padded_batch(generator, batch_size, n_A=4, n_B=30, n_query=50, device='cpu'):
    """Handles variable sequence lengths and appends queries to match T dim."""
    T_max_context = max(n_A, n_B)
    T_total = T_max_context + n_query

    batch = generator.sample_batch(batch_size, n_context=T_max_context, n_query=n_query, device=device)

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


class MetaTransferModel(nn.Module):
    def __init__(self, input_dim=1, dmodel=128):
        super().__init__()
        # Use MLPs for better representation as discussed
        self.y_proj = MLP(input_dim, dmodel, dmodel)
        self.x_proj = MLP(input_dim, dmodel, dmodel)
        self.transfer_layer = NewLayer(dmodel=dmodel)
        self.out_proj = MLP(dmodel, input_dim, dmodel)

    def forward(self, batch):
        # Data is already padded and concatenated in create_padded_batch
        A = self.y_proj(batch["y_cA"])
        B = self.y_proj(batch["y_cB"])
        C = self.y_proj(batch["y_cC"])

        hp_A = self.x_proj(batch["x_cA"])
        hp_B = self.x_proj(batch["x_cB"])
        hp_C = self.x_proj(batch["x_cC"])

        # Call NewLayer
        # Assuming your NewLayer still handles the sep splitting internally!
        _, _, C_out = self.transfer_layer(
            A, B, C, sep=batch["sep"], hp=(hp_A, hp_B, hp_C),
            mask_A=batch["mask_A"], mask_B=batch["mask_B"]
        )

        # We only care about predicting the query portion of C_out
        sep = batch["sep"]
        query_out = C_out[sep:, :, :]

        return self.out_proj(query_out)

# ==========================================
# 3. Training & Plotting Loop
# ==========================================
def plot_training_step(step, batch, y_pred, loss_history, n_A, n_B, save_path):
    # Fix: Ensure all tensors are float32 before converting to numpy
    # .float() converts bfloat16 -> float32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sep = batch["sep"]

    # 1. Extract and Cast Data
    # Adding .float() before .cpu().numpy() fixes the ScalarType error
    mask_A_np = batch["mask_A"][0, :sep].cpu().numpy()
    mask_B_np = batch["mask_B"][0, :sep].cpu().numpy()

    x_cA = batch["x_cA"][:sep, 0, 0].float().cpu().numpy()[~mask_A_np]
    y_cA = batch["y_cA"][:sep, 0, 0].float().cpu().numpy()[~mask_A_np]
    x_cB = batch["x_cB"][:sep, 0, 0].float().cpu().numpy()[~mask_B_np]
    y_cB = batch["y_cB"][:sep, 0, 0].float().cpu().numpy()[~mask_B_np]

    x_q = batch["x_qA"][:, 0, 0].float().cpu().numpy()
    y_qA_true = batch["y_qA_true"][:, 0, 0].float().cpu().numpy()
    y_qB_true = batch["y_qB_true"][:, 0, 0].float().cpu().numpy()

    # The fix for your specific error:
    y_p = y_pred[:, 0, 0].detach().float().cpu().numpy()

    # 2. Enhanced Visualization
    # Plotting the "Source" task B in the background
    ax1.plot(x_q, y_qB_true, color='red', linestyle='--', alpha=0.2, label='Source Task B (Full)')
    ax1.scatter(x_cB, y_cB, c='red', marker='x', s=20, alpha=0.4, label=f'Source Pts ({n_B})')

    # Plotting the "Target" task A
    ax1.plot(x_q, y_qA_true, color='blue', linewidth=1.5, alpha=0.6, label='Target Task A (GT)')
    ax1.scatter(x_cA, y_cA, c='blue', edgecolors='black', s=80, zorder=5, label=f'Target Pts ({n_A})')

    # The Prediction
    ax1.plot(x_q, y_p, color='green', linewidth=2.5, label='Model Prediction (A|B)')

    ax1.set_title(f"Step {step}: Meta-Transfer (VRAM Optimized)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc='upper right', fontsize='small', frameon=True)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # 3. Smoothed Loss Curve
    if len(loss_history) > 0:
        ax2.plot(loss_history, color='purple', alpha=0.3)
        # Add a moving average to see the trend through the noise
        if len(loss_history) > 50:
            avg_loss = np.convolve(loss_history, np.ones(50) / 50, mode='valid')
            ax2.plot(np.arange(49, len(loss_history)), avg_loss, color='purple', linewidth=2)

    ax2.set_title("Training Loss (MSE)")
    ax2.set_yscale('log')
    ax2.set_xlabel("Steps")
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()

    # Ensure directory exists and save
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"step_{step:05d}.png", dpi=150)
    plt.close(fig)
#
# def train_meta_model(steps=2000, batch_size=32, n_A=5, n_B=30, n_query=50, plot_every=200):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Training on: {device}")
#
#     generator = VectorizedComplexTaskGenerator()
#     model = MetaTransferModel(input_dim=1, dmodel=128).to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     # Add this right below the optimizer:
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
#     criterion = nn.MSELoss()
#
#     os.makedirs("training_plots", exist_ok=True)
#     loss_history = []
#
#     iterator = tqdm(range(steps + 1))
#
#     for step in iterator:
#         model.train()
#         optimizer.zero_grad()
#
#         batch = create_padded_batch(generator, batch_size, n_A, n_B, n_query, device)
#
#
#
#         y_pred = model(batch)
#
#         loss = criterion(y_pred, batch["y_qA_true"])
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#
#         loss_history.append(loss.item())
#
#         if step % 10 == 0:
#             iterator.set_description(f"Step {step}/{steps} | Loss: {loss.item():.4f}")
#
#         if step % plot_every == 0:
#             # print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")
#             model.eval()
#             with torch.no_grad():
#                 # Evaluate a clean batch for plotting
#                 eval_batch = create_padded_batch(generator, 1, n_A, n_B, n_query, device)
#                 eval_pred = model(eval_batch)
#                 plot_training_step(step, eval_batch, eval_pred, loss_history, n_A, n_B)

def train_meta_model(steps=2000, batch_size=32, n_A=5, n_B=30, n_query=50, save_path=Path('.'), plot_every=200, compile_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    generator = VectorizedComplexTaskGenerator()
    model = MetaTransferModel(input_dim=1, dmodel=128).to(device)

    # CHANGED: Add PyTorch compilation for massive speedups on Ampere GPUs
    if compile_model and torch.__version__.startswith('2.') and device.type == 'cuda':
        print("Compiling model for faster execution...")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    criterion = nn.MSELoss()

    # CHANGED: Initialize GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    os.makedirs("training_plots", exist_ok=True)
    loss_history = []

    iterator = tqdm(range(steps + 1))

    for step in iterator:
        model.train()
        optimizer.zero_grad()

        # Data generation is already perfectly on-device
        batch = create_padded_batch(generator, batch_size, n_A, n_B, n_query, device)

        # CHANGED: Wrap the forward pass and loss calculation in autocast (bfloat16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            y_pred = model(batch)
            loss = criterion(y_pred, batch["y_qA_true"])

        # CHANGED: Use the scaler to backward and step (prevents underflow in FP16/BF16)
        scaler.scale(loss).backward()

        # --- NEW: GRADIENT CLIPPING BLOCK ---
        # 1. Unscale the gradients so the norm is calculated correctly
        scaler.unscale_(optimizer)

        # 2. Clip the gradients (max_norm=1.0 is standard for Transformers/MLPs)
        # This prevents the "Exploding Gradient" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 0:
            mem_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            print(f"\n[GPU Monitor] Step 0: Max Allocated: {mem_allocated:.2f}GB | Max Reserved: {mem_reserved:.2f}GB")
            # Optional: Reset peak stats so you can check again later
            torch.cuda.reset_peak_memory_stats(device)

        scheduler.step()

        loss_history.append(loss.item())

        if step % 10 == 0:
            iterator.set_description(f"Step {step}/{steps} | Loss: {loss.item():.4f}")

        if step % plot_every == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate a clean batch for plotting
                eval_batch = create_padded_batch(generator, 1, n_A, n_B, n_query, device)

                # CHANGED: Ensure eval also runs in autocast if the model was trained in it
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    eval_pred = model(eval_batch)

                plot_training_step(step, eval_batch, eval_pred, loss_history, n_A, n_B, save_path)

# ==========================================
# 4. Run Execution
# ==========================================
if __name__ == "__main__":
    # We use n_A = 4 (sparse target) and n_B = 30 (dense source)
    train_meta_model(steps=25000, batch_size=8192, n_A=5, n_B=30, plot_every=500, save_path=Path("/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/local/"))
    print("Training complete! Check the 'training_plots' folder.")
