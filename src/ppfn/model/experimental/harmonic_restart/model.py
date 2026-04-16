import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor

from ppfn.model.mymodel.meta_context import ForwardMetaContext

"""
On Gating: "How do we ask for plausibility without overusing the data?"

If you use Domain C's points to compute the warp, and then you evaluate the NLL on those exact same C points to decide if the warp was "plausible," you will always get a false positive. The network has already perfectly overfit those anchors.

The Conceptual Solution: Evaluate the Transformation, Not the Data.
You don't need to look at Domain C to know if the transfer is plausible; you need to look at the energy of the deformation field itself.

Think of Domain B as a rubber sheet. To align it with a related Domain C, you might have to stretch it a little bit (low energy). To align it with an unrelated Domain C, you have to violently stretch, twist, and fold the sheet (high energy).

Instead of asking, "Does this warped B fit C?", you ask, "How hard did the network have to work to warp B?"

What is the magnitude of the predicted shift vector (mu)?

How erratic are the attention weights in the align_attn?

Does the shift vector change violently between adjacent tokens?

If the transformation requires extreme, high-frequency changes, it is mathematically implausible, indicating negative transfer.
"""

"""
reasons for mild nll improvement conditioning on unrelated contexts 
1. spurious correlation
2. The "Process of Elimination" (Your Second Point)Knowing what $A$ is not is mathematically valuable. If your model operates in a constrained space, and the unrelated context $B$ occupies a certain volume of that space, the model can confidently say, "Well, $A$ cannot be there." By eliminating a chunk of the probability mass, the remaining distribution is forced to squeeze into a smaller volume, artificially sharpening the confidence bounds.
3. overfitting & miscalibration
4. The Information Theory Trap: $H(A|B) \leq H(A)$In pure information theory, conditioning reduces entropy.
 The entropy of $A$ given $B$ can never be mathematically greater than the entropy of $A$ alone, on average.
 Many architectures (especially self-attention or conditional normalizations) are structurally biased to 
 behave this way. When you give the model an input $B$ to condition on, the network's internal representations
  contract. It assumes that because it was given a condition, it must use it to reduce the hypothesis space. 
  It tightens the bounds mechanically, unaware that $B$ is pure garbage.
"""


class PFNLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, ):
        super(PFNLayer, self).__init__()
        batch_first = False
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Shared Feedforward
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Pre-Norms
        self.norm1 = LayerNorm(d_model)  # Pre Self-Attention
        self.norm_cross = LayerNorm(d_model)  # Pre Cross-Attention
        self.norm2 = LayerNorm(d_model)  # Pre Feedforward

        self.dropout1 = Dropout(dropout)
        self.dropout_cross = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu

    def _apply_self_attention(self, src: Tensor, eval_pos: int, pad_mask: Optional[Tensor]) -> Tensor:
        train_part = src[:eval_pos, :, :]
        test_part = src[eval_pos:, :, :]
        train_pad_mask = pad_mask[:, :eval_pos] if pad_mask is not None else None

        train_out = self.self_attn(train_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]
        test_out = self.self_attn(test_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]

        return torch.cat([train_out, test_out], dim=0)

    def forward(
            self,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            single_eval_pos: int,
            pad_mask_A: Optional[Tensor] = None,
            pad_mask_B: Optional[Tensor] = None
    ):
        # Store original batch size to split them back apart later
        # Assumes A, B, and C all have the exact same shape: (Time, Batch, D)
        T, B_size, D = A.shape

        # ==========================================
        # 1. BATCH CONCATENATION
        # ==========================================
        # Stack along the Batch dimension (dim=1) -> Shape: (Time, 3 * Batch, D)
        combined = torch.cat([A, B, C], dim=1)

        # Handle padding masks (Shape: (Batch, Time))
        if pad_mask_A is not None or pad_mask_B is not None:
            device = combined.device
            # If one mask is provided but the other isn't, default the missing one to False
            if pad_mask_A is None:
                pad_mask_A = torch.zeros((B_size, T), dtype=torch.bool, device=device)
            if pad_mask_B is None:
                pad_mask_B = torch.zeros((B.shape[1], T), dtype=torch.bool, device=device)

            # C uses the exact same padding mask as A
            combined_mask = torch.cat([pad_mask_A, pad_mask_B, pad_mask_A], dim=0)
        else:
            combined_mask = None
        # ==========================================
        # 2. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        normed_combined = self.norm1(combined)

        # One massive attention call
        src2_combined = self._apply_self_attention(
            normed_combined,
            single_eval_pos,
            combined_mask
        )

        combined = combined + self.dropout1(src2_combined)

        # ==========================================
        # 3. SHARED FEED-FORWARD
        # ==========================================
        normed_ff_combined = self.norm2(combined)

        def ff_block(x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        combined = combined + self.dropout2(ff_block(normed_ff_combined))

        # ==========================================
        # 4. UNPACK BACK TO A, B, C
        # ==========================================
        # Split the combined tensor back into 3 distinct tensors along the batch dimension
        A_out = combined[:, :B_size, :]
        B_out = combined[:, B_size:-B_size, :]
        C_out = combined[:, -B_size:, :]

        # A_out, B_out, C_out = torch.split(combined, split_size_or_sections=B_size, dim=1)

        return A_out.contiguous(), B_out.contiguous(), C_out.contiguous()


class GumbelGate(nn.Module):
    def __init__(self, input_dim, hard=True):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, 1)
        self.hard = hard
        # Initialize bias so it starts "undecided" or slightly open
        nn.init.constant_(self.gate_linear.bias, 0.5)

    def forward(self, x, tau=1.0, training=True):
        logits = self.gate_linear(x)  # (Batch, 1)

        if training:
            # 1. Sample Gumbel noise
            unif = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(unif + 1e-20) + 1e-20)

            # 2. Apply Gumbel-Sigmoid trick
            # We treat the single logit as the difference between two gumbel samples
            y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

            if self.hard:
                # 3. Straight-Through Estimator
                # Forward pass: 0 or 1. Backward pass: gradient of y_soft
                y_hard = (y_soft > 0.5).float()
                gate = y_hard - y_soft.detach() + y_soft
            else:
                gate = y_soft
        else:
            # Inference: Just a deterministic threshold or sigmoid
            gate = (torch.sigmoid(logits) > 0.5).float() if self.hard else torch.sigmoid(logits)

        return gate, logits


class HardSigmoidGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d_model, 1)
        # Initialize bias to 0.5 to 1.0.
        # This ensures gate output starts around 0.6 - 0.7 (inside the linear gradient zone)
        nn.init.constant_(self.gate_linear.bias, 0.8)

    def forward(self, x):
        # F.hardsigmoid is built into modern PyTorch (1.7+)
        logits = self.gate_linear(x)
        ForwardMetaContext.set('gate_logits', logits)
        return F.hardsigmoid(logits)


class MetaWarpAttention(nn.Module):
    """
        MetaWarpAttention implements a coarse-to-fine, geometry-aware cross-attention mechanism.

        Standard relative attention assumes a static or zero-centered spatial relationship.
        When transferring features between two domains (e.g., Stream C querying Stream B)
        where one is a physically shifted or non-linearly warped variant of the other,
        standard attention either fails to align or suffers from massive uncertainty bounds.

        MetaWarp solves this "Domain Shift" problem dynamically in three stages:

        1. The Probe (Meta-Attention):
           Projects the features into a lightweight subspace (`d_probe`) to compute a fast,
           coarse correlation matrix between the query (C) and the key (B).

        2. The Inference (Geometry Extraction):
           Uses the coarse correlation matrix to compute the "Expected Physical Location"
           in B's coordinate space. By subtracting C's actual coordinate, it infers a
           dynamic, per-token physical shift (μ). A lightweight MLP then predicts the
           confidence/strictness of this shift (γ).

        3. The Transfer (Main Attention):
           Constructs a dynamic, Shifted Radial Basis Function (RBF) mask centered exactly
           on the inferred shift (μ) with width (γ). This mask restricts a full-capacity
           MultiheadAttention, forcing it to only pull features from the geometrically
           correct, dynamically inferred neighborhood.

        Args:
            d_model (int): Total dimension of the feature embeddings.
            nhead (int): Number of parallel attention heads.
            d_probe (int, optional): Dimension of the lightweight probing projection.
                Keep this small (e.g., 16 or 32) to minimize computational overhead. Defaults to 16.
            d_hp (int, optional): Dimension of the physical coordinates (Hyperparameters).
                Usually 1 for 1D signals (X-axis). Defaults to 1.
            dropout (float, optional): Dropout probability for the main attention. Defaults to 0.1.

        Inputs to forward():
            features_C (Tensor): Query features of shape (Batch, Seq_C, d_model).
            features_B_real (Tensor): Key features of shape (Batch, Seq_B_real, d_model),
                excluding non-physical tokens (like Attention Sinks) for the probe.
            features_B_full (Tensor): Key/Value features of shape (Batch, Seq_B_full, d_model),
                including all tokens for the final transfer.
            raw_hp_C (Tensor): Physical coordinates of C, shape (Batch, Seq_C, d_hp).
            raw_hp_B (Tensor): Physical coordinates of B, shape (Batch, Seq_B_real, d_hp).
            has_sink (bool, optional): Whether to append a learned bias column for an attention sink.
            key_padding_mask (Tensor, optional): Standard boolean mask for padded keys.
        """

    def __init__(self, d_model, nhead, d_probe=16, d_hp=1, dropout=0.1):
        super().__init__()
        self.num_heads = nhead

        # ==========================================
        # STAGE 1: The Probe (Meta-Attention)
        # ==========================================
        self.probe_q = nn.Linear(d_model, d_probe)
        self.probe_k = nn.Linear(d_model, d_probe)
        self.scale_probe = math.sqrt(d_probe)

        # Predicts gamma (search width) based on the inferred shift magnitude
        self.gamma_predictor = nn.Sequential(
            nn.Linear(d_hp, 16),
            nn.ReLU(),
            nn.Linear(16, nhead)
        )

        # ==========================================
        # STAGE 2: The Transfer (Main Attention)
        # ==========================================
        # We use batch_first=True internally for this module
        self.main_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.sink_bias = nn.Parameter(torch.zeros(nhead))

    def forward(self, features_C, features_B_real, features_B_full, raw_hp_C, raw_hp_B, has_sink=False,
                key_padding_mask=None):
        B, Tc, d_model = features_C.shape
        _, Tb_real, _ = features_B_real.shape

        # ==========================================
        # 1. THE PROBE: Where does C match in B?
        # ==========================================
        Q_probe = self.probe_q(features_C)  # (Batch, Tc, d_probe)
        K_probe = self.probe_k(features_B_real)  # (Batch, Tb_real, d_probe)

        probe_scores = torch.bmm(Q_probe, K_probe.transpose(1, 2)) / self.scale_probe

        # Softmax to get probabilities (ignoring padding for the probe for simplicity,
        # though you could apply key_padding_mask here if desired)
        probe_attn = F.softmax(probe_scores, dim=-1)

        # ==========================================
        # 2. THE INFERENCE: Extract Mu and Gamma
        # ==========================================
        # Expected physical location = Sum of (Probabilities * Physical Coordinates)
        inferred_B_locs = torch.bmm(probe_attn, raw_hp_B)  # (Batch, Tc, d_hp)

        # Shift = Where B is - Where C is
        inferred_shift = inferred_B_locs - raw_hp_C

        # Predict Gamma (Strictness)
        gamma = F.softplus(self.gamma_predictor(inferred_shift))  # (Batch, Tc, Heads)

        # ==========================================
        # 3. BUILD THE RBF MASK
        # ==========================================
        delta = raw_hp_C.unsqueeze(2) - raw_hp_B.unsqueeze(1)  # (B, Tc, Tb_real, d_hp)

        mu = inferred_shift.unsqueeze(2).view(B, 1, Tc, 1, -1)
        delta_expanded = delta.unsqueeze(1)

        shifted_dist_sq = ((delta_expanded - mu) ** 2).sum(dim=-1)  # (B, Heads, Tc, Tb_real)

        gamma_expanded = gamma.transpose(1, 2).unsqueeze(-1)  # (B, Heads, Tc, 1)

        bias = -gamma_expanded * shifted_dist_sq

        # Handle Sink appending for the mask
        if has_sink:
            sink_col = self.sink_bias.view(1, self.num_heads, 1, 1).expand(B, self.num_heads, Tc, 1)
            bias = torch.cat([bias, sink_col], dim=-1)  # (B, Heads, Tc, Tb_real + 1)

        bias_mask = bias.reshape(B * self.num_heads, Tc, -1)

        # ==========================================
        # 4. THE TRANSFER: Main Heavy Attention
        # ==========================================
        # We pass features_B_full here so the sink token is available to be attended to
        out, attn_weights = self.main_attn(
            query=features_C,
            key=features_B_full,
            value=features_B_full,
            key_padding_mask=key_padding_mask,
            attn_mask=bias_mask
        )

        return out, attn_weights


def apply_adain(content, style, eps=1e-5):
    """
    Applies AdaIN across the sequence dimension (dim=0).
    Expects shapes: (seq, batch, dim)
    """
    # 1. Compute statistics over the sequence length
    c_mean = content.mean(dim=0, keepdim=True)
    c_std = content.std(dim=0, keepdim=True) + eps

    s_mean = style.mean(dim=0, keepdim=True)
    s_std = style.std(dim=0, keepdim=True) + eps

    # 2. Normalize content, then scale and shift to match style
    normalized_content = (content - c_mean) / c_std
    return (normalized_content * s_std) + s_mean


# TODO what about gating against negative transfer?
class PreNormTriStreamTransformerLayer(nn.Module):
    """
    This class tries to learn the diffeomorphism between C and B
    """

    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=False, use_hp=False,
                 gate_type=None, use_gate=False, use_add_pfn=True, use_post_attn=False, cross_attn_type='gated_deform',
                 num_align_steps=3,
                 use_spectral_norm=False) -> None:
        super().__init__()
        batch_first = False
        self.use_hp = use_hp

        # Shared self-attention and conditional cross-attention
        self.use_post_attn = use_post_attn
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.cross_attn_type = cross_attn_type
        if cross_attn_type == 'meta':
            self.cross_attn = MetaWarpAttention(d_model, nhead, d_probe=64,
                                                d_hp=1)  # Using the full d_model as d_hp for maximum expressivity

        elif cross_attn_type == 'deform':
            self.num_align_steps = num_align_steps
            assert use_hp == False, 'use_hp == False, currently not supported!'
            # Attention to find correspondences between B and C
            self.align_attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)

            # MLP to predict the latent shift vector based on the correspondence
            if use_spectral_norm:
                import torch.nn.utils.parametrizations as param

                # Inside your cross_attn_type init:
                self.align_ffn = nn.Sequential(
                    param.spectral_norm(nn.Linear(d_model * 2, d_model * 2)),
                    nn.GELU(),
                    param.spectral_norm(nn.Linear(d_model * 2, d_model * 2))
                )
            else:
                self.align_ffn = nn.Sequential(
                    nn.Linear(d_model * 3, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model * 2)
                )
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        elif cross_attn_type == 'gated_deform':
            self.num_align_steps = num_align_steps
            assert use_hp == False, 'use_hp == False, currently not supported!'

            # Attention to find correspondences between B and C
            self.align_attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)

            self.valve_gain = nn.Parameter(torch.tensor(1.0))

            # MLP to predict the latent shift vector based on the correspondence
            if use_spectral_norm:
                import torch.nn.utils.parametrizations as param
                self.align_ffn = nn.Sequential(
                    param.spectral_norm(nn.Linear(d_model * 2, d_model * 2)),
                    nn.GELU(),
                    param.spectral_norm(nn.Linear(d_model * 2, d_model * 2))
                )
            else:
                self.align_ffn = nn.Sequential(
                    nn.Linear(d_model * 3, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model * 2)
                )

            # ==========================================
            # NEW: The Valve Controller
            # Inputs: [mean_log_var, mean_warp_energy] (Size 2)
            # Output: gate_scale (Size 1)
            # ==========================================
            self.valve_controller = nn.Sequential(
                nn.Linear(2, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                # nn.Softplus()  # Ensures the scale magnitude is always positive
            )

            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        elif cross_attn_type == 'cycle':
            assert use_hp == False, 'use_hp == False, currently not supported!'
            self.align_attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)

            # FIX: Input is only d_model * 2 (Token + Context)
            self.align_ffn = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model * 2)
            )
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        elif cross_attn_type == 'simple':
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        else:
            raise NotImplementedError("Unknown cross_attn_type")

        self.use_add_pfn = use_add_pfn
        if self.use_add_pfn:
            self.pfn_layer = PFNLayer(d_model, nhead, dim_feedforward=dim_feedforward)

        self.use_gate = use_gate
        self.gate_type = gate_type
        if self.gate_type == 'hard_sigmoid':
            self.gate_module = HardSigmoidGate(d_model)

        elif self.gate_type == 'gumbel':
            # 2 * d_model because input is concatenated
            self.gate_module = GumbelGate(2 * d_model, hard=False)  # Start soft, maybe flip to hard later
            self.tau = 1.0
        elif self.gate_type == "sigmoid":
            # Standard Sigmoid fallback
            self.gate_linear = Linear(2 * d_model, 1)
            nn.init.constant_(self.gate_linear.bias, 1.0)

        # Shared Feedforward
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Pre-Norms
        self.norm1 = LayerNorm(d_model)  # Pre Self-Attention
        self.norm_cross = LayerNorm(d_model)  # Pre Cross-Attention
        self.norm2 = LayerNorm(d_model)  # Pre Feedforward

        self.dropout1 = Dropout(dropout)
        self.dropout_cross = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.gate_linear = Linear(2 * d_model, 1)  # For the coherence check gating mechanism
        nn.init.constant_(self.gate_linear.bias,
                          1.0)  # Initialize bias to 1.0 to encourage initially trusting the cross-attention

        self.activation = F.relu

        self.use_B_attn_sink = use_B_attn_sink
        if use_B_attn_sink:
            self.B_attn_sink = nn.Parameter(
                torch.randn(1, 1, d_model))  # Learned token for C to attend to if it wants to ignore B

    def _apply_self_attention(self, src: Tensor, eval_pos: int, pad_mask: Optional[Tensor]) -> Tensor:
        train_part = src[:eval_pos, :, :]
        test_part = src[eval_pos:, :, :]
        train_pad_mask = pad_mask[:, :eval_pos] if pad_mask is not None else None

        train_out = self.self_attn(train_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]
        test_out = self.self_attn(test_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]

        return torch.cat([train_out, test_out], dim=0)

    def forward(
            self,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            hp_A: Tensor,
            hp_B: Tensor,
            hp_C: Tensor,
            sep: int,
            raw_hp_A: Tensor = None,
            raw_hp_B: Tensor = None,
            raw_hp_C: Tensor = None,
            pad_mask_A: Optional[Tensor] = None,
            pad_mask_B: Optional[Tensor] = None
    ):

        if self.use_hp:
            A = A + hp_A
            B = B + hp_B
            C = C + hp_C

        if self.use_add_pfn:
            A, B, C = self.pfn_layer(A, B, C, sep, pad_mask_A, pad_mask_B)

        # ==========================================
        # 1. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        if not self.use_post_attn:  # Consider: this is the most successful under the detached
            # normed_A = self.norm1(A)
            normed_B = self.norm1(B)
            normed_C = self.norm1(C)

            # C has the exact same structure/padding as A
            # FIXME: we could also just stack them and do one big attention call!
            # src2_A = self._apply_self_attention(normed_A, single_eval_pos, pad_mask_A)
            src2_B = self._apply_self_attention(normed_B, sep, pad_mask_B)
            src2_C = self._apply_self_attention(normed_C, sep, pad_mask_A)

            # A = A + self.dropout1(src2_A)
            B = B + self.dropout1(src2_B)
            C = C + self.dropout1(src2_C)

        # ==========================================
        # 2. CONDITIONAL CROSS-ATTENTION (Pre-Norm)
        # ==========================================
        # A and B do NOT cross attend. They bypass this block entirely.
        normed_cross_C = self.norm_cross(C)
        normed_cross_B = self.norm_cross(B)

        # B's train part serves as the memory
        B_train = normed_cross_B[:sep, :, :]
        pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None

        if self.use_B_attn_sink:
            batch_size = B_train.shape[1]
            # we append one learned token as an escape valve for C to not attend to B
            B_train = torch.cat([B_train, self.B_attn_sink.expand(1, batch_size, -1)], dim=0)

            if pad_mask_B is not None:
                pad_mask_B_train = torch.cat([pad_mask_B_train, torch.zeros(B_train.size(0), 1, ...)], dim=1)

        # TODO can we take the pre-sigmoid weights and use them as signal for the gating? wouldn't that measure the data support?
        if self.cross_attn_type == 'meta':
            # 1. Convert features to Batch-First: (Batch, Seq, Dim)
            features_C_batch = normed_cross_C.transpose(0, 1)
            features_B_full_batch = B_train.transpose(0, 1)  # Contains the sink if use_B_attn_sink=True

            # 2. Convert coordinates to Batch-First and slice B to match 'sep'
            raw_hp_C_batch = raw_hp_C.transpose(0, 1)
            raw_hp_B_batch = raw_hp_B[:sep, :, :].transpose(0, 1)

            # 3. Extract purely physical B features for the Probe (exclude the sink token)
            features_B_real_batch = features_B_full_batch[:, :sep, :]

            # 4. Execute Meta-Attention
            cross_C_batch, cross_attn_weights = self.cross_attn(
                features_C=features_C_batch,
                features_B_real=features_B_real_batch,  # Used to find the shift
                features_B_full=features_B_full_batch,  # Used for the actual feature transfer
                raw_hp_C=raw_hp_C_batch,
                raw_hp_B=raw_hp_B_batch,
                has_sink=self.use_B_attn_sink,
                key_padding_mask=pad_mask_B_train  # Pass the padding mask through!
            )

            # 5. Convert back to Seq-First: (Seq, Batch, Dim) for the residual connection
            cross_C = cross_C_batch.transpose(0, 1)

        elif self.cross_attn_type == 'deform':
            # Handle the training-time concatenated B
            if B.shape[1] != A.shape[1]:
                half_batch = B.shape[1] // 2
                B_warped = normed_cross_B[:, :half_batch, :]
                B_in_domain_A_true = normed_cross_B[:, half_batch:, :]
            else:
                B_warped = normed_cross_B
                B_in_domain_A_true = None

            B_train_pred = B_warped[:sep, :, :].clone()
            C_train = normed_cross_C[:sep, :, :]
            pad_mask_C_train = pad_mask_A[:, :sep] if pad_mask_A is not None else None

            # ==========================================
            # NEW: Pre-Conditioning with AdaIN
            # Transfer the global domain statistics of C onto B
            # ==========================================
            B_train_pred = apply_adain(content=B_train_pred, style=C_train)

            total_kl_loss = 0.0

            # ==========================================
            # THE RECURRENT ALIGNMENT LOOP
            # ==========================================
            for step in range(self.num_align_steps):
                # 1. Correspondence Search
                align_context, _ = self.align_attn(
                    query=B_train_pred,
                    key=C_train,
                    value=C_train,
                    key_padding_mask=pad_mask_C_train
                )

                # 2. Extract Global Context (Point 1)
                # Mean over the sequence dimension (0)
                global_context = align_context.mean(dim=0, keepdim=True)
                # Expand to match sequence length
                global_context_expanded = global_context.expand_as(B_train_pred)

                # 3. Combine Features
                # (Optional: If adding raw_hp_B, concatenate it here as a 4th element)
                align_features = torch.cat([
                    B_train_pred,
                    align_context,
                    global_context_expanded
                ], dim=-1)

                # 4. Predict Uncertainty (VIB)
                B_delta_params = self.align_ffn(align_features)
                mu, log_var = torch.chunk(B_delta_params, chunks=2, dim=-1)

                # 5. Reparameterization Trick & Shift
                if self.training:
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    B_delta = mu + eps * std

                    # Accumulate KL Divergence across recurrent steps
                    step_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
                    total_kl_loss += step_kl.mean()
                else:
                    B_delta = mu

                # 6. Apply Shift (Updates B_train_pred for the next loop)
                B_train_pred = B_train_pred + B_delta

            # ==========================================
            # END LOOP
            # ==========================================

            # Log to MetaContext
            if B_in_domain_A_true is not None:
                B_train_true = B_in_domain_A_true[:sep, :, :]

                # We track the final prediction and the accumulated KL loss
                ForwardMetaContext.set('B_in_A_domain', {
                    'pred': B_train_pred,
                    'true': B_train_true,
                    'kl_loss': total_kl_loss / self.num_align_steps  # Average KL per step
                })

            # Main Cross-Attention uses the fully refined, stochastic prediction
            pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None
            cross_C, cross_attn_weights = self.cross_attn(
                query=normed_cross_C,
                key=B_train_pred,
                value=B_train_pred,
                key_padding_mask=pad_mask_B_train
            )

        elif self.cross_attn_type == 'gated_deform':
            """
            Implements a Gated Deformable Cross-Attention mechanism using Variational Information 
            Bottleneck (VIB) statistics to dynamically route attention away from distorted tokens. 

            This branch aligns a source sequence (B) to a target sequence (A/C) through a recurrent 
            warping loop. It tracks the uncertainty (`log_var`) and shift energy (`B_delta`) of this 
            transformation. These statistics are fed into a learned 'Valve Controller' to scale an 
            orthogonalized Attention Sink (the 'Null Key'). 

            When the structural warp is highly uncertain or energetic, the Sink Key's magnitude 
            increases, overshadowing the distorted B tokens in the Softmax competition. Because the 
            Sink Token is paired with a strictly zero-valued Value vector, this effectively closes 
            the attention valve, allowing the original target sequence to bypass the cross-attention 
            unchanged via the module's residual connection.
            """
            # Handle the training-time concatenated B
            if B.shape[1] != A.shape[1]:
                half_batch = B.shape[1] // 2
                B_warped = normed_cross_B[:, :half_batch, :]
                B_in_domain_A_true = normed_cross_B[:, half_batch:, :]
            else:
                B_warped = normed_cross_B
                B_in_domain_A_true = None

            B_train_pred = B_warped[:sep, :, :].clone()
            C_train = normed_cross_C[:sep, :, :]
            pad_mask_C_train = pad_mask_A[:, :sep] if pad_mask_A is not None else None

            # ==========================================
            # NEW: Pre-Conditioning with AdaIN
            # Transfer the global domain statistics of C onto B
            # ==========================================
            B_train_pred = apply_adain(content=B_train_pred, style=C_train)

            total_kl_loss = 0.0

            # ==========================================
            # THE RECURRENT ALIGNMENT LOOP
            # ==========================================
            for step in range(self.num_align_steps):
                # 1. Correspondence Search
                align_context, _ = self.align_attn(
                    query=B_train_pred,
                    key=C_train,
                    value=C_train,
                    key_padding_mask=pad_mask_C_train
                )

                # 2. Extract Global Context
                global_context = align_context.mean(dim=0, keepdim=True)
                global_context_expanded = global_context.expand_as(B_train_pred)

                # 3. Combine Features
                align_features = torch.cat([
                    B_train_pred,
                    align_context,
                    global_context_expanded
                ], dim=-1)

                # 4. Predict Uncertainty (VIB)
                B_delta_params = self.align_ffn(align_features)
                mu, log_var = torch.chunk(B_delta_params, chunks=2, dim=-1)

                # 5. Reparameterization Trick & Shift
                if self.training:
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn_like(std)
                    B_delta = mu + eps * std

                    step_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
                    total_kl_loss += step_kl.mean()
                else:
                    B_delta = mu

                # 6. Apply Shift
                B_train_pred = B_train_pred + B_delta

            # Log to MetaContext
            if B_in_domain_A_true is not None:
                B_train_true = B_in_domain_A_true[:sep, :, :]
                ForwardMetaContext.set('B_in_A_domain', {
                    'pred': B_train_pred,
                    'true': B_train_true,
                    'kl_loss': total_kl_loss / self.num_align_steps
                })

            # ==========================================
            # THE NULL-ATTENTION VALVE (NEW)
            # ==========================================
            batch_size = B_train_pred.shape[1]

            # 1. Calculate Global Warp Statistics for the Valve Controller
            # We collapse across the sequence (dim=0) and feature (dim=2) dimensions
            # to get a single scalar per batch item.
            mean_log_var = log_var.mean(dim=[0, 2])  # Shape: (batch,)
            mean_energy = B_delta.pow(2).mean(dim=[0, 2])  # Shape: (batch,)

            # 2. Predict the Gate Scale
            valve_inputs = torch.stack([mean_log_var, mean_energy], dim=-1)  # (batch, 2)
            raw_gate = self.valve_controller(valve_inputs)  # (batch, 1)
            gate_scale = torch.exp(raw_gate * self.valve_gain).unsqueeze(0)
            # gate_scale = gate_scale.unsqueeze(0)  # (1, batch, 1) to broadcast over d_model

            # 3. Define the Sink Key Base (The Ideal Query)
            k_sink_base = normed_cross_C.mean(dim=0, keepdim=True)  # (1, batch, d_model)

            # 4. The "Anti-B" Mean Projection (Fast Orthogonalization)
            mu_B = B_train_pred.mean(dim=0, keepdim=True)  # (1, batch, d_model)

            # dot_product and norm_sq shape: (1, batch, 1)
            dot_product = (k_sink_base * mu_B).sum(dim=-1, keepdim=True)
            norm_sq = (mu_B * mu_B).sum(dim=-1, keepdim=True) + 1e-8  # Add epsilon

            # Subtract the component of k_sink_base that aligns with B
            k_null_dir = k_sink_base - (dot_product / norm_sq) * mu_B

            # 5. Apply the Valve Scale
            k_null = k_null_dir * gate_scale  # (1, batch, d_model)

            # 6. Define the Zero-Value Vector
            v_null = torch.zeros_like(k_null)  # (1, batch, d_model)

            # 7. Concatenate Keys and Values separately
            # B_train_pred is (seq, batch, dim)
            K_train_with_sink = torch.cat([B_train_pred, k_null], dim=0)  # (seq + 1, batch, dim)
            V_train_with_sink = torch.cat([B_train_pred, v_null], dim=0)  # (seq + 1, batch, dim)

            # 8. Update the Padding Mask (Don't mask the sink!)
            pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None
            if pad_mask_B_train is not None:
                # Append 'False' (0) for the sink token across the batch
                # Assuming pad_mask_B_train shape is (batch, seq)
                sink_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=pad_mask_B_train.device)
                pad_mask_B_train = torch.cat([pad_mask_B_train, sink_mask], dim=1)

            # ==========================================
            # FINAL CROSS-ATTENTION
            # ==========================================
            cross_C, cross_attn_weights = self.cross_attn(
                query=normed_cross_C,
                key=K_train_with_sink,  # Uses the appended k_null
                value=V_train_with_sink,  # Uses the appended v_null (zeros)
                key_padding_mask=pad_mask_B_train
            )

        elif self.cross_attn_type == 'cycle':
            # Adain for feature matching between B & C (originally used in neural style transfer)
            # 1. Handle training/inference split
            if B.shape[1] != A.shape[1]:
                half_batch = B.shape[1] // 2
                B_warped = normed_cross_B[:, :half_batch, :]
                B_in_domain_A_true = normed_cross_B[:, half_batch:, :]
            else:
                B_warped = normed_cross_B
                B_in_domain_A_true = None

            B_train = B_warped[:sep, :, :]
            C_train = normed_cross_C[:sep, :, :]
            pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None
            pad_mask_C_train = pad_mask_A[:, :sep] if pad_mask_A is not None else None

            # ==========================================
            # 1. FULL ADAIN GLOBAL ALIGNMENT (Mean & Variance)
            # ==========================================
            # Safely calculate Mean and Std for C (ignoring padded NaNs)
            if pad_mask_C_train is not None:
                valid_mask_C = (~pad_mask_C_train).unsqueeze(-1).float().transpose(0, 1)  # (Seq, Batch, 1)
                count_C = valid_mask_C.sum(dim=0, keepdim=True).clamp(min=1.0)

                mean_C = (C_train * valid_mask_C).sum(dim=0, keepdim=True) / count_C
                var_C = (((C_train - mean_C) ** 2) * valid_mask_C).sum(dim=0, keepdim=True) / count_C
                std_C = torch.sqrt(var_C + 1e-5)
            else:
                mean_C = C_train.mean(dim=0, keepdim=True)
                std_C = C_train.std(dim=0, keepdim=True) + 1e-5

            # B is dense, so standard calculations work
            mean_B = B_train.mean(dim=0, keepdim=True)
            std_B = B_train.std(dim=0, keepdim=True) + 1e-5

            # Full AdaIN Transformation: Match Mean AND Spread
            B_train_aligned = ((B_train - mean_B) / std_B) * std_C + mean_C

            # ==========================================
            # 2. FORWARD PASS (B -> C)
            # ==========================================
            align_context_fwd, _ = self.align_attn(
                query=B_train_aligned,
                key=C_train,
                value=C_train,
                key_padding_mask=pad_mask_C_train
            )

            fwd_features = torch.cat([B_train_aligned, align_context_fwd], dim=-1)
            mu_fwd, log_var_fwd = torch.chunk(self.align_ffn(fwd_features), 2, dim=-1)

            # ==========================================
            # 3. BACKWARD PASS (True Cycle Consistency)
            # ==========================================
            cycle_loss = torch.tensor(0.0, device=B_train.device)

            if self.training:
                # We map B_aligned to its intermediate "Fake C" position
                B_fake_C = B_train_aligned + mu_fwd

                # Backward Pass: Fake C looks back at the original B_aligned
                # Notice the Query is B_fake_C, but Key/Value is B_train_aligned
                align_context_back, _ = self.align_attn(
                    query=B_fake_C,
                    key=B_train_aligned,
                    value=B_train_aligned,
                    key_padding_mask=pad_mask_B_train
                )

                back_features = torch.cat([B_fake_C, align_context_back], dim=-1)
                mu_back, _ = torch.chunk(self.align_ffn(back_features), 2, dim=-1)

                # TRUE Cycle Constraint: mu_fwd and mu_back are evaluated on the exact
                # same physical token sequence (B_train), so element-wise MSE is perfect.
                cycle_loss = F.mse_loss(mu_fwd, -mu_back)

            # ==========================================
            # 4. REPARAMETERIZATION & LOGGING
            # ==========================================
            if self.training:
                std = torch.exp(0.5 * log_var_fwd)
                eps = torch.randn_like(std)
                B_delta = mu_fwd + eps * std
            else:
                B_delta = mu_fwd

            B_in_domain_A_pred = B_train_aligned + B_delta

            if B_in_domain_A_true is not None:
                kl_loss = -0.5 * torch.sum(1 + log_var_fwd - mu_fwd.pow(2) - log_var_fwd.exp(), dim=-1).mean()
                ForwardMetaContext.set('B_in_A_domain', {
                    'pred': B_in_domain_A_pred,
                    'true': B_in_domain_A_true[:sep, :, :],
                    'kl_loss': kl_loss,
                    'cycle_loss': cycle_loss
                })

            # ==========================================
            # 5. MAIN CROSS-ATTENTION
            # ==========================================
            pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None

            cross_C, _ = self.cross_attn(
                query=normed_cross_C,
                key=B_in_domain_A_pred,
                value=B_in_domain_A_pred,
                key_padding_mask=pad_mask_B_train
            )
        else:
            cross_C, cross_attn_weights = self.cross_attn(
                query=normed_cross_C,
                # Avoiding task based gradient interference, we detach B here and get better uncertainty bounds
                key=B_train,
                value=B_train,
                key_padding_mask=pad_mask_B_train
            )

        # # --------------------------------------
        # # Adjusted Attn Entropy (Thermodynamic Energy)
        # # --------------------------------------
        # # If the space is perfectly aligned (just shifted/scaled), C should look at B and instantly know exactly which
        # # token it corresponds to. The attention distribution will be a sharp peak (a Dirac delta).If the spaces are
        # # highly warped, the relationships are ambiguous. C will hedge its bets and attend softly to many tokens in B.
        # # In thermodynamics, this spreading of state is higher entropy.How to measure it: Calculate the Shannon Entropy
        # # of the cross-attention weights.
        # # $$E_{entropy} = -\frac{1}{N} \sum_{i} \sum_{j} A_{i,j} \log(A_{i,j} + \epsilon)$$
        #
        # # cross_attn_weights: (B, Heads, Tc, Tb)
        # epsilon = 1e-9
        #
        # # Entropy over the source/key dimension (Tb)
        # # Resulting shape: (B, Heads, Tc)
        # entropy = - (cross_attn_weights * torch.log(cross_attn_weights + epsilon)).sum(dim=-1)
        #
        # # Mean over all dimensions to get a single scalar for logging
        # thermodynamic_energy = entropy.mean()
        #
        # ForwardMetaContext.set('thermodynamic_energy', thermodynamic_energy)

        # --------------------------------------
        # Measure Transport Energy
        # --------------------------------------
        #  The Transport Work (The "Earth Mover's" Energy)Since you know the physical/hyperparameter locations
        #  (hp_A / hp_C and hp_B), you can measure the physical "distance" the attention mechanism has to reach
        #  across to find a match.If B is just a shifted A, a perfectly unwarped mapping would simply be a strict
        #  diagonal attention matrix offset by the shift $\Delta = hp_B - hp_C$. If the attention spreads out or
        #  reaches to completely wrong hp locations, the space is heavily warped.How to measure it:Treat the
        #  cross-attention matrix as a transport plan (or transition probability matrix), and calculate the
        #  Wasserstein-like expected distance:
        #  $$E_{transport} = \sum_{i} \sum_{j} A_{i,j} \cdot \mathcal{D}(hp_{C_i}, hp_{B_j})^2$$Where $A_{i,j}$
        #  is the cross-attention weight from C's $i$-th token to B's $j$-th token, and $\mathcal{D}$ is
        #  the distance between their known hyperparameter coordinates.
        #
        # # 1. Permute HP tensors to (Batch, Seq, Dhp) for cdist
        # hp_B_train = hp_B[:sep, :, :]
        #
        # # 1. Permute HP tensors to (Batch, Seq, Dhp) for cdist
        # hp_C_perm = hp_C.permute(1, 0, 2)
        # hp_B_train_perm = hp_B_train.permute(1, 0, 2)
        #
        # # 2. Compute the distance/cost matrix: (Batch, Tc, sep)
        # dist_matrix = torch.cdist(hp_C_perm, hp_B_train_perm, p=2)
        # cost_matrix = dist_matrix ** 2
        #
        # # 3. Check if cross_attn_weights is 3D or 4D
        # if cross_attn_weights.dim() == 4:
        #     # (B, Heads, Tc, sep+1) -> (B, Tc, sep+1)
        #     mean_attn = cross_attn_weights.mean(dim=1)
        # else:
        #     # (B, Tc, sep+1)
        #     mean_attn = cross_attn_weights
        #
        # # 4. Slice the last dimension to remove the sink
        # tb_len = cost_matrix.size(-1)  # This is now exactly `sep`
        # mean_attn_no_sink = mean_attn[:, :, :tb_len]
        #
        # # 5. Energy calculation
        # # Both tensors are now cleanly (Batch, Tc, sep)
        # transport_energy = (mean_attn_no_sink * cost_matrix).sum(dim=-1).mean()
        #
        # ForwardMetaContext.set('transport_energy', transport_energy)
        #
        # # ------------------------------------------
        #
        # # identify the sink relative weight
        # # if self.use_B_attn_sink:
        # ForwardMetaContext.set('cross_attn_weights', cross_attn_weights)  # Store for later analysis

        # ------------------------------------------
        # Update strength (Kinetic Energy)
        # ------------------------------------------
        # If you want to measure the energy strictly in the embedding space (rather than the sequence/hp space),
        # look at the residual update itself.The cross-attention outputs a vector cross_C which is then added to C.
        # This vector represents the literal mathematical "force" applied to C to pull it towards B's manifold.
        # How to measure it:Calculate the Relative L2 Norm of the update.
        # $$E_{kinetic} = \frac{1}{N} \sum \frac{\| \text{cross\_C} \|_2}{\| C \|_2}$$
        # C is the pre-residual state, cross_C is the attention output

        # # Calculate L2 norm along the feature dimension (dim=-1)
        # # These results will be (Tc, B)
        # update_norm = torch.norm(cross_C, p=2, dim=-1)
        # state_norm = torch.norm(C, p=2, dim=-1)
        #
        # # Relative magnitude of the warp
        # # Average across Sequence (dim=0) and Batch (dim=1)
        # kinetic_energy = (update_norm / (state_norm + 1e-8)).mean()
        #
        # ForwardMetaContext.set('kinetic_energy', kinetic_energy)

        # ==========================================
        # 3. THE TASK-LEVEL COHERENCE CHECK
        # ==========================================
        if self.use_gate:
            global_target = normed_cross_C.mean(dim=0)
            global_proposal = cross_C.mean(dim=0)
            gate_input = torch.cat([global_target, global_proposal], dim=-1)  # (Batch, 2D)

            # Evaluate the chosen gate type
            if self.gate_type == 'hard_sigmoid':
                # HardSigmoidGate handles setting the 'gate_logits' internally
                gate_scalar = self.gate_module(gate_input)

            elif self.gate_type == 'gumbel':
                gate_scalar, gate_logits = self.gate_module(gate_input, tau=self.tau, training=self.training)
                ForwardMetaContext.set('gate_logits', gate_logits)

            elif self.gate_type == 'sigmoid':
                gate_logits = self.gate_linear(gate_input)
                ForwardMetaContext.set('gate_logits', gate_logits)
                gate_scalar = torch.sigmoid(gate_logits)

            # Store the activated probability for analysis
            ForwardMetaContext.set('gate', gate_scalar)

            # Broadcast the scalar back across the sequence dimension -> (1, Batch, 1)
            gate_broadcast = gate_scalar.unsqueeze(0)

            # Apply the gated residual
            C = C + self.dropout_cross(gate_broadcast * cross_C)

        else:
            C = C + self.dropout_cross(cross_C)

        # ==========================================
        # 3. SHARED FEEDFORWARD (Pre-Norm)
        # ==========================================
        # normed_ff_A = self.norm2(A)
        normed_ff_B = self.norm2(B)
        normed_ff_C = self.norm2(C)

        # Helper for the MLP
        def ff_block(x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        # A = A + self.dropout2(ff_block(normed_ff_A))
        B = B + self.dropout2(ff_block(normed_ff_B))
        C = C + self.dropout2(ff_block(normed_ff_C))

        if self.use_post_attn:  # fixme: move this layer into the main model and detach C for it!
            normed_C = self.norm1(C)
            src2_C = self._apply_self_attention(normed_C, sep, pad_mask_A)
            C = C + self.dropout1(src2_C)

        return A, B, C


class FourierEncoder(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super().__init__()
        # Initialize random frequencies
        self.frequencies = nn.Parameter(torch.randn(1, d_model // 2) * sigma, requires_grad=False)
        self.linear = nn.Linear(d_model, d_model)  # Optional mixing layer

    def forward(self, x):
        # x is (Seq, Batch, 1)
        scaled_x = 2 * torch.pi * x @ self.frequencies
        fourier_features = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return self.linear(fourier_features)


class TriHarmonicModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_B_attn_sink=False, use_freq_enc_x=True,
                 use_post_attn=True):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = FourierEncoder(d_model) if use_freq_enc_x else nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        self.pfn_layer = PFNLayer(d_model, nhead=nhead, dropout=dropout)
        self.layer = PreNormTriStreamTransformerLayer(d_model, nhead=nhead, use_B_attn_sink=use_B_attn_sink)

        self.use_post_attn = use_post_attn

        if self.use_post_attn:
            self.pfn_layer2 = PFNLayer(d_model, nhead=nhead, dropout=dropout)

        # Final norm for Pre-Norm architecture
        self.final_norm = LayerNorm(d_model)

        # Shared decoder
        self.decoder = nn.Linear(d_model, num_bars)

    def forward(self, batch):
        X_train_A, Y_train_A = batch['train']['X_A'], batch['train']['Y_A']
        X_train_B, Y_train_B = batch['train']['X_B'], batch['train']['Y_B']
        X_test_A = batch['test']['X_A']
        X_test_B = batch['test']['X_B']

        single_eval_pos = batch['train']['X_B'].shape[0]

        # Concat train and test X
        X_A = torch.cat([X_train_A, X_test_A], dim=0)
        X_B = torch.cat([X_train_B, X_test_B], dim=0)

        pad_mask_A = torch.isnan(X_A).transpose(0, 1)

        # Clean NaNs
        X_A_clean = torch.nan_to_num(X_A, nan=0.0).unsqueeze(-1)
        X_B_clean = torch.nan_to_num(X_B, nan=0.0).unsqueeze(-1)
        Y_A_train_clean = torch.nan_to_num(Y_train_A, nan=0.0).unsqueeze(-1)
        Y_B_train_clean = torch.nan_to_num(Y_train_B, nan=0.0).unsqueeze(-1)

        # Encode
        emb_X_A = self.x_encoder(X_A_clean)
        emb_X_B = self.x_encoder(X_B_clean)
        emb_Y_A = self.y_encoder(Y_A_train_clean)
        emb_Y_B = self.y_encoder(Y_B_train_clean)

        # ==========================================
        # FIX: Inject Y into train positions without truncating the sequence
        # ==========================================
        A = emb_X_A.clone()
        B = emb_X_B.clone()

        A[:single_eval_pos, :, :] += emb_Y_A
        B[:single_eval_pos, :, :] += emb_Y_B

        C = A.clone()

        if 'X_B_in_A' in batch['train'].keys():
            # We are in training on the prior and can compute all of the above for this, it has the same pad as B
            X_B_train_in_A = batch['train']['X_B_in_A']
            X_B_test_in_A = batch['test']['X_B_in_A']

            X_B_train_in_A_clean = torch.nan_to_num(X_B_train_in_A, nan=0.0).unsqueeze(-1)
            X_B_test_in_A_clean = torch.nan_to_num(X_B_test_in_A, nan=0.0).unsqueeze(-1)

            X_B_in_A = torch.cat([X_B_train_in_A_clean, X_B_test_in_A_clean], dim=0)

            emb_X_B_in_A = self.x_encoder(X_B_in_A)

            Y_B_train_in_A = batch['train']['Y_B_in_A']
            Y_B_train_in_A_clean = torch.nan_to_num(Y_B_train_in_A, nan=0.0).unsqueeze(-1)
            emb_Y_B_in_A = self.y_encoder(Y_B_train_in_A_clean)

            B_in_A = emb_X_B_in_A.clone()
            B_in_A[:single_eval_pos, :, :] += emb_Y_B_in_A

            B = torch.cat([B, B_in_A], dim=1)  # parallel processing of the two.

        # Pass through the Marginal PFN Layer
        A, B, C = self.pfn_layer(A, B, C, single_eval_pos, pad_mask_A)

        # ==========================================
        # FIX: Cleaned up kwargs to match PreNormTriStreamTransformerLayer signature
        # ==========================================
        _, _, C = self.layer(
            A.detach(), B.detach(), C.detach(),
            hp_A=emb_X_A.detach(), hp_B=emb_X_B.detach(), hp_C=emb_X_A.detach(),
            sep=single_eval_pos,
            raw_hp_A=X_A_clean,
            raw_hp_B=X_B_clean,
            raw_hp_C=X_A_clean,
            pad_mask_A=pad_mask_A,
            pad_mask_B=None
        )

        if 'X_B_in_A' in batch['train'].keys():
            # drop the auxiliary that we attached to stream B
            B = B[:, :A.shape[1], :]

        if self.use_post_attn:
            A, B, C = self.pfn_layer2(A, B, C, single_eval_pos, pad_mask_A)

        # Apply final norm
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)

        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        if self.training:
            return logits_A, logits_B, logits_C

        else:
            raise NotImplementedError("Energy-Penalized BMA is still in development")

            """
            Forward pass with dynamic Energy-Penalized Bayesian Model Averaging (BMA).

            This method processes a target sequence (A/C) alongside a set of related 
            contexts (B_i) passed through the batch dimension. During inference, it 
            dynamically routes probability mass between the available conditional 
            contexts and the unconditional baseline based on their predictive confidence 
            and structural deformation cost.

            Args:
                batch (dict): Dictionary containing train/test splits for X_A, Y_A, X_B, Y_B.
                bma_lambda (float, optional): The energy penalty multiplier. Controls how 
                    harshly a context is penalized for requiring heavy structural warping. 
                    Setting this to 0.0 results in pure Predictive Entropy BMA. Defaults to 1.0.
                bma_temp (float, optional): Temperature for the BMA Softmax selection. 
                    Higher values smoothly blend contexts; lower values force a sharp, 
                    argmax-like selection of the single best context. Defaults to 1.0.

            Returns:
                tuple:
                    - logits_A (Tensor): Unconditional predictions (A only).
                    - logits_B (Tensor): Auxiliary target predictions.
                    - logits_C (Tensor): Conditional predictions (A | B).
                    - bma_probs (Tensor, inference only): The final model-averaged 
                      probability distribution across all contexts and the baseline.

            BMA Mechanics (Inference Only):
                1. Calculates the token-level Shannon entropy of the Posterior Predictive 
                   Distribution (PPD) for all conditional contexts and the unconditional baseline.
                2. Retrieves the Variational Information Bottleneck (VIB) statistics (`log_var` 
                   and `B_delta`) from the `gated_deform` cross-attention layer to compute 
                   the 'Warp Energy' required to align each context.
                3. Computes log-weights by penalizing the predictive entropy with the warp 
                   energy: Weight = -(Entropy + lambda * Energy) / Temperature.
                4. If all contexts require excessive energy, the Softmax denominator naturally 
                   routes probability mass to the unconditional baseline, which has zero 
                   warp energy by definition.
            """

            # TODO: in the deform
            #  if not self.training:
            #     # Log the energy statistics for BMA at inference
            #     ForwardMetaContext.set('vib_stats', {
            #         'mean_log_var': log_var.mean(dim=[0, 2]), # Shape: (batch,)
            #         'mean_energy': B_delta.pow(2).mean(dim=[0, 2]) # Shape: (batch,)
            #     })

            # ==========================================
            # ENERGY-PENALIZED BMA (Inference Only)
            # ==========================================

            seq_len, batch_size, num_bars = logits_C.shape

            # 1. Convert Logits to Probabilities
            probs_C = F.softmax(logits_C, dim=-1)  # Conditional (A|B_i)
            probs_A = F.softmax(logits_A, dim=-1)  # Unconditional (A)

            # Since A is expanded in the batch dim, all batch items for A are identical.
            # We just need one copy of the unconditional probability.
            probs_A_single = probs_A[:, 0:1, :]  # Shape: (seq_len, 1, num_bars)

            # 2. Calculate Predictive Entropy (Token-level)
            # H = - sum(p * log(p))
            # Adding a tiny epsilon to prevent log(0)
            entropy_C = -torch.sum(probs_C * torch.log(probs_C + 1e-9), dim=-1)  # (seq, batch)
            entropy_A = -torch.sum(probs_A_single * torch.log(probs_A_single + 1e-9), dim=-1)  # (seq, 1)

            # 3. Retrieve Warp Energy from MetaContext
            vib_stats = ForwardMetaContext.get('vib_stats')
            if vib_stats is not None:
                # Shape: (1, batch) so it broadcasts over seq_len
                mean_log_var = vib_stats['mean_log_var'].unsqueeze(0)
                mean_energy = vib_stats['mean_energy'].unsqueeze(0)

                # Total Energy Penalty for each context B_i
                # Softplus ensures the penalty is strictly positive
                warp_energy_C = F.softplus(mean_log_var + mean_energy)
            else:
                # Fallback if no warp occurred
                warp_energy_C = torch.zeros((1, batch_size), device=logits_C.device)

            # The unconditional model (A) has zero warp energy by definition
            warp_energy_A = torch.zeros((1, 1), device=logits_A.device)

            # 4. Formulate the Unnormalized Log-Weights
            # Weight = -(Entropy + lambda * Energy) / Temperature
            log_w_C = -(entropy_C + bma_lambda * warp_energy_C) / bma_temp  # (seq, batch)
            log_w_A = -(entropy_A + bma_lambda * warp_energy_A) / bma_temp  # (seq, 1)

            # 5. Concatenate all models (N conditionals + 1 unconditional)
            # Shape: (seq, batch + 1)
            all_log_weights = torch.cat([log_w_C, log_w_A], dim=1)

            # 6. Normalize weights across the model dimension using Softmax
            bma_weights = F.softmax(all_log_weights, dim=1)  # (seq, batch + 1)

            # Extract the weights for C and A
            weights_C = bma_weights[:, :-1].unsqueeze(-1)  # (seq, batch, 1)
            weights_A = bma_weights[:, -1:].unsqueeze(-1)  # (seq, 1, 1)

            # 7. Compute the Final BMA Probability Distribution
            # Sum the weighted conditional probabilities
            weighted_C_sum = torch.sum(probs_C * weights_C, dim=1, keepdim=True)  # (seq, 1, num_bars)

            # Add the weighted unconditional probability
            bma_probs = weighted_C_sum + (probs_A_single * weights_A)  # (seq, 1, num_bars)

            # Return standard logits for training, but add the BMA probs for inference
            return logits_A, logits_B, logits_C, bma_probs

