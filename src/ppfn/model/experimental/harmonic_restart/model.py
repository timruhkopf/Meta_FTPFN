import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor

from ppfn.model.mymodel.meta_context import ForwardMetaContext


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


# TODO what about gating against negative transfer?
class PreNormTriStreamTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=False, use_hp=False,
                 gate_type=None, use_gate=False, use_add_pfn=True, use_post_attn=False, cross_attn_type='cycle', num_align_steps=3) -> None:
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
            self.num_align_steps=num_align_steps
            assert use_hp == False, 'use_hp == False, currently not supported!'
            # Attention to find correspondences between B and C
            self.align_attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)

            # MLP to predict the latent shift vector based on the correspondence
            self.align_ffn = nn.Sequential(
                nn.Linear(d_model * 3, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model * 2)
            )
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)


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

        else: raise NotImplementedError("Unknown cross_attn_type")

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
        elif self.cross_attn_type == 'cycle':
             # Adain for feature matching between B & C (originally used in neural style transfer)
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

            # --- 1. GLOBAL ALIGNMENT (AdaIN Style) ---
            global_C = C_train.mean(dim=0, keepdim=True)
            global_B = B_train.mean(dim=0, keepdim=True)
            B_train_aligned = B_train + (global_C - global_B)

            # --- 2. BATCHED CORRESPONDENCE PASS ---
            cycle_loss = torch.tensor(0.0, device=B_train.device)

            if self.training:
                # BATCHING: Stack Forward and Backward queries along the Batch Dimension (dim=1)
                # Forward Q: B, K/V: C  |  Backward Q: C, K/V: B
                batched_Q = torch.cat([B_train_aligned, C_train], dim=1)
                batched_K = torch.cat([C_train, B_train_aligned], dim=1)

                # Batch the padding masks (Batch dimension is dim=0 for pad masks)
                if pad_mask_B_train is not None and pad_mask_C_train is not None:
                    batched_mask = torch.cat([pad_mask_C_train, pad_mask_B_train], dim=0)
                else:
                    batched_mask = None

                # ONE single attention call for both directions!
                batched_context, _ = self.align_attn(batched_Q, batched_K, batched_K, key_padding_mask=batched_mask)

                # ONE single FFN call!
                batched_features = torch.cat([batched_Q, batched_context], dim=-1)
                batched_params = self.align_ffn(batched_features)

                # SPLIT the results back into Forward and Backward
                params_fwd, params_back = torch.chunk(batched_params, chunks=2, dim=1)

                # Extract mu and log_var
                mu_fwd, log_var_fwd = torch.chunk(params_fwd, chunks=2, dim=-1)
                mu_back, _ = torch.chunk(params_back, chunks=2, dim=-1)

                # Calculate Cycle Constraint
                cycle_loss = F.mse_loss(mu_fwd, -mu_back)

            else:
                # INFERENCE ONLY: Just run the forward pass normally to save compute
                align_context_fwd, _ = self.align_attn(B_train_aligned, C_train, C_train,
                                                       key_padding_mask=pad_mask_C_train)
                fwd_features = torch.cat([B_train_aligned, align_context_fwd], dim=-1)
                params_fwd = self.align_ffn(fwd_features)
                mu_fwd, log_var_fwd = torch.chunk(params_fwd, chunks=2, dim=-1)

            # --- 3. REPARAMETERIZATION & LOGGING ---
            # Sample Delta
            delta_fwd = mu_fwd + torch.randn_like(mu_fwd) * torch.exp(0.5 * log_var_fwd) if self.training else mu_fwd
            B_in_domain_A_pred = B_train_aligned + delta_fwd

            if B_in_domain_A_true is not None:
                kl_loss = -0.5 * torch.sum(1 + log_var_fwd - mu_fwd.pow(2) - log_var_fwd.exp(), dim=-1).mean()
                ForwardMetaContext.set('B_in_A_domain', {
                    'pred': B_in_domain_A_pred,
                    'true': B_in_domain_A_true[:sep, :, :],
                    'kl_loss': kl_loss,
                    'cycle_loss': cycle_loss
                })

            # --- 4. MAIN CROSS-ATTENTION ---
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
    def __init__(self, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_B_attn_sink=False, use_freq_enc_x=True, use_post_attn=True):
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

            B = torch.cat([B, B_in_A], dim=1) # parallel processing of the two.

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

        return logits_A, logits_B, logits_C
