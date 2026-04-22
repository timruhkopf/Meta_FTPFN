from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, MultiheadAttention

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.harmonic_restart.pfn_layer import PFNLayer


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


class TriStreamLayer(nn.Module):
    """
    This class tries to learn the diffeomorphism between C and B
    """

    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=False, use_hp=False,
                 use_add_pfn=True, use_post_attn=False, cross_attn_type='gated_deform',
                 num_align_steps=3,
                 use_spectral_norm=False) -> None:
        super().__init__()
        batch_first = False
        self.use_hp = use_hp

        # Shared self-attention and conditional cross-attention
        self.use_post_attn = use_post_attn
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.cross_attn_type = cross_attn_type
        if cross_attn_type == 'deform':
            assert num_align_steps >= 1, "num_align_steps should be >= 1 (2 is first recursion)"
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

            assert use_hp == False, 'use_hp == False, currently not supported!'

            # Layer 1: First Cross-Attention
            self.align_attn = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout)

            # Match the capacity of deform's align_ffn (~10 * d_model^2 params).
            # We use a standard FFN expansion of 4x, which yields ~8 * d_model^2 params.
            self.align_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
            self.norm_inter = nn.LayerNorm(d_model)

            # Layer 2: Second Cross-Attention
            self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        else:
            raise NotImplementedError("Unknown cross_attn_type")

        self.use_add_pfn = use_add_pfn
        if self.use_add_pfn:
            self.pfn_layer = PFNLayer(d_model, nhead, dim_feedforward=dim_feedforward)

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
        if self.cross_attn_type == 'deform':
            # Handle the training-time concatenated B (where we actually know what B would look without corruption; i.e.
            # when B is observed in the domain of A. This obviously can not be used during inference)
            # but during training, we can actually provide a supervision signal, of where we would expect B to be on the
            # manifold at this exact point!
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

        elif self.cross_attn_type == 'simple':

            # ==========================================
            # FIX: Handle the training-time concatenated B
            # B's batch size is 2x C's batch size during training
            # ==========================================
            if B.shape[1] != A.shape[1]:
                half_batch = B.shape[1] // 2
                B_warped = normed_cross_B[:, :half_batch, :]
            else:
                B_warped = normed_cross_B

            # Extract the memory tokens
            B_train = B_warped[:sep, :, :]
            pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None
            # 1. First Cross-Attention Pass (C queries B)
            cross_C_1, _ = self.align_attn(
                query=normed_cross_C,
                key=B_train,
                value=B_train,
                key_padding_mask=pad_mask_B_train
            )

            # 2. Intermediate Processing (Matches align_ffn parameter capacity)
            # Add & Norm, then FFN, then Add.
            inter_C = self.norm_inter(normed_cross_C + cross_C_1)
            inter_C = inter_C + self.align_ffn(inter_C)

            # 3. Second Cross-Attention Pass (Refined C queries B again)
            cross_C, cross_attn_weights = self.cross_attn(
                query=inter_C,
                key=B_train,
                value=B_train,
                key_padding_mask=pad_mask_B_train
            )

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
