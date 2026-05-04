import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.harmonic_restart.pfn_layer import PFNLayer

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
    def __init__(self, cross_attn_layer, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_freq_enc_x=True,
                 use_post_attn=True, use_cross_attn=True):
        super().__init__()
        self.num_bars = num_bars
        # reducing sigma might smooth the high frequency detail in A and B?
        self.x_encoder = FourierEncoder(d_model, sigma=0.25) if use_freq_enc_x else nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        self.pfn_layer = PFNLayer(d_model, nhead=nhead, dropout=dropout)
        self.cross_layer = cross_attn_layer

        self.use_cross_attn = use_cross_attn  # to allow baseline calculations for nll scores (once)
        self.use_post_attn = use_post_attn

        if self.use_post_attn:
            self.pfn_layer2 = PFNLayer(d_model, nhead=nhead, dropout=dropout)

        # Final norm for Pre-Norm architecture
        self.final_norm = LayerNorm(d_model)

        # Shared decoder
        self.decoder = nn.Linear(d_model, num_bars)

    @property
    def is_pretraining(self):
        return not next(self.pfn_layer.parameters()).requires_grad == False


    def forward(self, batch):
        X_train_A, Y_train_A = batch['train']['X_A'], batch['train']['Y_A']
        X_train_B, Y_train_B = batch['train']['X_B'], batch['train']['Y_B']
        X_test_A = batch['test']['X_A']
        X_test_B = batch['test']['X_B']

        single_eval_pos = batch['train']['X_B'].shape[0]

        # Concat train and test X
        # fixme: fix prior to give unsqueezed version here
        X_A = torch.cat([X_train_A, X_test_A], dim=0)
        X_B = torch.cat([X_train_B, X_test_B], dim=0)

        pad_mask_A = torch.isnan(X_A).transpose(0, 1)

        # Clean NaNs
        X_A_clean = torch.nan_to_num(X_A, nan=0.0).unsqueeze(-1)
        X_B_clean = torch.nan_to_num(X_B, nan=0.0).unsqueeze(-1)
        Y_A_train_clean = torch.nan_to_num(Y_train_A, nan=0.0).unsqueeze(-1)
        Y_B_train_clean = torch.nan_to_num(Y_train_B, nan=0.0).unsqueeze(-1)

        # FIXME: use ADAIN here?

        # Encode
        emb_X_A = self.x_encoder(X_A_clean)
        emb_X_B = self.x_encoder(X_B_clean)
        emb_Y_A = self.y_encoder(Y_A_train_clean)
        emb_Y_B = self.y_encoder(Y_B_train_clean)

        A = emb_X_A.clone()
        B = emb_X_B.clone()

        batch_size = A.shape[1]

        A[:single_eval_pos, :, :] += emb_Y_A
        B[:single_eval_pos, :, :] += emb_Y_B

        if next(self.cross_layer.parameters()).requires_grad == True and 'X_A_in_B' in batch['train'].keys():
            # We want to find what the backend thinks about the in-prior known transformed version of A living in B domain.
            # This way, we can try to find an invariant representation in the cross layer
            # We are in training on the prior and can compute all of the above for this, it has the same pad as B
            # FIXME: we will want to use the previous layer's output instead!
            X_A_train_in_B = batch['train']['X_A_in_B']
            X_A_test_in_B = batch['test']['X_A_in_B']

            X_A_train_in_B_clean = torch.nan_to_num(X_A_train_in_B, nan=0.0).unsqueeze(-1)
            X_A_test_in_B_clean = torch.nan_to_num(X_A_test_in_B, nan=0.0).unsqueeze(-1)

            X_A_in_B = torch.cat([X_A_train_in_B_clean, X_A_test_in_B_clean], dim=0)

            emb_X_A_in_B = self.x_encoder(X_A_in_B)

            Y_A_train_in_B = batch['train']['Y_A_in_B']
            Y_A_train_in_B_clean = torch.nan_to_num(Y_A_train_in_B, nan=0.0).unsqueeze(-1)
            emb_Y_A_in_B = self.y_encoder(Y_A_train_in_B_clean)

            A_in_B = emb_X_A_in_B.clone()
            A_in_B[:single_eval_pos, :, :] += emb_Y_A_in_B

            A = torch.cat([A, A_in_B], dim=1)  # parallel processing of the two.
            pad_mask_A = torch.concat([pad_mask_A, pad_mask_A],
                                      dim=0)  # double the pad mask for the concatenated version
            # for cross-attn processing
            X_A = torch.cat([X_A_clean, X_A_in_B], dim=1)  # for cross-attn processing
            X_C = X_A.clone()

        if next(self.cross_layer.parameters()).requires_grad == True and 'X_B_in_A' in batch['train'].keys():
            # Same as above!
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
            # FIXME:pad_mask_B = torch.concat([pad_mask_B, pad_mask_B], dim=0)
            X_B = torch.cat([X_B_clean, X_B_in_A], dim=1)

        C = A.clone()

        # Pass through the Marginal PFN Layer
        A, B, C = self.pfn_layer(A, B, C, single_eval_pos, pad_mask_A)

        # ==========================================
        # FIX: Cleaned up kwargs to match PreNormTriStreamTransformerLayer signature
        # ==========================================
        # fixme: this will prevent joint training, but be more efficient in warmup
        if self.use_cross_attn and not self.is_pretraining:
            # overwriting A, B with pass throughs, but purged of shadow batch, which is available during training
            A, B, C = self.cross_layer(
                A.detach(), B.detach(), C.detach(),
                hp_A=emb_X_A.detach(), hp_B=emb_X_B.detach(), hp_C=emb_X_A.detach(),
                sep=single_eval_pos,
                raw_hp_A=X_A,
                raw_hp_B=X_B,
                raw_hp_C=X_C,
                pad_mask_A=pad_mask_A,
                pad_mask_B=None
            )

            # You still must manually truncate the pad_mask_A because it wasn't returned!
            true_batch_size = A.shape[1]  # A is now safely 256
            pad_mask_A = pad_mask_A[:true_batch_size, :]
            # FIXME: what about mask_B

        if 'X_B_in_A' in batch['train'].keys():
            # drop the auxiliary A in B that we attached to stream B
            B = B[:, :batch_size, :]
            # fixme: pad_mask_B = pad_mask_B[:, :batch_size] if pad_mask_B is not None else None

        if 'X_A_in_B' in batch['train'].keys():
            A = A[:, :batch_size, :]
            C = C[:, :batch_size, :]
            pad_mask_A = pad_mask_A[:batch_size]

        if self.use_post_attn:
            A, B, C = self.pfn_layer2(A, B, C, single_eval_pos, pad_mask_A) # fixme: mask_B padding!

        # Apply final norm
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)

        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        # if self.training:
        return logits_A, logits_B, logits_C

        # else:
        #     raise NotImplementedError("Energy-Penalized BMA is still in development")
        #
        #     """
        #     Forward pass with dynamic Energy-Penalized Bayesian Model Averaging (BMA).
        #
        #     This method processes a target sequence (A/C) alongside a set of related
        #     contexts (B_i) passed through the batch dimension. During inference, it
        #     dynamically routes probability mass between the available conditional
        #     contexts and the unconditional baseline based on their predictive confidence
        #     and structural deformation cost.
        #
        #     Args:
        #         batch (dict): Dictionary containing train/test splits for X_A, Y_A, X_B, Y_B.
        #         bma_lambda (float, optional): The energy penalty multiplier. Controls how
        #             harshly a context is penalized for requiring heavy structural warping.
        #             Setting this to 0.0 results in pure Predictive Entropy BMA. Defaults to 1.0.
        #         bma_temp (float, optional): Temperature for the BMA Softmax selection.
        #             Higher values smoothly blend contexts; lower values force a sharp,
        #             argmax-like selection of the single best context. Defaults to 1.0.
        #
        #     Returns:
        #         tuple:
        #             - logits_A (Tensor): Unconditional predictions (A only).
        #             - logits_B (Tensor): Auxiliary target predictions.
        #             - logits_C (Tensor): Conditional predictions (A | B).
        #             - bma_probs (Tensor, inference only): The final model-averaged
        #               probability distribution across all contexts and the baseline.
        #
        #     BMA Mechanics (Inference Only):
        #         1. Calculates the token-level Shannon entropy of the Posterior Predictive
        #            Distribution (PPD) for all conditional contexts and the unconditional baseline.
        #         2. Retrieves the Variational Information Bottleneck (VIB) statistics (`log_var`
        #            and `B_delta`) from the `gated_deform` cross-attention layer to compute
        #            the 'Warp Energy' required to align each context.
        #         3. Computes log-weights by penalizing the predictive entropy with the warp
        #            energy: Weight = -(Entropy + lambda * Energy) / Temperature.
        #         4. If all contexts require excessive energy, the Softmax denominator naturally
        #            routes probability mass to the unconditional baseline, which has zero
        #            warp energy by definition.
        #     """
        #
        #     # TODO: in the deform
        #     #  if not self.training:
        #     #     # Log the energy statistics for BMA at inference
        #     #     ForwardMetaContext.set('vib_stats', {
        #     #         'mean_log_var': log_var.mean(dim=[0, 2]), # Shape: (batch,)
        #     #         'mean_energy': B_delta.pow(2).mean(dim=[0, 2]) # Shape: (batch,)
        #     #     })
        #
        #     # ==========================================
        #     # ENERGY-PENALIZED BMA (Inference Only)
        #     # ==========================================
        #
        #     seq_len, batch_size, num_bars = logits_C.shape
        #
        #     # 1. Convert Logits to Probabilities
        #     probs_C = F.softmax(logits_C, dim=-1)  # Conditional (A|B_i)
        #     probs_A = F.softmax(logits_A, dim=-1)  # Unconditional (A)
        #
        #     # Since A is expanded in the batch dim, all batch items for A are identical.
        #     # We just need one copy of the unconditional probability.
        #     probs_A_single = probs_A[:, 0:1, :]  # Shape: (seq_len, 1, num_bars)
        #
        #     # 2. Calculate Predictive Entropy (Token-level)
        #     # H = - sum(p * log(p))
        #     # Adding a tiny epsilon to prevent log(0)
        #     entropy_C = -torch.sum(probs_C * torch.log(probs_C + 1e-9), dim=-1)  # (seq, batch)
        #     entropy_A = -torch.sum(probs_A_single * torch.log(probs_A_single + 1e-9), dim=-1)  # (seq, 1)
        #
        #     # 3. Retrieve Warp Energy from MetaContext
        #     vib_stats = ForwardMetaContext.get('vib_stats')
        #     if vib_stats is not None:
        #         # Shape: (1, batch) so it broadcasts over seq_len
        #         mean_log_var = vib_stats['mean_log_var'].unsqueeze(0)
        #         mean_energy = vib_stats['mean_energy'].unsqueeze(0)
        #
        #         # Total Energy Penalty for each context B_i
        #         # Softplus ensures the penalty is strictly positive
        #         warp_energy_C = F.softplus(mean_log_var + mean_energy)
        #     else:
        #         # Fallback if no warp occurred
        #         warp_energy_C = torch.zeros((1, batch_size), device=logits_C.device)
        #
        #     # The unconditional model (A) has zero warp energy by definition
        #     warp_energy_A = torch.zeros((1, 1), device=logits_A.device)
        #
        #     # 4. Formulate the Unnormalized Log-Weights
        #     # Weight = -(Entropy + lambda * Energy) / Temperature
        #     log_w_C = -(entropy_C + bma_lambda * warp_energy_C) / bma_temp  # (seq, batch)
        #     log_w_A = -(entropy_A + bma_lambda * warp_energy_A) / bma_temp  # (seq, 1)
        #
        #     # 5. Concatenate all models (N conditionals + 1 unconditional)
        #     # Shape: (seq, batch + 1)
        #     all_log_weights = torch.cat([log_w_C, log_w_A], dim=1)
        #
        #     # 6. Normalize weights across the model dimension using Softmax
        #     bma_weights = F.softmax(all_log_weights, dim=1)  # (seq, batch + 1)
        #
        #     # Extract the weights for C and A
        #     weights_C = bma_weights[:, :-1].unsqueeze(-1)  # (seq, batch, 1)
        #     weights_A = bma_weights[:, -1:].unsqueeze(-1)  # (seq, 1, 1)
        #
        #     # 7. Compute the Final BMA Probability Distribution
        #     # Sum the weighted conditional probabilities
        #     weighted_C_sum = torch.sum(probs_C * weights_C, dim=1, keepdim=True)  # (seq, 1, num_bars)
        #
        #     # Add the weighted unconditional probability
        #     bma_probs = weighted_C_sum + (probs_A_single * weights_A)  # (seq, 1, num_bars)
        #
        #     # Return standard logits for training, but add the BMA probs for inference
        #     return logits_A, logits_B, logits_C, bma_probs
        #
