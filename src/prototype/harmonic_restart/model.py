import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.harmonic_restart.pfn_layer import PFNLayer



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
    """
        Tri-Stream Posterior Foundation Network (PFN) for Cross-Domain Alignment.

        This model is designed to perform Bayesian inference across multiple related
        functional streams. It learns to map a sparse, potentially distorted observation
        stream (Domain A) against a dense, canonical prior stream (Domain B) in order
        to improve predictive log-likelihoods on a target evaluation stream (Domain C).

        Core Architectural Concepts:
            1. Tri-Stream Marginal Processing: The network maintains three distinct
               representational streams (A, B, and C). Initial layers process these
               streams marginally, allowing the model to perform valid Bayesian inference
               for each domain independently before attempting cross-domain alignment.

            2. Cross-Domain Alignment (The "Translation" Objective): A dedicated
               cross-attention layer allows Stream A to query Stream B. This enables
               the model to resolve spatial distortions (e.g., affine shifts, spatial
               warping) by copying relevant features from the canonical domain to
               reduce the final negative log-likelihood (NLL) of Stream C.

            3. Shadow Batches & Auxiliary Guidance: During training, the model ingests
               synthetic cross-domain projections ('A_in_B' and 'B_in_A'). These "shadow
               batches" are appended to the sequence dimension. They act as anchor
               points to guide the cross-attention mechanism, allowing the calculation
               of an auxiliary identity-matrix loss (similar to CLIP alignment) between
               the distorted and canonical domains. To prevent data leakage during
               inference, these shadow projections are truncated before final decoding.

        Args:
            cross_attn_layer (nn.Module): The module responsible for computing attention
                between Stream A and Stream B (and their respective shadow batches).
            d_model (int, optional): Dimensionality of the latent embeddings. Defaults to 64.
            nhead (int, optional): Number of attention heads in the PFN layers. Defaults to 4.
            dropout (float, optional): Dropout probability for the PFN layers. Defaults to 0.1.
            num_bars (int, optional): Size of the output logit vector (e.g., discrete bins
                for the target bar distribution). Defaults to 100.
            use_freq_enc_x (bool, optional): If True, applies a Fourier Encoder to the X
                coordinates to capture high-frequency details. Otherwise, uses a standard
                linear projection. Defaults to True.
            use_post_attn (bool, optional): If True, applies an additional marginal PFN layer
                after the cross-attention step to refine the aligned representations.
                Defaults to True.
            use_cross_attn (bool, optional): If True, activates cross-domain alignment
                and shadow batch processing. If False, the model acts as a baseline
                independent PFN (useful for ablation/baseline NLL scoring). Defaults to True.
        """
    def __init__(self, cross_attn_layer, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_freq_enc_x=False,
                 use_post_attn=True, use_cross_attn=True):
        super().__init__()
        self.num_bars = num_bars
        # reducing sigma might smooth the high frequency detail in A and B?
        self.x_encoder = FourierEncoder(d_model, sigma=1) if use_freq_enc_x else nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        self.pfn_layer = PFNLayer(d_model, nhead=nhead, dropout=dropout)
        self.cross_layers = cross_attn_layer

        self.use_cross_attn = use_cross_attn  # to allow baseline calculations for nll scores (once)
        self.use_post_attn = use_post_attn

        if self.use_post_attn:
            self.pfn_layer2 = PFNLayer(d_model, nhead=nhead, dropout=dropout)

        # Final norm for Pre-Norm architecture
        self.final_norm = LayerNorm(d_model)

        # Shared decoder
        self.decoder = nn.Linear(d_model, num_bars)

    def append_cross_domain_features(self, batch, suffix, base_emb, base_pad_mask, base_X, single_eval_pos):
        """
        Processes cross-domain features and appends them to the base representations.

        Args:
            batch (dict): The data batch.
            suffix (str): The identifier suffix, e.g., 'A_in_B' or 'B_in_A'.
            base_emb (Tensor): The main embedding tensor to append to (e.g., A or B).
            base_pad_mask (Tensor): The padding mask to duplicate.
            base_X (Tensor): The main X tensor to append to (e.g., X_A or X_B).
            single_eval_pos (int): Position index for adding the Y embeddings.

        Returns:
            Tuple of updated (base_emb, base_pad_mask, base_X)
        """
        # 1. Dynamically generate the dictionary keys
        x_key = f"X_{suffix}"
        y_key = f"Y_{suffix}"

        # 2. Extract and format X features
        x_train = torch.nan_to_num(batch['train'][x_key], nan=0.0)
        x_test = torch.nan_to_num(batch['test'][x_key], nan=0.0)
        x_concat = torch.cat([x_train, x_test], dim=0)

        # 3. Encode X
        emb_x = self.x_encoder(x_concat)

        # 4. Extract, format, and encode Y features
        y_train = torch.nan_to_num(batch['train'][y_key], nan=0.0)
        emb_y = self.y_encoder(y_train)

        # 5. Combine embeddings
        emb_combined = emb_x.clone()
        emb_combined[:single_eval_pos, :, :] += emb_y

        # 6. Concatenate with the base representations
        out_emb = torch.cat([base_emb, emb_combined], dim=1)
        out_X = torch.cat([base_X, x_concat], dim=1)

        # Double the pad mask for the concatenated version
        if base_pad_mask is not None:
            out_pad_mask = torch.cat([base_pad_mask, base_pad_mask], dim=0)
        else:
            out_pad_mask = None

        return out_emb, out_pad_mask, out_X

    def forward(self, batch):
        #FIXME: make these arguments!
        X_train_A, Y_train_A = batch['train']['X_A'], batch['train']['Y_A']
        X_train_B, Y_train_B = batch['train']['X_B'], batch['train']['Y_B']
        X_test_A = batch['test']['X_A']
        X_test_B = batch['test']['X_B']
        pad_mask_A = batch['train']['padding_mask_A']
        pad_mask_B = batch['train']['padding_mask_B']
        single_eval_pos = batch['train']['X_B'].shape[0]

        # Concat train and test X
        # fixme: fix prior to give unsqueezed version here
        X_A = torch.cat([X_train_A, X_test_A], dim=0)
        X_B = torch.cat([X_train_B, X_test_B], dim=0)

        # # FIXME: use ADAIN here?

        # Encode
        emb_X_A = self.x_encoder(X_A)
        emb_X_B = self.x_encoder(X_B)
        emb_Y_A = self.y_encoder(Y_train_A)
        emb_Y_B = self.y_encoder(Y_train_B)

        A = emb_X_A.clone()
        B = emb_X_B.clone()

        batch_size = A.shape[1]

        A[:single_eval_pos, :, :] += emb_Y_A
        B[:single_eval_pos, :, :] += emb_Y_B

        if self.use_cross_attn: # fixme: the keys will not exist during inference!
            # Process A_in_B and append to A
            A, pad_mask_A, X_A = self.append_cross_domain_features(
                batch=batch,
                suffix='A_in_B',
                base_emb=A,
                base_pad_mask=pad_mask_A,
                base_X=X_A,
                single_eval_pos=single_eval_pos
            )

            # Process B_in_A and append to B
            B, pad_mask_B, X_B = self.append_cross_domain_features(
                batch=batch,
                suffix='B_in_A',
                base_emb=B,
                base_pad_mask=pad_mask_B,
                base_X=X_B,
                single_eval_pos=single_eval_pos
            )

        X_C = X_A.clone()
        C = A.clone() # this must be after the potential edit on A in the above branch!

        # Pass through the Marginal PFN Layer
        A, B, C = self.pfn_layer(A, B, C, single_eval_pos, pad_mask_A, pad_mask_B)

        # ==========================================
        # FIX: Cleaned up kwargs to match PreNormTriStreamTransformerLayer signature
        # ==========================================
        # fixme: this will prevent joint training, but be more efficient in warmup
        if self.use_cross_attn:
            # overwriting A, B with pass throughs, but purged of shadow batch, which is available during training
            A, B, C = self.cross_layers(
                # A.detach(), B.detach(), C.detach(),
                # hp_A=emb_X_A.detach(), hp_B=emb_X_B.detach(), hp_C=emb_X_A.detach(),
                A.detach(), B.detach(), C.detach(),
                # FIXME: these are not in appropriate size!
                # hp_A=emb_X_A, hp_B=emb_X_B, hp_C=emb_X_A,  # C gets the same positional info as A since it's the "anchor"
                sep=single_eval_pos,
                raw_hp_A=X_A,
                raw_hp_B=X_B,
                raw_hp_C=X_C,
                pad_mask_A=pad_mask_A,
                pad_mask_B=pad_mask_B
            )

            # You still must manually truncate the pad_mask_A because it wasn't returned!
            true_batch_size = A.shape[1]  # A is now safely 256
            pad_mask_A = pad_mask_A[:true_batch_size, :]
            # FIXME: what about mask_B

            if 'X_B_in_A' in batch['train'].keys():
                # drop the auxiliary A in B that we attached to stream B
                B = B[:, :batch_size, :]
                pad_mask_B = pad_mask_B[:batch_size]

                A = A[:, :batch_size, :]
                C = C[:, :batch_size, :]
                pad_mask_A = pad_mask_A[:batch_size]

        if self.use_post_attn:
            A, B, C = self.pfn_layer2(A, B, C, single_eval_pos, pad_mask_A, pad_mask_B)

        # Apply final norm
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)

        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        if any([torch.any(torch.isnan(t)) for t in (logits_A, logits_B, logits_C)]):
            import pdb
            pdb.set_trace()
        return logits_A, logits_B, logits_C
