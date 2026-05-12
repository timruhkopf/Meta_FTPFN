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
        self.x_encoder = FourierEncoder(d_model, sigma=1) if use_freq_enc_x else nn.Linear(1, d_model)
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

        if self.use_cross_attn: #
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
            A, B, C = self.cross_layer(
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

        # if self.training:
        return logits_A, logits_B, logits_C
