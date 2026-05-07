from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Linear, Dropout, MultiheadAttention, LayerNorm


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
            pad_mask_A: Optional[Tensor] = None, # train part only!
            pad_mask_B: Optional[Tensor] = None  # train part only!
    ):
        # Store original batch size to split them back apart later
        # Assumes A, B, and C all have the exact same shape: (Time, Batch, D)
        T, B_size, D = A.shape

        # ==========================================
        # 1. BATCH CONCATENATION
        # ==========================================
        # Stack along the Batch dimension (dim=1) -> Shape: (Time, 3 * Batch, D)
        combined = torch.cat([A, B, C], dim=1)

        # ==========================================
        # 2. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        normed_combined = self.norm1(combined)
        combined_mask = torch.cat([pad_mask_A, pad_mask_B, pad_mask_A], dim=0)

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
