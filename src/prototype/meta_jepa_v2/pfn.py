import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple
import copy
import threading



class UnifiedPFNLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, eval_pos: int, pad_mask: Optional[Tensor] = None) -> Tensor:
        normed_x = self.norm1(x)
        train_part = normed_x[:eval_pos]
        test_part = normed_x[eval_pos:]

        train_pad_mask = pad_mask[:, :eval_pos] if pad_mask is not None else None

        # Context self-attends
        train_out = self.self_attn(
            query=train_part, key=train_part, value=train_part,
            key_padding_mask=train_pad_mask, need_weights=False
        )[0]

        # Query cross-attends using identical projection weights
        if test_part.shape[0] > 0:
            test_out = self.self_attn(
                query=test_part, key=train_part, value=train_part,
                key_padding_mask=train_pad_mask, need_weights=False
            )[0]
            attn_out = torch.cat([train_out, test_out], dim=0)
        else:
            attn_out = train_out

        x = x + self.dropout1(attn_out)
        normed_ff = self.norm2(x)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(normed_ff))))
        return x + self.dropout2(ff_out)


class PFNStack(nn.Module):
    def __init__(self, d_model: int, num_layers: int, nhead: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([UnifiedPFNLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, context: Tensor, queries: Optional[Tensor] = None, pad_mask: Optional[Tensor] = None):
        eval_pos = context.shape[0]
        combined = torch.cat([context, queries], dim=0) if queries is not None else context

        for layer in self.layers:
            combined = layer(combined, eval_pos, pad_mask)

        out_ctx = combined[:eval_pos]
        out_q = combined[eval_pos:] if queries is not None else None
        return out_ctx, out_q