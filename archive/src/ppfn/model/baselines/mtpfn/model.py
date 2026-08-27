"""
MTPFN architecture, following Figure 2 and Section 3.2 exactly.

Pipeline per episode:

  Feature Encoder
    Each task's raw sequence [(x_1,y_1), ..., (x_n,y_n)] (query points have
    y masked out with a learned "missing value" embedding) is embedded and
    prepended with a single SHARED learnable "[TASK]" token (the same
    initial parameter vector for every task -- it becomes task-specific
    only through what it attends to, which is exactly what lets the model
    "naturally handle inputs of varying lengths... generalize to any
    number of tasks" per Section 3.2).

  Repeated (x N) hierarchical block:
    Intra-Task Encoder  -- standard transformer layer, self-attention over
                            one task's own sequence (including its [TASK]
                            token), applied independently per task
                            (O(T * D^2) total across T tasks).
    Inter-Task Encoder  -- standard transformer layer, self-attention only
                            over the T tasks' [TASK]-token summaries
                            (O(T^2)); the updated token is scattered back
                            into position 0 of each task's sequence for the
                            next block.

  One final Intra-Task Encoder (outside the repeated block, per Figure 2's
  layout, immediately below the Output Layer).

  Output Layer: a linear head applied at the query-point position(s) of
  the target task (task 0), producing (mu, sigma^2).

Section 5's reported configuration: 23 total attention layers = 12
intra-task + 11 inter-task (interleaved, i.e. N=11 repeated blocks + 1
final intra layer), 4 heads, hidden size 512. These are exposed as
constructor defaults below.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gaussian_head import split_head_output


class EncoderLayer(nn.Module):
    """A standard pre-LN transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.ln1(h + self.dropout(attn_out))
        h = self.ln2(h + self.dropout(self.ff(h)))
        return h


class MTPFN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128, # 512,
        n_heads: int = 4,
        d_ff: int = 2048,
        num_intra_layers: int = 4, # 12,
        num_inter_layers: int = 3, #11,
        dropout: float = 0.0,
        n_out: int = 100 # number of bins to project to
    ):
        super().__init__()
        assert num_intra_layers == num_inter_layers + 1, (
            "Paper's configuration interleaves as Intra, Inter, Intra, ..., Intra "
            "so there is always exactly one more intra- than inter-task layer."
        )
        self.d_model = d_model

        # --- Feature encoder ---
        self.x_encoder = nn.Linear(input_dim, d_model)
        self.y_encoder = nn.Linear(1, d_model)
        self.missing_y_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        # Single shared learnable [TASK] token, prepended to every task's
        # sequence -- NOT a per-task-index embedding (see module docstring).
        self.task_token = nn.Parameter(torch.randn(d_model) * 0.02)

        self.intra_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_intra_layers)]
        )
        self.inter_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_inter_layers)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.head =  self.output_projection = nn.Sequential(
            nn.Linear(self.input_size, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_out),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, valid_mask: torch.Tensor,
                query_mask: torch.Tensor) -> torch.Tensor:
        """
        x           : [B, T, L, input_dim]
        y           : [B, T, L]        (ignored where NOT valid, or where query_mask is True)
        valid_mask  : [B, T, L]        True where a real (possibly query) point occupies this slot
        query_mask  : [B, T, L]        True where y is unknown / to be predicted (task 0 only)

        Returns raw_head_output at EVERY slot: [B, T, L, 2] (mu, raw_var);
        caller reads off the query-task positions of interest.
        """
        B, T, L, _ = x.shape
        device = x.device

        observed = valid_mask & ~query_mask
        y_in = torch.where(observed, y, torch.zeros_like(y)).unsqueeze(-1)
        h_points = self.x_encoder(x) + self.y_encoder(y_in)
        h_points = torch.where(
            observed.unsqueeze(-1), h_points, self.missing_y_embed.view(1, 1, 1, -1).expand_as(h_points)
        )
        # zero out fully-padded (invalid & not-a-query) slots so they don't leak signal
        keep = valid_mask.unsqueeze(-1)
        h_points = h_points * keep

        task_tok = self.task_token.view(1, 1, 1, -1).expand(B, T, 1, -1)
        h = torch.cat([task_tok, h_points], dim=2)  # [B, T, 1+L, d]

        # padding mask for intra-task attention: [TASK] token (pos 0) is
        # always real; data slots are padding wherever valid_mask is False.
        task_pad = torch.zeros(B, T, 1, dtype=torch.bool, device=device)
        pad_mask = torch.cat([task_pad, ~valid_mask], dim=2)  # True = ignore

        h = h.reshape(B * T, 1 + L, self.d_model)
        pad_mask = pad_mask.reshape(B * T, 1 + L)

        n_inter = len(self.inter_layers)
        for i in range(n_inter):
            h = self.intra_layers[i](h, key_padding_mask=pad_mask)

            h = h.view(B, T, 1 + L, self.d_model)
            task_summaries = h[:, :, 0, :]  # [B, T, d]
            task_summaries = self.inter_layers[i](task_summaries, key_padding_mask=None)
            h = torch.cat([task_summaries.unsqueeze(2), h[:, :, 1:, :]], dim=2)
            h = h.reshape(B * T, 1 + L, self.d_model)

        # one final intra-task layer, outside the repeated (intra, inter) block
        h = self.intra_layers[-1](h, key_padding_mask=pad_mask)

        h = h.view(B, T, 1 + L, self.d_model)[:, :, 1:, :]  # drop [TASK] token, keep data slots
        h = self.out_norm(h)
        return self.head(h)  # [B, T, L, 2]


