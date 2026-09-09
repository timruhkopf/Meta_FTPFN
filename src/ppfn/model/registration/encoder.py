"""Encoder (cloud B / E) -- ARCHITECTURE.md §2.2. Full bidirectional
self-attention over the encoder cloud's tokens, `L_enc` pre-LN blocks. "The
encoder never sees A. This is what makes M independent of A's warp and A's
design, and it is what makes the severed-encoder baseline exact."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PreLNEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: [B,N,d_model]. key_padding_mask: [B,N] bool, True=real (inverted
        for `nn.MultiheadAttention`, which wants True=ignore)."""
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=~key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.0
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                PreLNEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: [B,N,d_model]. Returns (M [B,N,d_model], g_B [B,d_model]
        -- the masked mean over real tokens, ARCHITECTURE.md §2.2)."""
        x = tokens
        for block in self.blocks:
            x = block(x, key_padding_mask)
        m = self.out_ln(x)
        mask_f = key_padding_mask.unsqueeze(-1).to(m.dtype)
        g_b = (m * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return m, g_b


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = Encoder(d_model=16, n_heads=2, d_ff=32, n_layers=3)
    tokens = torch.rand(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    m, g_b = enc(tokens, mask)
    print("M:", m.shape, "g_B:", g_b.shape)
