"""Input embeddings -- ARCHITECTURE.md §2.1. Separate `Linear_A`/`Linear_B`
projections: "Cloud identity is carried by *which projection and which
stack* a token goes through, never by an added tag -- the additive-tag
cross-term problem is avoided by construction."
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_max: int, d_model: int):
        super().__init__()
        self.linear_b = nn.Linear(d_max + 1, d_model)  # [x-tilde^B, y-tilde^B]
        self.linear_a = nn.Linear(
            d_max + 2, d_model
        )  # [x-tilde^A, y-tilde^A, labelled-flag]

    def encoder_tokens(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: [B,N,d_max]  y: [B,N] -> [B,N,d_model]."""
        return self.linear_b(torch.cat([x, y.unsqueeze(-1)], dim=-1))

    def decoder_context_tokens(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Labelled: [x-tilde, y-tilde, 1]."""
        b, n, _ = x.shape
        flag = x.new_ones(b, n, 1)
        return self.linear_a(torch.cat([x, y.unsqueeze(-1), flag], dim=-1))

    def decoder_query_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Query: [x-tilde, 0, 0]."""
        b, n, _ = x.shape
        zeros = x.new_zeros(b, n, 2)
        return self.linear_a(torch.cat([x, zeros], dim=-1))


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = InputEmbeddings(d_max=5, d_model=16)
    x = torch.rand(2, 4, 5)
    y = torch.rand(2, 4)
    print("encoder_tokens:", emb.encoder_tokens(x, y).shape)
    print("decoder_context_tokens:", emb.decoder_context_tokens(x, y).shape)
    print("decoder_query_tokens:", emb.decoder_query_tokens(x).shape)
