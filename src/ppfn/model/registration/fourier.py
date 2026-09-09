"""Fourier feature map phi on coordinates -- ARCHITECTURE.md §2.3(c): "8
frequencies per axis, so that proximity in the warped frame is expressible
at multiple scales." Fixed (non-learned) frequencies, NeRF-style geometric
spacing 2^0..2^7 -- the spec doesn't pin down the spacing, and geometric
spacing is the standard choice for covering both fine and coarse scales
with few frequencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_frequencies: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32) * torch.pi
        self.register_buffer("freqs", freqs)  # [n_frequencies]
        self.in_dim = in_dim
        self.out_dim = in_dim * n_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., in_dim * n_frequencies * 2]."""
        # [..., in_dim, n_frequencies]
        angles = x.unsqueeze(-1) * self.freqs.view(*([1] * x.dim()), -1)
        feats = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )  # [..., in_dim, 2*n_freq]
        return feats.flatten(-2, -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    ff = FourierFeatures(in_dim=5, n_frequencies=8)
    x = torch.rand(4, 10, 5)
    out = ff(x)
    print("input", x.shape, "-> output", out.shape, "(expect out_dim=", ff.out_dim, ")")
    assert out.shape[-1] == ff.out_dim
