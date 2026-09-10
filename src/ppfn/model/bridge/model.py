"""Plain encoder-decoder PFN -- no gating, no transport head, no curriculum,
no coordinate-free first layer. Standard transformer decoder cross-attention
(every layer, both context AND query tokens) to a plain transformer
encoder's memory, layered on top of the existing single-stream PFN's
train/test self-attention pattern
(`ppfn.model.pfn.pfn.MaskedMHA`/`PFNBlock`).

Verifies the cross-attention pathway can carry information from an abundant
B cloud into A's predictions at all, before any registration machinery
(warp/transport/coupling/affine) is layered on top -- see
`ppfn.prior.bridge.dataset`'s module docstring for the matching same-domain,
no-warp prior this is meant to run against.

Reuses `ppfn.model.registration.embeddings.InputEmbeddings` (already plain
per-token-type Linear projections -- "cloud identity carried by which
projection, never an added tag", exactly the PFN-style embedding this
experiment wants) and `ppfn.model.registration.encoder.Encoder` (plain
bidirectional self-attention, nothing fancy) as-is; only the decoder block
is new.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.pfn.pfn import MaskedMHA
from ppfn.model.registration.embeddings import InputEmbeddings
from ppfn.model.registration.encoder import Encoder
from ppfn.model.registration.heads import TailBarDistribution
from ppfn.prior.bridge.dataset import BridgeBatch, D_MAX


class BridgeDecoderBlock(nn.Module):
    """(a) train-train self-attn / test-train cross-attn -- `PFNBlock`'s own
    pattern, ARCHITECTURE.md-unrelated, this project's A-only baseline.
    (b) BOTH context and query tokens cross-attend to the encoder memory,
    plain, every layer, no gate, no transport-conditioned queries -- the one
    thing this block adds on top of a plain single-stream PFN. (c) FFN.
    (a) and (b) each share one set of weights between their two calls
    (ctx/qry), matching `PFNBlock`'s own within-block weight sharing."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MaskedMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MaskedMHA(d_model, n_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_ctx: torch.Tensor,
        h_qry: torch.Tensor,
        ctx_mask: torch.Tensor,
        m: torch.Tensor,
        enc_mask: torch.Tensor,
        use_encoder: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_n, qry_n = self.ln1(h_ctx), self.ln1(h_qry)
        h_ctx = h_ctx + self.self_attn(ctx_n, key_padding_mask=ctx_mask)
        h_qry = h_qry + self.self_attn(qry_n, kv_input=ctx_n, key_padding_mask=ctx_mask)

        if use_encoder:
            ctx_n, qry_n = self.ln2(h_ctx), self.ln2(h_qry)
            h_ctx = h_ctx + self.cross_attn(ctx_n, kv_input=m, key_padding_mask=enc_mask)
            h_qry = h_qry + self.cross_attn(qry_n, kv_input=m, key_padding_mask=enc_mask)

        h_ctx = h_ctx + self.ff(self.ln3(h_ctx))
        h_qry = h_qry + self.ff(self.ln3(h_qry))
        return h_ctx, h_qry


class BridgePFN(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers_enc: int = 6,
        n_layers_dec: int = 8,
        n_bins: int = 64,
        dropout: float = 0.0,
        d_max: int = D_MAX,
    ):
        super().__init__()
        self.d_max = d_max
        self.embeddings = InputEmbeddings(d_max, d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers_enc, dropout)
        self.decoder_layers = nn.ModuleList(
            [
                BridgeDecoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers_dec)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)
        self.predictive_dist = TailBarDistribution(n_bins)
        self.predictive_head = nn.Linear(d_model, self.predictive_dist.num_logits)

    def dim_mask(self, d_real: torch.Tensor, n: int) -> torch.Tensor:
        idx = torch.arange(self.d_max, device=d_real.device).view(1, 1, -1)
        return idx < d_real.view(-1, 1, 1)

    def forward(self, batch: BridgeBatch, use_encoder: bool = True) -> dict:
        """`use_encoder=False` skips every cross-attention call, leaving the
        decoder as a plain A-only PFN -- a free severed/lower-bound
        comparison against the same weights, no separate model needed."""
        enc_tok = self.embeddings.encoder_tokens(batch.enc_x, batch.enc_y)
        m, _ = self.encoder(enc_tok, batch.enc_mask) if use_encoder else (None, None)

        h_ctx = self.embeddings.decoder_context_tokens(batch.dec_ctx_x, batch.dec_ctx_y)
        h_qry = self.embeddings.decoder_query_tokens(batch.dec_qry_x)

        for layer in self.decoder_layers:
            h_ctx, h_qry = layer(
                h_ctx, h_qry, batch.dec_ctx_mask, m, batch.enc_mask, use_encoder=use_encoder
            )

        h_qry = self.out_ln(h_qry)
        predictive_logits = self.predictive_head(h_qry)
        return {"predictive_logits": predictive_logits}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run a forward pass, check query-mask isolation and the use_encoder
    toggle both work as intended."""
    import torch

    from ppfn.prior.bridge.dataset import (
        BridgeStreamDataset,
        collate_bridge_batch,
    )

    torch.manual_seed(0)
    dataset = BridgeStreamDataset(seed=0, d=1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_bridge_batch(items)

    model = BridgePFN(d_model=32, n_heads=4, d_ff=64, n_layers_enc=2, n_layers_dec=3)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("predictive_logits:", out["predictive_logits"].shape)

    out_severed = model(batch, use_encoder=False)
    print(
        "bridged vs severed logits differ (expect True):",
        not torch.allclose(out["predictive_logits"], out_severed["predictive_logits"]),
    )

    # Query-isolation check: perturbing one query point must not change any
    # OTHER query point's final logits.
    batch_pert = collate_bridge_batch(items)
    batch_pert.dec_qry_x[:, 0, :] += 1.0
    out_pert = model(batch_pert)
    other_diff = (
        (out["predictive_logits"][:, 1:] - out_pert["predictive_logits"][:, 1:])
        .abs()
        .max()
        .item()
    )
    print(
        "max diff at OTHER query points after perturbing query 0 (~0 expected):",
        other_diff,
    )
