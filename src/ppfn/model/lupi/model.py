"""LUPIPFN -- build spec §4 (architecture) + §5.1 (two positioning modes on
one trunk). See `ppfn.model.lupi`'s package docstring for how this differs
from `ppfn.model.registration`.

Per-layer structure (spec §4.3's pseudocode, `b` cached / computed once,
only `u = [A_ctx ; A_qry]` evolves across layers):

    u = u + CrossAttn(Q=u, KV=b)         # match + read B, shared weights across ctx/qry (spec §4.4)
    u = u + SelfAttn(Q=u, KV=A_ctx)      # PFN train/test mask -- reuses ppfn.model.pfn.pfn.PFNBlock verbatim
    u = u + FFN(u)                       # folded into the PFNBlock call above

`CrossAttn` is one `MaskedMHA` call over the WHOLE concatenated `u` tensor
against `b` -- this is what makes "shared weights, ctx and query read B the
same way" (spec §4.4's default) automatic rather than a separate design
choice: there's only one Q projection and it's applied to one tensor.

`SelfAttn + FFN` is exactly `PFNBlock`'s existing train/test split
(`ppfn.model.pfn.pfn.PFNBlock`, reused verbatim, matching
`ppfn.model.baselines.id_token_pfn`'s own precedent of reusing it outside
`ppfn.model.pfn.pfn.PFN`): A-context tokens are "train" (bidirectional
self-attention among themselves), A-query tokens are "test" (cross-attend to
A-context only, never to each other) -- this is what spec §4.3 calls
"A-self-attention lets neighbouring A tokens agree on a locally coherent
displacement," and is where the PPD-stays-a-function invariant (no
test-test attention) already built into `PFNBlock` carries over for free.

B's own encoder stack (`EncB`, spec §4.3: "CACHED") is a small separate
self-attention-only stack, computed ONCE per batch via `encode_b` and
reused for BOTH positioning modes within one training step -- this is what
keeps the two-modes-per-batch cost "well under 2x" (spec §5.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.pfn.bar_distribution import BarDistribution, uniform_bin_borders
from ppfn.model.pfn.pfn import MaskedMHA, PFNBlock
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class EncoderLayer(nn.Module):
    """Plain pre-LN self-attention + FFN block, no test-token split -- the
    "train-only half" of `PFNBlock`, factored out because B's own encoder
    stack (`EncB`) never has a second (query) token stream to cross-attend
    against; running B through `PFNBlock` itself would need a throwaway
    dummy test tensor just to get a return value nobody reads."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


class LUPIPFN(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc_b: int = 3,
        n_layers_align: int = 6,
        d_ff: int = 512,
        n_bins_predictive: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_max = d_max
        self.n_layers_align = n_layers_align

        # Shared coordinate embedding (spec §4.2: "one shared coordinate
        # embedding across all three token types") -- reused verbatim for
        # B's own position, A's own (student-mode) position, AND the
        # injected ground-truth B-frame position (oracle mode): in oracle
        # mode an A/test token's coordinate input IS T(x), so it goes
        # through this exact same projection as a real B token would.
        self.coord_embed = nn.Linear(d_max, d_model)
        self.value_embed = nn.Linear(1, d_model)
        self.no_value = nn.Parameter(torch.zeros(d_model))
        # id=0 -> A-role token (context or query), id=1 -> B token.
        self.domain_embed = nn.Embedding(2, d_model)

        self.enc_b_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_enc_b)]
        )

        self.cross_ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers_align)])
        self.cross_attn = nn.ModuleList([MaskedMHA(d_model, n_heads) for _ in range(n_layers_align)])
        self.align_blocks = nn.ModuleList(
            [PFNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_align)]
        )

        self.out_ln = nn.LayerNorm(d_model)
        self.bar_dist = BarDistribution(uniform_bin_borders(n_bins_predictive, 0.0, 1.0))
        self.pred_head = nn.Linear(d_model, self.bar_dist.num_bars)

    def encode_b(self, batch: LUPIBatch) -> torch.Tensor:
        """-> [B, n_enc, d_model]. Computed once per batch, reused by both
        `align(..., mode="student")` and `align(..., mode="oracle")` calls
        in the same training step (spec §5.1's "under 2x" claim)."""
        b = self.coord_embed(batch.enc_x) + self.value_embed(batch.enc_z.unsqueeze(-1)) + self.domain_embed.weight[1]
        for layer in self.enc_b_layers:
            b = layer(b, key_padding_mask=batch.enc_mask)
        return b

    def align(self, batch: LUPIBatch, b: torch.Tensor, mode: str) -> dict:
        """mode: "student" (position input = A's own coordinates, alignment
        must be inferred purely through cross-attention against `b`) or
        "oracle" (position input = the ground-truth B-frame coordinate
        `T(x)`, injected directly -- spec §5.1's privileged-information
        signal). -> {"predictive_logits": [B, n_qry, n_bins]}."""
        assert mode in ("student", "oracle")
        if mode == "student":
            ctx_pos, qry_pos = batch.dec_ctx_x, batch.dec_qry_x
        else:
            ctx_pos, qry_pos = batch.dec_ctx_oracle_bpos, batch.dec_qry_oracle_bpos

        ctx_tok = (
            self.coord_embed(ctx_pos) + self.value_embed(batch.dec_ctx_z.unsqueeze(-1))
            + self.domain_embed.weight[0]
        )
        qry_tok = self.coord_embed(qry_pos) + self.no_value.view(1, 1, -1) + self.domain_embed.weight[0]

        n_ctx = ctx_tok.shape[1]
        for l in range(self.n_layers_align):
            u = torch.cat([ctx_tok, qry_tok], dim=1)
            cross_out = self.cross_attn[l](
                self.cross_ln[l](u), kv_input=b, key_padding_mask=batch.enc_mask
            )
            u = u + cross_out
            ctx_tok, qry_tok = u[:, :n_ctx], u[:, n_ctx:]
            ctx_tok, qry_tok = self.align_blocks[l](
                ctx_tok, qry_tok, train_key_padding_mask=batch.dec_ctx_mask
            )

        qry_tok = self.out_ln(qry_tok)
        logits = self.pred_head(qry_tok)
        return {"predictive_logits": logits}

    def forward(self, batch: LUPIBatch, mode: str = "student") -> dict:
        """Standalone single-mode forward (e.g. inference, where no oracle
        signal exists) -- `LUPILoss` calls `encode_b`/`align` directly
        instead, to share `b` across both modes within one training step."""
        return self.align(batch, self.encode_b(batch), mode=mode)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run both positioning modes on a real batch, confirm shapes, confirm the
    loss/backward path works end to end, and confirm B-only padding is
    invisible (same style of check as ppfn.model.baselines.id_token_pfn)."""
    import dataclasses

    import torch

    from ppfn.loss.lupi_loss import LUPILoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = LUPIPFN(
        d_model=32, n_heads=4, n_layers_enc_b=2, n_layers_align=2, d_ff=64, n_bins_predictive=16
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out_student = model(batch, mode="student")
    out_oracle = model(batch, mode="oracle")
    print("student logits:", out_student["predictive_logits"].shape)
    print("oracle logits:", out_oracle["predictive_logits"].shape)

    criterion = LUPILoss()
    loss, metrics = criterion(model, batch)
    print("loss/metrics:", metrics)

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None: {n_none}")

    # Padding invisibility: one extra all-zero, masked-out B point must not
    # change either mode's predictive_logits.
    padded_enc_x = torch.cat([batch.enc_x, torch.zeros_like(batch.enc_x[:, :1])], dim=1)
    padded_enc_z = torch.cat([batch.enc_z, torch.zeros_like(batch.enc_z[:, :1])], dim=1)
    padded_enc_mask = torch.cat([batch.enc_mask, torch.zeros_like(batch.enc_mask[:, :1])], dim=1)
    padded_batch = dataclasses.replace(
        batch, enc_x=padded_enc_x, enc_z=padded_enc_z, enc_mask=padded_enc_mask
    )
    out_padded = model(padded_batch, mode="student")
    diff = (out_student["predictive_logits"] - out_padded["predictive_logits"]).abs().max().item()
    print("max diff from one masked-out padding point in B (~0 expected):", diff)
