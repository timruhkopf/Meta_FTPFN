"""Lower/upper bound PFN for `ppfn.prior.lupi` -- one plain, single-stream,
NO-ID PFN trained on the pooled context `[A_ctx ; B_inA]` (`B_inA` =
`ppfn.prior.lupi.dataset.LUPIBatch.enc_x_inA`/`enc_z_inA`, B fully
registered -- position via T, value via h -- into A's frame, no inversion
needed; see those fields' docstrings), scored on A's query tokens. Reuses
`ppfn.model.pfn.pfn.PFNBlock`.

`severed` (a forward-time argument, not a per-item prior flag) toggles
whether the B_inA portion of the pooled context is masked out entirely:

    - `severed=True`  -> context = A_ctx alone  -- the LOWER bound
    - `severed=False` -> context = [A_ctx ; B_inA] -- the UPPER bound

Both are computed from the SAME trained weights (`ppfn.loss.lupi_bounds_loss
.BoundsLoss` runs both every training step, mirroring `LUPILoss`'s own
"both modes every batch" pattern) -- one checkpoint yields both bounds.

No domain/additive-ID tag (2026-09-16, at the user's direction -- an
earlier version of this class had one, copied from
`ppfn.model.baselines.id_token_pfn.IDTokenPFN`'s pattern, matching that
class's own naming but not its purpose here): B_inA is ALREADY fully
registered into A's frame and scale, so there's no alignment problem left
to distinguish a cloud identity FOR -- unlike `IDTokenPFN`, which pools B
in B's OWN (unregistered) frame and genuinely needs the tag to keep the
two clouds' differing scales separable. Lower and upper bound are just "a
regular decoder-only PFN conditioned on different context compositions/
sizes" -- exactly what a PFN already amortizes over, so ONE set of weights
trained on a MIX of both regimes covers both, rather than two separately
trained models (this class previously backed two separate experiments,
`plain_pfn_bound_a_alone`/`plain_pfn_bound_oracle`, now retired in favor
of this one, reporting both curves from a single checkpoint by varying
`severed` at inference time)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.baselines.calibration import sample_calibration_borders
from ppfn.model.pfn.bar_distribution import FullSupportBarDistribution
from ppfn.model.pfn.pfn import PFNBlock
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class BoundsPFN(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        n_bins_predictive: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_max = d_max
        self.train_embed = nn.Linear(d_max + 1, d_model)
        self.test_x_embed = nn.Linear(d_max, d_model)
        self.test_placeholder = nn.Parameter(torch.zeros(d_model))

        self.blocks = nn.ModuleList(
            [PFNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        borders = sample_calibration_borders(n_bins_predictive)
        self.bar_dist = FullSupportBarDistribution(borders)
        self.pred_head = nn.Linear(d_model, self.bar_dist.num_bars)

    def forward(self, batch: LUPIBatch, severed: bool) -> dict:
        """-> {"predictive_logits": [B, n_qry, n_bins]}."""
        pooled_x = torch.cat([batch.dec_ctx_x, batch.enc_x_inA], dim=1)
        pooled_z = torch.cat([batch.dec_ctx_z, batch.enc_z_inA], dim=1)

        b_mask = torch.zeros_like(batch.enc_mask) if severed else batch.enc_mask
        pooled_mask = torch.cat([batch.dec_ctx_mask, b_mask], dim=1)

        train_tok = self.train_embed(
            torch.cat([pooled_x, pooled_z.unsqueeze(-1)], dim=-1)
        )

        test_tok = self.test_x_embed(batch.dec_qry_x) + self.test_placeholder.view(1, 1, -1)

        for block in self.blocks:
            train_tok, test_tok = block(
                train_tok, test_tok, train_key_padding_mask=pooled_mask
            )

        test_tok = self.out_ln(test_tok)
        logits = self.pred_head(test_tok)
        return {"predictive_logits": logits}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run both severed/pooled forward passes on a real batch, confirm shapes,
    confirm the loss/backward path works, and confirm B padding is
    invisible in the pooled (severed=False) pass."""
    import dataclasses

    import torch

    from ppfn.loss.lupi_bounds_loss import BoundsLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = BoundsPFN(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out_lower = model(batch, severed=True)
    out_upper = model(batch, severed=False)
    print("lower (A-only) logits:", out_lower["predictive_logits"].shape)
    print("upper (A + B_inA) logits:", out_upper["predictive_logits"].shape)

    criterion = BoundsLoss()
    loss, metrics = criterion(model, batch)
    print("loss/metrics:", metrics)

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None: {n_none}")

    padded_enc_x_inA = torch.cat([batch.enc_x_inA, torch.zeros_like(batch.enc_x_inA[:, :1])], dim=1)
    padded_enc_z_inA = torch.cat([batch.enc_z_inA, torch.zeros_like(batch.enc_z_inA[:, :1])], dim=1)
    padded_enc_mask = torch.cat([batch.enc_mask, torch.zeros_like(batch.enc_mask[:, :1])], dim=1)
    padded_batch = dataclasses.replace(
        batch, enc_x_inA=padded_enc_x_inA, enc_z_inA=padded_enc_z_inA, enc_mask=padded_enc_mask
    )
    out_padded = model(padded_batch, severed=False)
    diff = (out_upper["predictive_logits"] - out_padded["predictive_logits"]).abs().max().item()
    print("max diff from one masked-out padding point in B_inA (~0 expected):", diff)
