"""Plain (untagged) `ppfn.model.pfn.pfn.PFN` reference bounds for the LUPI
id-token comparison -- an architecturally INDEPENDENT cross-check of
`AAloneIDTokenPFN`/`OracleIDTokenPFN` (`lupi_id_token_pfn.py`): same data
(`ppfn.prior.lupi`), no additive domain-id tag at all, no separate x/y
encoder split for train vs test the way `IDTokenPFN` now has -- just the
existing single-stream PFN, unmodified, fed a `LUPIBatch` instead of its
usual `BNNPrior` batch.

Per the user's explicit correction (2026-09-15): these are meant to be
properly meta-trained via the normal Hydra pipeline and loaded from a
checkpoint for inference, NOT fit from scratch inside a notebook -- a
per-instance SGD fit on one draw's points is not what a PFN's posterior
is, and doesn't test the thing this whole comparison exists to test
(in-context registration, no gradient steps at inference time).

`PFN.__init__` builds a bounded-`[0,1]` `BarDistribution` by default,
which is wrong for `ppfn.prior.lupi`'s raw (unnormalized, see
`ppfn.prior.lupi.sampler`'s own comment) `z` targets -- both classes below
swap it for a `FullSupportBarDistribution` with quantile-fit borders
(matching `IDTokenPFN`'s own head, `ppfn.model.baselines.calibration`),
resizing `out_head` to match its bin count. An earlier version used
`ppfn.model.registration.heads.TailBarDistribution` (fixed `[-4,4]` body) --
reassessed 2026-09-16 after that fixed range turned out to be a poor match
for this raw target's actual scale (see `ppfn.model.pfn.bar_distribution`'s
own docstring).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.baselines.calibration import sample_calibration_borders
from ppfn.model.pfn.bar_distribution import FullSupportBarDistribution
from ppfn.model.pfn.pfn import PFN
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class _PlainPFNBound(nn.Module):
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
        self.pfn = PFN(
            max_x_dim=d_max, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, n_bins=n_bins_predictive, dropout=dropout,
        )
        borders = sample_calibration_borders(n_bins_predictive)
        self.pfn.bar_dist = FullSupportBarDistribution(borders)
        self.pfn.out_head = nn.Linear(d_model, self.pfn.bar_dist.num_bars)

    @property
    def predictive_dist(self):
        return self.pfn.bar_dist


class AAlonePlainPFN(_PlainPFNBound):
    """Lower bound: `x_train=dec_ctx_x, y_train=dec_ctx_z` (A's context
    only), scored on `dec_qry_x`. No B, ever."""

    def forward(self, batch: LUPIBatch) -> dict:
        n_features = batch.d_real
        logits = self.pfn(
            batch.dec_ctx_x, batch.dec_ctx_z, batch.dec_qry_x,
            n_features=n_features, train_key_padding_mask=batch.dec_ctx_mask,
        )
        return {"predictive_logits": logits}


class OraclePlainPFN(_PlainPFNBound):
    """Upper-bound reference: `[A_ctx ; B_inA]` pooled as ONE undifferentiated
    train set (no domain tag -- unlike `OracleIDTokenPFN`, this architecture
    has no mechanism to distinguish the two clouds at all), scored on
    `dec_qry_x`. Independent architectural check of `OracleIDTokenPFN`.

    B_inA here means FULLY registered: position `enc_x_inA` (T) paired with
    value `enc_z_inA` (h), NOT `enc_z` (B's raw, un-h'd value) -- otherwise
    this "upper bound" would itself still carry an unsolved value-scale
    mismatch, undermining its role as the ceiling the student/oracle-decoder
    gap is measured against (2026-09-16 correction, see
    `LUPIPair.z_b_inA`'s docstring)."""

    def forward(self, batch: LUPIBatch) -> dict:
        x_train = torch.cat([batch.dec_ctx_x, batch.enc_x_inA], dim=1)
        y_train = torch.cat([batch.dec_ctx_z, batch.enc_z_inA], dim=1)
        mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)
        logits = self.pfn(
            x_train, y_train, batch.dec_qry_x,
            n_features=batch.d_real, train_key_padding_mask=mask,
        )
        return {"predictive_logits": logits}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md."""
    from ppfn.loss.id_token_loss import IDTokenLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    criterion = IDTokenLoss()
    for name, cls in [("AAlonePlainPFN", AAlonePlainPFN), ("OraclePlainPFN", OraclePlainPFN)]:
        model = cls(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
        out = model(batch)
        loss, metrics = criterion(model, batch, out)
        loss.backward()
        n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
        print(f"{name}: logits {tuple(out['predictive_logits'].shape)}  loss={loss.item():.4f}  grad=None: {n_none}")
