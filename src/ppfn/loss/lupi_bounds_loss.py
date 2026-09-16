"""Loss for `ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN` -- both bounds
every batch (`severed=True` for the lower bound, `severed=False` for the
upper), mirroring `ppfn.loss.lupi_loss.LUPILoss`'s "both modes every batch"
pattern so one trained model always yields a consistent bracket rather than
depending on a stochastic per-item mask ever being drawn.
"""

from __future__ import annotations

import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


class BoundsLoss(nn.Module):
    def forward(self, model: nn.Module, batch: LUPIBatch, output: dict | None = None) -> tuple:
        if output is None:
            out_lower = model(batch, severed=True)
            out_upper = model(batch, severed=False)
        else:
            out_lower = output["out_lower"]
            out_upper = output["out_upper"]

        nll_lower = model.bar_dist(out_lower["predictive_logits"], batch.dec_qry_z)
        nll_upper = model.bar_dist(out_upper["predictive_logits"], batch.dec_qry_z)

        mask = batch.dec_qry_mask.to(nll_lower.dtype)
        denom = mask.sum().clamp_min(1.0)
        loss_lower = (nll_lower * mask).sum() / denom
        loss_upper = (nll_upper * mask).sum() / denom
        loss = loss_lower + loss_upper

        metrics = {
            "loss/total": loss.item(),
            "loss/lower_nll": loss_lower.item(),
            "loss/upper_nll": loss_upper.item(),
            "loss/bounds_gap": (loss_lower - loss_upper).item(),
        }
        return loss, metrics
