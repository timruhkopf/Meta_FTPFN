"""Loss for `ppfn.model.baselines.lupi_flow_matching_pfn.FlowMatchingVelocityField`
-- conditional flow matching (Lipman et al. 2023): plain regression against
the straight-line interpolant's velocity target, which is CONSTANT in `t`
(`z_1 - z_0`), so nothing here needs the `t` the model happened to sample --
only `output`'s predicted velocities and `batch`'s two known endpoints.

Masked two ways, mirroring `ppfn.model.registration.heads.TransportHead.nll`'s
own convention: `batch.enc_mask` drops padded B tokens, `model.dim_mask`
drops the zero-padded position channels beyond each draw's own `d_real`
(both endpoints are zero there by construction -- see
`ppfn.prior.lupi.dataset._pad_rescale_coords` -- so the raw target is
already exactly zero on those channels; the mask exists so an untrained
model isn't penalized for whatever it predicts there, not because the
target itself needs correcting)."""

from __future__ import annotations

import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


class FlowMatchingLoss(nn.Module):
    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        """-> (loss, metrics). `model` is used for `model.dim_mask` only.
        `progress` accepted-but-unused, matching `IDTokenLoss`'s own
        calling-convention parity with `IDTokenTrainer` (no curriculum
        here)."""
        del progress
        target_pos = batch.enc_x_inA - batch.enc_x  # [B,n_enc,d_max]
        target_val = batch.enc_z_inA - batch.enc_z  # [B,n_enc]

        token_mask = batch.enc_mask.to(target_val.dtype)  # [B,n_enc]
        dim_mask = model.dim_mask(batch.d_real, target_pos.shape[1]).to(target_pos.dtype)  # [B,n_enc,d_max]
        pos_mask = dim_mask * token_mask.unsqueeze(-1)

        pos_sq_err = (output["velocity_pred_pos"] - target_pos) ** 2
        pos_loss = (pos_sq_err * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)

        val_sq_err = (output["velocity_pred_val"] - target_val) ** 2
        val_loss = (val_sq_err * token_mask).sum() / token_mask.sum().clamp_min(1.0)

        loss = pos_loss + val_loss
        return loss, {
            "loss/total": loss.item(),
            "loss/velocity_pos": pos_loss.item(),
            "loss/velocity_val": val_loss.item(),
        }
