"""Loss for `ppfn.model.baselines.id_token_pfn.IDTokenPFN` -- a single
bar-distribution NLL term on A's query tokens (`batch.dec_qry_mask`) only,
deliberately with no transport/coupling/affine/distillation terms (see that
module's docstring for why this baseline exists). There is no
`role_swapped` masking needed here since this baseline never predicts B in
the first place (`IDTokenPFN.forward` only builds query tokens from
`batch.dec_qry_x`), and `ppfn.prior.lupi` doesn't do role randomization at
all (A and B are fixed roles, unlike `ppfn.prior.registration`).
"""

from __future__ import annotations

import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


class IDTokenLoss(nn.Module):
    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        """-> (loss, metrics). `model` is unused beyond
        `model.predictive_dist` -- kept as an explicit argument to match
        `RegistrationLoss.forward`'s calling convention
        (`criterion(model, batch, output, ...)`), even though this loss
        never needs a second forward pass the way `L_distil`/`L_pathway` do.
        `progress` is accepted-but-unused -- IDTokenTrainer calls every
        criterion with it (LUPIIDTokenLoss's own student-weight/temperature
        curriculum needs it), and this loss has no curriculum of its own."""
        del progress
        nll = model.predictive_dist(output["predictive_logits"], batch.dec_qry_z)  # [B, n_qry]
        mask = batch.dec_qry_mask.to(nll.dtype)
        loss = (nll * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, {"loss/pred_nll": loss.item()}
