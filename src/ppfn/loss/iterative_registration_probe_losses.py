"""Losses for the component-isolation probes,
`ppfn.model.baselines.iterative_registration_probes`. Same deep-supervision
(`w_k ∝ k`) + final-iteration distributional-loss pattern as
`ppfn.loss.iterative_registration_loss`, just scoped to ONE of the two
auxiliary targets each, since each probe only has that one mechanism to
verify."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


def _dim_mask(d_real: torch.Tensor, d_max: int) -> torch.Tensor:
    idx = torch.arange(d_max, device=d_real.device).view(1, d_max)
    return idx < d_real.view(-1, 1)


class ValueStepProbeLoss(nn.Module):
    def forward(self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0) -> tuple:
        enc_mask = batch.enc_mask.to(torch.float32)
        denom = enc_mask.sum().clamp_min(1.0)

        deep = torch.zeros((), device=enc_mask.device)
        w_total = 0.0
        for k, h_hat_k in enumerate(output["h_hat_per_iter"]):
            w_k = float(k + 1)
            se = (h_hat_k - batch.enc_z_inA) ** 2
            deep = deep + w_k * (se * enc_mask).sum() / denom
            w_total += w_k
        deep = deep / w_total

        mu, log_sigma = output["h_hat_final"], output["log_sigma_final"]
        sigma = log_sigma.exp().clamp_min(1e-4)
        gaussian_nll = 0.5 * math.log(2 * math.pi) + log_sigma + 0.5 * ((batch.enc_z_inA - mu) / sigma) ** 2
        # beta-NLL (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic
        # Uncertainty Estimation") -- plain NLL's gradient w.r.t. mu is
        # implicitly down-weighted by 1/sigma^2, so once sigma drifts too
        # large somewhere, mu stops getting corrected there and inflating
        # sigma further becomes the cheapest remaining way to reduce loss --
        # empirically confirmed here (2026-09-17): sigma drifted
        # 0.87 -> 13-15 over 40 steps on a fixed batch while the point
        # estimate (this same loss's own "deep" MSE term) kept improving
        # fine, i.e. genuinely a training-dynamics pathology, not the
        # underlying mechanism being wrong. Reweighting by sigma^2 (detached
        # -- it's a per-sample scale on the loss, not something to
        # differentiate through here) restores a well-behaved gradient on mu.
        beta_nll_weight = (sigma.detach() ** 2).pow(0.5)
        gaussian_nll = beta_nll_weight * gaussian_nll
        final = (gaussian_nll * enc_mask).sum() / denom

        total = deep + final
        return total, {"loss/total": total.item(), "loss/h_deep": deep.item(), "loss/h_final_nll": final.item()}


class PositionStepProbeLoss(nn.Module):
    def forward(self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0) -> tuple:
        enc_mask = batch.enc_mask.to(torch.float32)
        denom = enc_mask.sum().clamp_min(1.0)
        d_max = batch.dec_ctx_x.shape[-1]
        dmask = _dim_mask(batch.d_real, d_max).to(torch.float32).unsqueeze(1)  # [B, 1, d_max]
        dmask_sum = dmask.sum(-1).clamp_min(1.0)

        deep = torch.zeros((), device=enc_mask.device)
        w_total = 0.0
        for k, T_hat_k in enumerate(output["T_hat_per_iter"]):
            w_k = float(k + 1)
            se = ((T_hat_k - batch.enc_x_inA) ** 2 * dmask).sum(-1) / dmask_sum
            deep = deep + w_k * (se * enc_mask).sum() / denom
            w_total += w_k
        deep = deep / w_total

        n_B = batch.enc_x.shape[1]
        dmask_bool = _dim_mask(batch.d_real, d_max).unsqueeze(1).expand(-1, n_B, -1)
        transport_nll = model.transport_head.nll(output["transport_logits"], batch.enc_x_inA, dmask_bool)
        final = (transport_nll * enc_mask).sum() / denom

        total = deep + final
        return total, {"loss/total": total.item(), "loss/T_deep": deep.item(), "loss/T_final_nll": final.item()}
