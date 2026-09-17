"""Loss for `ppfn.model.baselines.iterative_registration_pfn.IterativeRegistrationPFN`.

Four terms:

    L = NLL(predictive_logits, dec_qry_z)                                    # final prediction, A's query only
        + lambda_T * [ deep-supervised MSE(T_hat^(k), enc_x_inA), w_k ∝ k    # cheap, every iteration -- RAFT's own
                        + TransportHead.nll(transport_logits, enc_x_inA) ]   # calibrated, last iteration only
        + lambda_h * [ deep-supervised MSE(h_hat^(k), enc_z_inA), w_k ∝ k
                        + Gaussian_NLL(aux_value_mu, aux_value_log_sigma, enc_z_inA) ]

Per-iteration deep supervision uses plain MSE, not a distributional loss --
matching RAFT's (Teed & Deng 2020) own per-iteration flow supervision, which
is exactly this kind of "cheap signal at every step, no per-step uncertainty
modeling" pattern. The genuine-uncertainty concern the labbook's §2 raises
(a point loss can't express that the student may not be able to know `T`/`h`
exactly) is answered by the LAST-iteration terms specifically: `TransportHead`
is already a proper (autoregressive-over-axes, potentially multimodal) bar
distribution, and the Gaussian NLL on `h` lets the model widen `sigma` instead
of being forced into one overconfident guess. Intermediate iterations don't
need that same treatment -- they're a cheap "point in the right direction"
signal guiding convergence, not the number anything gets evaluated against.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


def _dim_mask(d_real: torch.Tensor, d_max: int) -> torch.Tensor:
    idx = torch.arange(d_max, device=d_real.device).view(1, d_max)
    return idx < d_real.view(-1, 1)


class IterativeRegistrationLoss(nn.Module):
    def __init__(self, lambda_T: float = 0.2, lambda_h: float = 0.2):
        super().__init__()
        self.lambda_T = lambda_T
        self.lambda_h = lambda_h

    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        qry_mask = batch.dec_qry_mask.to(torch.float32)
        qry_denom = qry_mask.sum().clamp_min(1.0)
        nll = model.predictive_dist(output["predictive_logits"], batch.dec_qry_z)
        loss_pred = (nll * qry_mask).sum() / qry_denom

        enc_mask = batch.enc_mask.to(torch.float32)
        enc_denom = enc_mask.sum().clamp_min(1.0)
        d_max = batch.dec_ctx_x.shape[-1]
        dmask = _dim_mask(batch.d_real, d_max).to(torch.float32).unsqueeze(1)  # [B, 1, d_max]
        dmask_sum = dmask.sum(-1).clamp_min(1.0)  # [B, 1]

        deep_T = torch.zeros((), device=nll.device)
        deep_h = torch.zeros((), device=nll.device)
        w_total = 0.0
        for k, (T_hat_k, h_hat_k) in enumerate(zip(output["T_hat_per_iter"], output["h_hat_per_iter"])):
            w_k = float(k + 1)
            se = ((T_hat_k - batch.enc_x_inA) ** 2 * dmask).sum(-1) / dmask_sum  # [B, n_B]
            deep_T = deep_T + w_k * (se * enc_mask).sum() / enc_denom
            se_h = (h_hat_k - batch.enc_z_inA) ** 2  # [B, n_B]
            deep_h = deep_h + w_k * (se_h * enc_mask).sum() / enc_denom
            w_total += w_k
        deep_T = deep_T / w_total
        deep_h = deep_h / w_total

        n_B = batch.enc_x.shape[1]
        dmask_bool = _dim_mask(batch.d_real, d_max).unsqueeze(1).expand(-1, n_B, -1)  # [B, n_B, d_max]
        transport_nll = model.transport_head.nll(
            output["transport_logits"], batch.enc_x_inA, dmask_bool
        )  # [B, n_B]
        final_T = (transport_nll * enc_mask).sum() / enc_denom

        mu, log_sigma = output["aux_value_mu"], output["aux_value_log_sigma"]
        sigma = log_sigma.exp().clamp_min(1e-4)
        gaussian_nll = (
            0.5 * math.log(2 * math.pi)
            + log_sigma
            + 0.5 * ((batch.enc_z_inA - mu) / sigma) ** 2
        )  # [B, n_B]
        # beta-NLL reweighting (Seitzer et al. 2022) -- see
        # ppfn.loss.iterative_registration_probe_losses.ValueStepProbeLoss's
        # own comment for the empirically-confirmed runaway-variance
        # pathology this fixes (same head, same mechanism, here too).
        gaussian_nll = (sigma.detach() ** 2).pow(0.5) * gaussian_nll
        final_h = (gaussian_nll * enc_mask).sum() / enc_denom

        loss_T = deep_T + final_T
        loss_h = deep_h + final_h
        total = loss_pred + self.lambda_T * loss_T + self.lambda_h * loss_h

        return total, {
            "loss/total": total.item(),
            "loss/pred_nll": loss_pred.item(),
            "loss/T_deep": deep_T.item(),
            "loss/T_final_nll": final_T.item(),
            "loss/h_deep": deep_h.item(),
            "loss/h_final_nll": final_h.item(),
        }
