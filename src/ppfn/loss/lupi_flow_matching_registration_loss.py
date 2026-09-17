"""Loss for `ppfn.model.baselines.lupi_flow_matching_pfn.FlowMatchingRegistrationPFN`
-- four terms, mirroring `LUPIIDTokenLoss`'s student/teacher/CE structure
plus the velocity-regression term `FlowMatchingLoss` already owns:

    L = lambda_flow * L_flow(v_phi)                                   # trains v_phi only
        + NLL(student_mixture_logits, dec_qry_z)                      # trains predictor, against the SDE's own registration
        + NLL(teacher_logits, dec_qry_z)                               # trains predictor, upper-1 (true B_inA)
        + lambda_ce * CE(teacher_logits.detach(), student_mixture_logits)

All predictive terms scored on `dec_qry_mask` only (CLAUDE.md invariant #6:
predictive loss scores decoder-cloud queries, never encoder-cloud targets).
`lambda_flow`, `lambda_ce` fixed constants, no curriculum -- this is the
first cut of this pathway, and per this repo's own precedent (the labbook's
"start with small fixed constants ... before tuning a schedule on top of a
schedule"), isolating whether the coupling helps at all comes before tuning
how it's weighted."""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.loss.lupi_flow_matching_loss import FlowMatchingLoss
from ppfn.prior.lupi.dataset import LUPIBatch

_LOG_PROB_CLAMP_MIN = -50.0  # see LUPIIDTokenLoss's own docstring for why


class FlowMatchingRegistrationLoss(nn.Module):
    def __init__(self, lambda_flow: float = 1.0, lambda_ce: float = 1.0):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_ce = lambda_ce
        self.flow_loss = FlowMatchingLoss()

    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        del progress
        flow_loss, flow_metrics = self.flow_loss(model.flow_field, batch, output["fm_output"])

        mask = batch.dec_qry_mask.to(torch.float32)
        denom = mask.sum().clamp_min(1.0)

        nll_student_tok = model.bar_dist(output["student_logits"], batch.dec_qry_z)
        nll_teacher_tok = model.bar_dist(output["teacher_logits"], batch.dec_qry_z)
        nll_student = (nll_student_tok * mask).sum() / denom
        nll_teacher = (nll_teacher_tok * mask).sum() / denom

        log_p_student = torch.log_softmax(
            output["student_logits"], dim=-1
        ).clamp_min(_LOG_PROB_CLAMP_MIN)
        p_teacher = torch.softmax(output["teacher_logits"].detach(), dim=-1)
        ce_tok = -(p_teacher * log_p_student).sum(-1)
        ce = (ce_tok * mask).sum() / denom

        total = self.lambda_flow * flow_loss + nll_student + nll_teacher + self.lambda_ce * ce
        metrics = {
            "loss/total": total.item(),
            "loss/nll_student": nll_student.item(),
            "loss/nll_teacher": nll_teacher.item(),
            "loss/ce_distil": ce.item(),
            "loss/upper1_gap": (nll_student - nll_teacher).item(),
            **{f"flow/{k.split('/', 1)[1]}": v for k, v in flow_metrics.items()},
        }
        return total, metrics
