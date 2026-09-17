"""Loss for `ppfn.model.baselines.lupi_flow_matching_pfn.FlowMatchingRegistrationPFN`
-- four terms, mirroring `LUPIIDTokenLoss`'s student/teacher/CE structure
plus the velocity-regression term `FlowMatchingLoss` already owns:

    L = lambda_flow * L_flow(v_phi)                                          # trains v_phi only
        + student_weight(progress) * NLL(student_mixture_logits, dec_qry_z)  # trains predictor, against the SDE's own registration
        + NLL(teacher_logits, dec_qry_z)                                      # trains predictor, upper-1 (true B_inA)
        + lambda_ce * CE(teacher_logits.detach()/T(progress), student_logits/T(progress))

All predictive terms scored on `dec_qry_mask` only (CLAUDE.md invariant #6:
predictive loss scores decoder-cloud queries, never encoder-cloud targets).

`student_weight`/temperature curriculum added 2026-09-17 (REVISION -- the
first cut fixed both at 1.0/no-schedule, "isolate whether the coupling
helps before tuning how it's weighted"; this is that follow-up tuning pass,
not a design reversal). Real-run evidence from
`01-pretraining-lupi-flow-matching-registration` epochs 0-20 motivated it:
`loss/ce_distil` (~2.6-4.0) dwarfs both NLL terms (~-2 to 0) in
`loss/total`'s own magnitude for the entire observed window, while
`loss/upper1_gap` (`nll_student - nll_teacher`) grew fast early then
plateaued noisily around 0.35-0.5 rather than shrinking -- consistent with
`nll_student`'s own proper-scoring-rule gradient (the signal that should
directly improve the STUDENT's real predictive quality) being swamped by
the CE term pulling the shared backbone toward matching the teacher's
distribution SHAPE instead. Exactly the failure mode `LUPIIDTokenLoss`'s
own docstring already names and fixes for the analogous `IDTokenPFN`
architecture -- same fix, same default hyperparameters (`floor=0.15`,
`temperature_max=2.0`, `ramp_frac=0.2`), reused here rather than re-derived,
for direct comparability against that precedent. See
`docs/labbook/2026-09-17-flow-matching-registration-real-run-launch.md` for
the full epoch-by-epoch evidence this responds to."""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.loss.lupi_flow_matching_loss import FlowMatchingLoss
from ppfn.prior.lupi.dataset import LUPIBatch

_LOG_PROB_CLAMP_MIN = -50.0  # see LUPIIDTokenLoss's own docstring for why


class FlowMatchingRegistrationLoss(nn.Module):
    def __init__(
        self,
        lambda_flow: float = 1.0,
        lambda_ce: float = 1.0,
        student_weight_floor: float = 0.15,
        temperature_max: float = 2.0,
        ramp_frac: float = 0.2,
    ):
        super().__init__()
        self.lambda_flow = lambda_flow
        self.lambda_ce = lambda_ce
        self.student_weight_floor = student_weight_floor
        self.temperature_max = temperature_max
        self.ramp_frac = ramp_frac
        self.flow_loss = FlowMatchingLoss()

    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        flow_loss, flow_metrics = self.flow_loss(model.flow_field, batch, output["fm_output"])

        mask = batch.dec_qry_mask.to(torch.float32)
        denom = mask.sum().clamp_min(1.0)

        ramp = min(max(progress, 0.0) / self.ramp_frac, 1.0)
        student_weight = self.student_weight_floor + (1.0 - self.student_weight_floor) * ramp
        temperature = 1.0 + (self.temperature_max - 1.0) * (1.0 - ramp)

        nll_student_tok = model.bar_dist(output["student_logits"], batch.dec_qry_z)
        nll_teacher_tok = model.bar_dist(output["teacher_logits"], batch.dec_qry_z)
        nll_student = (nll_student_tok * mask).sum() / denom
        nll_teacher = (nll_teacher_tok * mask).sum() / denom

        log_p_student = torch.log_softmax(
            output["student_logits"] / temperature, dim=-1
        ).clamp_min(_LOG_PROB_CLAMP_MIN)
        p_teacher = torch.softmax(output["teacher_logits"].detach() / temperature, dim=-1)
        ce_tok = -(p_teacher * log_p_student).sum(-1)
        ce = (ce_tok * mask).sum() / denom

        total = (
            self.lambda_flow * flow_loss + student_weight * nll_student + nll_teacher
            + self.lambda_ce * ce
        )
        metrics = {
            "loss/total": total.item(),
            "loss/nll_student": nll_student.item(),
            "loss/nll_teacher": nll_teacher.item(),
            "loss/ce_distil": ce.item(),
            "loss/upper1_gap": (nll_student - nll_teacher).item(),
            "train/student_weight": student_weight,
            "train/ce_temperature": temperature,
            **{f"flow/{k.split('/', 1)[1]}": v for k, v in flow_metrics.items()},
        }
        return total, metrics
