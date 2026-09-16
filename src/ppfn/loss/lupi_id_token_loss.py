"""Loss for `ppfn.model.baselines.lupi_id_token_pfn.LUPIIDTokenPFN` --
three terms, all on A's query tokens only (`batch.dec_qry_mask`):

    L = student_weight(progress) * NLL(student_logits, dec_qry_z)
        + NLL(teacher_logits, dec_qry_z)
        + lambda_ce * CE(teacher_logits.detach() / T(progress), student_logits / T(progress))

See that module's docstring for why plain cross-entropy rather than the
project's usual forward KL (`ppfn.loss.registration_loss._categorical_kl`):
mathematically identical gradient w.r.t. the student once the teacher is
detached (CE = KL + teacher entropy, and the entropy term doesn't depend on
the student), and structurally immune to one of that helper's two documented
NaN-underflow failure modes (docs/labbook/2026-09-10-categorical-kl-nan-underflow.md)
since CE never evaluates `log(p_teacher)`. The other failure mode (student
underflowing a bin's probability to exactly zero while the teacher assigns
it nonzero mass) is guarded the same way that fix did: clamping the
student's log-softmax before multiplying by the teacher's probabilities.

`student_weight`/temperature curriculum (2026-09-16, at the user's
direction): both weights share every parameter, so early in training the
shared attention/embedding stack has to simultaneously serve two different
input distributions (raw B vs. registered B) while ALSO being pulled to
make the raw-B (student) output resemble the registered-B (teacher)
output. `NLL(student)` against real, noisy A-targets is a higher-variance,
harder-won signal before the student pathway has learned to read raw B's
registration cues at all -- ramping it in from a floor (never exactly 0,
same discipline as `lambda_T`'s floor in the registration loss, CLAUDE.md
invariant #5: a term that CAN be safely ignored is a stable, reachable
attractor) lets the shared backbone first build up whatever "read raw B,
predict what a registered view would say" computation the CE term is
already asking for, using the teacher's own (better-calibrated, since it's
a full distribution rather than one noisy scalar) target -- before also
asking it to nail A's targets directly.

Deliberately faster than `sample_rho_curriculum`'s own ~30%-of-training
ramp (see that function's docstring) -- if both curricula finished
easing in at the same time, the genuinely hard regime (hard rho AND real
NLL pressure on the student, simultaneously) would only ever be trained
on very late. `ramp_frac` defaults to 0.2 for exactly this reason.

Temperature anneal on the CE term (2.0 -> 1.0 over the same ramp,
standard knowledge-distillation practice -- both student and teacher
logits are divided by the SAME temperature before the cross-entropy, per
Hinton et al. 2015, not just the teacher: softening only one side would
compare distributions at mismatched sharpness and bias the gradient):
an early, undertrained teacher's distribution can be overconfident in the
wrong place; softening makes an early bad target less committal instead
of actively misleading, independent of and complementary to the
student-weight ramp above.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch

_LOG_PROB_CLAMP_MIN = -50.0  # ~2e-22 probability; see module docstring


class LUPIIDTokenLoss(nn.Module):
    def __init__(
        self,
        lambda_ce: float = 1.0,
        student_weight_floor: float = 0.15,
        temperature_max: float = 2.0,
        ramp_frac: float = 0.2,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.student_weight_floor = student_weight_floor
        self.temperature_max = temperature_max
        self.ramp_frac = ramp_frac

    def forward(
        self, model: nn.Module, batch: LUPIBatch, output: dict, progress: float = 1.0
    ) -> tuple:
        mask = batch.dec_qry_mask.to(torch.float32)
        denom = mask.sum().clamp_min(1.0)

        ramp = min(max(progress, 0.0) / self.ramp_frac, 1.0)
        student_weight = self.student_weight_floor + (1.0 - self.student_weight_floor) * ramp
        temperature = 1.0 + (self.temperature_max - 1.0) * (1.0 - ramp)

        nll_student_tok = model.predictive_dist(output["student_logits"], batch.dec_qry_z)
        nll_teacher_tok = model.predictive_dist(output["teacher_logits"], batch.dec_qry_z)
        nll_student = (nll_student_tok * mask).sum() / denom
        nll_teacher = (nll_teacher_tok * mask).sum() / denom

        log_p_student = torch.log_softmax(
            output["student_logits"] / temperature, dim=-1
        ).clamp_min(_LOG_PROB_CLAMP_MIN)
        p_teacher = torch.softmax(output["teacher_logits"].detach() / temperature, dim=-1)
        ce_tok = -(p_teacher * log_p_student).sum(-1)  # [B, n_qry]
        ce = (ce_tok * mask).sum() / denom

        total = student_weight * nll_student + nll_teacher + self.lambda_ce * ce
        return total, {
            "loss/total": total.item(),
            "loss/nll_student": nll_student.item(),
            "loss/nll_teacher": nll_teacher.item(),
            "loss/ce_distil": ce.item(),
            "train/student_weight": student_weight,
            "train/ce_temperature": temperature,
        }
