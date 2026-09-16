"""Loss for `ppfn.model.baselines.lupi_id_token_pfn.LUPIIDTokenPFN` --
three terms, all on A's query tokens only (`batch.dec_qry_mask`):

    L = NLL(student_logits, dec_qry_z) + NLL(teacher_logits, dec_qry_z)
        + lambda_ce * CE(teacher_logits.detach(), student_logits)

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
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch

_LOG_PROB_CLAMP_MIN = -50.0  # ~2e-22 probability; see module docstring


class LUPIIDTokenLoss(nn.Module):
    def __init__(self, lambda_ce: float = 1.0):
        super().__init__()
        self.lambda_ce = lambda_ce

    def forward(self, model: nn.Module, batch: LUPIBatch, output: dict) -> tuple:
        mask = batch.dec_qry_mask.to(torch.float32)
        denom = mask.sum().clamp_min(1.0)

        nll_student_tok = model.predictive_dist(output["student_logits"], batch.dec_qry_z)
        nll_teacher_tok = model.predictive_dist(output["teacher_logits"], batch.dec_qry_z)
        nll_student = (nll_student_tok * mask).sum() / denom
        nll_teacher = (nll_teacher_tok * mask).sum() / denom

        log_p_student = torch.log_softmax(output["student_logits"], dim=-1).clamp_min(
            _LOG_PROB_CLAMP_MIN
        )
        p_teacher = torch.softmax(output["teacher_logits"].detach(), dim=-1)
        ce_tok = -(p_teacher * log_p_student).sum(-1)  # [B, n_qry]
        ce = (ce_tok * mask).sum() / denom

        total = nll_student + nll_teacher + self.lambda_ce * ce
        return total, {
            "loss/total": total.item(),
            "loss/nll_student": nll_student.item(),
            "loss/nll_teacher": nll_teacher.item(),
            "loss/ce_distil": ce.item(),
        }
