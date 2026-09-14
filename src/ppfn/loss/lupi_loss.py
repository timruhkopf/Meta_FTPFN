"""Loss for `ppfn.model.lupi.model.LUPIPFN` -- build spec §5:

    L = L_student_NLL + lambda_o * L_oracle_NLL + lambda_d * L_distil

Uses `model.bar_dist` (`ppfn.model.pfn.bar_distribution.BarDistribution`,
per user instruction to reuse the existing bar-distribution implementation
rather than `ppfn.model.registration.heads.TailBarDistribution`) for NLL --
a good fit here specifically because targets are already quantile-normalized
to [0,1] (`ppfn.prior.lupi.ecdf`), which is exactly `BarDistribution`'s fixed
support, unlike the z-scored-but-unbounded targets `TailBarDistribution`
exists for.

Distillation is forward KL(p_oracle || p_student) = cross-entropy up to a
p_student-independent constant (spec §5.2), computed directly on the bar
distribution's bin PROBABILITIES rather than densities: `BarDistribution`
here uses uniform-width bins, so the missing `-log(bucket_width)` term is
the SAME additive constant for both p_o and p_s and cancels in the CE
exactly -- no need for spec's optional equal-mass binning refinement.
`p_oracle` is detached (spec: "p_o detached") -- the oracle branch is
already grounded by its own real NLL term via `lambda_o`; distillation must
not additionally pull the oracle toward the student.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch


class LUPILoss(nn.Module):
    def __init__(self, lambda_o: float = 1.0, lambda_d: float = 1.0):
        super().__init__()
        self.lambda_o = lambda_o
        self.lambda_d = lambda_d

    def forward(self, model: nn.Module, batch: LUPIBatch, output: dict | None = None) -> tuple:
        """`output` is accepted-but-unused (matches `IDTokenLoss`'s calling
        convention, `criterion(model, batch, output)`) -- this loss needs
        BOTH positioning modes, which a single upstream `model(batch)` call
        can't provide, so it drives `model.encode_b`/`model.align` itself."""
        b = model.encode_b(batch)
        out_student = model.align(batch, b, mode="student")
        out_oracle = model.align(batch, b, mode="oracle")

        logits_student = out_student["predictive_logits"]
        logits_oracle = out_oracle["predictive_logits"]

        nll_student = model.bar_dist(logits_student, batch.dec_qry_z)  # [B, n_qry]
        nll_oracle = model.bar_dist(logits_oracle, batch.dec_qry_z)

        mask = batch.dec_qry_mask.to(nll_student.dtype)
        denom = mask.sum().clamp_min(1.0)
        loss_student = (nll_student * mask).sum() / denom
        loss_oracle = (nll_oracle * mask).sum() / denom

        p_oracle = torch.softmax(logits_oracle.detach(), dim=-1)
        log_p_student = torch.log_softmax(logits_student, dim=-1)
        distil_ce = -(p_oracle * log_p_student).sum(-1)  # [B, n_qry]
        loss_distil = (distil_ce * mask).sum() / denom

        loss = loss_student + self.lambda_o * loss_oracle + self.lambda_d * loss_distil

        metrics = {
            "loss/total": loss.item(),
            "loss/student_nll": loss_student.item(),
            "loss/oracle_nll": loss_oracle.item(),
            "loss/distil_ce": loss_distil.item(),
            # Spec §7.1's headline gap, the "irreducible cost of not knowing T":
            "loss/oracle_student_gap": (loss_student - loss_oracle).item(),
        }
        return loss, metrics


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: run the loss on a
    small model/batch and sanity-check the oracle branch starts out no
    worse than the student's (it has strictly more information at
    initialization already, before any training) -- not a hard invariant
    post-training, but a useful sign nothing is wired backwards."""
    import torch

    from ppfn.model.lupi.model import LUPIPFN
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(8)]
    batch = collate_lupi_batch(items)

    model = LUPIPFN(d_model=32, n_heads=4, n_layers_enc_b=2, n_layers_align=2, d_ff=64, n_bins_predictive=16)
    criterion = LUPILoss(lambda_o=1.0, lambda_d=1.0)
    loss, metrics = criterion(model, batch)
    print("metrics:", metrics)
