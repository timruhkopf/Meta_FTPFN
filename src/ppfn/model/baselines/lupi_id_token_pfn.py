"""LUPI (Learning Using Privileged Information)-style wrapper around
`IDTokenPFN` -- the user's follow-up to the plain additive-ID-token
baseline: registering two tasks purely in-context (`IDTokenPFN` alone) may
still be hard, so this checks how much of the gap closes when a
*privileged* view of B (already pushed into A's frame via the TRUE,
ground-truth transport -- `LUPIBatch.enc_x_inA`, valid at any rho, see
`ppfn.prior.lupi.dataset`/`ppfn.prior.lupi.sampler`) is available as a
training-time-only signal.

Runs against `ppfn.prior.lupi`'s prior specifically (not the plain
`ppfn.prior.registration` prior this module originally targeted) -- see
`ppfn.model.baselines.id_token_pfn`'s module docstring for why: the plain
prior has no y-distortion `h` and no acquisition-biased A design, so it
can't test the actual question this experiment is for.

One shared-weight `IDTokenPFN` backbone, run on two views of the SAME
underlying pair in a single batch-stacked forward pass (student items and
teacher items concatenated along the batch dimension, split back after --
half the forward-pass overhead of two separate calls):

  * student: `[A_context ; B]` (raw, unregistered B) -- what the model
    actually gets to use at inference.
  * teacher: `[A_context ; B_inA]` (B pushed into A's frame via the true
    transport) -- privileged information, training-time only.

Both score their own bar-distribution NLL on A's query tokens (never B's --
same invariant as `IDTokenLoss`). Student is additionally pulled toward the
teacher's (detached) predictive distribution via a plain cross-entropy term
-- not the project's usual forward KL (`ppfn.loss.registration_loss._categorical_kl`):
CE(p_teacher, q_student) = KL(p_teacher || q_student) + H(p_teacher), and
H(p_teacher) has zero gradient w.r.t. the student's parameters once the
teacher is detached, so this trains identically to forward KL while being
simpler and, per docs/labbook/2026-09-10-categorical-kl-nan-underflow.md,
strictly less NaN-prone: that entry's failure mode 2 (teacher underflows a
bin to exactly zero probability while its log-prob is -inf) can't arise
here at all, since plain CE never evaluates `log(p_teacher)`. Failure mode
1 (student underflows a bin to log-prob -inf while the teacher assigns it
nonzero mass) still applies, so the same clamp fix is applied here too.

`AAloneIDTokenPFN`/`OracleIDTokenPFN` below are the two reference bounds
the user asked for -- same `IDTokenPFN` backbone, trained as SEPARATE,
dedicated runs (not read off this wrapper's own student/teacher pathways,
which are entangled with the joint CE objective and wouldn't be a clean
bound): B masked out entirely (lower bound, "A alone" -- CLAUDE.md build
order step 3's severed baseline) and B always replaced by the oracle
`enc_x_inA` (upper bound, the same privileged view the teacher above uses,
but trained on its own rather than jointly with a student).
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from ppfn.model.baselines.id_token_pfn import IDTokenPFN
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


def _cat_batches(a: LUPIBatch, b: LUPIBatch) -> LUPIBatch:
    """Concatenate two same-shape-family LUPIBatches along the batch
    dimension. Both inputs must already be padded to the same per-field
    point counts (true here: `b` is `a` with only `enc_x` replaced, see
    `LUPIIDTokenPFN.forward` -- every other field is identical, so no
    re-padding is needed)."""
    fields = {}
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, torch.Tensor):
            fields[f.name] = torch.cat([va, vb], dim=0)
        elif isinstance(va, list):
            fields[f.name] = va + vb
        else:
            fields[f.name] = va
    return LUPIBatch(**fields)


class LUPIIDTokenPFN(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        n_bins_predictive: int = 64,
        dropout: float = 0.0,
        bounded01: bool = False,
    ):
        super().__init__()
        self.backbone = IDTokenPFN(
            d_max=d_max, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            d_ff=d_ff, n_bins_predictive=n_bins_predictive, dropout=dropout,
            bounded01=bounded01,
        )

    @property
    def predictive_dist(self):
        return self.backbone.predictive_dist

    def forward(self, batch: LUPIBatch) -> dict:
        """-> {"student_logits", "teacher_logits"}, each [B, n_qry, n_bins_predictive]."""
        # Teacher = FULLY registered B: position via enc_x_inA (T) AND value
        # via enc_z_inA (h) -- both replaced together. Using enc_x_inA with
        # the unchanged enc_z (B's raw, un-h'd value) would leave the
        # teacher itself with an unsolved value-calibration problem, which
        # breaks the whole point of distilling the student toward it: the
        # CE term is only a meaningful "registration" signal if the teacher
        # it pulls toward is what a model would predict GIVEN registration
        # is fully solved (2026-09-16, at the user's explicit correction --
        # an earlier version of this pathway used enc_z here, matching a
        # since-corrected LUPIPair.z_b_inA bug, see that field's docstring).
        teacher_batch = dataclasses.replace(batch, enc_x=batch.enc_x_inA, enc_z=batch.enc_z_inA)
        stacked = _cat_batches(batch, teacher_batch)
        out = self.backbone(stacked)
        b = batch.dec_ctx_x.shape[0]
        logits = out["predictive_logits"]
        return {"student_logits": logits[:b], "teacher_logits": logits[b:]}


class AAloneIDTokenPFN(IDTokenPFN):
    """Lower-bound reference: B masked out entirely, every context token is
    A. Same backbone as the student/teacher above, trained standalone."""

    def forward(self, batch: LUPIBatch) -> dict:
        batch = dataclasses.replace(batch, enc_mask=torch.zeros_like(batch.enc_mask))
        return super().forward(batch)


class OracleIDTokenPFN(IDTokenPFN):
    """Upper-bound reference: B always replaced by its FULLY registered
    (position AND value, enc_x_inA/enc_z_inA) view. Same privileged view the
    LUPI teacher above uses, but trained on its own rather than jointly with
    a student + CE term."""

    def forward(self, batch: LUPIBatch) -> dict:
        batch = dataclasses.replace(batch, enc_x=batch.enc_x_inA, enc_z=batch.enc_z_inA)
        return super().forward(batch)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md."""
    import torch

    from ppfn.loss.lupi_id_token_loss import LUPIIDTokenLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = LUPIIDTokenPFN(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("student_logits:", out["student_logits"].shape)
    print("teacher_logits:", out["teacher_logits"].shape)

    criterion = LUPIIDTokenLoss(lambda_ce=1.0)
    loss, metrics = criterion(model, batch, out)
    print("loss/metrics:", metrics)

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None: {n_none}")

    # rho=0 invariant: T is the identity at rho=0, so enc_x_inA == enc_x --
    # but h is sampled independently of rho, so enc_z_inA != enc_z in
    # general even here. Student and teacher logits are therefore NOT
    # expected to match at rho=0 (corrected 2026-09-16 -- an earlier version
    # of this check asserted logit equality, which only held because the
    # teacher was, at the time, wrongly paired with enc_z instead of
    # enc_z_inA; see LUPIPair.z_b_inA's docstring). What's still an
    # invariant is the POSITION-only piece.
    from ppfn.prior.lupi.dataset import build_training_item
    import numpy as np

    rng = np.random.default_rng(0)
    rho0_items = [build_training_item(rng, progress=0.5, s_max=0.1, force_rho_zero=True) for _ in range(4)]
    rho0_batch = collate_lupi_batch(rho0_items)
    pos_diff = (rho0_batch.enc_x_inA - rho0_batch.enc_x).abs().max().item()
    print("max |enc_x_inA - enc_x| at rho=0 (~0 expected, T is the identity):", pos_diff)
    with torch.no_grad():
        rho0_out = model(rho0_batch)
    diff = (rho0_out["student_logits"] - rho0_out["teacher_logits"]).abs().max().item()
    print("max |student - teacher| logits at rho=0 (NOT expected to be 0 -- h still differs enc_z vs enc_z_inA):", diff)

    # Bound models: forward shape + masking sanity.
    a_alone = AAloneIDTokenPFN(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
    oracle = OracleIDTokenPFN(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
    print("AAloneIDTokenPFN predictive_logits:", a_alone(batch)["predictive_logits"].shape)
    print("OracleIDTokenPFN predictive_logits:", oracle(batch)["predictive_logits"].shape)
