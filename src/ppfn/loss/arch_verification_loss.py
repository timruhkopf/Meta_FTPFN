"""Loss for the "architecture verification" experiment -- CLAUDE.md
build-order step 4, stripped to just what that question needs: can a query
token in A's frame recover what pooling A and B would give it, on
same-domain (rho=0, un-warped) data. Deliberately not
`ppfn.loss.registration_loss.RegistrationLoss`: no transport/coupling/affine
supervision (identity transport is already forced structurally via
`RegistrationTrainer.force_identity_transport`, so those heads would just be
learning an irrelevant, unsupervised-by-this-experiment side task) and no
`weight_schedule` ramp over training progress -- that ramp exists to protect
the full registration curriculum (CLAUDE.md invariants #4/#5) from
collapsing onto an easy attractor over a long run; this experiment never
turns registration on at all, so there's no attractor to guard against and
nothing to schedule.

    L = L_pred + lambda_pathway * L_pathway

Both fixed weights. See `ppfn.monitor.arch_verification` for the companion
three-way NLL comparison (lower bound / upper-2 / encoder-decoder) this
L_pathway term is meant to close -- CLAUDE.md: "Always report three gaps,
never a bare NLL."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.prior.registration.dataset import RegistrationBatch, build_pooled_context


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def _categorical_kl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    """KL(teacher || student), per-token -- see
    ppfn.loss.registration_loss's identical helper for why forward KL."""
    log_p_t = torch.log_softmax(teacher_logits, dim=-1)
    log_p_s = torch.log_softmax(student_logits, dim=-1)
    p_t = log_p_t.exp()
    return (p_t * (log_p_t - log_p_s)).sum(-1)


class ArchVerificationLoss(nn.Module):
    def __init__(self, lambda_pathway: float = 1.0):
        super().__init__()
        self.lambda_pathway = lambda_pathway

    def forward(
        self, model: nn.Module, batch: RegistrationBatch, output: dict, progress: float
    ) -> tuple[torch.Tensor, dict]:
        del progress  # no schedule -- see module docstring
        metrics: dict[str, float] = {}

        # Named to match ppfn.monitor.arch_verification's val-batch
        # counterparts (nll/encoder_decoder) with a train/ prefix -- this is
        # the SAME predictive NLL, just on the training minibatch (in train
        # mode) rather than the held-out validation batch. Kept as a
        # separate, clearly-labeled metric rather than reusing the bare
        # `loss/pred` name from ppfn.loss.registration_loss, which reads as
        # a fourth, unrelated number next to the three held-out NLLs this
        # experiment is actually about.
        pred_nll = model.predictive_dist(output["predictive_logits"], batch.y_qry)
        token_mask = batch.dec_qry_mask & (~batch.role_swapped).unsqueeze(-1)
        l_pred = _masked_mean(pred_nll, token_mask)
        metrics["train/nll_encoder_decoder"] = l_pred.item()

        pooled_batch = build_pooled_context(batch)
        with torch.no_grad():
            pooled_out = model(pooled_batch, severed_mask=torch.ones_like(batch.severed))
        kl = _categorical_kl(
            pooled_out["predictive_logits"].detach(), output["predictive_logits"]
        )
        l_pathway = _masked_mean(kl, token_mask)
        metrics["train/pathway_kl"] = l_pathway.item()

        total = l_pred + self.lambda_pathway * l_pathway
        metrics["train/loss_total"] = total.item()
        return total, metrics


if __name__ == "__main__":
    """Diagnostic: run ArchVerificationLoss against a real model+batch,
    print both loss components, confirm total.backward() populates
    gradients only on the params this experiment actually trains (not the
    transport/affine heads, which get no signal here)."""
    import torch

    from ppfn.model.registration.model import RegistrationPFN
    from ppfn.prior.registration.dataset import (
        RegistrationStreamDataset,
        collate_registration_batch,
    )

    torch.manual_seed(0)
    dataset = RegistrationStreamDataset(seed=2, s_max=0.1, force_rho_zero=True)
    items = [next(iter(dataset)) for _ in range(6)]
    batch = collate_registration_batch(items)

    model = RegistrationPFN(
        d_model=32, n_heads=4, d_ff=64, n_layers_enc=2, n_layers_dec=3
    )
    criterion = ArchVerificationLoss()

    transport_override = (batch.dec_ctx_x, batch.dec_qry_x)
    output = model(batch, transport_override=transport_override)
    loss, metrics = criterion(model, batch, output, progress=0.0)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    model.zero_grad()
    loss.backward()
    transport_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.transport_head.parameters()
    )
    print(f"transport_head received gradient (expect False): {transport_head_has_grad}")
