"""Minimal, three-number monitor set for the "architecture verification"
experiment (CLAUDE.md build-order step 4, reduced) -- deliberately NOT
`ppfn.monitor.registry`'s full dashboard (transport/coupling per layer, gate
magnitudes, weight schedule, rho-anchor split): those are about the
registration curriculum this experiment doesn't run, and would just be
noise here.

All three NLLs are computed by pushing the SAME fixed validation batch's
SAME query tokens through three configurations of one model -- CLAUDE.md:
"Always report three gaps, never a bare NLL":

  nll/lower_severed    -- decoder-only, A_test attends to A_train only
                          (severed_mask=True on the plain batch). The lower
                          bound: no information from B at all.
  nll/upper2_pooled     -- decoder-only on the pooled context [A ; B_inA]
                          (severed_mask=True on the pooled batch,
                          ppfn.prior.registration.dataset.build_pooled_context).
                          The oracle target this experiment's L_pathway
                          trains toward.
  nll/encoder_decoder   -- the actual two-stream model, cross-attention
                          active (severed_mask=False). What we're actually
                          asking: did the model learn to use its memory of
                          B, and does doing so via cross-attention close the
                          gap to the pooled oracle above.
"""

from __future__ import annotations

import torch

from ppfn.monitor.registry import MonitorContext
from ppfn.prior.registration.dataset import build_pooled_context


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return ((values * mask_f).sum() / denom).item()


def compute_arch_verification_metrics(ctx: MonitorContext) -> dict:
    batch = ctx.val_batch
    token_mask = batch.dec_qry_mask & (~batch.role_swapped).unsqueeze(-1)

    out_lower = ctx.model(batch, severed_mask=torch.ones_like(batch.severed))
    nll_lower = ctx.model.predictive_dist(out_lower["predictive_logits"], batch.y_qry)

    pooled_batch = build_pooled_context(batch)
    out_upper2 = ctx.model(pooled_batch, severed_mask=torch.ones_like(batch.severed))
    nll_upper2 = ctx.model.predictive_dist(out_upper2["predictive_logits"], batch.y_qry)

    out_encdec = ctx.model(batch, severed_mask=torch.zeros_like(batch.severed))
    nll_encdec = ctx.model.predictive_dist(out_encdec["predictive_logits"], batch.y_qry)

    return {
        "nll/lower_severed": _masked_mean(nll_lower, token_mask),
        "nll/upper2_pooled": _masked_mean(nll_upper2, token_mask),
        "nll/encoder_decoder": _masked_mean(nll_encdec, token_mask),
    }


if __name__ == "__main__":
    import torch

    from ppfn.model.registration.model import RegistrationPFN
    from ppfn.prior.registration.dataset import (
        RegistrationStreamDataset,
        collate_registration_batch,
    )

    torch.manual_seed(0)
    dataset = RegistrationStreamDataset(seed=3, s_max=0.1, force_rho_zero=True)
    items = [next(iter(dataset)) for _ in range(8)]
    batch = collate_registration_batch(items)

    model = RegistrationPFN(d_model=32, n_heads=4, d_ff=64, n_layers_enc=2, n_layers_dec=3)
    model.eval()
    with torch.no_grad():
        metrics = compute_arch_verification_metrics(MonitorContext(model=model, val_batch=batch))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
