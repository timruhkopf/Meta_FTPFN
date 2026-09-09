"""Training objective -- ARCHITECTURE.md §3.

    L = L_pred
      + lambda_T   * L_transport   (§3.2, deep-supervised, context+query)
      + lambda_C   * L_coupling    (§3.3, barycentric head-0, every layer)
      + lambda_aff * L_affine      (§2.4, global affine, context only)
      + lambda_D   * L_distil      (§3.4, oracle teacher, ramped from 20%)
      + lambda_P   * L_pathway     (§3.5, pooled oracle, rho=0 slice only)

`L_distil`/`L_pathway` need a SECOND (stop-gradient) forward pass of the
same model under a different configuration -- teacher-forced transport, and
severed+pooled-context respectively. That's why `RegistrationLoss.forward`
takes `model` as an argument rather than only a precomputed `output`: the
repo's usual `criterion(output, batch=...)` convention (see
`archive/src/ppfn/model/mymodel/multistream_objective.py`) assumes one
forward pass is enough, which isn't true here (ARCHITECTURE.md §3.4: "Cost
is one extra decoder forward pass per step"). Both extra passes are skipped
entirely (not just zero-weighted) whenever their current lambda is 0 or no
batch item qualifies -- most of training, per the §3.6 schedule -- so the
steady-state per-step cost matches the spec's "one extra pass", not three.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from ppfn.model.registration.heads import GlobalAffineHead
from ppfn.prior.registration.dataset import RegistrationBatch


def deep_supervision_weights(n_layers: int, device=None) -> torch.Tensor:
    """w_l ~ l, sum(w_l) = 1, for l = 1..n_layers -- ARCHITECTURE.md §3.2/§3.3."""
    w = torch.arange(1, n_layers + 1, dtype=torch.float32, device=device)
    return w / w.sum()


def weight_schedule(progress: float) -> dict:
    """ARCHITECTURE.md §3.6. `progress`: fraction of total training steps
    completed, in [0, 1]."""
    t30 = min(progress / 0.30, 1.0)
    lambda_t = 3.0 + t30 * (0.5 - 3.0)
    lambda_c = 1.0 + t30 * (0.2 - 1.0)
    lambda_aff = 0.3
    if progress < 0.20:
        lambda_d = 0.0
    else:
        t_d = min((progress - 0.20) / 0.20, 1.0)
        lambda_d = t_d * 1.0
    lambda_p = 1.0
    return {
        "lambda_T": lambda_t,
        "lambda_C": lambda_c,
        "lambda_aff": lambda_aff,
        "lambda_D": lambda_d,
        "lambda_P": lambda_p,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def _categorical_kl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor
) -> torch.Tensor:
    """KL(teacher || student), per-token -- ARCHITECTURE.md §3.4: "Forward
    KL, deliberately... penalizes the student placing low mass where the
    teacher places high mass." Exact: both distributions are the same
    `TailBarDistribution`'s bins, so the categorical KL over bin
    probabilities equals the KL of the two piecewise-constant densities
    (within-bin density ratio == probability ratio, since bin widths match)."""
    log_p_t = torch.log_softmax(teacher_logits, dim=-1)
    log_p_s = torch.log_softmax(student_logits, dim=-1)
    p_t = log_p_t.exp()
    return (p_t * (log_p_t - log_p_s)).sum(-1)


class RegistrationLoss(nn.Module):
    def __init__(self, affine_weight_is_lambda_aff: bool = True):
        super().__init__()
        # affine_weight_is_lambda_aff kept as a named, documented no-op
        # switch point: lambda_aff is constant per §3.6, so there is
        # currently nothing to toggle, but the weight schedule is read from
        # `weight_schedule` either way -- see that function to change it.
        del affine_weight_is_lambda_aff

    def forward(
        self, model: nn.Module, batch: RegistrationBatch, output: dict, progress: float
    ) -> tuple[torch.Tensor, dict]:
        weights = weight_schedule(progress)
        d_max = model.d_max
        dim_mask_ctx = model.dim_mask(batch.d_real, batch.dec_ctx_x.shape[1])
        dim_mask_qry = model.dim_mask(batch.d_real, batch.dec_qry_x.shape[1])

        metrics: dict[str, float] = {}

        # --- L_pred: query tokens, decoder cloud, NEVER on role-swapped
        # items (ARCHITECTURE.md §4.3: "Swapped passes contribute
        # L_transport and L_coupling only, never L_pred").
        pred_nll = model.predictive_dist(
            output["predictive_logits"], batch.y_qry
        )  # [B,n_qry]
        pred_token_mask = batch.dec_qry_mask & (~batch.role_swapped).unsqueeze(-1)
        l_pred = _masked_mean(pred_nll, pred_token_mask)
        metrics["loss/pred"] = l_pred.item()

        # --- L_transport: deep-supervised, context + query.
        n_layers = len(output["transport_logits_ctx"])
        w = deep_supervision_weights(n_layers, device=l_pred.device)
        l_transport = l_pred.new_zeros(())
        for ell, (logits_ctx, logits_qry) in enumerate(
            zip(output["transport_logits_ctx"], output["transport_logits_qry"])
        ):
            nll_ctx = model.transport_head.nll(
                logits_ctx, batch.transport_ctx, dim_mask_ctx
            )  # [B,n_ctx]
            nll_qry = model.transport_head.nll(
                logits_qry, batch.transport_qry, dim_mask_qry
            )  # [B,n_qry]
            layer_term = 0.5 * (
                _masked_mean(nll_ctx, batch.dec_ctx_mask)
                + _masked_mean(nll_qry, batch.dec_qry_mask)
            )
            l_transport = l_transport + w[ell] * layer_term
        metrics["loss/transport"] = l_transport.item()

        # --- L_coupling: barycentric head-0 output, deep-supervised.
        l_coupling = l_pred.new_zeros(())
        for ell, (bary_ctx, bary_qry) in enumerate(
            zip(output["bary_ctx_layers"], output["bary_qry_layers"])
        ):
            se_ctx = ((bary_ctx - batch.transport_ctx) ** 2 * dim_mask_ctx).sum(
                -1
            ) / dim_mask_ctx.sum(-1).clamp_min(1)
            se_qry = ((bary_qry - batch.transport_qry) ** 2 * dim_mask_qry).sum(
                -1
            ) / dim_mask_qry.sum(-1).clamp_min(1)
            layer_term = 0.5 * (
                _masked_mean(se_ctx, batch.dec_ctx_mask)
                + _masked_mean(se_qry, batch.dec_qry_mask)
            )
            l_coupling = l_coupling + w[ell] * layer_term
        metrics["loss/coupling"] = l_coupling.item()

        # --- L_affine: global affine, context tokens only.
        affine_pred = GlobalAffineHead.apply(
            output["a_g"], output["b_g"], batch.dec_ctx_x
        )
        affine_se = ((affine_pred - batch.transport_ctx) ** 2 * dim_mask_ctx).sum(
            -1
        ) / dim_mask_ctx.sum(-1).clamp_min(1)
        l_affine = _masked_mean(affine_se, batch.dec_ctx_mask)
        metrics["loss/affine"] = l_affine.item()

        total = (
            l_pred
            + weights["lambda_T"] * l_transport
            + weights["lambda_C"] * l_coupling
            + weights["lambda_aff"] * l_affine
        )

        # --- L_distil: oracle teacher (transport teacher-forced), stop-grad.
        l_distil = l_pred.new_zeros(())
        if weights["lambda_D"] > 0.0:
            with torch.no_grad():
                teacher_out = model(
                    batch, transport_override=(batch.transport_ctx, batch.transport_qry)
                )
            kl = _categorical_kl(
                teacher_out["predictive_logits"].detach(), output["predictive_logits"]
            )  # [B,n_qry]
            l_distil = _masked_mean(kl, batch.dec_qry_mask)
            metrics["loss/distil"] = l_distil.item()
            total = total + weights["lambda_D"] * l_distil

        # --- L_pathway: pooled oracle (encoder severed, context = pooled),
        # rho=0 slice only.
        rho_zero_mask = (batch.rho == 0.0) & batch.dec_qry_mask.any(dim=-1)
        if bool(rho_zero_mask.any()):
            pooled_x = torch.cat([batch.dec_ctx_x, batch.enc_x], dim=1)
            pooled_y = torch.cat([batch.dec_ctx_y, batch.enc_y], dim=1)
            pooled_mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)
            # Dummy same-shape transport target for the pooled context: the
            # model always runs the (teacher-forced) transport head
            # internally regardless of severed mode, so this needs SOME
            # value of the right shape -- its content is irrelevant here,
            # since L_pathway only reads `predictive_logits` from this pass
            # and severed_mask=True makes every cross-attention gate (and
            # therefore every use of transport) a no-op anyway.
            pooled_transport = torch.cat(
                [batch.transport_ctx, torch.zeros_like(batch.enc_x)], dim=1
            )
            pooled_batch = dataclasses.replace(
                batch,
                dec_ctx_x=pooled_x,
                dec_ctx_y=pooled_y,
                dec_ctx_mask=pooled_mask,
                transport_ctx=pooled_transport,
            )
            with torch.no_grad():
                pooled_out = model(
                    pooled_batch, severed_mask=torch.ones_like(batch.severed)
                )
            kl_pool = _categorical_kl(
                pooled_out["predictive_logits"].detach(), output["predictive_logits"]
            )
            token_mask = batch.dec_qry_mask & rho_zero_mask.unsqueeze(-1)
            l_pathway = _masked_mean(kl_pool, token_mask)
            metrics["loss/pathway"] = l_pathway.item()
            total = total + weights["lambda_P"] * l_pathway

        metrics["loss/total"] = total.item()
        metrics.update({f"weights/{k}": v for k, v in weights.items()})
        metrics["gate/mean_abs"] = output["gates"].abs().mean().item()
        for ell, g in enumerate(output["gates"].tolist()):
            metrics[f"gate/layer_{ell}"] = g

        return total, metrics


if __name__ == "__main__":
    """Diagnostic: run RegistrationLoss against a real model+batch, print
    every loss component and confirm `total.backward()` populates gradients."""
    import torch

    from ppfn.model.registration.model import RegistrationPFN
    from ppfn.prior.registration.dataset import (
        RegistrationStreamDataset,
        collate_registration_batch,
    )

    torch.manual_seed(0)
    dataset = RegistrationStreamDataset(seed=2, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(6)]
    batch = collate_registration_batch(items)

    model = RegistrationPFN(
        d_model=32, n_heads=4, d_ff=64, n_layers_enc=2, n_layers_dec=3
    )
    criterion = RegistrationLoss()

    for progress in (0.0, 0.25, 0.5):
        output = model(batch)
        loss, metrics = criterion(model, batch, output, progress=progress)
        print(f"progress={progress}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        model.zero_grad()
        loss.backward()
        n_none = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is None
        )
        print(f"  params with grad=None: {n_none}")
