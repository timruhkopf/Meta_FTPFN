"""Flexible monitor registry -- ARCHITECTURE.md §4.5's live-monitor table,
and CLAUDE.md's "Keep the monitor set declared in one module; nothing gets
logged that isn't declared there, or the dashboard stops being readable."

Deliberately a flat function registry, not a class hierarchy: each monitor
is a plain `(MonitorContext) -> dict[str, float]` function, registered with
`@register_monitor("name")` anywhere in the codebase (this module is where
the built-in §4.5 monitors live, but a downstream experiment can register
its own in its own file just by importing this module and decorating a
function -- nothing here needs subclassing or a new lifecycle hook). This
replaces the older `AbstractCallback`/`CallbackHandler` pattern
(`ppfn.trainer.callbacks`) for THIS purpose specifically: that scheme's
many lifecycle hooks (`on_epoch_end`, `on_step_end`, `log_on_epoch_end`,
...) are overkill for "run this batch of pure functions on a validation
batch and log whatever they return," which is all the monitor set needs.
The training-run lifecycle itself (starting/ending an MLflow run, tagging
git provenance) still goes through the existing, tested
`ppfn.trainer.callbacks.mlflow_cb.MLflowCallback` machinery where it's a
natural fit -- see `ppfn.trainer.registration_trainer`.

Adding a monitor the project didn't have yet is exactly:

    @register_monitor("my_new_diagnostic")
    def _my_new_diagnostic(ctx: MonitorContext) -> dict[str, float]:
        ...
        return {"my/diagnostic": value}

No other file needs to change; `compute_all_monitors` picks it up automatically.

Deliberately NOT implemented here (out of scope for the current build --
CLAUDE.md's build order places these at step 9, "Bounds, fold calibration,
baselines, spending law", after the go/no-go, transport, coupling, affine
and distillation steps this build covers): fold_fraction (§5.2), the
Fisher-conditioning diagnostic (§5.3), and the upper-1/upper-2 KL gaps
(§5.1's three-way decomposition). The registry makes each a one-function
addition once that step is reached.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch
import torch.nn as nn

from ppfn.prior.registration.dataset import RegistrationBatch

MonitorFn = Callable[["MonitorContext"], dict]

_REGISTRY: dict[str, MonitorFn] = {}


def register_monitor(name: str):
    def _decorator(fn: MonitorFn) -> MonitorFn:
        _REGISTRY[name] = fn
        return fn

    return _decorator


@dataclasses.dataclass
class MonitorContext:
    model: nn.Module
    val_batch: RegistrationBatch


def compute_all_monitors(ctx: MonitorContext) -> dict[str, float]:
    out: dict[str, float] = {}
    with torch.no_grad():
        for fn in _REGISTRY.values():
            out.update(fn(ctx))
    return out


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return ((values * mask_f).sum() / denom).item()


@register_monitor("gates")
def _gate_magnitudes(ctx: MonitorContext) -> dict:
    """mean |gamma_l| per layer -- ARCHITECTURE.md §4.5: "how much B is
    being used... drifting to 0 -> collapse." Read directly off the
    decoder's own parameters -- no forward pass needed."""
    out = {}
    gates = [
        layer.cross_attn.gate.detach().abs().item()
        for layer in ctx.model.decoder.layers
    ]
    for ell, g in enumerate(gates):
        out[f"gate/layer_{ell}"] = g
    out["gate/mean_abs"] = sum(gates) / len(gates)
    return out


@register_monitor("transfer_gap")
def _transfer_gap(ctx: MonitorContext) -> dict:
    """NLL(severed) - NLL(full) -- ARCHITECTURE.md §4.5: "value extracted
    from B... closing toward 0 -> collapse." Only over non-role-swapped
    items/tokens, matching L_pred's own definition (§3.1/§4.3)."""
    batch = ctx.val_batch
    token_mask = batch.dec_qry_mask & (~batch.role_swapped).unsqueeze(-1)

    out_full = ctx.model(batch, severed_mask=torch.zeros_like(batch.severed))
    nll_full = ctx.model.predictive_dist(out_full["predictive_logits"], batch.y_qry)

    out_severed = ctx.model(batch, severed_mask=torch.ones_like(batch.severed))
    nll_severed = ctx.model.predictive_dist(
        out_severed["predictive_logits"], batch.y_qry
    )

    gap = _masked_mean(nll_severed, token_mask) - _masked_mean(nll_full, token_mask)
    return {
        "bounds/nll_full": _masked_mean(nll_full, token_mask),
        "bounds/nll_severed": _masked_mean(nll_severed, token_mask),
        "bounds/transfer_gap": gap,
    }


@register_monitor("transport_and_coupling_per_layer")
def _transport_and_coupling_per_layer(ctx: MonitorContext) -> dict:
    """Transport NLL and coupling MSE per layer -- ARCHITECTURE.md §4.5:
    "is coarse-to-fine happening... flat across l -> refinement is
    nominal" / "is attention a real correspondence... flat/high -> head 0
    is diffuse." """
    batch = ctx.val_batch
    dim_mask_qry = ctx.model.dim_mask(batch.d_real, batch.dec_qry_x.shape[1])
    out = ctx.model(batch)

    result = {}
    for ell, logits_qry in enumerate(out["transport_logits_qry"]):
        nll = ctx.model.transport_head.nll(
            logits_qry, batch.transport_qry, dim_mask_qry
        )
        result[f"transport_nll/layer_{ell}"] = _masked_mean(nll, batch.dec_qry_mask)
    for ell, bary_qry in enumerate(out["bary_qry_layers"]):
        se = ((bary_qry - batch.transport_qry) ** 2 * dim_mask_qry).sum(
            -1
        ) / dim_mask_qry.sum(-1).clamp_min(1)
        result[f"coupling_mse/layer_{ell}"] = _masked_mean(se, batch.dec_qry_mask)
    return result


@register_monitor("rho_anchor_holding")
def _rho_anchor_holding(ctx: MonitorContext) -> dict:
    """NLL at rho=0 vs rho~1 -- ARCHITECTURE.md §4.5: "is the anchor
    holding... rho=0 degrading over training -> pathway drift." Splits the
    validation batch by its own (already-drawn) rho rather than redrawing,
    so this stays a pure function of the given batch."""
    batch = ctx.val_batch
    out = ctx.model(batch)
    nll = ctx.model.predictive_dist(out["predictive_logits"], batch.y_qry)
    token_mask_base = batch.dec_qry_mask & (~batch.role_swapped).unsqueeze(-1)

    near_zero = batch.rho < 0.05
    near_one = batch.rho > 0.8
    result = {}
    if bool(near_zero.any()):
        result["bounds/nll_rho_near_0"] = _masked_mean(
            nll, token_mask_base & near_zero.unsqueeze(-1)
        )
    if bool(near_one.any()):
        result["bounds/nll_rho_near_1"] = _masked_mean(
            nll, token_mask_base & near_one.unsqueeze(-1)
        )
    return result
