"""Diagnostics for `ppfn.model.lupi.model.LUPIPFN` -- mirrors
`ppfn.monitor.registry`'s flat function-registry pattern (own registry, own
`MonitorContext`, since that module's is typed to `RegistrationBatch`) rather
than extending it, so this package stays independent of the registration
monitor set per CLAUDE.md's "Keep the monitor set declared in one module."

Two questions this answers, both requested directly rather than inferred:
- **Is the model actually using B, and where?** `_qry_source_breakdown`
  splits query-token NLL by `batch.dec_qry_source` (spec §6.2's 3-way query
  mixture: uniform / near-B / near-A-context, already sampled into every
  batch, just not previously surfaced past the prior). Near-B queries land
  where B has dense evidence and A's own context is uninformative by
  construction -- a small NLL there specifically (not just in the
  aggregate) is the direct evidence that the B-path is being read, not
  ignored.
- **Does the tiny aggregate oracle-student gap hold up once ρ=0 (and
  low-ρ, curriculum-favored) items are excluded?** `_rho_stratified_gap`
  mirrors `ppfn.monitor.registry._rho_anchor_holding`'s near-0/near-1 split
  for the same reason that one exists: a batch-averaged number dominated by
  the easy (ρ≈0) regime says nothing about the hard one.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch
import torch.nn as nn

from ppfn.prior.lupi.dataset import LUPIBatch

LUPIMonitorFn = Callable[["LUPIMonitorContext"], dict]

_REGISTRY: dict[str, LUPIMonitorFn] = {}


def register_lupi_monitor(name: str):
    def _decorator(fn: LUPIMonitorFn) -> LUPIMonitorFn:
        _REGISTRY[name] = fn
        return fn

    return _decorator


@dataclasses.dataclass
class LUPIMonitorContext:
    model: nn.Module
    val_batch: LUPIBatch


def compute_all_lupi_monitors(ctx: LUPIMonitorContext) -> dict[str, float]:
    out: dict[str, float] = {}
    with torch.no_grad():
        for fn in _REGISTRY.values():
            out.update(fn(ctx))
    return out


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    mask_f = mask.to(values.dtype)
    denom = mask_f.sum()
    if denom.item() == 0:
        return None
    return ((values * mask_f).sum() / denom).item()


_SOURCE_NAMES = {0: "uniform", 1: "near_b", 2: "near_a"}


@register_lupi_monitor("qry_source_breakdown")
def _qry_source_breakdown(ctx: LUPIMonitorContext) -> dict:
    batch = ctx.val_batch
    b = ctx.model.encode_b(batch)
    out_student = ctx.model.align(batch, b, mode="student")
    out_oracle = ctx.model.align(batch, b, mode="oracle")

    nll_student = ctx.model.bar_dist(out_student["predictive_logits"], batch.dec_qry_z)
    nll_oracle = ctx.model.bar_dist(out_oracle["predictive_logits"], batch.dec_qry_z)

    result = {}
    for src_id, name in _SOURCE_NAMES.items():
        src_mask = batch.dec_qry_mask & (batch.dec_qry_source == src_id)
        n = int(src_mask.sum().item())
        result[f"lupi_qry/{name}/n_tokens"] = n
        student_val = _masked_mean(nll_student, src_mask)
        oracle_val = _masked_mean(nll_oracle, src_mask)
        if student_val is not None:
            result[f"lupi_qry/{name}/student_nll"] = student_val
        if oracle_val is not None:
            result[f"lupi_qry/{name}/oracle_nll"] = oracle_val
        if student_val is not None and oracle_val is not None:
            result[f"lupi_qry/{name}/oracle_student_gap"] = student_val - oracle_val
    return result


@register_lupi_monitor("rho_stratified_gap")
def _rho_stratified_gap(ctx: LUPIMonitorContext) -> dict:
    batch = ctx.val_batch
    b = ctx.model.encode_b(batch)
    out_student = ctx.model.align(batch, b, mode="student")
    out_oracle = ctx.model.align(batch, b, mode="oracle")

    nll_student = ctx.model.bar_dist(out_student["predictive_logits"], batch.dec_qry_z)
    nll_oracle = ctx.model.bar_dist(out_oracle["predictive_logits"], batch.dec_qry_z)

    bands = {
        "rho_eq_0": batch.rho < 1e-6,
        "rho_low": (batch.rho >= 1e-6) & (batch.rho < 0.3),
        "rho_mid": (batch.rho >= 0.3) & (batch.rho < 0.7),
        "rho_high": batch.rho >= 0.7,
    }
    result = {}
    for band_name, item_mask in bands.items():
        tok_mask = batch.dec_qry_mask & item_mask.unsqueeze(-1)
        n = int(tok_mask.sum().item())
        result[f"lupi_rho/{band_name}/n_tokens"] = n
        student_val = _masked_mean(nll_student, tok_mask)
        oracle_val = _masked_mean(nll_oracle, tok_mask)
        if student_val is not None:
            result[f"lupi_rho/{band_name}/student_nll"] = student_val
        if oracle_val is not None:
            result[f"lupi_rho/{band_name}/oracle_nll"] = oracle_val
        if student_val is not None and oracle_val is not None:
            result[f"lupi_rho/{band_name}/oracle_student_gap"] = student_val - oracle_val
    return result
