"""Component-isolation probes for `ppfn.model.baselines.
iterative_registration_pfn.IterativeRegistrationPFN` -- requested 2026-09-17,
directly implementing the incremental verification path the design doc
itself calls for (docs/labbook/2026-09-16-lupi-registration-mechanism-and-
architecture-survey.md): verify the value-step and position-step mechanisms
EACH IN ISOLATION, with the other unknown teacher-forced to ground truth,
before trusting them coupled in the full iterative model.

`ValueStepProbePFN`: `T_hat` is pinned to the true `enc_x_inA` at every
iteration (the position-step computation never runs at all) -- tests
whether the value-recalibration mechanism (evidence formation, h-slot
update, RQS readout) alone can learn to recover `h`, given a
best-case-scenario, already-solved position estimate. Trained against
`ppfn.prior.lupi.sampler.sample_pair(..., rho=0.0)` (`T` is the identity
there, so pinning `T_hat` to ground truth is trivial/exact and there's
nothing left for a position mechanism to do anyway) at `d=1`.

`PositionStepProbePFN`: `h_hat` is pinned to the true `enc_z_inA` at every
iteration (the RQS/h-slot machinery never runs). `T_hat` still starts at
`enc_x` (the identity prior) and evolves via the real position-step
computation (value-informed pull toward A's anchors + smoothness pull
among B's own evolving estimates) -- tests whether that mechanism alone
converges to the true registered position, given already-solved value
calibration. Trained against `sample_pair(..., force_h_identity=True)`
(`h` is the identity there) at `d=1`, so there's nothing left for a value
mechanism to do -- but note `force_h_identity` does NOT trivialize the
position problem the way `rho=0` trivializes the value problem: T is still
a genuine, non-identity warp whenever `rho>0`, which is exactly what this
probe needs to actually exercise the position mechanism.

Both reuse `IterativeRegistrationPFN`'s own private `_SelfAttnBlock`
directly (not duplicated) -- these probes are tightly-coupled unit tests of
that model's own internals, not independent architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.baselines.iterative_registration_pfn import RQSValueHead, _SelfAttnBlock
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class ValueStepProbePFN(nn.Module):
    def __init__(
        self, d_max: int = D_MAX, d_model: int = 128, n_heads: int = 4, n_ff: int = 256,
        n_iters: int = 4, n_hslots: int = 4, rqs_bins: int = 8, dropout: float = 0.0,
    ):
        super().__init__()
        self.n_iters, self.n_hslots = n_iters, n_hslots
        self.x_embed = nn.Linear(d_max, d_model)
        self.y_embed = nn.Linear(1, d_model)
        self.hslot_init = nn.Parameter(torch.randn(n_hslots, d_model) * 0.02)
        self.value_head = RQSValueHead(d_model, n_bins=rqs_bins)
        self.evidence_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.hslot_update = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.self_attn = _SelfAttnBlock(d_model, n_heads, n_ff, dropout)
        self.log_sigma_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, batch: LUPIBatch) -> dict:
        B, n_A = batch.dec_ctx_x.shape[:2]
        n_B = batch.enc_x.shape[1]
        T_true = batch.enc_x_inA  # pinned -- NEVER updated

        a_tok = self.x_embed(batch.dec_ctx_x) + self.y_embed(batch.dec_ctx_z.unsqueeze(-1))
        b_tok = self.x_embed(T_true) + self.y_embed(batch.enc_z.unsqueeze(-1))
        hslots = self.hslot_init.unsqueeze(0).expand(B, -1, -1)
        hidden = torch.cat([a_tok, b_tok, hslots], dim=1)
        pooled_mask = torch.cat(
            [batch.dec_ctx_mask, batch.enc_mask, batch.enc_mask.new_ones(B, self.n_hslots)], dim=1
        )

        neg_inf = torch.finfo(T_true.dtype).min
        pos_dist = torch.cdist(T_true, batch.dec_ctx_x)
        pos_scores = (-pos_dist.pow(2)).masked_fill(~batch.dec_ctx_mask.unsqueeze(1), neg_inf)
        corr = torch.softmax(pos_scores, dim=-1)  # fixed, since T_true is fixed -- recomputed each iter for clarity, not efficiency
        y_tilde = torch.einsum("bja,ba->bj", corr, batch.dec_ctx_z)
        confidence = corr.amax(dim=-1)

        h_hat_per_iter = []
        for _ in range(self.n_iters):
            hslot_hidden = hidden[:, n_A + n_B :, :]
            hslot_summary = hslot_hidden.mean(dim=1)
            evidence = self.evidence_mlp(torch.stack([batch.enc_z, y_tilde, confidence], dim=-1))
            conf_w = confidence * batch.enc_mask.float()
            conf_w = conf_w / conf_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            evidence_global = torch.einsum("bj,bjd->bd", conf_w, evidence)
            delta = self.hslot_update(torch.cat([hslot_summary, evidence_global], dim=-1))
            hslot_summary_new = hslot_summary + delta
            hslot_hidden_new = hslot_hidden + delta.unsqueeze(1)

            h_hat = self.value_head(hslot_summary_new, batch.enc_z)
            h_hat_per_iter.append(h_hat)

            b_tok = self.x_embed(T_true) + self.y_embed(h_hat.unsqueeze(-1))
            pooled = torch.cat([a_tok, b_tok, hslot_hidden_new], dim=1)
            hidden = self.self_attn(pooled + hidden, pooled_mask)

        b_hidden_final = hidden[:, n_A : n_A + n_B, :]
        # Clamped -- an unclamped log-sigma head trained purely on NLL is
        # prone to runaway variance growth (empirically confirmed 2026-09-17:
        # mean sigma drifted 0.87 -> 15.2 over 40 steps on a fixed toy batch,
        # with a loss spike to 131 mid-run, while the underlying point
        # estimate kept improving fine) -- the classic heteroscedastic-NLL
        # instability (Kendall & Gal 2017 and others note the same failure
        # mode). Range chosen so sigma in ~[0.01, 20], generous for this
        # target's scale without leaving the optimizer room to run away.
        log_sigma = self.log_sigma_head(b_hidden_final).squeeze(-1).clamp(-5.0, 3.0)
        return {"h_hat_per_iter": h_hat_per_iter, "h_hat_final": h_hat_per_iter[-1], "log_sigma_final": log_sigma}


class PositionStepProbePFN(nn.Module):
    def __init__(
        self, d_max: int = D_MAX, d_model: int = 128, n_heads: int = 4, n_ff: int = 256,
        n_iters: int = 4, n_hslots: int = 4, dropout: float = 0.0,
    ):
        super().__init__()
        self.n_iters, self.n_hslots, self.d_max = n_iters, n_hslots, d_max
        self.x_embed = nn.Linear(d_max, d_model)
        self.y_embed = nn.Linear(1, d_model)
        self.hslot_init = nn.Parameter(torch.randn(n_hslots, d_model) * 0.02)
        self.self_attn = _SelfAttnBlock(d_model, n_heads, n_ff, dropout)
        self.log_tau_val = nn.Parameter(torch.tensor(0.0))
        self.log_tau_smooth = nn.Parameter(torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(0.0))
        from ppfn.model.registration.heads import TransportHead

        self.transport_head = TransportHead(d_model, d_max, n_bins=32)

    def forward(self, batch: LUPIBatch) -> dict:
        B, n_A = batch.dec_ctx_x.shape[:2]
        n_B = batch.enc_x.shape[1]
        h_true = batch.enc_z_inA  # pinned -- the RQS/h-slot machinery never runs
        T_hat = batch.enc_x.clone()  # init: identity prior, same as the full model

        a_tok = self.x_embed(batch.dec_ctx_x) + self.y_embed(batch.dec_ctx_z.unsqueeze(-1))
        hslots = self.hslot_init.unsqueeze(0).expand(B, -1, -1)
        neg_inf = torch.finfo(T_hat.dtype).min
        tau_val = F.softplus(self.log_tau_val) + 1e-3
        tau_smooth = F.softplus(self.log_tau_smooth) + 1e-3
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)

        T_hat_per_iter = []
        hidden = None
        for _ in range(self.n_iters):
            val_dist = (h_true.unsqueeze(-1) - batch.dec_ctx_z.unsqueeze(1)).pow(2)
            val_scores = (-val_dist / tau_val).masked_fill(~batch.dec_ctx_mask.unsqueeze(1), neg_inf)
            corr_val = torch.softmax(val_scores, dim=-1)
            pos_pull = torch.einsum("bja,bad->bjd", corr_val, batch.dec_ctx_x)

            b_dist = torch.cdist(T_hat, T_hat)
            eye_mask = torch.eye(n_B, device=T_hat.device, dtype=torch.bool).unsqueeze(0)
            b_key_mask = batch.enc_mask.unsqueeze(1) & ~eye_mask
            b_scores = (-b_dist.pow(2) / tau_smooth).masked_fill(~b_key_mask, neg_inf)
            w_smooth = torch.softmax(b_scores, dim=-1)
            smooth_pull = torch.einsum("bjk,bkd->bjd", w_smooth, T_hat)

            target = beta * pos_pull + (1 - beta) * smooth_pull
            T_hat = (1 - alpha) * T_hat + alpha * target
            T_hat_per_iter.append(T_hat)

            b_tok = self.x_embed(T_hat) + self.y_embed(h_true.unsqueeze(-1))
            pooled = torch.cat([a_tok, b_tok, hslots], dim=1)
            pooled_mask = torch.cat(
                [batch.dec_ctx_mask, batch.enc_mask, batch.enc_mask.new_ones(B, self.n_hslots)], dim=1
            )
            hidden = self.self_attn(pooled if hidden is None else pooled + hidden, pooled_mask)

        b_hidden_final = hidden[:, n_A : n_A + n_B, :]
        transport_logits = self.transport_head(b_hidden_final, teacher_targets=batch.enc_x_inA)
        return {"T_hat_per_iter": T_hat_per_iter, "T_hat_final": T_hat_per_iter[-1], "transport_logits": transport_logits}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: confirm both probes
    train (loss decreases over a handful of local optimizer steps) on a
    fixed d=1 batch before ever launching a real run."""
    import numpy as np
    import torch

    from ppfn.loss.iterative_registration_probe_losses import PositionStepProbeLoss, ValueStepProbeLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    ds_value = LUPIStreamDataset(seed=0, s_max=0.1, force_rho_zero=True, d=1)
    batch_value = collate_lupi_batch([next(iter(ds_value)) for _ in range(8)])
    model_v = ValueStepProbePFN(d_model=32, n_heads=4, n_ff=64, n_iters=3, n_hslots=4, rqs_bins=6)
    opt_v = torch.optim.Adam(model_v.parameters(), lr=3e-3)
    crit_v = ValueStepProbeLoss()
    losses_v = []
    for _ in range(40):
        opt_v.zero_grad()
        out = model_v(batch_value)
        loss, _ = crit_v(model_v, batch_value, out)
        loss.backward()
        opt_v.step()
        losses_v.append(loss.item())
    print(f"ValueStepProbePFN (T pinned to truth, rho=0, d=1): loss[0]={losses_v[0]:.4f} -> loss[-1]={losses_v[-1]:.4f}")

    ds_pos = LUPIStreamDataset(seed=0, s_max=0.1, force_h_identity=True, d=1)
    batch_pos = collate_lupi_batch([next(iter(ds_pos)) for _ in range(8)])
    model_p = PositionStepProbePFN(d_model=32, n_heads=4, n_ff=64, n_iters=3, n_hslots=4)
    opt_p = torch.optim.Adam(model_p.parameters(), lr=3e-3)
    crit_p = PositionStepProbeLoss()
    losses_p = []
    for _ in range(40):
        opt_p.zero_grad()
        out = model_p(batch_pos)
        loss, _ = crit_p(model_p, batch_pos, out)
        loss.backward()
        opt_p.step()
        losses_p.append(loss.item())
    print(f"PositionStepProbePFN (h pinned to truth, force_h_identity, d=1): loss[0]={losses_p[0]:.4f} -> loss[-1]={losses_p[-1]:.4f}")
