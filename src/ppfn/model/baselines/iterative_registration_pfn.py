"""Iterative Registration Transformer (IRT) -- docs/labbook/2026-09-16-lupi-
registration-mechanism-and-architecture-survey.md's concrete synthesis:
`IDTokenPFN`/`LUPIIDTokenPFN` ask generic attention depth to discover
registration implicitly, purely as a side effect of a final-output loss.
This replaces that with an explicit, weight-tied (Universal Transformer,
Dehghani et al. 2019) iteration that is architecturally an unrolled
Iterative-Closest-Point-style alternation: given a current position estimate
for B, recalibrate B's values against A's nearby anchors; given the
recalibrated values, refine B's position estimate against A's anchors AND
against neighboring B points' own current estimates (exploiting `T`'s
smoothness, per the labbook's own §1 diagnosis). Deep supervision (`w_k ∝
k`) applies LUPI's privileged targets (`enc_x_inA`, `enc_z_inA`) to this
EXPLICIT state at every iteration, not just the final prediction -- directly
analogous to RAFT's (Teed & Deng 2020) iterative flow refinement with
per-iteration supervision against ground-truth flow, no teacher-forcing of
the running estimate between iterations (train/test parity: the model must
live with and improve its own prior estimate, exactly as at inference).

Correspondence steps are deliberately NOT learned Q/K attention -- they're
temperature-softmax kernels directly on the numeric, interpretable state
(current position/value estimates), matching classical ICP's "soft nearest
neighbor" rather than an abstract embedding-space similarity. This also
sidesteps the raw-logit blowup failure mode the labbook's §3 diagnoses:
every correspondence here is a proper softmax over real candidates, never an
unbounded target. A generic residual self-attention pass (ordinary learned
MHA) sits alongside the explicit steps each iteration, so whatever the
explicit mechanism doesn't capture still has a fallback path.

No additive domain tag: A and B already play structurally distinct
computational roles here (A = fixed anchors, B = the thing being iteratively
registered), unlike `IDTokenPFN`'s naive undifferentiated pool, which needs
a tag specifically because it has no other way to tell the clouds apart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.baselines.calibration import sample_calibration_borders
from ppfn.model.pfn.bar_distribution import FullSupportBarDistribution
from ppfn.model.pfn.pfn import MaskedMHA, PFNBlock
from ppfn.model.pfn.rqs import rational_quadratic_spline_forward
from ppfn.model.registration.heads import TransportHead
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


def _dim_mask(d_real: torch.Tensor, d_max: int) -> torch.Tensor:
    """[B] int -> [B, d_max] bool, True for real (non-padded) axes."""
    idx = torch.arange(d_max, device=d_real.device).view(1, d_max)
    return idx < d_real.view(-1, 1)


class _SelfAttnBlock(nn.Module):
    """Plain pre-norm self-attention + FFN over ONE pooled stream -- the
    "residual generic attention pass" each iteration takes, distinct from
    `PFNBlock`'s two-stream (train/test) design since there's no query
    stream involved until the final readout (see `IterativeRegistrationPFN.
    forward`'s last step)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


class RQSValueHead(nn.Module):
    """Monotone value-recalibration head -- `ppfn.model.pfn.rqs`'s forward-
    only rational-quadratic spline, wrapped in a learned positive affine
    (`s > 0, b`) so the head isn't stuck at the identity outside the
    spline's active region (only the raw spline is; composing an increasing
    affine with an increasing spline is still increasing). ONE set of
    (spline, affine) params per batch ITEM, read off a pooled h-slot
    summary -- `h` is one function per draw, not one per B-token."""

    def __init__(self, d_model: int, n_bins: int = 8, tail_bound: float = 5.0, hidden: int | None = None):
        super().__init__()
        self.n_bins = n_bins
        self.tail_bound = tail_bound
        hidden = hidden or d_model
        # widths, heights, (n_bins-1) internal derivatives, log-scale, shift
        out_dim = n_bins + n_bins + (n_bins - 1) + 1 + 1
        self.hypernet = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, h_slot_summary: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """h_slot_summary: [B, d_model]  y: [B, n_B] (raw B values) ->
        recalibrated mean estimate [B, n_B]."""
        params = self.hypernet(h_slot_summary)  # [B, out_dim]
        uw, uh, ud, log_s, b = torch.split(
            params, [self.n_bins, self.n_bins, self.n_bins - 1, 1, 1], dim=-1
        )
        n_b = y.shape[1]
        uw = uw.unsqueeze(1).expand(-1, n_b, -1)
        uh = uh.unsqueeze(1).expand(-1, n_b, -1)
        ud = ud.unsqueeze(1).expand(-1, n_b, -1)
        s = F.softplus(log_s)  # [B, 1], positive -- increasing-affine-of-increasing-spline stays increasing
        spline_out = rational_quadratic_spline_forward(y, uw, uh, ud, tail_bound=self.tail_bound)
        return s * spline_out + b


class RegistrationIterationBlock(nn.Module):
    """One weight-tied iteration -- applied K times with the SAME instance
    (Universal-Transformer style) by `IterativeRegistrationPFN`, not K
    independently-parametrized blocks."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_max: int, rqs_bins: int = 8, dropout: float = 0.0):
        super().__init__()
        self.value_head = RQSValueHead(d_model, n_bins=rqs_bins)
        self.evidence_mlp = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.hslot_update = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        # Softplus'd -- temperatures/step-sizes must stay positive; kept as
        # single shared scalars for this first cut (see the labbook: start
        # with fixed/simple, add a coarse-to-fine schedule only if the
        # simple version earns it).
        self.log_tau_pos = nn.Parameter(torch.tensor(0.0))
        self.log_tau_val = nn.Parameter(torch.tensor(0.0))
        self.log_tau_smooth = nn.Parameter(torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid -> step size in (0,1)
        self.beta_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid -> anchor-vs-smoothness mix in (0,1)
        self.x_embed = nn.Linear(d_max, d_model)
        self.y_embed = nn.Linear(1, d_model)
        self.self_attn = _SelfAttnBlock(d_model, n_heads, d_ff, dropout)

    def forward(
        self,
        T_hat: torch.Tensor,  # [B, n_B, d_max] current position estimate
        enc_z: torch.Tensor,  # [B, n_B] raw B values, fixed throughout
        enc_mask: torch.Tensor,  # [B, n_B] bool
        dec_ctx_x: torch.Tensor,  # [B, n_A, d_max]
        dec_ctx_z: torch.Tensor,  # [B, n_A]
        dec_ctx_mask: torch.Tensor,  # [B, n_A] bool
        hidden: torch.Tensor,  # [B, n_A + n_B + K_h, d_model] -- previous iteration's pooled hidden state
        n_A: int,
        n_hslots: int,
    ) -> dict:
        neg_inf = torch.finfo(T_hat.dtype).min
        tau_pos = F.softplus(self.log_tau_pos) + 1e-3
        tau_val = F.softplus(self.log_tau_val) + 1e-3
        tau_smooth = F.softplus(self.log_tau_smooth) + 1e-3
        alpha = torch.sigmoid(self.alpha_raw)
        beta = torch.sigmoid(self.beta_raw)

        # --- value step: T_hat (current) -> locally-expected A-scale value -> h_hat ---
        pos_dist = torch.cdist(T_hat, dec_ctx_x)  # [B, n_B, n_A]
        pos_scores = -pos_dist.pow(2) / tau_pos
        pos_scores = pos_scores.masked_fill(~dec_ctx_mask.unsqueeze(1), neg_inf)
        corr = torch.softmax(pos_scores, dim=-1)  # [B, n_B, n_A], row-stochastic over A anchors
        y_tilde = torch.einsum("bja,ba->bj", corr, dec_ctx_z)  # [B, n_B]
        confidence = corr.amax(dim=-1, keepdim=True)  # [B, n_B, 1] -- how peaked the correspondence is

        evidence = self.evidence_mlp(
            torch.stack([enc_z, y_tilde, confidence.squeeze(-1)], dim=-1)
        )  # [B, n_B, d_model]
        conf_w = (confidence.squeeze(-1) * enc_mask.float())
        conf_w = conf_w / conf_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        evidence_global = torch.einsum("bj,bjd->bd", conf_w, evidence)  # [B, d_model]

        hslot_hidden = hidden[:, n_A + T_hat.shape[1] :, :]  # [B, K_h, d_model]
        hslot_summary = hslot_hidden.mean(dim=1)  # [B, d_model]
        hslot_update_in = torch.cat(
            [hslot_summary, evidence_global], dim=-1
        )  # [B, 2*d_model]
        hslot_summary_new = hslot_summary + self.hslot_update(hslot_update_in)
        hslot_hidden_new = hslot_hidden + self.hslot_update(hslot_update_in).unsqueeze(1)

        h_hat = self.value_head(hslot_summary_new, enc_z)  # [B, n_B]

        # --- position step: h_hat -> value-informed correspondence -> refined T_hat ---
        val_dist = (h_hat.unsqueeze(-1) - dec_ctx_z.unsqueeze(1)).pow(2)  # [B, n_B, n_A]
        val_scores = -val_dist / tau_val
        val_scores = val_scores.masked_fill(~dec_ctx_mask.unsqueeze(1), neg_inf)
        corr_val = torch.softmax(val_scores, dim=-1)
        pos_pull = torch.einsum("bja,bad->bjd", corr_val, dec_ctx_x)  # [B, n_B, d_max]

        b_dist = torch.cdist(T_hat, T_hat)  # [B, n_B, n_B]
        b_scores = -b_dist.pow(2) / tau_smooth
        eye_mask = torch.eye(T_hat.shape[1], device=T_hat.device, dtype=torch.bool).unsqueeze(0)
        b_key_mask = enc_mask.unsqueeze(1) & ~eye_mask
        b_scores = b_scores.masked_fill(~b_key_mask, neg_inf)
        w_smooth = torch.softmax(b_scores, dim=-1)
        smooth_pull = torch.einsum("bjk,bkd->bjd", w_smooth, T_hat)

        target = beta * pos_pull + (1 - beta) * smooth_pull
        T_hat_new = (1 - alpha) * T_hat + alpha * target

        # --- residual generic self-attention over the whole pooled set ---
        a_tok = self.x_embed(dec_ctx_x) + self.y_embed(dec_ctx_z.unsqueeze(-1))
        b_tok = self.x_embed(T_hat_new) + self.y_embed(h_hat.unsqueeze(-1))
        pooled = torch.cat([a_tok, b_tok, hslot_hidden_new], dim=1)
        pooled_mask = torch.cat(
            [dec_ctx_mask, enc_mask, enc_mask.new_ones(enc_mask.shape[0], n_hslots)], dim=1
        )
        hidden_new = self.self_attn(pooled + hidden, pooled_mask)

        return {"T_hat": T_hat_new, "h_hat": h_hat, "hidden": hidden_new}


class IterativeRegistrationPFN(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_ff: int = 512,
        n_bins_predictive: int = 64,
        n_iters: int = 4,
        n_hslots: int = 4,
        rqs_bins: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_max = d_max
        self.n_iters = n_iters
        self.n_hslots = n_hslots
        self.hslot_init = nn.Parameter(torch.randn(n_hslots, d_model) * 0.02)
        self.rib = RegistrationIterationBlock(d_model, n_heads, n_ff, d_max, rqs_bins=rqs_bins, dropout=dropout)

        # Final query readout -- same two-stream cross-attention PFNBlock
        # pattern as every other baseline here, reading the LAST iteration's
        # pooled hidden state as the train stream.
        self.readout_x_embed = nn.Linear(d_max, d_model)
        self.readout_block = PFNBlock(d_model, n_heads, n_ff, dropout)
        self.out_ln = nn.LayerNorm(d_model)
        borders = sample_calibration_borders(n_bins_predictive)
        self.predictive_dist = FullSupportBarDistribution(borders)
        self.predictive_head = nn.Linear(d_model, self.predictive_dist.num_bars)

        # LUPI auxiliary heads -- transport reused as-is, value head mirrors
        # RQSValueHead's construction but scores a proper NLL, not a point
        # estimate (see the labbook's §2b: a point loss can't express
        # genuine registration uncertainty).
        self.transport_head = TransportHead(d_model, d_max, n_bins=32)
        self.aux_value_head = RQSValueHead(d_model, n_bins=rqs_bins)
        self.aux_value_log_sigma = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, batch: LUPIBatch) -> dict:
        B, n_A = batch.dec_ctx_x.shape[:2]
        n_B = batch.enc_x.shape[1]
        device = batch.dec_ctx_x.device

        T_hat = batch.enc_x.clone()  # init: T_hat^(0) = raw B position (identity prior)
        a_tok0 = self.rib.x_embed(batch.dec_ctx_x) + self.rib.y_embed(batch.dec_ctx_z.unsqueeze(-1))
        b_tok0 = self.rib.x_embed(T_hat) + self.rib.y_embed(batch.enc_z.unsqueeze(-1))
        hslots0 = self.hslot_init.unsqueeze(0).expand(B, -1, -1)
        hidden = torch.cat([a_tok0, b_tok0, hslots0], dim=1)

        T_hat_per_iter, h_hat_per_iter = [], []
        for _ in range(self.n_iters):
            out = self.rib(
                T_hat, batch.enc_z, batch.enc_mask,
                batch.dec_ctx_x, batch.dec_ctx_z, batch.dec_ctx_mask,
                hidden, n_A, self.n_hslots,
            )
            T_hat, hidden = out["T_hat"], out["hidden"]
            T_hat_per_iter.append(T_hat)
            h_hat_per_iter.append(out["h_hat"])

        # --- LUPI auxiliary heads, off the LAST iteration's B hidden states ---
        b_hidden_final = hidden[:, n_A : n_A + n_B, :]
        transport_logits = self.transport_head(b_hidden_final, teacher_targets=batch.enc_x_inA)
        hslot_summary_final = hidden[:, n_A + n_B :, :].mean(dim=1)
        aux_mu = self.aux_value_head(hslot_summary_final, batch.enc_z)
        aux_log_sigma = self.aux_value_log_sigma(b_hidden_final).squeeze(-1)

        # --- final predictive readout: query cross-attends into the last iteration's pooled state ---
        test_tok = self.readout_x_embed(batch.dec_qry_x)
        train_tok = hidden[:, : n_A + n_B, :]
        train_mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)
        train_tok, test_tok = self.readout_block(train_tok, test_tok, train_key_padding_mask=train_mask)
        test_tok = self.out_ln(test_tok)
        logits = self.predictive_head(test_tok)

        return {
            "predictive_logits": logits,
            "T_hat_per_iter": T_hat_per_iter,  # list of [B, n_B, d_max], len n_iters
            "h_hat_per_iter": h_hat_per_iter,  # list of [B, n_B], len n_iters
            "transport_logits": transport_logits,  # [B, n_B, d_max, n_bins] -- last-iteration-only aux head
            "aux_value_mu": aux_mu,  # [B, n_B]
            "aux_value_log_sigma": aux_log_sigma,  # [B, n_B]
        }


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md, including the
    incremental checks the labbook's design discussion called for BEFORE
    trusting the coupled iterative model: (1) shapes + backward pass work
    end to end, (2) padding invisibility, (3) the position/value steps are
    each individually well-behaved (finite, and move the RIGHT direction on
    a toy case) before trusting them coupled."""
    import dataclasses

    import numpy as np
    import torch

    from ppfn.loss.iterative_registration_loss import IterativeRegistrationLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = IterativeRegistrationPFN(
        d_model=32, n_heads=4, n_ff=64, n_bins_predictive=16, n_iters=3, n_hslots=4, rqs_bins=6,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("predictive_logits:", out["predictive_logits"].shape)
    print("T_hat_per_iter:", len(out["T_hat_per_iter"]), out["T_hat_per_iter"][0].shape)
    print("h_hat_per_iter:", len(out["h_hat_per_iter"]), out["h_hat_per_iter"][0].shape)
    print("transport_logits:", out["transport_logits"].shape)

    criterion = IterativeRegistrationLoss()
    loss, metrics = criterion(model, batch, out)
    print("loss/metrics:", {k: round(v, 4) for k, v in metrics.items()})

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None (expect 0): {n_none}")

    # Padding-invisibility check: pad B with one extra all-zero, masked-out point.
    padded_enc_x = torch.cat([batch.enc_x, torch.zeros_like(batch.enc_x[:, :1])], dim=1)
    padded_enc_z = torch.cat([batch.enc_z, torch.zeros_like(batch.enc_z[:, :1])], dim=1)
    padded_enc_x_inA = torch.cat([batch.enc_x_inA, torch.zeros_like(batch.enc_x_inA[:, :1])], dim=1)
    padded_enc_z_inA = torch.cat([batch.enc_z_inA, torch.zeros_like(batch.enc_z_inA[:, :1])], dim=1)
    padded_enc_mask = torch.cat([batch.enc_mask, torch.zeros_like(batch.enc_mask[:, :1])], dim=1)
    padded_batch = dataclasses.replace(
        batch, enc_x=padded_enc_x, enc_z=padded_enc_z, enc_x_inA=padded_enc_x_inA,
        enc_z_inA=padded_enc_z_inA, enc_mask=padded_enc_mask,
    )
    with torch.no_grad():
        out_padded = model(padded_batch)
    diff = (out["predictive_logits"] - out_padded["predictive_logits"]).abs().max().item()
    print(f"max diff from one masked-out padding point in B (~0 expected): {diff:.6f}")

    # rho=0 invariant: x_j^{B->A} == x_j^B exactly there.
    from ppfn.prior.lupi.dataset import build_training_item

    rng = np.random.default_rng(0)
    rho0_items = [build_training_item(rng, progress=0.5, s_max=0.1, force_rho_zero=True) for _ in range(4)]
    rho0_batch = collate_lupi_batch(rho0_items)
    pos_diff = (rho0_batch.enc_x_inA - rho0_batch.enc_x).abs().max().item()
    print(f"max |enc_x_inA - enc_x| at rho=0 (~0 expected): {pos_diff:.6f}")

    # Incremental check: does the VALUE step alone converge if position is
    # exactly right? Feed T_hat = enc_x_inA directly (teacher-forced) as a
    # one-off, standalone RIB call, and check L_h drops over a few local
    # optimizer steps on this one fixed batch (diagnostic only -- the real
    # model never teacher-forces T_hat between iterations, see module
    # docstring; this is purely to isolate step correctness before trusting
    # the coupled loop).
    rib_probe = RegistrationIterationBlock(d_model=32, n_heads=4, d_ff=64, d_max=D_MAX, rqs_bins=6)
    opt = torch.optim.Adam(rib_probe.parameters(), lr=1e-2)
    hidden0 = torch.randn(4, batch.dec_ctx_x.shape[1] + batch.enc_x.shape[1] + 4, 32)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out_probe = rib_probe(
            batch.enc_x_inA, batch.enc_z, batch.enc_mask,
            batch.dec_ctx_x, batch.dec_ctx_z, batch.dec_ctx_mask,
            hidden0, batch.dec_ctx_x.shape[1], 4,
        )
        l_h = ((out_probe["h_hat"] - batch.enc_z_inA) ** 2 * batch.enc_mask).sum() / batch.enc_mask.sum()
        l_h.backward()
        opt.step()
        losses.append(l_h.item())
    print(f"value-step-alone probe, T_hat teacher-forced to truth: L_h[0]={losses[0]:.4f} -> L_h[-1]={losses[-1]:.4f} (expect a clear decrease)")
