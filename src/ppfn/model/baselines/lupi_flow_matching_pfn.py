"""Registration as a learned stochastic transport -- the flow-matching/SDE
direction logged in
docs/labbook/2026-09-17-lupi-registration-sde-flow-matching-direction.md,
superseding the ICP-shaped `IterativeRegistrationPFN` (branch
`lupi-iterative-registration`, discarded -- see
docs/labbook/2026-09-17-irt-discarded-isolation-probe-evidence.md).

Core idea: define a continuous path in JOINT position-value space from B's
raw observation to its fully registered image,

    z_0 = (x_j^B, y_j^B)          batch.enc_x, batch.enc_z
    z_1 = (x_j^{B->A}, y_j^{B->A})  batch.enc_x_inA, batch.enc_z_inA

(both already produced by `ppfn.prior.lupi.sampler.sample_pair` -- no prior
change needed, unlike the IRT branch, which needed nothing extra here
either). A transformer `v_phi(z_t, t, context)` is trained to predict the
velocity `dz_t/dt` along the straight-line interpolant `z_t = (1-t) z_0 + t
z_1`, via **conditional flow matching** (Lipman et al. 2023): since the
target velocity along a straight line is the CONSTANT `z_1 - z_0`
regardless of `t`, training is plain regression, simulation-free -- no
backprop through an ODE/SDE solver, unlike classical Neural-ODE training.
The solver only appears at inference (`FlowMatchingVelocityField.integrate`
below), where sampling the corresponding SDE with independent noise
realizations gives a genuine, non-Gaussian distribution over the registered
endpoint per point -- uncertainty as a property of the stochastic process
itself, not a separate head bolted onto a point estimate (the gap that
motivated discarding the ICP-shaped alternative: see the labbook entry
above for the precise comparison against `IterativeRegistrationPFN`'s
`alpha~0.095` position-step result).

T and h are recovered JOINTLY here -- one `z=(x,y)` state, one learned
drift -- rather than as two separately-parameterized mechanisms the way
`RegistrationIterationBlock` split them into a position-step and a
value-step. This is a deliberate change from that design, not an oversight:
it's what lets the model represent correlated T/h uncertainty (e.g. "if the
position warp went this way, the value recalibration must have gone that
way to stay consistent with what's observed") that two independent heads
structurally cannot.

Architecture: A's context cloud (fixed throughout, doesn't move) and B's
current interpolated state `z_t` are pooled into ONE token set, tagged by
an additive domain embedding exactly like `IDTokenPFN`'s (id=0 for A, id=1
for B) plus an additive time embedding on B's tokens only (A never depends
on `t`). Unlike `IDTokenPFN`, there is no train/test asymmetry here at
all -- every token attends to every other bidirectionally
(`_JointAttnBlock`, not `PFNBlock`): there's no held-out query set to
protect from leakage, since the only output this model produces is a
velocity for B's own tokens, read directly off their post-attention hidden
states.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.pfn.pfn import MaskedMHA
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class _JointAttnBlock(nn.Module):
    """Plain bidirectional self-attention block over one pooled token set --
    no train/test split (see module docstring). `MaskedMHA` reused as-is;
    only its self-attention call (`kv_input=None`) is used here, never its
    cross-attention path."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedMHA(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


class FlowMatchingVelocityField(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_max = d_max
        self.x_embed = nn.Linear(d_max, d_model)
        self.y_embed = nn.Linear(1, d_model)
        # Same additive-tag construction and orthogonal init as IDTokenPFN's
        # domain_embed (see that module's docstring for the rationale) --
        # id=0 -> A (fixed context), id=1 -> B (the state that moves).
        self.domain_embed = nn.Embedding(2, d_model)
        nn.init.orthogonal_(self.domain_embed.weight, gain=d_model**0.5)
        # Time conditioning: a small MLP on the raw scalar t in [0,1],
        # added only to B's tokens (A's tokens never depend on t -- they're
        # the fixed endpoint of every trajectory, not part of the moving
        # state). Simplest starting point (matches this codebase's own
        # "fewest extra parameters first" precedent, e.g. the h-value head's
        # Gaussian-before-bar-distribution choice) -- a sinusoidal/Fourier
        # time feature is the natural upgrade if plain-MLP conditioning
        # turns out to undertrain near t=0/1.
        self.time_embed = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        self.blocks = nn.ModuleList(
            [_JointAttnBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        # Joint position+value velocity, one linear head: d_max position
        # channels + 1 value channel, exactly the layout of z=(x,y) itself.
        self.velocity_head = nn.Linear(d_model, d_max + 1)

    def dim_mask(self, d_real: torch.Tensor, n: int) -> torch.Tensor:
        """Same convention as `ppfn.model.registration.model.RegistrationModel.dim_mask`
        -- [B] long -> [B,n,d_max] bool, True on the first d_real[b] real
        coordinate channels."""
        idx = torch.arange(self.d_max, device=d_real.device).view(1, 1, -1)
        return idx < d_real.view(-1, 1, 1)

    def _velocity(
        self,
        batch: LUPIBatch,
        zt_x: torch.Tensor,
        zt_y: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """zt_x: [B,n_enc,d_max]  zt_y,t: [B,n_enc] -> velocity [B,n_enc,d_max+1]
        (first d_max channels: position velocity; last channel: value
        velocity). Shared by `forward` (training, t~Unif(0,1) throughout the
        interpolant) and `integrate` (inference, t stepped explicitly)."""
        ctx_tok = (
            self.x_embed(batch.dec_ctx_x)
            + self.y_embed(batch.dec_ctx_z.unsqueeze(-1))
            + self.domain_embed.weight[0].view(1, 1, -1)
        )
        b_tok = (
            self.x_embed(zt_x)
            + self.y_embed(zt_y.unsqueeze(-1))
            + self.domain_embed.weight[1].view(1, 1, -1)
            + self.time_embed(t.unsqueeze(-1))
        )
        pooled = torch.cat([ctx_tok, b_tok], dim=1)
        pooled_mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)

        for block in self.blocks:
            pooled = block(pooled, key_padding_mask=pooled_mask)

        pooled = self.out_ln(pooled)
        n_ctx = batch.dec_ctx_x.shape[1]
        b_hidden = pooled[:, n_ctx:]
        return self.velocity_head(b_hidden)

    def forward(self, batch: LUPIBatch, t_override: torch.Tensor | None = None) -> dict:
        """Training-time forward: samples `t` per (batch item, B token) --
        NOT one shared `t` per batch item -- since every B point is its own
        independent (z_0,z_1) coupling for conditional flow matching, and
        sampling per-point maximizes the number of distinct (z_t,t) pairs
        seen per step, same spirit as this codebase's other per-token
        random draws (e.g. `dec_qry_source`'s per-query mixture).
        `t_override`: for the rho=0/force_h_identity degenerate-path check
        (`__main__` below) and for `integrate`'s reuse of this method at an
        explicit `t`. -> {"velocity_pred_pos": [B,n_enc,d_max],
        "velocity_pred_val": [B,n_enc], "t": [B,n_enc]}. The straight-line
        interpolant's target velocity (`enc_x_inA - enc_x`,
        `enc_z_inA - enc_z`) is CONSTANT in `t`, so the loss recomputes it
        directly from `batch` -- nothing about it depends on the sampled
        `t` returned here."""
        B, n_enc, _ = batch.enc_x.shape
        if t_override is None:
            t = torch.rand(B, n_enc, device=batch.enc_x.device, dtype=batch.enc_x.dtype)
        else:
            t = t_override
        t_x = t.unsqueeze(-1)
        zt_x = (1 - t_x) * batch.enc_x + t_x * batch.enc_x_inA
        zt_y = (1 - t) * batch.enc_z + t * batch.enc_z_inA

        velocity = self._velocity(batch, zt_x, zt_y, t)
        return {
            "velocity_pred_pos": velocity[..., : self.d_max],
            "velocity_pred_val": velocity[..., self.d_max],
            "t": t,
        }

    @torch.no_grad()
    def integrate(
        self, batch: LUPIBatch, n_steps: int = 20, sigma: float = 0.0, n_samples: int = 1
    ) -> dict:
        """Euler-Maruyama integration of `dz_t = v_phi(z_t,t,context) dt +
        sigma dW_t` from `t=0` (`batch.enc_x`/`enc_z`, B's raw observation)
        to `t=1` -- `sigma=0` reduces to a deterministic Euler ODE
        integration (the mean-transport estimate; `n_samples` beyond 1 is
        then wasted, all trajectories coincide). `sigma>0` with
        `n_samples>1` produces genuinely different trajectories per sample,
        whose empirical spread at `t=1` IS the model's conveyed uncertainty
        -- see the labbook entry's "where uncertainty comes from" section.
        A's context is fixed throughout; only B's `(x,y)` state evolves.
        -> {"x1": [n_samples,B,n_enc,d_max], "y1": [n_samples,B,n_enc]}."""
        assert n_steps >= 1
        device, dtype = batch.enc_x.device, batch.enc_x.dtype
        B, n_enc, _ = batch.enc_x.shape
        dt = 1.0 / n_steps

        x1_samples, y1_samples = [], []
        for _ in range(n_samples):
            x_t = batch.enc_x.clone()
            y_t = batch.enc_z.clone()
            for step in range(n_steps):
                t_val = step * dt
                t = torch.full((B, n_enc), t_val, device=device, dtype=dtype)
                velocity = self._velocity(batch, x_t, y_t, t)
                v_pos, v_val = velocity[..., : self.d_max], velocity[..., self.d_max]
                noise_pos = torch.randn_like(x_t) if sigma > 0 else 0.0
                noise_val = torch.randn_like(y_t) if sigma > 0 else 0.0
                x_t = x_t + v_pos * dt + sigma * (dt**0.5) * noise_pos
                y_t = y_t + v_val * dt + sigma * (dt**0.5) * noise_val
            x1_samples.append(x_t)
            y1_samples.append(y_t)
        return {"x1": torch.stack(x1_samples, dim=0), "y1": torch.stack(y1_samples, dim=0)}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run it on a real batch from LUPIStreamDataset, print shapes, confirm the
    loss/backward path works end to end, and check the rho=0/
    force_h_identity degenerate-path invariant (z_0==z_1 exactly there, so
    the target velocity is exactly zero -- a cheap sanity check on the loss
    wiring before trusting it on real, non-degenerate draws)."""
    import numpy as np
    import torch

    from ppfn.loss.lupi_flow_matching_loss import FlowMatchingLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = FlowMatchingVelocityField(d_model=32, n_heads=4, n_layers=2, d_ff=64)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("velocity_pred_pos:", out["velocity_pred_pos"].shape)
    print("velocity_pred_val:", out["velocity_pred_val"].shape)

    criterion = FlowMatchingLoss()
    loss, metrics = criterion(model, batch, out)
    print("loss/metrics:", metrics)

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None: {n_none}")

    print("\nrho=0, force_h_identity degenerate-path check "
          "(z_0 == z_1 exactly -> target velocity == 0 everywhere real):")
    degenerate_dataset = LUPIStreamDataset(seed=0, s_max=0.1, force_rho_zero=True, force_h_identity=True)
    degenerate_items = [next(iter(degenerate_dataset)) for _ in range(4)]
    degenerate_batch = collate_lupi_batch(degenerate_items)
    max_pos_gap = (degenerate_batch.enc_x_inA - degenerate_batch.enc_x).abs().max().item()
    max_val_gap = (degenerate_batch.enc_z_inA - degenerate_batch.enc_z).abs().max().item()
    print(f"max |x_b_inA - x_b| (expect ~0): {max_pos_gap:.6f}")
    print(f"max |z_b_inA - z_b| (expect ~0): {max_val_gap:.6f}")
    degenerate_out = model(degenerate_batch)
    degenerate_loss, degenerate_metrics = criterion(model, degenerate_batch, degenerate_out)
    print("degenerate-path loss/metrics (pre-training, NOT expected to be ~0 yet "
          "-- this only checks the TARGET is ~0, not that the untrained model already predicts it):",
          degenerate_metrics)

    print("\nintegrate() smoke test: n_steps=5, sigma=0 (deterministic), n_samples=1")
    integrated = model.integrate(batch, n_steps=5, sigma=0.0, n_samples=1)
    print("x1:", integrated["x1"].shape, "y1:", integrated["y1"].shape)
    print("integrate() smoke test: n_steps=5, sigma=0.1, n_samples=3 (stochastic)")
    integrated_sde = model.integrate(batch, n_steps=5, sigma=0.1, n_samples=3)
    spread = integrated_sde["y1"].std(dim=0).mean().item()
    print(f"mean per-point std across the 3 SDE samples (expect > 0): {spread:.4f}")
