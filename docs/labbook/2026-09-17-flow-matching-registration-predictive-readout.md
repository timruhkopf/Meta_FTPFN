# Closing the loop: a predictive readout on top of `FlowMatchingVelocityField`

`commit: pending`

## What prompted this

[2026-09-17-lupi-registration-sde-flow-matching-direction.md](2026-09-17-lupi-registration-sde-flow-matching-direction.md)
built `FlowMatchingVelocityField` (`ppfn.model.baselines.lupi_flow_matching_pfn`,
commit `ddd17f0`): a transformer `v_φ` trained by conditional flow matching to
transport B's raw `(x,y)` into `\hat{z}_1 = (\hat x^{B→A}, \hat y^{B→A})`. That
entry left "coupling to the predictive stack" explicitly open — `v_φ` alone
registers B, it doesn't predict anything for A's queries. This session built
the missing piece: `FlowMatchingRegistrationPFN`, in the same module.

## What was built

- **`FlowMatchingRegistrationPFN`** wraps `v_φ` with
  `ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN`'s pooled-`[A_ctx;B_inA]`
  decoder-only readout, reused unmodified. At inference/training, `v_φ`'s own
  `integrate()` (Euler–Maruyama, already `@torch.no_grad()`) draws `K`
  independent SDE trajectories per B token; each is substituted for
  `enc_x_inA`/`enc_z_inA` and read out by `BoundsPFN` (`severed=False`).
  Student and teacher (the TRUE `enc_x_inA`/`enc_z_inA`, i.e. upper-1) share
  one `BoundsPFN` backbone, `K+1` batch copies stacked for a single forward
  call — generalizes `LUPIIDTokenPFN`'s own student/teacher stacking trick.
- **Mixture in probability space, not logit space.** The `K` student
  bar-distribution logits are combined as `log(mean_k softmax(logits_k))`
  (`logsumexp` over `log_softmax`, minus `log K`) — averaging logits directly
  would not average the underlying distributions. This quantity turns out to
  be idempotent under `BarDistribution`'s own internal `log_softmax`
  (`log_softmax(log p) == log p` when `p` already sums to 1), so no new
  NLL/CE code was needed anywhere downstream — the mixture logits feed
  `bar_dist`'s existing machinery unchanged. Verified in the `__main__`
  diagnostic: `logsumexp(student_logits, dim=-1)` is ~0 everywhere (max abs
  `2.4e-7`), confirming it's a valid log-distribution.
- **`v_φ` trains only from its own `FlowMatchingLoss` term.** `integrate()`
  stays `@torch.no_grad()` — deliberately not backpropagating the predictive
  NLL through `n_steps` of Euler-Maruyama, which would reintroduce exactly
  the simulation-in-the-loop cost flow matching exists to avoid. `v_φ`'s own
  `forward()` (not `integrate()`) is called separately, grad-tracked, purely
  to supply `L_flow` — verified by checking `grad is None` counts per
  parameter group after `.backward()` (all params get a gradient; `v_φ`'s
  path is through its own forward call, not through the no-grad integration).
- **`FlowMatchingRegistrationLoss`**
  (`ppfn.loss.lupi_flow_matching_registration_loss`): four terms —
  `L_flow` (unmodified `FlowMatchingLoss`, called on `model.flow_field`) +
  student-mixture NLL + teacher (upper-1) NLL + CE distillation
  (student pulled toward detached teacher, same plain-CE-not-forward-KL
  convention as `LUPIIDTokenLoss`, for the same NaN-avoidance reason). Fixed
  `lambda_flow=lambda_ce=1.0`, no curriculum — first cut, isolate whether the
  coupling helps before tuning a schedule on top of a schedule.
- Hydra configs (`configs/model/lupi_flow_matching_registration.yaml`,
  `configs/loss/lupi_flow_matching_registration.yaml`,
  `configs/trainer/lupi_flow_matching_registration.yaml` — reuses
  `IDTokenTrainer` as-is, same rationale as the bare flow-matching config —
  and `configs/experiment/lupi_flow_matching_registration_debug.yaml`).
  Default config ships `n_transport_samples=1, sde_sigma=0.0` (deterministic
  ODE mean, mixture is a no-op at `K=1`) — the genuinely stochastic SDE
  (`K>1`, `sigma>0`) is the next isolated experiment on top of this one, not
  bundled into the same first cut.

## Verification

- `__main__` diagnostic (`python -m ppfn.model.baselines.lupi_flow_matching_pfn`):
  shapes check out, loss finite, gradients reach every trainable parameter,
  mixture logits are a valid log-distribution, and the `rho=0`/
  `force_h_identity` degenerate batch runs through the full model without
  error.
- End-to-end Hydra run (`experiment=lupi_flow_matching_registration_debug`,
  2 epochs × 5 steps, CPU): all four loss terms log correctly through
  `IDTokenTrainer`/MLflow, and `loss/total` drops 11.93 → 8.05 over the two
  epochs on this tiny smoke config (not a real training result, just
  confirms the wiring learns something rather than being inert).

## Open, not addressed here

- `K>1`/`sigma>0` (the actual stochastic-SDE mixture, the main point of the
  original design) — not yet run at real scale; only smoke-tested (`K=3`,
  `sigma=0.1`) in the `__main__` diagnostic.
- Whether `v_φ` should eventually get *some* gradient signal from the
  predictive loss (e.g. via a differentiable-but-truncated integration, or
  REINFORCE-style score-function estimator on the SDE noise) is deliberately
  not addressed — current design keeps the two training signals fully
  decoupled, which is the simplest thing that could work and matches this
  repo's "isolate before trusting coupled" convention.
- No real-sized run launched yet.
