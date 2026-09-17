# Prior [0,1]-normalization in progress; real run prepped, not launched

`commit: pending`

## Prior change (user's own, in progress -- not touched by this entry's author)

The user is reworking `ppfn.prior.lupi` so its outputs are normalized to
`[0,1]` for `[0,1]` inputs, while still preserving genuine, informative
variation in `T` and `h` *between* draws -- i.e. not normalizing away the
exact registration difficulty this whole project measures. Nothing else
about the prior's contract is changing (still `z -> Φ_0(z)`/`Φ_ρ(z)` for
position, `h(f(z))` for A's value, same LUPI privileged fields
`enc_x_inA`/`enc_z_inA`). Not implemented by this session; logged here as
project context because it's a live dependency for the run prepped below.

**Why this matters for anything already built:** confirmed
`ppfn.model.baselines.calibration.sample_calibration_borders` (used by both
`BoundsPFN` and, transitively, `FlowMatchingRegistrationPFN`'s predictor)
fits `FullSupportBarDistribution`'s bin borders from a FRESH sample drawn
from the live prior at model-construction time -- it does not hardcode an
assumed target range anywhere. So this normalization change needs **no**
config or code change on the model/loss side to keep working; the borders
will simply refit to whatever the new, normalized target scale turns out to
be the next time a model is constructed. This was checked, not assumed, by
reading that module directly.

## Rundown of the model this branch has built so far

(Full version given directly to the user in conversation; summarized here
for the record.) Pipeline: `ppfn.prior.lupi` (shared latent `z`, `Φ_0`/`Φ_ρ`
diffeomorphisms, value distortion `h`, privileged `enc_x_inA`/`enc_z_inA`
known only at training time) -> `FlowMatchingVelocityField` (`v_φ`, conditional
flow matching, simulation-free training, `integrate()` for SDE sampling at
inference) -> `FlowMatchingRegistrationPFN` (wraps `v_φ` with `BoundsPFN`'s
pooled-`[A_ctx;B_inA]` decoder-only readout, `K`-sample probability-space
mixture into one `FullSupportBarDistribution` PPD, student on
`v_φ`'s own estimate vs. teacher on the true `B_inA`) -> `FlowMatchingRegistrationLoss`
(velocity regression + student-mixture NLL + teacher/upper-1 NLL + CE
distillation, all predictive terms on `dec_qry_mask` only). See
[2026-09-17-flow-matching-registration-predictive-readout.md](2026-09-17-flow-matching-registration-predictive-readout.md)
for what was verified (module `__main__` diagnostic + a CPU debug Hydra run,
loss 11.9->8.0 over 2 epochs).

## Real-run config prepped, NOT launched

`configs/experiment/lupi_flow_matching_registration.yaml` -- sized on the
sibling real runs (`lupi_bounds.yaml`, `lupi_id_token_baseline.yaml`):
`n_a_range=[8,100]`, `n_b_range=[8,100]`, `n_qry_range=[8,64]`,
`warp_grid_n=4`, 100 epochs x 500 steps, `bf16`, same
optimizer/scheduler (`adamw_registration`, warmup 5000, `eta_min=3e-5`).
Two deliberate first-run choices, both untuned:

- `batch_size=24` (vs. 32-64 in the sibling configs) -- conservative,
  because this model pays extra cost the siblings don't: `integrate()`
  (`n_integration_steps` sequential no-grad forward passes through `v_φ`)
  runs every training step, on top of the `K+1`-stacked `BoundsPFN` forward
  (`K=1` here, so 2x-stacked, same multiplier as `lupi_id_token_baseline`'s
  own student/teacher stacking).
- `n_integration_steps=10` (vs. the model's own default of 20) -- halved for
  training-time cost; this is an integration-accuracy/wall-clock trade at
  TRAINING time only (the model's `integrate()` target during training is
  "good enough registered B for the predictor to learn from," not a
  final-eval-quality trajectory), revisit once one epoch's actual wall-clock
  on ulysses is known.
- `n_transport_samples=1`, `sde_sigma=0.0` kept at the model config's own
  default (deterministic ODE mean, mixture is a no-op at `K=1`) --
  deliberate, not an oversight: `K>1`/`sigma>0` is the next isolated
  experiment on top of this one once the deterministic pipeline's own
  numbers are in, matching this repo's "isolate before trusting coupled"
  convention.

Verified by dry-run Hydra compose only (`hydra.compose` with
`experiment=lupi_flow_matching_registration`, no training) -- resolves to
the right `_target_`s and sizes, no interpolation errors.

**Not yet launched, and shouldn't be until:**
1. the prior's `[0,1]`-normalization change lands and passes its own
   `rho=0`/`force_h_identity` invariant checks (launching against a
   mid-edit prior burns real compute against a data-generating process
   that's about to change under it), and
2. the user gives the go-ahead to actually submit it to ulysses (a
   real, shared, hard-to-fully-reverse compute commitment -- not something
   to launch on this session's own initiative).

commit: pending
