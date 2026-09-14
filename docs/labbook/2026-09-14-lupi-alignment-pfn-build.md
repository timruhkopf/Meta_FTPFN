# LUPI-distilled alignment-aware multi-task PFN: first build

*Log, 2026-09-14. Implements the LUPI/JEPA-thread build spec
(`2026-09-14-lupi-aligned-multitask-pfn-build-spec.md`, consolidated from the
JEPA/LUPI discussion) as a new, self-contained prior/model/loss/trainer under
`ppfn.prior.lupi`, `ppfn.model.lupi`, `ppfn.loss.lupi_loss`,
`ppfn.trainer.lupi_trainer`. Additive to the existing registration prior/model
(`ppfn.prior.registration`, `ppfn.model.registration`) — nothing there was
touched — and to `ppfn.model.baselines.id_token_pfn`, whose `PFNBlock` reuse
pattern this follows.

## What was built

- `ppfn.prior.lupi`: T reuses `ppfn.prior.registration.warp` verbatim
  (velocity-field diffeomorphisms, `sample_warp_pair`/`flow_rk4`/
  `mix_velocity`/`declared_box`) and `.region`/`.function_prior` for the
  shared latent function and A's support restriction. Two genuinely new
  pieces: `monotone.py` (the y-distortion h, an `a*y+b+c*asinh(d*y)` family
  that's monotone by construction for a,c,d>=0 — no rejection sampling) and
  `acquisition.py` (a simulated-acquisition A-design: rank-weighted sampling
  without replacement from a uniform pool within A's region, `beta`
  controlling aggressiveness, `beta=0` w.p. 0.2 to keep uniform designs in
  the mixture). `ecdf.py` implements build-spec option (b) from §3.2
  literally: A's own ECDF is fit from exactly the noisy, acquisition-biased
  sample it has (the pathology being taught, not corrected here); B's ECDF
  is fit from noiseless `f(z_B)` since `z_B ~ Uniform` already matches the
  declared-domain reference measure this project's invariant #8 uses
  elsewhere for coordinates — no separate dense grid needed (a deliberate
  simplification vs. spec §3.1's literal "surrogate on a grid" instruction,
  justified in `ecdf.py`'s docstring).

- `ppfn.model.lupi.model.LUPIPFN`: single shared trunk, two positioning
  modes (student: A-frame coordinates only; oracle: the true B-frame
  position `T(x)` injected through the SAME shared coordinate embedding).
  Per align-layer: one `MaskedMHA` cross-attention call over the whole
  `[A_ctx;A_qry]` tensor against B (reused/cached `b`, computed once via
  `encode_b`), then `ppfn.model.pfn.pfn.PFNBlock` reused verbatim for the
  "A-context self-attends, A-query cross-attends to A-context only, never
  to itself" step + FFN — this is exactly spec §4.3's per-layer structure,
  and reusing `PFNBlock` here (rather than reimplementing the train/test
  mask) is the same reuse `ppfn.model.baselines.id_token_pfn` already
  established as this repo's convention.

- `ppfn.loss.lupi_loss.LUPILoss`: `L = L_student + lambda_o*L_oracle +
  lambda_d*L_distil`, all three via `model.bar_dist`
  (`ppfn.model.pfn.bar_distribution.BarDistribution`, per explicit
  instruction to reuse it rather than
  `ppfn.model.registration.heads.TailBarDistribution` — a good fit since
  targets are quantile-normalized to [0,1], `BarDistribution`'s native
  support). Distillation is forward-KL-as-cross-entropy on the bar
  distribution's bin probabilities directly; uniform bin widths make the
  missing `-log(bucket_width)` term cancel between oracle and student
  exactly, so spec §5.2's optional equal-mass-binning refinement isn't
  needed.

- `ppfn.trainer.lupi_trainer.LUPITrainer`: near-verbatim sibling of
  `ppfn.trainer.id_token_trainer.IDTokenTrainer` (same fixed-validation-batch
  convention, same checkpoint schema).

- Configs: `configs/{prior,model,loss,trainer}/lupi*.yaml` +
  `configs/experiment/lupi_baseline{,_debug}.yaml`, wired into the existing
  generic `ppfn.pipelines.train` entry point. Real-run config points
  `callbacks.mlflow.experiment_name` at a SEPARATE MLflow experiment
  (`lupi-alignment-pfn`, `-debug` suffix for the smoke-test config) per
  request, rather than reusing `ppfn_training`/`id-token-baseline`.

## Deliberate simplifications vs. the build spec (not yet built)

- No GlobalAffineHead / coordinate-free layer 1 / gated cross-attention —
  those are `ppfn.model.registration`-specific invariants (CLAUDE.md) that
  this spec explicitly doesn't require (it never outputs a transport
  estimate at all).
- Query-mix fractions (spec §6.2: 40% uniform / 40% near-B / 20% near-A) are
  implemented as `prior.frac_uniform`/`prior.frac_near_b` — "optional
  preferential near-B sampling" is `frac_near_b=0.0` as an ablation switch,
  per the request that this stay configurable/off-able rather than hardwired
  on.
- No equal-mass bar-distribution bins, no §7 diagnostics (attention
  dispersion vs. identifiability, path attribution, CDF diagnostics) — out
  of scope for this pass per "let's not worry too much about additional
  metrics just yet."
- `h` and the acquisition sampler are simplifications of the spec's own
  suggestions (asinh family instead of an unconstrained monotone spline;
  rank-weighted single-pool sampling instead of a real EI-on-GP reference
  acquisition) — both documented in their own module docstrings.

## Verification

Local smoke tests (`MPLBACKEND=Agg`, CPU):
- rho=0 invariant: `max|oracle_bpos - x| = 0.0` exactly, both for A-context
  and A-query tokens, before AND after batching/collation.
- Model: gradients flow to every trainable parameter (`n_none=0`); one
  masked-out padding token in B changes `predictive_logits` by <4e-7 (float
  noise); permuting A-context token order changes them by the same order of
  magnitude (permutation invariance of the train/test PFN mask, inherited
  from `PFNBlock`).
- Full pipeline (`python -m ppfn.pipelines.train experiment=lupi_baseline_debug
  experiment_name=00-debug-lupi ~benchmark`, 2 epochs x 5 steps, CPU): runs
  end to end, logs to MLflow (`lupi-alignment-pfn-debug`), loss decreases
  epoch to epoch (student_nll 0.228 -> 0.103, oracle_nll 0.227 -> 0.104).
  Not a claim of a working model yet, just that the data/loss/optimizer path
  is wired correctly.

Not yet run: the real-sized `lupi_baseline` config, and no comparison against
an A-only baseline or the registration model's own upper bounds (build spec
§7.1's three-NLL report) — next step, on ulysses.

commit: pending
