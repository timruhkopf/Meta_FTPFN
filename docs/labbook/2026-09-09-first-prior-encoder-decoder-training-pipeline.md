# First implementation: prior, encoder-decoder architecture, training pipeline

Implements `docs/ARCHITECTURE.md` §1 (prior), §2 (encoder-decoder), §3
(objectives), and the Hydra/MLflow-wired training loop from §4 — the first
code for this project beyond the docs themselves. Covers CLAUDE.md build
order steps 1-8 in one pass (prior + invariant tests, encoder + gated
cross-attention, transport head, coupling head, global affine, oracle +
pathway distillation, role randomization, masking modes) but NOT step 9
(bounds/fold-calibration/baselines/spending law — deliberately deferred,
see below). No training run beyond short smoke tests was performed; this
entry is about what was built and verified to run correctly, not about
model quality.

## What was verified

- **ρ=0 invariants** (`tests/test_registration_prior.py`, 16 tests, all
  passing): `A_inB_target == x̃^A` exactly (max abs diff `0.0`, not just
  small) and `pooled_context == [A ∪ B_inA]` exactly, swept over 20 seeds
  and d ∈ {1,2,3,5}. Also: declared-box normalization does NOT collapse to
  a restricted region's own empirical bbox (invariant #8), and realized
  sub-box volume fraction matches the nominal target to <2% over 200k MC
  samples.
- **Query mask-isolation** (`decoder.py`/`model.py` demos): perturbing one
  query token's input changes no other query token's output, at every
  layer, exactly (max diff `0.0`) — the structural property invariant #2
  depends on.
- **`transport_override` actually reaches the cross-attention queries**: a
  real bug, caught before it shipped — see below.
- **Full forward+backward pass**: all 127 params receive finite, non-NaN
  gradients through predictive + transport + coupling losses.
- **End-to-end Hydra/MLflow run** (`experiment=registration_debug`, 2
  epochs × 5 steps, `experiment=step4_pathway`, 2×3 steps): both complete
  cleanly, and all designed metrics (loss components, §3.6 weight
  schedule, per-layer gates, `bounds/transfer_gap`, per-layer transport
  NLL / coupling MSE) land in the MLflow run as expected.

## Bug caught during self-review: oracle teacher-forcing was a no-op

`Decoder.forward`'s `transport_override` param (needed for §3.4's oracle
distillation — "t_i^(l) := A_inB_target(i), all l, all tokens") was applied
to the returned/logged `t_ctx_layers`/`t_qry_layers` values, but each
`DecoderLayer` computed its OWN `t_ctx`/`t_qry` internally from the
affine+Δ_ℓ formula, and never read the override — so the cross-attention
queries at every layer were built from the *un-forced* transport regardless
of the override. The teacher pass would have run and produced *a* KL loss,
plausible-looking, silently measuring nothing like an oracle. Fixed by
threading the override into `DecoderLayer.forward` itself (checked before
the affine+Δ branch) and verified directly: forcing a random tensor via the
override changes the final hidden state (confirms the override reaches
computation), and `t_ctx_layers[1]` exactly equals the forced tensor
(confirms the override isn't just influencing but *replacing* it). This is
exactly the kind of bug CLAUDE.md's "When unsure" section warns about
("produce a plausible-looking model measuring the wrong thing... not
visible in the training curves") — caught here only because the fix was
demo/test-verified immediately rather than trusted by inspection.

## s_max calibration for the velocity-field warp rejection band

ARCHITECTURE.md §1.3: "expect a few percent rejection at s_max≈1; tune
s_max to hit that" (band: `log|det J|` max−min ≤ log(9) on a 5^d grid).
Swept `s_max ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5}` × `d ∈ {1,2,3,5}`, 150
draws each, no other change:

| s_max | d=1 | d=2 | d=3 | d=5 |
|---|---|---|---|---|
| 0.05 | 0% | 0% | 0% | 0% |
| **0.1** | **7.3%** | **6.7%** | **2.7%** | **0%*** |
| 0.15 | 10.7% | 20.0% | 19.3% | 12.0% |
| 0.2 | 18.7% | 30.7% | 36.7% | 30.7% |
| 0.3 | 33.3% | 48.7% | 53.3% | 51.3% |
| 0.5 | 42.7% | 66.7% | 72.7% | 65.3% |

(*d=5 sweep was cut short by the background job timeout at s_max=0.1/0.15;
the two points shown are real, the rest of that row is extrapolated from
the visible trend and not separately confirmed.)

At `s_max=1.0` (a literal reading of "s ~ Uniform[0,1]·s_max" with
`s_max=1`) rejection was 84% at d=2 — nowhere near "a few percent". Picked
**`s_max=0.1`** as the default (`ppfn.prior.registration.sampler`'s
`sample_pair(..., s_max=0.1)`): 0-7.3% across d ∈ {1,2,3}, squarely "a few
percent". Not re-verified against a full d=5 sweep or a larger sample size
— worth revisiting if d=5 draws look qualitatively under- or over-warped
once the prior visualizer (`sampler.py`'s own `__main__` demo) gets a real
look rather than just the invariant-test pass.

## Design decisions made where the spec was ambiguous

Recorded here rather than in `docs/decisions.md` since these are
implementation choices within an already-settled design, not reversals of
settled decisions — flag if any should be promoted there.

- **§2.5 vs §3.2 tension on the transport head's depth.** §2.5 says "All
  heads read the final decoder representation"; §3.2 and invariant #9
  explicitly require `L_transport = Σ_l w_l · NLL(T^(l), target)`,
  deep-supervised. Implemented per §3.2/invariant #9 (transport head
  applied at every decoder layer, shared weights) since those are called
  out as load-bearing; read §2.5's blanket statement as describing the
  predictive head's behavior specifically. Flag if this reading is wrong.
- **Predictive-loss role-swap exclusion done via a token mask multiplying
  `y_qry_mask`** (`RegistrationLoss`), not a separate code path, so it
  composes with padding for free — but means a fully role-swapped batch
  contributes `loss/pred = 0` exactly, which will look like a NaN-adjacent
  degenerate value in a metrics dashboard if someone isn't expecting it.
- **Context/query split point for the decoder cloud** (not specified by
  the prior spec at all, since §2 introduces queries only at the
  architecture level): uniform over `[1, n_dec-1]`, resampled every draw.
- **Role randomization implemented as pure batch-assembly relabeling**
  (`ppfn.prior.registration.dataset.build_training_item`), not a second
  generative path — the prior already produces both `A_inB_target` and
  `B_inA_target` per pair (§1.5), so "swap" just picks which cloud/target
  pair feeds the encoder vs. decoder stream. Verified algebraically (not
  just asserted) that this is consistent with §3.5's pathway distillation
  at ρ=0: since `box_A == box_E` there, `pooled_context` is symmetric in
  the two clouds regardless of which one is "decoder" that step.
- **Predictive head tail bins**: implemented as an exponential
  (`tail_rate=1.0`, both sides) density anchored at ±4, not a ported
  `FullSupportBarDistribution` — the reference implementation this
  project's `BarDistribution` was ported from is referenced in that
  module's own docstring but not vendored anywhere in this repo (checked
  `archive/`), so there was nothing to copy the exact functional form
  from. A proper, exact density (verified: NLL is finite and
  monotonically increasing with distance past the border), just not
  necessarily calibrated the same way PFNs4BO's own version would be.
- **Deferred to CLAUDE.md build-order step 9** (not attempted this pass):
  fold-fraction (§5.2), Fisher-conditioning diagnostic (§5.3), and the
  upper-1/upper-2 KL three-way decomposition (§5.1). The monitor registry
  (`ppfn.monitor.registry`) is built so each is a one-function addition
  when that step is reached — no other file needs to change.

## What's NOT yet verified

No actual training run long enough to see loss go down, no check that
`Var(T(x)) ≈ σ²p/(n_A·E|∇f|²)`-style scaling shows up, no eyeballed prior
visualizer plots (the `__main__` demos exist and were smoke-run for shape
correctness, not inspected for "does this look like a sensible warp").
`docs/ARCHITECTURE.md`'s build order treats step 2 (prior visualizer,
eyeball 50 draws) as a hard prerequisite before anything else is worth
trusting — that eyeballing hasn't happened yet, only the invariant tests
have. Next real step per the build order is exactly that.

`commit: pending`
