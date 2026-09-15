# Quantile-normalization-under-A is structurally bottlenecked; a decoder-only, task-ID-tagged alternative

*Log, 2026-09-15. Follow-up to
[Reference-measure mismatch: why naive pooling can fail completely, not just a little](2026-09-15-lupi-reference-measure-mismatch.md).
Captures the argument that led from "the ECDF diagnostic checks out" to
"the whole quantile-normalization design may be the wrong approach," plus a
proposed simpler architecture. This is a design-discussion log, not a
result -- nothing here has been built or tested yet except where noted.*

## The meta-argument: A's own ECDF is unfalsifiable from A's own data

Today's diagnostics (previous two entries) independently converged on the
same root cause:

- `z_a_ctx` is *exactly* `rank/(n_a+1)` -- verified exactly. The value
  channel is pure rank; only the permutation across `x` carries information.
- Every A context point can land at quantile 0 under B's own ECDF
  (confirmed: zero overlap between A's raw `h(f(.))` range and B's raw
  `f(.)` range, for a draw with only mild acquisition bias, `beta=0.59`).
- A naive attempt to measure the "true" gauge mismatch via a shared
  z-reference came back tautologically zero -- exposing that `h`'s
  rank-preservation cancels exactly whenever both ECDFs share a reference
  ensemble, which says nothing about the mismatch that matters (uniform in
  each task's own x-space, not uniform in z).

The unifying point, stated plainly: **A's own small, acquisition-biased
sample cannot self-diagnose its own bias.** Whether A's observed range is
the true distribution's bulk or an extreme tail is not recoverable from A's
data alone -- there is no signal in A's own sample telling you this. Any
scheme that normalizes A purely against A's own evidence (current
option-(b) design) is therefore structurally bottlenecked, not just
under-resolved. This isn't a new implementation bug to fix; it's a
property of the design.

## Proposal 1: make "inpainting via B" an explicit objective, not an incidental one

Current design (`ppfn.prior.lupi.acquisition`, §6.2's query mixture) already
biases A's design toward good regions and places some test queries near B.
But that still scores against A's own (broken) ECDF, and doesn't test the
sharper claim: **A's design may never have visited its own best region at
all** -- not sparsely sampled, never touched. The only evidence for "there
might be something better than anything A has seen" is B's broader
coverage. That's a qualitatively different capability than "predict A's
value at x" -- closer to "does B's landscape suggest a basin A hasn't
found" -- and worth making an explicit, separately-reported objective/eval
slice rather than folding it into aggregate NLL. Not yet built.

## Proposal 2: replace per-task quantile normalization with a shared reference, keep bounded output

Clarification on "raw" (this was initially proposed as removing
normalization entirely; corrected): values should still land in `[0,1]`
(the bounded `BarDistribution` stays a good fit) -- but via a **shared**
reference, not A's own biased sample. In the synthetic prior this can be a
true reference measure (dense grid, or B's own large-sample ECDF, already
implemented as `LUPIPairInternals.value_under_b_ecdf`). Consequence worth
being explicit about: if A's true range sits entirely outside B's observed
range, A's points collapse toward quantile 0 under this scheme -- but
that's not a failure mode to route around, it's the "inpainting" signal
from Proposal 1 (a collapse toward 0 *is* "A's optimum is likely better
than anything B has explicitly seen"). At deployment, no true reference
measure is available; the practical fallback is B's own empirical CDF
(B is the large, "abundant, well-explored" side) -- an open question, not
resolved here.

## Proposal 3: a simpler decoder-only architecture, no explicit registration machinery

Reuse `ppfn.model.baselines.id_token_pfn.IDTokenPFN`'s pattern (pooled
tokens, additive task-id embedding, `TailBarDistribution`) rather than
`LUPIPFN`'s bespoke cross-attention-with-injected-oracle-position machinery:

- **Oracle**: context = `[A_ctx ; B_inA]` (B already registered via the
  true `T` -- privileged, `x_b_inA`/`enc_x_inA` already exist).
- **Student**: context = `[A_ctx ; B]` (B in its own native, unregistered
  frame; task-id tag is the only hint the two streams differ).
- Both scored by NLL on `A_test`; student additionally trained by
  forward-KL/CE distillation against the oracle's predictive distribution
  -- same objective shape as `ppfn.loss.lupi_loss.LUPILoss`, simpler
  underlying model.

**Explicitly ruled out for this variant:** the "layer 1 cross-attends on
`y` only" trick (`ppfn.model.lupi.model`'s architecture doesn't use this
either, but it was raised as a candidate mechanism during discussion and
explicitly rejected for the decoder-only variant) -- feed `x`, `y`, and the
task-id token together from layer 1, no special coordinate-blocking.

**Risks/open questions flagged before building, not objections:**

1. Trades a guarantee for an empirical bet. Quantile-normalization's
   original purpose was "`h` is absorbed by construction," provable. Raw
   (shared-reference-normalized) values + task-id drops that guarantee;
   the model must *learn*, from data and the distillation signal alone,
   that two streams can be on different calibrations. Plausible, not
   proven.
2. The "mismatched geometry naturally suppresses early cross-stream
   attention" mechanism (the task-id token as a "valve") is plausible --
   consistent with the *intuition* behind this repo's other architecture's
   own layer-1 invariant, though that invariant itself is being rejected
   here -- but unverified. Worth an attention-visualization check once
   built, not assumed.
3. Doesn't conflict with CLAUDE.md's "no unconstrained transport head"
   precedent for the other (registration) architecture -- if anything it's
   a purer instance of "match distributions, never vectors," since there's
   no intermediate registration output at all.

## Status

Not yet built by this session. **A parallel Claude Code session is already
implementing this** -- coordinated via cross-session message (see this
session's own transcript) to make sure it: reuses `ppfn.prior.lupi`'s
existing `h`/`T` machinery rather than building a new prior, refines
`ppfn.prior.lupi.acquisition`'s biased-A sampling to guarantee a
non-collapsing probability tail toward higher-performing regions (not hard
truncation), and drops the layer-1 y-only idea per the ruling above.

commit: pending
