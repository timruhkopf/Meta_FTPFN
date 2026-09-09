# Milestone review — 2026-08-27

Scope: `docs/ROADMAP.md`, `docs/milestones/M0`–`M14`, `.claude/rules/*.md`,
`src/ppfn/model/unwarp.py`, `src/ppfn/prior/bnn/{bnn_prior,mlp}.py`,
`configs/prior/bnn.yaml`, and the installed
`tabpfn/architectures/{tabpfn_v2_5,shared/bar_distribution}.py`. This is a
documentation/planning review — code-diff review was out of scope (too large
for `/ultrareview`; see the command output). Findings are graded:
**[BLOCKING]** will silently produce wrong results or stall work if
unaddressed, **[DESIGN]** a real choice worth a second look before
implementing, **[MINOR]** small/nice-to-have.

## 1. Overall pipeline coherence

The M0→M14 dependency graph is acyclic and the layering is sound: M1 (loss) →
M2 (prior contract + toy) → M3/M4 (harder priors) → M5 (marginal baseline) →
M6/M7/M8 (comparisons) → M9 (real model) → M10 (real benchmarks), with
M11–M14 as cross-cutting infra correctly gated on the milestones that produce
the interfaces they need. No milestone claims a dependency it doesn't need or
silently omits one it does, as far as the written specs go. M8's explicit
carve-out ("doesn't need M6 to run its core comparisons") is correctly
reasoned, not just asserted.

**[DESIGN] The roadmap's own stated sequencing philosophy is already broken
by the roadmap itself.** `docs/ROADMAP.md` says, in its own words: "Don't
skip ahead... the whole point of M1–M8 is... a clear answer on whether M9
needs to exist at all in its planned form." But `M9-proposed-model.md` is
written in full implementation-ready detail (checkpoint loading, cross-attention
module, KL-distillation branch, padding, tensor-shape tests — real scope), and
its own prototype (`src/ppfn/model/unwarp.py`) already lives in the **active**
`src/` tree, not gated behind anything, with its own `__main__` training loop
already run. If M8's finding is "plain attention already solves this," the
time already sunk into `unwarp.py`'s bridge/broadcast/thinking-row machinery
was not gated by that finding. Either the "validate before building" framing
is aspirational rather than actually governing what work happens, or M9 needs
an explicit "prototype freely, but the *milestone's acceptance criteria*
(checkpoint init from M5, full training run, M7 comparison) wait for M8" split
that the roadmap doesn't currently draw. Worth deciding explicitly rather than
leaving the contradiction implicit.

**[MINOR]** M13 is correctly flagged as the one M11–M14 item that can start
immediately (it doesn't depend on model/prior work) — good, and matches how
concretely specified it already is relative to M11/M12/M14's "sketch" status.

## 2. Per-milestone feasibility — repo/coding perspective

**M0.** Consistent with the current archive state; no issues.

**M1.** Verified: `tabpfn.architectures.shared.bar_distribution` really does
define `BarDistribution`/`FullSupportBarDistribution` as claimed, and the
class has more useful surface than the milestone credits — `has_equal_borders(other)`
and `get_probs_for_different_borders(...)` already exist (lines ~54, ~108).
**[MINOR]** M1 doesn't mention `has_equal_borders` even though M7's and M9's
"identical borders" requirement is exactly what it checks — wire it in
directly rather than re-deriving an equality check. There is **no built-in KL
divergence method** — relevant to M9, see §4.

**M2.** The contract is well-specified for the happy path. **[DESIGN]** it
never states whether the A↔B transform must be invertible/monotonic — see §3,
this is a real gap, not a style nit, because `A_inB`/`B_inA` are only
well-defined for transforms that can actually be run backward.

**M3.** Feasible as scoped; correctly treats `harmonics`/`harmonics_fix`
reconciliation as a real decision rather than assuming an answer.

**M4.** Feasible for option 1 (fixed weights + monotonic warp). **Option 2
(two BNN instantiations + weight-space interpolation) has a real mathematical
problem the milestone doesn't flag — see §3, this is the single most
concrete "will silently produce wrong results" finding in the prior stack.**

**M5.** Feasible against the real `PPFNTrainer`/`TabPFNV2p5`. **[BLOCKING, but
easy to fix]** `.claude/rules/checkpoints.md` documents that this repo has
**two incompatible checkpoint schemas** (`PPFNTrainer._save_checkpoint` vs.
`CheckpointCallback`'s snapshot). M5's deliverable #4 ("a saved checkpoint,
loadable per the contract") and acceptance criterion ("checkpoint saves and
reloads to identical eval NLL") never say **which** of the two schemas is
authoritative. Since M9's decoder-initialization and M7's callback both load
"the M5 checkpoint," an unstated choice here becomes two teams (or two future
sessions) guessing differently. Pin this down in M5 itself: state explicitly
that `CheckpointCallback`'s schema is the one M7/M9 load (it's the one
actually wired via `configs/callbacks/`), or say otherwise.

**M6.** The `share_unrelated` ↔ MTPFN's `p` correspondence is **actually
sound**, not just asserted — MTPFN's Algorithm A.1 defines `p` per source
task; with exactly one source task (B), "probability B is drawn
independently" is exactly `share_unrelated`. Good. The reintegration plan
(pull from `archive/`, don't rebuild) is well-founded — the archived
`mtpfn/README.md` is real, substantive prior art, not a claim to take on
faith, and the milestone correctly treats it as ground truth to preserve.

**M7.** Solid. The self-comparison-should-be-~0 acceptance criterion is
exactly the right correctness check for "same context" machinery, and the
provenance-check acceptance criterion is the first real enforcement of an
invariant that's been documented-but-unenforced since `checkpoints.md` was
written — good that M7 is where it finally becomes code.

**M8.** Well-isolated ablation; the "not MTPFN" distinction is drawn
correctly against the real MTPFN mechanism (prepended token + specialized
attention topology vs. this milestone's additive embedding into the existing
unmodified stack).

**M9.** See §4 — this is where the deepest technical review is.
**[BLOCKING]** the milestone's own "first concrete task" (verify `_decode`
exposes contextualized thinking-row embeddings) has a knowable answer that
the milestone treats as open: it doesn't. `TabPFNV2p5._decode`'s own
docstring states `train_start`/`train_end` "delimit the **(non-thinking)**
training rows" — thinking rows (indices `0:T`) are explicitly excluded from
what `_decode` returns, and even the rows it does return are sliced to only
the last feature-group column (`x_BRCD[..., -1]`, the target column), not the
full per-feature-group cell embedding. **`_decode` cannot supply what M9
needs; a new seam into (or around) `TabPFNV2p5.forward`'s block loop is
required.** This isn't a minor detail — it changes deliverable 1 from "verify
an assumption" to "build a documented extension point," which is more work
and should be scoped as such.

**M10–M14.** Feasible as sketches; correctly gate on real interfaces rather
than guessing at them. M13's stress-test-driven methodology (reusing
`archive/slurm/sqlite_attempt/`'s own tooling) is the strongest-grounded
milestone in the whole set — it's the only one whose "done" criterion is an
actual empirical test against a known-hard constraint rather than a
code-level check.

## 3. The prior — math and conceptual soundness

**Overall**: the core idea (A sparse/anchor, B dense/warped-but-related,
`A_inB`/`B_inA` as oracle-transported data for training signal) is coherent
and the M2 contract captures the shape of it correctly. Two real gaps:

**[BLOCKING] Transform invertibility is never required at the contract
level.** M2 says the prior "knows the ground-truth transform for related
pairs" and can therefore produce `A_inB`/`B_inA`. But "transported data" is
only well-defined if the transform can be run in both directions — for a
non-injective warp (e.g. anything that folds or clips the domain), mapping a
B-side point backward into A's domain has no unique answer. M4's own option 1
explicitly requires a **monotonic** warp — correctly, because monotonic ⇒
invertible in 1D — but this requirement is only stated for the BNN prior, not
promoted to the M2 contract itself. The toy prior (M2) says "warp / shift /
scale / otherwise distorted" without the same constraint. Recommendation:
require invertibility (monotonicity in 1D; a documented equivalent in higher
`p`) as part of the M2 contract itself, not something each prior
re-derives — otherwise M3 (harmonics) or a future prior can silently pick a
non-invertible transform and produce `A_inB`/`B_inA` that are well-formed
tensors but not actually meaningful transported data, and nothing in M2's
acceptance criteria would catch it (there's no test proposed that checks
transform invertibility).

**[BLOCKING] M4's weight-space interpolation option is very likely
ill-posed, and the milestone presents it as a symmetric design choice against
output-space interpolation when it isn't one.** Independently-sampled neural
networks (not fine-tuned from a shared initialization, not permutation-aligned)
generically live in unrelated regions of weight space — this is the
well-documented lack of *linear mode connectivity* between independent
solutions (permutation/scaling symmetry of hidden units means "the same
function" has many unrelated weight-space representations). Linearly
interpolating two such weight vectors at `alpha=0.5` does not generally
produce "half of function 1, half of function 2" — it typically produces
something with no principled relationship to either function, and can be
degenerate (e.g. internal cancellation across mismatched hidden-unit
orderings driving intermediate activations toward the origin, or arbitrary
high-frequency artifacts). M4 phrases this as "weight-space or output-space —
decide and document which," implying either is a reasonable choice; it is
not. **Output-space interpolation** (`f_B(x) = (1-alpha) f_1(x) + alpha
f_2(x)`, interpolating the two networks' *outputs*, not their weights) is
trivially well-defined and continuous in `alpha` by construction and should
be the recommended default; weight-space interpolation should be called out
as a known-risky research direction requiring alignment machinery (e.g.
Git Re-Basin-style permutation matching) that's out of scope for "pick a
relatedness knob," not a plain alternative.

**`share_unrelated` vs. MTPFN's `p`**: sound, see §2 — no finding here beyond
what's already noted.

**[DESIGN] M9's decoder distribution-shift risk is real and underweighted.**
M5's decoder is trained on oracle `B_inA` — data with whatever regularity the
prior's exact transport produces. Early in M9's joint training, the
encoder's output will look nothing like that (freshly initialized weights,
no signal yet about what "translated B" should resemble). Feeding
out-of-distribution context into a model that has only ever seen
well-behaved oracle context is a standard exposure-bias/covariate-shift
problem, and it directly threatens the mechanism M9 relies on to train the
encoder at all: if the decoder's response to OOD context is uninformative or
degenerate, the gradient signal flowing back into the encoder through it may
not teach the encoder anything useful, independent of whether the decoder is
frozen or fine-tuned. M9's milestone treats "frozen vs. fine-tuned" as the
open question to resolve empirically; that framing undersells the risk,
since fine-tuning on garbage-early-on inputs can also just degrade the
decoder without producing a useful signal (a cold-start problem, not a
frozen/unfrozen binary). Worth considering explicitly before implementation:
a warm-start/curriculum (blend oracle and encoder-produced `B_inA` early in
training, anneal toward pure encoder output) is a standard mitigation for
exactly this failure mode and isn't mentioned as an option.

## 4. The model — critical architectural review

Grounded directly in `src/ppfn/model/unwarp.py` and the installed
`tabpfn_v2_5.py` — not just the M9 doc's prose.

**[BLOCKING, concrete] The `_decode` extension point M9 proposes checking
does not work — see §2.** `_decode`'s docstring explicitly excludes thinking
rows and only returns the target-column slice. Any inducing-point/cross-attention
design that needs the *contextualized thinking-row cell embeddings* (all
feature groups, not just the target column) must pull them out of the
`self.blocks` loop directly — i.e., either factor a protected seam into
`TabPFNV2p5` (a real, if small, upstream-style change) or reimplement enough
of `forward()`'s orchestration (embed → `add_thinking_rows` → block loop) as
a parallel path that calls the *same* sub-modules
(`feature_group_embedder`, `target_embedder`, `add_thinking_rows`, `blocks`)
without copying the whole 370-line method. This is buildable — `unwarp.py`
already does something like the latter (constructs its own `TabPFNBlock`
list and drives it directly) — but the M9 doc should stop calling this a
"verify an assumption" task and scope it as "build the seam," which is where
the real implementation risk lives.

**[DESIGN] Two-pass decoder structure is implied but not spelled out, and
has a real compute-cost/consistency question attached.** The architecture
needs (a) a decoder pass over `A_train` alone to produce the
thinking-row summary the encoder cross-attends to, and (b) a second decoder
pass over `[A_train, encoder_output(B)]` with `A_test` as query for the final
prediction. These cannot trivially share KV-cache state:
`AlongColumnAttention`'s `cached_kv`/`return_kv` path (verified in the
installed source) is built for "test rows attend to a fixed train KV cache,"
not "extend the train block with new rows and recompute" — inserting the
encoder's translated-B rows between passes (a) and (b) means pass (b) is a
fresh forward pass over the whole `[A_train, B_inA]` context, not an
extension of pass (a)'s cache. Not a correctness bug, but 2x decoder compute
per training step that the milestone doesn't budget for, and worth stating
explicitly rather than leaving as an implicit consequence of the design.

**[MINOR, reassurance rather than risk]** The KV-cache/multi-query-attention
optimization for test rows (`AlongColumnAttention`, all test-row query heads
sharing one cached KV head) is orthogonal to what M9 needs and not a landmine
— it only activates on the `cached_kv`/`return_kv` path, which the new
cross-domain cross-attention module doesn't need to touch as long as it's
built as a genuinely separate `Attention` subclass (as `.claude/rules/model-prototyping.md`
already directs) rather than trying to route cross-domain attention through
`AlongColumnAttention` itself.

**KL-distillation math**: buildable, but two things the milestone should add
to its acceptance criteria, not just its prose:
1. `FullSupportBarDistribution` has **no built-in KL method** — verified by
   reading the class's full method list. The KL term must be hand-built from
   `compute_scaled_log_probs` (which does exist) for both passes, reduced
   manually. `has_equal_borders` (also real, see §2) should be an explicit
   runtime assertion guarding this computation, not just documentation.
2. **No loss-weighting scheme between the primary NLL and the KL term is
   specified anywhere.** A distillation-style auxiliary loss that isn't
   explicitly weighted (and checked for not dominating/collapsing the
   primary gradient) is a common, quiet failure mode in exactly this kind of
   joint-objective setup — worth adding as an acceptance criterion (e.g. "the
   primary NLL still improves with the KL term enabled, not just decoder
   output-matching without underlying correctness").

**Reading `unwarp.py`'s current `forward()` against the plan**: the
bridge/broadcast attention's flattening of `(thinking_rows, features)` into
one sequence (confirmed at `Q_bridge = T_B.reshape(B_batch, seq_len_bridge, D)`
etc.) is exactly the issue M9 already identifies — no new finding there
beyond confirming the milestone's own diagnosis is accurate. One additional
note: `stream_A_blocks`/`stream_B_blocks` are currently invoked with
`single_eval_pos=None` (full bidirectional self-attention, no train/test
split) — correct for the current "contextualize-only" phase 1, since
`cells_A`/`cells_B` are built from `batch['train']` only. But once
`stream_A_blocks` becomes "the decoder, loaded from an M5 checkpoint," it
must additionally support the masked `single_eval_pos != None` calling
convention (M5's decoder was trained with a real train/test split) — this is
a genuine change to how the block stack gets invoked, not just a
weight-loading swap, and should be named as its own deliverable.

## 5. Prioritized findings

1. **[BLOCKING]** `_decode` does not expose contextualized thinking-row
   embeddings (confirmed via its own docstring: thinking rows explicitly
   excluded, and only the target-column slice is returned). M9's plan needs
   a real extension point into `TabPFNV2p5`'s block loop, not a
   verification step. Resolve before starting M9 implementation.
2. **[BLOCKING]** M4's weight-space-interpolation BNN relatedness option is
   very likely to produce degenerate, non-smooth relatedness due to the lack
   of linear mode connectivity between independently-sampled networks.
   Default to output-space interpolation; treat weight-space as a flagged
   research risk, not a plain alternative.
3. **[BLOCKING]** M2's prior contract doesn't require the A↔B transform to
   be invertible, but `A_inB`/`B_inA` are only well-defined for invertible
   transforms. Promote M4's monotonicity requirement to the M2 contract
   itself so every prior (not just BNN) is held to it.
4. **[BLOCKING, easy fix]** M5 doesn't say which of the two documented,
   incompatible checkpoint schemas (`checkpoints.md`) it produces, and M7/M9
   both depend on loading "the M5 checkpoint." Pin this down explicitly in
   M5's deliverables.
5. **[DESIGN]** The decoder-was-never-trained-on-encoder-output
   distribution-shift risk in M9 is underweighted relative to how central it
   is to whether joint training converges at all. Consider a warm-start/
   curriculum mitigation explicitly rather than treating "frozen vs.
   fine-tuned" as the only open question.
6. **[DESIGN]** The roadmap's stated "validate before building M9" philosophy
   is already contradicted by M9's own scope and the live prototype's
   presence in the active tree. Decide explicitly what's gated on M8 vs. what
   isn't, rather than leaving the tension implicit.
7. **[DESIGN]** M9's two-pass decoder structure has a real, unbudgeted 2x
   compute cost per training step (KV-caching across the two passes doesn't
   trivially apply). Worth stating as a known cost, not a silent consequence.
8. **[MINOR]** M1 should use `BarDistribution.has_equal_borders` (already
   exists) directly for M7/M9's "identical borders" requirement instead of
   re-deriving an equality check.
9. **[MINOR]** The KL-distillation term needs an explicit loss-weighting
   scheme and an acceptance criterion that checks it isn't dominating the
   primary NLL loss.
10. **[MINOR, active-code finding, not milestone-doc]** `BNNPrior.ensure_ecdf_loaded`
    caches its ECDF sample file keyed by nothing — if `num_inputs`/`num_outputs`
    change between runs (plausible once M4 varies `p`), a stale cache from a
    different dimensionality gets silently reused. Worth a guard when M4
    touches this file.
