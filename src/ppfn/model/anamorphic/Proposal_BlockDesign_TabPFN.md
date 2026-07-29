# In-Context Coordinate Alignment for Prior-Data Fitted Networks

## Project Proposal

---

## 1. Problem Statement

Standard tabular PFNs (TabPFN and derivatives) assume that a feature's coordinate meaning is stable: column 3 means the
same physical quantity across every row of a dataset, and across the train/test split. Multi-task and multi-source
settings routinely violate this. Two data sources, A and B, may report what is nominally "the same" underlying quantity
under different measurement conventions — different scaling, different units, a monotonic re-parameterization (e.g.
`log(lr)` vs `lr`), partial dropout of dimensions, or lossy/non-invertible transforms. The raw coordinates of B are not
directly comparable to A's, even when both are drawn from causally related processes.

We want a PFN that, given only the raw observations of A and B (never told the transform, never given a task-identity
label), can:

1. **Detect** whether B's data is relevant to A's task at all (robustness to negative transfer), and
2. **Infer**, in-context, the implicit correspondence between A's and B's coordinate systems well enough to use B's data
   to improve inference on A — particularly when `N_A ≪ N_B`, i.e. the domain we actually care about is data-starved and
   the domain we want to borrow strength from is data-rich.

The shared target variable `y` is the only variable that is *not* domain-specific — it is present, in principle, for
both A and B in the same generative sense — and is therefore the anchor ("Rosetta stone") that must carry the alignment
signal.

[//]: # (There is no task-ID column in our formulation: task identity has to emerge implicitly from data geometry, both)
[//]: # (because it isn't generally observable in the real setting we're targeting and because giving the model an explicit ID)
[//]: # (risks it learning a shortcut &#40;"if task=B, ignore A"&#41; rather than the general alignment skill we want.)

We control the generative process (SCM) in simulation, which gives us privileged access to ground truth (true `B_in_A`,
the coordinates B's samples would have if measured in A's frame) for training and diagnosis — but any deployed system
must work without that privilege.

---

## 2. What We Already Know

### 2.1 A zero-shot diagnostic is possible without any training

Before committing to either architecture below, we built a **zero-shot, teacher-free evidence-gap diagnostic** using a
pretrained TabPFN checkpoint's *native* interface — no fine-tuning, no custom heads, no hidden-state hacking. The key
design correction that made this both simple and honest: `Y` is kept in its own stream (as TabPFN expects), and
cross-domain missingness is expressed as literal `NaN` in `X`, which TabPFN v2 is already trained to handle as a normal
missing-value pattern. This let us directly compare, for a batch of synthetic (shift+scale) A/B pairs, the model's
predictive NLL on held-out `A_test` under four conditions:

- **Baseline**: `A_train` alone (no B in context); which is the reference; unconditional model
- **Block-diagonal, distorted B** (the actual use case: B needs alignment)
- **Block-diagonal, aligned B** (B already in A's coordinate frame — isolates the cost of the block-diagonal
  *representation* itself, independent of alignment difficulty)
- **Oracle / direct concatenation of aligned B** (privileged-information lower bound — no block structure, no NaNs,
  "cheat" case)

### 2.2 Empirical results (N=100 batch items, shift+scale transform)

| Condition                   | Mean NLL gap vs. baseline | SEM  | n(inf) |
|-----------------------------|---------------------------|------|--------|
| Oracle / concat (aligned B) | −2.17                     | 0.17 | 2/100  |
| Block design, aligned B     | −0.21                     | 0.03 | 0/100  |
| Block design, distorted B   | −0.15                     | 0.03 | 0/100  |

**Paired alignment cost** (distorted − aligned, computed per-item before averaging to control for shared A-draw
variance): **+0.063 ± 0.019 nats**, >3 SEM from zero — a small but statistically real cost.

Notice, that these Numbers are in fact already negative on average, but depending on the scale and shift magnitudes, the
model might just have gotten easy to guess answers -- precisely because the data wasn't that far off from A's data. This
is a shortcut, that may bias the results and introduce lucky shot smaller confidence bounds. significant change on this
prior usually is in the range of 1 nat.

Two findings stand out:

1. **The dominant cost is the block-diagonal presentation itself, not the alignment task.** The gap between the oracle
   and the block-format-with-trivial-alignment condition (≈2 nats) is roughly 30× larger than the additional cost of
   needing to infer a real shift/scale on top of that format (≈0.06 nats). This suggests the pretrained checkpoint's
   bottleneck is primarily about **recognizing this specific missingness pattern** — large, contiguous,
   row-group-correlated blocks of missing columns — rather than a fundamental incapacity to reason about coordinate
   correspondence. TabPFN's native NaN-handling was almost certainly trained on scattered, cell-level missingness, not
   this structured, group-correlated pattern.
2. **The block-diagonal representation is numerically more robust than naive concatenation.** The two `inf` NLLs
   occurred exclusively under the oracle/concat condition — never under either block-diagonal condition. The most likely
   mechanism: TabPFN's output head uses an adaptive discretization fit to the empirical range of in-context `y` values;
   pooling A's and B's `y` together (as concatenation does) can shift those bin borders in a way that assigns near-zero
   density to some `A_test` points, whereas keeping A and B in separate rows (as the block design does) appears to
   protect against this. This is a genuinely useful finding independent of the alignment question: **the block-diagonal
   interface is not just "not obviously worse" than raw concatenation, it may be structurally safer as a way to compose
   contexts from heterogeneous sources.**

### 2.3 Implication for prioritization

Because most of the recoverable NLL is locked up in "format exposure" rather than "alignment reasoning," a **cheap,
targeted fine-tuning phase that exposes the model to block-diagonal-shaped contexts (even under trivial, identity
alignment)** should recover most of the available headroom before any alignment-specific machinery is needed. This
directly shapes the phased plan in Section 4.

https://arxiv.org/pdf/2410.01565? Also points towards the fact, that we are passing it data that is unlikely under the
prior -- because nan's usually originated from an iid distribution, and in turn don't hold this structural meaning. It
has zero prior probability mass assigned to massive, contiguous, row-group-correlated missingness. Fine-tuning is
basically a way to augment the prior to include this new missingness pattern

### 2.4 Relevant related work: MTPFN (Li, Daulton, Müller, Wilson, Bakshy)

MTPFN tackles a related but distinct problem — robust multi-task transfer for Bayesian optimization — with a
hierarchical attention design: intra-task blocks do full self-attention within each task's own points and summarize the
task into a single learnable `[TASK]` token; inter-task blocks then attend only across these `T` pooled tokens (cost
`O(TD² + T²)` instead of `O(T²D²)`). Robustness to negative transfer is taught explicitly via a data-generation
hyperparameter `p` (probability an auxiliary task is unrelated).

Two takeaways for us:

- Their pooling-then-compare design is well suited to *whole-task* relatedness (e.g., "do these tasks share a
  lengthscale") but is a poor fit for *coordinate-level* correspondence — the kind of column-by-column comparison our
  problem needs happens before pooling would need to occur, and squeezing everything through one task-summary vector
  first likely discards exactly the signal we need. If we ever need MTPFN-style scaling, the natural fix is pooling into
  several feature-group tokens rather than one per task.
- Their `p`-parameter finding is directly transferable as a training-recipe lesson: the *capacity* to ignore irrelevant
  sources (via low column-attention coupling, in our architecture) has to be explicitly trained by exposing the model to
  genuinely unrelated B-blocks during training, or the capability likely won't emerge on its own.

---

## 3. Candidate Architectures

Both approaches target the same underlying problem — training a model to solve the coordinate-alignment sub-task — but
inject supervision at different points in the pipeline. We are not choosing one over the other; both are worth building
out, and the diagnostic above is designed to keep them honest against each other and against the cheap zero-shot
baseline.

### 3.1 Approach A — Direct NLL, end-to-end on the block design

Train (or fine-tune) the full `[block-diagonal design] → y_test` pipeline against a single scalar log-likelihood,
exactly matching deployment. Row/column attention is left free to discover whatever alignment machinery helps minimize
prediction error, with no architectural bottleneck forcing an explicit "alignment representation" to exist.

**Strengths:** optimizes exactly what we evaluate; no architectural surgery; consistent with how TabPFN itself was
trained; the zero-shot diagnostic above already operates in this paradigm and gives us a free, trainable-on-top
baseline.

**Known risks:**

- With `N_A ≪ N_B`, the loss is computed on relatively few A-test points while the forward computation is dominated by
  B; the alignment-relevant gradient is diffuse and indirect. Notably, PFNs are known to lack localization, meaning the
  bigger B is relative to A, the more will we potentially wash out A's signal.
- Multiple alignments can produce similarly low NLL when B's transform is lossy/non-invertible (non-identifiability),
  and nothing in the objective discriminates a "correct" alignment from an accidentally-correlated one.
- Fine-tuning on a narrow synthetic block-design distribution risks quietly degrading the pretrained model's broader
  tabular capability.

**Cheap variant worth trying first:** rather than fine-tuning the whole pipeline on the downstream NLL, directly
supervise an alignment sub-module with MSE/NLL against the *known* ground-truth `B_in_A` (available because we control
the SCM). This is a plain supervised regression/density problem with no teacher and no predictor in the loop, so the
gradient is maximally direct — it sidesteps the "diffuse gradient through the full pipeline" concern entirely, at the
cost of not (by itself) teaching the model when *not* to align (see Section 2.4's point on needing negative examples).

### 3.2 Approach B — Privileged-information distillation (JEPA / LUPI style)

A frozen, correct **teacher** sees the privileged input `[A, true B_in_A, A_test]` and produces a predictive
distribution over `A_test`'s target — by construction, this teacher has *no* alignment uncertainty, only the SCM's
intrinsic noise and finite-sample uncertainty. A **student**, given only the raw block design (`A`, un-aligned `B`),
must produce a representation `hat{B}_in_A` that, spliced into an otherwise-identical predictor, reproduces the
teacher's predictive distribution (matched via forward KL, `KL(teacher ‖ student)`, chosen deliberately over the reverse
direction).

**Why forward KL specifically:** it is mass-covering rather than mode-seeking. It penalizes the student heavily for
failing to cover the teacher's mode, but is largely indifferent to the student assigning *extra*, honest spread beyond
the teacher's — which matters because the student's epistemically correct posterior should generally be *wider* than the
teacher's (it must marginalize over plausible alignments the teacher never had to consider). Reverse KL would instead
actively punish that honest extra uncertainty, training the student toward false confidence.

**Known risks:**

- Interface mismatch: a frozen predictor was never trained to consume a continuous, uncertainty-bearing embedding in the
  B-slot.
- Representation collapse / shortcut solutions: matching output distributions through a (possibly later-unfrozen)
  predictor is a many-to-one mapping; the student can find something that fools the KL objective without encoding a
  genuine alignment.
- Once the predictor is unfrozen for joint fine-tuning, a *different* risk appears — not classical EMA-style
  bootstrapping collapse (the teacher here is externally fixed and never moves, so that specific failure mode doesn't
  apply), but a co-adaptation shortcut where student and predictor jointly settle on an idiosyncratic code that isn't a
  real alignment.

**Refinement adopted:** rather than splicing an internal representation into a frozen predictor, feed the student's
**final output value** through a learnable deterministic token, and unfreeze the predictor to fine-tune jointly on real
NLL against `y`. This produces a hybrid — call it **bottlenecked NLL** — that keeps an explicit, probeable `hat{B}_in_A`
channel (unlike vanilla Approach A) while avoiding the brittleness of matching a density through a predictor that was
never built to consume one. A sensible curriculum: pretrain the student's production skill via the teacher/KL objective
while the predictor is frozen, then unfreeze and switch to plain NLL for joint fine-tuning.

### 3.3 How the two approaches compare on the axis we can already measure

The zero-shot diagnostic (Section 2) currently only tells us what a pretrained-but-unmodified checkpoint can do; it does
not yet distinguish Approach A from Approach B, since no fine-tuning has happened. What it does tell us is *where* to
spend the fine-tuning budget first: given that ~97% of the measured gap is format-exposure rather than
alignment-reasoning, the cheap NLL-only alignment-module pretraining variant of Approach A is the natural first
fine-tuning experiment, before either full end-to-end NLL or the fuller JEPA pipeline.


---

## 4. Proposed Plan

**Phase 0 — Diagnostic (done).** Zero-shot evidence-gap measurement across (aligned/distorted B) ×
(block-diagonal/concat), establishing the pretrained checkpoint's baseline capability and decomposing the gap into
format-cost vs. alignment-cost.

**Phase 1 — Cheap alignment-module NLL pretraining.** Train the alignment sub-module alone against known ground-truth
`B_in_A` (no teacher, no predictor). Cheapest possible signal; establishes whether the small residual "alignment cost"
from Phase 0 shrinks with direct, targeted supervision.

**Phase 2a — Block-format exposure fine-tuning.** Since Phase 0 attributes most of the gap to the model not recognizing
the block-diagonal missingness pattern, fine-tune (lightly) on block-diagonal-shaped contexts under *trivial* (identity)
alignment only. This isolates and tests the "format exposure" hypothesis directly before spending effort on
alignment-specific machinery.

**Phase 2b — Full end-to-end NLL fine-tuning (Approach A).** Fine-tune the whole pipeline on the downstream NLL
objective, across a range of `N_A/N_B` ratios and transform families, as the direct-optimization baseline.

**Phase 2c — Distillation refinement (Approach B).** Build the teacher/student/predictor pipeline, forward-KL
pretraining phase, then the bottlenecked-NLL joint fine-tuning phase, as an enhancement layered on top of whichever of
1/2a/2b provides the best warm start.

**Phase 3 — Evaluation across both approaches**, using:

- **Alignment gap vs. `N_A/N_B`** sweep (the original motivating hypothesis: does JEPA's per-example privileged
  supervision dilute less than NLL's diffuse gradient as `N_A` shrinks?).
- **Transform-family sweep**, from identity → smooth monotonic (log/affine, already tested) → non-monotonic/lossy, since
  the current 30:1 format:alignment ratio was measured on the easiest possible transform and may not generalize.
- **Calibration/entropy diagnostics** comparing student vs. teacher width across `N_A`, to check the student isn't being
  trained toward artificial overconfidence (the "penalizing accurately-wrong" concern raised during design).
- **Representation probes** on `hat{B}_in_A` (does it linearly decode the true transform's parameters?) and **canary
  tests** (shuffled/randomized B) to catch representation collapse.
- **Deployment-time evidence proxy**, extending the zero-shot NLL-gap diagnostic itself: since no teacher is available
  at deployment, use the held-out-marginal NLL gap (A+B context vs. A-alone context) as the teacher-free "bending
  energy" signal for down-stream Bayesian model averaging across candidate auxiliary sources.
- **Numerical-robustness check**, following up on the oracle/concat `inf` finding: confirm (or refute) the
  bin-border-miscalibration mechanism directly, and treat "robustness to context composition" as its own evaluation axis
  alongside raw NLL.

---

## 5. Open Risks Carried Forward

- The 30:1 format-cost-to-alignment-cost ratio is measured on the easiest transform in our test suite (shift + scale)
  and a single sample size regime; it should not be assumed to generalize to harder or non-identifiable transforms
  without direct testing.
- Forward-KL distillation mitigates but does not fully eliminate the risk of the student being trained toward
  overconfidence relative to its true epistemic state under alignment ambiguity.
- Any joint fine-tuning of student + predictor reopens a co-adaptation/shortcut risk distinct from (and requiring
  different mitigations than) classical self-distillation collapse.
- We have not yet trained the model to actively decline transfer when B is genuinely unrelated (MTPFN's `p`-parameter
  lesson) — the current experiments all assume B is at least a candidate-relevant source.

## 6. MISC Ideas:

### 6.1 Using Privileged information as dedicated location loss

consider tracking X_B_in_A, Y_B_in_A as additional quers / loss. This way in a fine-tuning scenario, we can guide the
model with privileged information where the information matters most!
it's just Approach A's end-to-end NLL with extra, well-localized query rows added during training, using privileged
X-locations (available only in simulation) purely to sharpen where the loss is evaluated. It's the cheapest possible way
to inject privileged information without touching the architecture at all, and it composes cleanly with everything else
here.

### 6.2 Feature-wise attention for alignment

TabPFN's attention is already column-permutation-invariant by design — it was built on the premise that a column's
identity isn't intrinsically meaningful and has to be inferred from how it behaves in context. That's structurally close
to what our problem needs

[//]: # (### 6.3 In-context Tokenization for Alignment)

[//]: # (The MTPFN paper suggests hierarchical attention pooling via [`TASK`] tokens, and inter task attention solely on these tokens. )

[//]: # (Originally, this is taken from long-context summarization in NLP, where basically all sentences of a long document)

[//]: # (are &#40;obviously written in the same language&#41; processed with this task token, before hierarchical attention is applied on )

[//]: # (the pooled sentence representation. In our case, it looks more like having one Book, where sentences or chapters are in different)

[//]: # (languages, and we only have a patched version of the book needing to decipher first the individual meaning and where it belongs)

[//]: # (before we can think about the overall meaning of the book. )

### 6.4 Feature (sub-) spaces:

If we organize everything in a block design; even if the spaces are measured with the exact same features, this closely
resembles the construction of a mixed effect model. The model can in theory at least discover that they describe the
same underlying process, meaning we just can add the colums respectively and recover the appended solution We can also
discover subspaces that align or introduce / explain new variability. We could of course introduce a shared column
section, where both tasks overlap, which will probably be a strong anchor for exchange of infomration between the
dataset.

### 6.5 Can we do in-context DTW for warping-alignment?

Dynamic Time Warping (DTW) is a well-known technique for aligning sequences that may vary in speed or timing. In our
context, we could explore whether an amortized version of DTW

### 6.6 Y is our Rosetta Stone in Mixture Effects models.

Can we estimate the latents that are shared between two tasks?

**In Bayesian terms,** $y$ provides the likelihood update that collapses the prior over the feature spaces. Here is how
this physically manifests in TabPFN's feature-wise attention, and how it relates to extracting the shared latent
structure we discussed earlier:

The Unaligned Prior: When the model looks at $X_A$ and $X_B$ in isolation, the feature-wise attention (which is
permutation invariant) computes similarities based solely on the marginal distributions $P (X_A)$
and $P (X_B)$. If $B$ is a warped version of $A$, these marginals might look somewhat similar, but the uncertainty is
massive.

* The Likelihood Update via $y$: Because $y$ is concatenated into the context, the attention mechanism doesn't just
  compute $P (X)$; it computes the joint $P (X, y)$.

* In-Context Alignment: The Queries ($Q$) and Keys ($K$) in the attention layers will align columns not by their raw
  values, but by their gradients relative to $y$. If Feature 3 in Block A and Feature 7 in Block B both explain the
  variance in $y$ identically, their attention scores will spike.This is exactly how you can extract the shared latent
  structure (your point 2.1 from our earlier chat). By extracting the final $QK^T$ attention matrix from a PFN trained
  on this block-diagonal SCM, you will literally be looking at a posterior covariance matrix of feature alignments.

**The Mixed-Effects Connection:**
Your realization in 6.4 brings this full circle. A mixed-effects model partitions variance into fixed effects (shared)
and random effects (block-specific). By structuring your PFN inputs as a block-diagonal matrix with a shared target $y$,
you are forcing the Transformer to act as a non-linear, non-parametric mixed-effects model.

* The attention across the aligned, equivalent columns acts as the fixed effect (pooling data across $A$ and $B$ to
  learn the global rule).
* The attention heads that isolate columns unique to $B$ (or unalignable noise) act as the random effects (capturing
  block-specific variance).

