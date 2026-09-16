# LUPI decoder-only registration: is bare attention the right mechanism, and a richer use of privileged supervision

`commit: pending`

## What prompted this

After scaling `lupi_bounds`/`lupi_id_token_baseline` for parallel training and
adding a fade-in/temperature curriculum to `LUPIIDTokenLoss` (student NLL
weight 0.15→1.0, CE temperature 2.0→1.0 over the first 20% of training — see
`src/ppfn/loss/lupi_id_token_loss.py`), the user asked for a step back: is
bare multi-head attention + an additive domain tag structurally the right way
to *learn* in-context registration (`T` and `h`) at all, or is there a
concept that would make better use of the privileged information LUPI already
gives us at training time? This covers that discussion, a previously-tried-
and-abandoned idea of the user's own (attention-logit-level LUPI supervision,
which blew up), and a survey of attention/in-context-learning results raised
in conversation — revised after a first round of the user's own pushback on
the initial draft.

Nothing here has been implemented. `lupi_bounds`/`lupi_id_token_baseline` kept
running untouched throughout. This entry is deliberately kept as a design
document/idea log — the "research landscape" section is inspiration for
architectural thinking, not a backlog to work through in order.

---

## 0. Notation (extends CLAUDE.md's own table, doesn't replace it)

CLAUDE.md already fixes `z` (latent), `y = f(z) + ε`, `T` (A→B), `ρ`. LUPI adds
one thing on top: `h`, a value distortion applied to A only. Everything below
maps 1:1 to code identifiers so "`enc_x_inA`" never has to be parsed cold again.

| Symbol | Meaning | Code |
|---|---|---|
| `z ~ Unif([0,1]^d)` | shared latent, the one thing both clouds secretly agree on | `zlat_*` in `sample_pair` |
| `Φ_0` | A's own coordinate warp | `to_a` (`internals.to_a`) |
| `Φ_ρ` | B's coordinate warp, a `(1-ρ, ρ)` mix of `Φ_0`'s velocity field and an independent one | `to_b` |
| `f` | shared latent function (the thing both clouds are secretly observing) | `internals.f` |
| `h` | A-only monotone value distortion (asinh-family) | `internals.h` |
| `x_i^A = Φ_0(z_i^A)`, `y_i^A = h(f(z_i^A)) + ε_i^A` | A's context, `z_i^A` acquisition-biased (aggressiveness `β`) | `dec_ctx_x`, `dec_ctx_z` |
| `x_j^B = Φ_ρ(z_j^B)`, `y_j^B = f(z_j^B) + ε_j^B` | B, uniform/unbiased | `enc_x`, `enc_z` |
| `x_j^{B→A} := Φ_0(z_j^B)` | B's position, *transported* into A's frame (forward eval of `Φ_0`, never an inversion) | `enc_x_inA` |
| `y_j^{B→A} := h(y_j^B)` | B's value, *recalibrated* into A's scale | `enc_z_inA` |
| `x_k^*, y_k^*` | A's query/test points, same generative family as A's context | `dec_qry_x`, `dec_qry_z` |

`(x_j^{B→A}, y_j^{B→A})` together are "B_inA" in conversation — the fully
registered, privileged view the teacher pathway is handed and the oracle bound
is trained on.

---

## 1. Is bare attention + additive tag well-matched to this problem?

Short answer: it's a reasonable universal-approximator baseline (which is
exactly why it's worth having as an experiment), but its native inductive bias
is mismatched to both pieces of what it has to recover, and shouldn't be
expected to be the best *achievable* mechanism even if it clears a
respectable bar.

- **`T` is smooth and volume-bounded** (RK4 flow of a velocity field,
  rejection-sampled so the local log-det-Jacobian band stays under `log 9`).
  Recovering a smooth coordinate warp is the kind of thing that benefits from
  *iterative refinement* — propose a correspondence, check consistency,
  adjust — which is precisely why the main registration architecture has a
  dedicated, depth-wise-refined transport head (`w_ℓ ∝ ℓ`) instead of asking
  generic residual depth to discover it as a side effect of minimizing a
  final loss. `IDTokenPFN` has none of that.
- **`T` is also *coupled* across the `d` coordinates** — a generic velocity
  field doesn't act independently per axis, so a good transport estimate for
  axis 1 may genuinely depend on where axis 0 landed. (Matters for §2a.)
- **`h` is monotone**, which attention has no native way to exploit. A plain
  value channel has to re-derive "preserve rank order" from data on every
  draw; an architecture that knows monotonicity a priori doesn't spend
  capacity re-learning a constraint it could have for free.

So: attention *can* probably learn something registration-shaped given enough
scale/exposure — that's a legitimate thing to measure, and it's exactly what's
currently training. But the more interesting move is giving privileged
information a *richer* job than softening the final loss.

---

## 2. Proposed extension: supervise the intermediate registration directly

Right now LUPI's privileged signal (`enc_x_inA`, `enc_z_inA`) only shapes the
**final output distribution**, via CE-distillation. Ground truth for the
intermediate quantities exists too, and nothing stops us from supervising
them directly, on the **student pathway only** (the teacher's B tokens
already *are* `enc_x_inA`/`enc_z_inA`, so an auxiliary task there is a
trivial identity, not a useful signal).

**The pushback that shaped this, and why it matters:** encouraging the
student to match "what the teacher would have at that point" runs into a real
problem if done via a plain point-matching penalty — the student, seeing only
raw B, often *cannot* know `T` and `h` exactly (early in a draw especially,
with few A points, more than one `(T, h)` could be consistent with what's
been observed). A point loss forces one committed guess and punishes it
quadratically for being "wrong" even when hedging would be correct — related
to, but distinct from, why the earlier attention-logit experiment blew up
(§3). Both auxiliary heads below are distributional, not point-estimate,
specifically to answer this.

Worth being precise about *why* a distributional loss fixes this and a point
loss doesn't, since it's easy to conflate with the (different, also real)
failure mode in §3: **NLL against a realized true value is a proper scoring
rule** — exactly how the project's own main predictive loss already works
everywhere, and it correctly incentivizes calibrated uncertainty in
expectation over many draws, even though any *single* draw's NLL looks
"harsh" if the truth was a low-probability tail. An MSE point loss doesn't
have that property: its only correct target is the conditional mean, with no
way to express "I'm unsure." §3's failure mode is a different, more severe
problem (raw unnormalized logits with no shared reference scale, gameable by
uniformly inflating magnitudes) — fixing §3 needs normalization; fixing this
concern needs a distribution instead of a point.

### 2a. Transport auxiliary head (recovers `x_j^{B→A}`) — reuse, don't reinvent

`ppfn.model.registration.heads.TransportHead` already exists and already does
almost exactly the right thing — worth documenting precisely, since it's more
sophisticated than "a regression head" and directly answers the coupling
concern from §1:

**How it actually works:** it's an *autoregressive-over-axes* bar
distribution, not `d_max` independent per-axis heads. For axis `k`, its own
small MLP (`axis_heads[k]`, not weight-shared across axes) takes
`[h_j^(L), known]`, where `known` is a `d_max`-length vector that's zero
everywhere except axes `< k`, which hold the *true* values of those earlier
axes (teacher-forced during training; the running bar-distribution *mean* of
its own prediction at inference, since the truth isn't available then). Each
axis outputs `n_bins` logits scored by a shared bounded `BarDistribution`.
Concretely, for `d_max=5`: axis 0's distribution is predicted from `h_j^(L)`
alone; axis 1's distribution is predicted from `h_j^(L)` *and axis 0's true
value*; axis 2 sees axes 0 and 1; and so on. That's a proper autoregressive
factorization of the full joint distribution over the `d_max`-dimensional
target — it can represent correlated, multimodal uncertainty across axes *by
construction*, which is exactly what a coupled, non-separable warp needs and
what `d_max` independent marginal heads couldn't give you.

```
h_j^(L)  := B-token j's hidden state, last pooled layer, STUDENT pathway
logits_T,j = TransportHead(h_j^(L), teacher_targets=x_j^{B→A})   # existing class, unmodified, called with teacher forcing at train time
L_T = (1/|B|) Σ_j TransportHead.nll(logits_T,j, x_j^{B→A}, dim_mask=d_real)
```
`dim_mask=d_real` already masks out the padded axes beyond the draw's real
dimensionality — no extra plumbing needed, the same mask convention used
elsewhere in this codebase (`_pad_rescale_coords`, etc.).

This already gives calibrated, potentially multimodal uncertainty over the
transported position — no redesign needed here, just this write-up.

### 2b. Value-recalibration head (recovers `y_j^{B→A} = h(y_j^B)`) — revised

The first draft of this used a monotone-spline point estimate scored by MSE —
exactly the point-loss problem flagged above, and worth being honest that a
full Neural-Spline-Flow-style rational-quadratic spline (Durkan et al. 2019)
is also more machinery than this needs: NSF's RQS is built to be an
*invertible* bijection with a tractable Jacobian, because a normalizing flow
needs exact densities under change-of-variables. Nothing here needs to invert
this map or compute an exact density that way — it's a monotone *regression*
target, not a flow layer. Keeping RQS's *forward* construction (bounded
active region with linear/identity tails outside, so it's well-behaved on
`y_j^B`'s unbounded range for free; monotonicity guaranteed architecturally
via non-negative bin widths/heights/derivatives) while dropping the
log-det-Jacobian bookkeeping entirely gets the useful part without the
unneeded part:

```
knots_j = Hypernet(h_j^(L))                    # RQS bin widths/heights/derivatives; positive by construction -> mu_j(.) monotone increasing
mu_j    = RQSpline_forward(y_j^B; knots_j)      # deterministic MEAN of the predicted recalibrated value -- forward-only, no inverse, no Jacobian
log_sigma_j = SmallHead(h_j^(L))                # scalar spread, unconstrained sign
L_h = (1/|B|) Σ_j NLL( Normal(mu_j, exp(log_sigma_j)), y_j^{B→A} )
```

Starting point: a location-scale Gaussian NLL around the monotone mean —
fewest extra parameters, and it directly fixes the point-loss problem (the
model can now widen `σ_j` instead of being forced into a single overconfident
guess when the draw genuinely doesn't disambiguate `h` yet). If residual
uncertainty around `mu_j` turns out to be visibly skewed or heavy-tailed in
practice, the fallback is swapping the Gaussian for a light
`FullSupportBarDistribution` head conditioned on `mu_j` — more expressive,
more parameters, only worth it if the Gaussian demonstrably isn't enough.

### 2c. Loss integration

```
L_total = student_weight(t)·NLL(student) + NLL(teacher)
          + λ_ce · CE(teacher.detach()/T(t), student/T(t))
          + λ_T · L_T + λ_h · L_h
```

Start with `λ_T`, `λ_h` as small fixed constants (e.g. 0.1–0.3), not their own
curriculum — the first experiment should isolate *whether this helps at all*
before tuning a schedule on top of a schedule. Attach both heads after the
**last** pooled layer only for the first cut, matching the project's own
incremental build-order philosophy.

### 2d. Where this plugs in

- `IDTokenPFN.forward` gains a `return_hidden: bool = False` flag (mirrors
  `PFN.forward`'s own existing convention) returning pooled B-token hidden
  states alongside `predictive_logits`.
- A new wrapper, sibling to `LUPIIDTokenPFN` (e.g. `LUPIIDTokenAuxPFN`),
  reads the student pathway's B-hidden-states and applies `TransportHead`
  (reused as-is) + the revised monotone-mean-plus-spread `h` head.
- A new loss, `LUPIIDTokenAuxLoss`, wraps `LUPIIDTokenLoss` and adds `L_T`,
  `L_h`.
- Nothing about `IDTokenTrainer` needs to change — same calling convention.

---

## 3. The rejected idea: attention-logit-level LUPI supervision, laid out explicitly

The user's own earlier experiment: force raw attention scores to be high
between a query and B-tokens known (via privileged info) to be in the true
registered vicinity, even when nominally "in the other domain" — promising
early, then logit drift and blowup.

**Diagnosis, precisely:** pre-softmax attention logits are unbounded, and a
loss pushing a target logit "up" with no compensating normalization hands the
model a trivial cheat — uniformly scale `‖q‖·‖k‖` everywhere. That raises the
target logit's raw value without improving anything *discriminative*, because
the loss never saw competing candidates on a shared scale. This is a
genuinely different, more basic problem than §2's "point loss can't express
uncertainty" — no amount of "account for uncertainty" fixes an unbounded,
unnormalized target; it needs normalization, full stop.

**The concrete fix — a dedicated, normalized coupling head:**

Don't touch the internal Q/K logits the real `PFNBlock`s use for
representation-building at all. Add a **separate** small scoring function,
its own bilinear or small-MLP head, producing a full score matrix between
query points and B-tokens:

```
score_{ik} = CouplingHead(h_i^{query,(L)}, h_k^{B,(L)})            # separate head, own weights -- not the PFNBlocks' internal attention
P_i        = softmax_k(score_{i,:})                                # row-stochastic -- CLAUDE.md's own "row-stochastic softmax is correct here", never doubly-stochastic/Sinkhorn (A is a strict subregion of B's support)
```

Target — a **soft**, kernel-weighted correspondence in `z`-space (never a hard
binary "this one, not that one" — a hard binary target reintroduces exactly
the over-commitment problem, just one level removed from the raw-logit
version):

```
target_{ik} ∝ exp(- ‖z_i^A − z_k^B‖² / 2σ²),   row-normalized over k
L_couple = (1/n_A) Σ_i CE( target_{i,:}, P_i )
```

**Why this is structurally immune to the original blowup:** both `target` and
`P_i` are proper probability distributions over the *same* support (they sum
to 1 over `k`). There's no way to raise one entry's probability without
necessarily lowering others — the uniform-scale-inflation cheat that broke
the raw-logit version simply isn't available to a loss defined on normalized
distributions. `σ` is a genuine, principled knob: wide early (coarse,
forgiving), narrowed over training or over depth (`w_ℓ ∝ ℓ`-style, one
instance of this head per layer, decreasing `σ` with layer index) — turning
the "iterative refinement across depth" idea from §1/§4 into something
literal rather than just an analogy.

This is precisely CLAUDE.md's own already-planned build-order step 6
("Coupling head 0 + coupling loss ... it should look like a soft monotone
correspondence") — plausibly informed by exactly this earlier experiment.
Worth reviving specifically as *that* head, never as a patch on the
general-purpose MHA blocks.

---

## 4. The research landscape

### 4a. Worth investigating further

**Transformers learn operators in-context.** Worth investigating concretely:
"In-Context Operator Learning with Data Prompts for Differential Equation
Problems" (Yang, Liu, Osher et al., PNAS 2023, "ICON") is the paper to read
next here specifically for *how* they encode input-output example pairs as
"data prompts" — that encoding choice is the part most likely to transfer
usefully to how `A_ctx`/`B` get tokenized, independent of the differential-
equation specifics. This problem genuinely *is* in-context operator learning
(learn `T`, `h` from a handful of examples, no gradient steps at inference),
just for a coordinate-transform operator instead of a PDE solution operator.

**The iteration-shaped stack.** Concretely: weight-tie some or all of the
pooled `PFNBlock`s (Universal Transformer, Dehghani et al. 2019, is the direct
precedent — same block applied repeatedly rather than `K` independently
-parametrized blocks), optionally with an adaptive iteration count (ACT,
Graves 2016) rather than a fixed depth. Directly justified by existing
results, not just an analogy: Garg et al. 2022 ("What Can Transformers Learn
In-Context? A Case Study of Simple Function Classes") and the line of work on
transformers implementing (preconditioned) gradient descent in-context show
stacked linear self-attention can implement an *unrolled fixed-point
iteration* — a Neumann-series-style approximation to `(X'X)⁻¹` is literally
one instance of "each layer ≈ one more term/step." If a registration
refinement procedure is genuinely iteration-shaped, a stack that's
*architecturally* an iteration (shared weights, explicit iteration count) is
a better match than one that merely has enough depth to approximate one
incidentally. Worth a deliberate, isolated experiment.

**Thinking rows (MTPFN-style learnable extra tokens) — yes, with a caveat:**
cheap, well-precedented (register tokens: "Vision Transformers Need
Registers," Darcet et al. 2023; "pause tokens" in LLMs), and complements the
iteration story directly — scratch space for a running registration estimate
that isn't pinned to any one token's own hidden state. **Needs a dedicated
ablation, not just "add and see"** — the obvious confound is that `K` extra
tokens also means more parameters and more compute per forward pass, so any
improvement needs to be checked against a parameter/compute-matched control
(e.g. widening `d_ff` by an equivalent amount, no extra tokens) before
crediting the *tokens* specifically rather than "the model just got bigger."

### 4b. The translation analogy, in more depth

In vanilla seq2seq (Vaswani 2017), the encoder computes a representation of
the *complete* source sentence once, and that representation never depends on
how much of the target has been decoded — a source word's meaning is
decoder-state-independent. Cross-attention only flows one way: decoder
queries into a static, precomputed memory.

That's not true here, and it's not just an analogy — it's already encoded
elsewhere in this project for an *adjacent* reason. CLAUDE.md's own "settled
against" list rules out **KV-caching across warp samples**, with exactly this
justification: *"B's representations depend on A's coordinates whenever both
share a context. Warp-independence is purchased by the encoder/decoder split,
not inherited."* That's the same fact — B's *usefulness*, i.e. what it
corresponds to, isn't a property of B in isolation, so a representation of B
computed without reference to A can't be the right one. `IDTokenPFN`'s
bidirectional pooled self-attention (A and B attend to each other
symmetrically before the query cross-attends) is already the correct response
to this, not an accident.

What the analogy leaves genuinely open, worth stating as a live question
rather than resolving: does B need to attend to *all* of A to register
itself, or would a cheaper two-hop scheme work as well or better — B
summarizes itself once, A conditions on that summary, B then re-attends to
A's now-A-conditioned summary? That's a real architectural fork (full
symmetric bidirectional attention over the whole pooled set, vs. a structured
two-pass scheme), not something this analogy settles by itself.

### 4c. Considered and set aside

**DeepONet-style branch/trunk split** and **geometry-aware/RBF-biased
attention** — both dropped, at the user's direction. On the latter
specifically, thinking it through further makes the rejection sharper than
just "not exciting": an RBF bias based on raw Euclidean distance between
`x_i^A` and `x_k^B` is close to circular. "Nearby" presupposes a shared
metric, and the entire problem is that A's and B's coordinate metrics are
warped relative to each other *before* registration — comparing raw
positions across domains with a fixed-metric bias doesn't mean anything until
the thing you're trying to learn already exists. Not a fit here.

**Slot Attention** (Locatello et al. 2020) — kept, downgraded. Still a
plausible concrete *mechanism* for the coupling head in §3 (iterative,
competitive soft-assignment with an explicit iteration count is
architecturally close to what a coupling head wants), but not independently
exciting enough to chase before the more basic §3 design is even tried.

**Gradient-based test-time adaptation (MAML-style)** — noted for contrast,
not recommended: a few gradient steps on the context set at inference would
break the "one forward pass, no gradient steps" PFN premise this whole
project is built on.

### 4d. TabPFN-style row+feature (cell) attention as a basis architecture

Asked directly whether adopting this as a basis architecture to toy with
(not just a bolt-on) is worth it.

**The registration-specific motivation is weaker than it looks.** TabPFN's
column/feature attention is well-suited to tabular data because real table
columns are often genuinely, or close to, independent — cross-feature
attention there is largely about discovering redundancy/interaction between
otherwise-separable columns. Here, `T` comes from one shared velocity field
acting *jointly* across all `d` coordinates (§1) — it is not, in general,
separable per-axis. Treating each coordinate as an independently-attendable
cell doesn't obviously help recover a non-separable coupling, and might
mildly cut against it, unless the feature-attention layers end up
reconstructing the coupling anyway (plausible with enough depth, but then the
benefit over a plain per-point linear projection is less clear).

**The actually compelling motivation is orthogonal to registration, and
stronger:** `d_real` varies per draw (1, 2, 3, or 5, `D_CHOICES`), currently
handled by zero-padding to `d_max` and masking (`_pad_rescale_coords`,
`n_features`) — workable, but every draw's linear `x_embed` projection has to
implicitly serve both a `d=1` draw and a `d=5` draw through the same weight
matrix. A cell-encoding treats "how many feature-tokens per point" as a
first-class variable, exactly the way TabPFN handles tables with different
numbers of columns — no padding needed at all, just `d_real` cells for that
draw. That's the same structural problem (variable, unordered dimensionality)
TabPFN's own encoding exists to solve, independent of anything about
registration specifically.

**A genuine unification this would buy:** the additive domain tag could stop
being a special-cased embedding and become just another feature *column* —
one synthetic cell per token carrying the domain id, native to the
feature-attention framework instead of bolted on.

**Open engineering questions, not yet answered:** how query tokens (which
have `x`-cells but no `y`-cells) fit the same 2D grid; whether domain identity
attaches per-row or as its own cell (previous paragraph suggests the latter);
and the shape-mismatch concern already raised in conversation (`n` varies
across A/B, `d_real` varies across draws — the grid is genuinely ragged in
both directions at once, more so than a single tabular dataset ever is).

**Verdict:** worth prototyping as its own separate basis architecture, not a
bolt-on to §2 — the padding-elegance argument alone justifies trying it,
independent of whether it turns out to help registration specifically. Bigger
lift than §2 or thinking rows; its own track.

---

## Status and how to use this

Nothing here is implemented. `lupi_bounds`/`lupi_id_token_baseline` ran
throughout, untouched. §4 is inspiration for architectural thinking, not a
backlog to execute in order — whether/when any of §2 (aux heads), thinking
rows, the iteration-shaped stack, the operator-learning framing, or the
TabPFN cell-encoding track turns into an actual experiment is a later,
separate decision, to be revisited once the current two runs' results are in.

**If/when §2 specifically does become an experiment:** prototype it alone
first — `TransportHead` reused as-is, the revised monotone-mean-plus-spread
`h` head, fixed small `λ_T`/`λ_h`, last-layer-only — as a *new* experiment
sitting alongside `lupi_id_token_baseline`, not replacing it. Whatever else
from §4 gets tried later, keep each addition isolated in its own run so
individual effects don't get confounded, and check the thinking-rows ablation
against a parameter/compute-matched control specifically (§4a).

**Verification, mirroring existing repo conventions, for whenever §2 is
built:**
- `rho=0` invariant: at `rho=0`, `x_j^{B→A} == x_j^B` exactly (already
  asserted elsewhere) — `TransportHead`'s target degenerates to the identity
  there, a cheap sanity check the head's wired correctly before trusting it
  at `rho>0`.
- A `__main__` diagnostic on the new model class (matching every existing
  baseline's own convention): confirm `L_T`/`L_h` are finite and decrease
  over a handful of local optimizer steps on a fixed synthetic batch, before
  ever launching a real training run.
- If/when §3's coupling head gets built: visualize its output at `d=1`, per
  CLAUDE.md's own build order ("it should look like a soft monotone
  correspondence") — same diagnostic already planned there, reused here.
