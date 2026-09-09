# Amortized In-Context Registration and Transfer Between Deformed Point Clouds

A roadmap for using a PFN to transfer information from an abundant task B to a scarce task A when the two live in non-uniformly deformed input coordinates.

*Revision 2. Corrections in this revision: no KV caching under bidirectional PFN attention (§5.1); PFN training description fixed (§3); the kNN-descriptor / "GW-in-context" coupling head retracted and replaced with iterative y-anchored refinement (§5.2); transport head changed from Gaussian mixture to bar distribution (§5.3); folding promoted from nuisance to specification test (§5.4); coupling loss defined (§6.3); λ₁ annealing replaced with floor plus architectural bottleneck (§6.5).*

*Revision 3 adds §1.6 on selective sampling and design ignorability, which reclassifies the rank-feature and quantile-matching baselines as biased rather than weak under adaptive designs, and replaces the y-spread identifiability diagnostic with Fisher conditioning (§8.5).*

---

## 1. Problem statement

### 1.1 Setting

Two labelled point clouds:

- **A** (scarce): {(xᵢ^A, yᵢ^A)}, i = 1..n_A, with n_A small.
- **B** (abundant): {(xⱼ^B, yⱼ^B)}, j = 1..n_B, with n_B >> n_A.

No point correspondence. No shared input frame. The two clouds are related by an unknown diffeomorphism of the input space. Responses y are assumed to live in a shared, undeformed scale (§1.5 relaxes this).

**Goal.** Predict p(y | x*) for x* given in A-coordinates, using B's data to sharpen the prediction beyond what A alone supports, without the deformation causing negative transfer.

### 1.2 Why divergence-based framing was dropped

Relative entropy between the two clouds is only defined once they share a measurable space, so any divergence number is a joint measurement of task difference *and* misregistration. Since registration is the actual object of interest, it should be an estimand, not a nuisance folded into a scalar. Divergences remain useful downstream as diagnostics (residual after registration, α-sweep for tail-vs-core structure), not as the objective.

### 1.3 Gauge fixing

T and f are confounded in general: warping the input of a flexible function class produces another member of that class. The problem becomes identifiable only under an explicit assumption:

    f_A = f_B ∘ T

One latent function, two coordinate systems. B is abundant, so B's frame is the reference and f_B is well determined by B alone. T is then the only free object.

**Consequence.** T is identified only where B has coverage *and* |∇f| is appreciable. Elsewhere no amount of A-data constrains it, and the warp family should be pinned to the identity there rather than left free.

### 1.4 Direction of the pushforward: A into B

Commit to T : X_A → X_B and write the predictor as

    p( y | T(x*), context = B ∪ T#A )

**This does not cost you the prediction you actually want.** y is not transformed by T, so the above *is* p(y | x*) for a query named in A-coordinates. It is a change of the conditioning coordinate, not of the predicted variable, and the predictive density over y therefore needs no Jacobian correction. The A→B direction is a computational and gauge choice, not a restriction on the estimand.

Three places where the direction does show up, all handled:

- **Under P3 (y-distortion).** The head returns p(y_B | ·) and A's observable is y_A = h(y_B). Push through the estimated monotone h, with its 1-D Jacobian. Explicit, cheap, but do not forget it.
- **If you want a standalone model in A's frame** (handing someone a callable function in A-coordinates), you need T⁻¹. The latent-z construction (§2) supplies it: the B_inA head is trained alongside A_inB at no extra generative cost.
- **If you ever want a density over x** rather than a conditional, Jacobians matter. Out of scope here; the model is conditional throughout.

**Rationale for choosing A→B rather than B→A.** The warp then acts on the small set (n_A points plus queries) rather than the large one, so every warp sample re-embeds O(n_A) tokens instead of O(n_B). Under the two-stream design of §5.1 this compounds into genuine warp-independence of the B-side computation; under a single-stream design it is still the cheaper direction, just not free.

### 1.5 The two distortions

- **x-distortion** (primary): the diffeomorphism T. Non-uniform, possibly axis-coupling.
- **y-distortion** (secondary): y_A = h(f(T(x))) for unknown monotone h.

Handle x first. y is separable because monotonicity means rank statistics of y are exactly h-invariant, which yields both a robust estimator and a test (§8.6).

### 1.6 Selective sampling and the marginal/conditional split

Nothing above assumed A and B sample the *same region* with the *same design*. In the motivating case they do not. In transfer BO, A is an early run whose acquisition has visited a subregion and whose responses concentrate in one band, while B is a completed run covering the domain.

This is not a mild degradation. It is a bias, and it separates the candidate estimators cleanly.

**Marginal-matching estimators are biased under selective sampling.** Within-cloud rank is warp-invariant but support-*equivariant*: A's ranks span [0,1] over whatever subregion the design visited, B's over the whole domain, so equal ranks denote different locations. Quantile matching absorbs the design difference into T̂ with no diagnostic. The same argument applies to plain OT on x-marginals and to GW, which reads intra-cloud geometry that sampling density also controls. Under adaptive design these fail by producing confident misregistration, not by degrading gracefully.

**Conditional-likelihood estimators are not.** An acquisition rule depends only on past observations, so it is ignorable for p(y | x) in the standard missing-at-random sense: the design chose where to look, not what the function does there. Maximizing Σᵢ log p̂(yᵢ | T_θ(xᵢ), B) is therefore design-invariant.

**Consequences for this project.**

- The rank-feature baseline (§8.4) is not a weak-but-safe fallback under adaptive design; it is wrong. Report it stratified by design type, and expect the p = 0 point on the spending curve to be unavailable in the BO regime.
- §5.2's rank features must be dropped, or replaced by domain-box normalization (below), whenever the prior includes selective designs.
- The prior (§4.3) must sample designs, not just sizes: uniform, clustered, and acquisition-like. A model trained only on uniform designs will inherit the marginal-matching bias implicitly.

**Declared domain bounds are a free, design-immune anchor.** Where the search space is specified — as it is in essentially all BO and HPO settings — normalizing both clouds by their *declared* box gives a coarse frame alignment that empirical sampling cannot contaminate. The residual estimation problem is then only the non-uniform part of the warp, which is substantially smaller than starting from unregistered clouds. This also supplies the provisional shared frame that §5.2's layer 1 otherwise lacks under P2. Treat it as the default preprocessing wherever bounds exist, and keep the no-bounds case as the harder variant.

**Unbalanced transport becomes mandatory, not preferred.** A covers a strict subregion of B's support by construction here. Sinkhorn balancing would impose that A's mass spreads across all of B and drag T̂ outward. Row-stochastic softmax, or an explicitly partial formulation with a mass-creation penalty.

**The honest limitation.** In BO you want predictions in unexplored regions, which are exactly where T is least identified. There is a partial reprieve: acquisition functions consume uncertainty, so a warp-marginalized PPD that widens in unregistered regions inflates EI/UCB there and encourages sampling precisely what would improve registration — a self-correcting loop. The failure mode is warp uncertainty dominating function uncertainty, giving exploration driven by not knowing where you are rather than by not knowing the function. Decomposing predictive variance into warp and function components would diagnose this; with a bar-distribution transport head there is no clean way to do that decomposition, so it is an open problem (§Open items), not a solved one.

---

## 2. Generative model: the latent-z construction

Sample a latent configuration and two diffeomorphisms from it, rather than one warp relating the clouds directly.

    z ~ p(z),   S_A, S_B ~ p(S)
    x^A = S_A(z),   x^B = S_B(z)
    T = S_B ∘ S_A⁻¹,   T⁻¹ = S_A ∘ S_B⁻¹
    y = f(S_B(z)) + ε          (f lives in B's frame, per §1.3)

**Rationale.**

- **No inversion anywhere.** Both clouds and both pushforward targets — A_inB(xᵢ^A) = S_B(zᵢ^A) and B_inA(xⱼ^B) = S_A(zⱼ^B) — come from forward passes on the same z. This is what makes the auxiliary supervision of §6.2 cheap enough to use on every prior sample, and it removes the pressure to pick a warp family with a tractable inverse.
- **Symmetry.** Neither cloud is privileged in the generative process; the gauge choice is made separately and explicitly by where f is placed.
- **Controllable overlap.** Drawing z^A and z^B from different latent regions produces partial support overlap, which is the realistic case and the one where transfer must be gated rather than applied globally.

ε is added in B's frame. If measurement noise is physically a property of the A instrument, add a separate per-cloud noise scale; do not let one σ silently serve both.

**Optional: shared-z anchor points.** Draw a small subset of z shared between the two clouds. Those points then have *known* correspondence, which enables a stronger form of the coupling loss (§6.3). Used at training time only; nothing at inference depends on it. Keep the shared fraction small and randomized, or the model learns to expect exact matches that real data will not provide.

---

## 3. The PFN

### 3.1 What it is, stated correctly

A prior-fitted network trained on samples from a BNN prior: sample an architecture (width, depth, activation), sample weights to obtain f, sample inputs, emit (x, y) pairs.

Training is **not** autoregressive next-target prediction. A single multi-head self-attention stack processes train and test points together, with a mask such that:

- train (context) points attend bidirectionally among themselves;
- test points attend to train points but not to each other and not to themselves.

The test-point outputs are projected to logits over a discretized **bar distribution** (piecewise-constant Riemann density over a fixed set of bins), and the loss is the NLL of the true target under that distribution. One forward pass therefore yields a full PPD for every test point, each conditioned on the context and independent of the other test points.

### 3.2 Three consequences for this project

**(a) Registration must be read out from A-as-context tokens.** Query tokens are mutually isolated by the mask. A single A-point supplies only one codimension-1 constraint on T, so registration requires A-points to pool information; that pooling only happens among context tokens. Put A in the context (it is labelled, so it belongs there anyway) and attach the transport head to those positions.

**(b) The bar distribution is the natural output form for the transport head too** (§5.3). No new machinery, and it inherits the multimodality handling that a Gaussian head lacks.

**(c) There is no KV cache.** Bidirectional context attention means B's representations depend on A's coordinates whenever both sit in the same context. Warp-independence of the B-side must be purchased architecturally (§5.1); it is not a property of the base model.

### 3.3 Suspected reason the dataset-ID token alone did not help

A smooth input warp composed with a BNN sample is approximately another BNN sample, absorbable into the first layer. Under a BNN prior, "same f, warped coordinates" and "different f" therefore have comparable plausibility, and a well-calibrated model correctly declines to commit to the former. If so, the deficit is in the prior, not the tokens. Experiment 0 (§8.1) tests this before any architecture work.

---

## 4. Priors: a nested ladder

Train and evaluate on a ladder, not a single prior. Each rung isolates one mechanism and provides the reference point for the next.

| Rung | Warp family | Purpose |
|---|---|---|
| **P0** | Identity | Sanity: transfer must work when there is nothing to register. Ceiling check. |
| **P1** | Elementwise monotone rational-quadratic spline, K bins/axis | Axis-aligned regime. Rank features are exactly invariant here, giving a strong parameter-free baseline. |
| **P2** | Stationary velocity field (primary) or coupling INN (alternate) | Axis-coupling deformation. The regime where quantile matching, rank features and GW break, so it discriminates between methods. |
| **P3** | P2 + monotone h on y | Joint x- and y-distortion. |

### 4.1 Warp parameterizations

**Elementwise RQ spline (P1).** Analytic inverse, analytic log-det, nests the identity, single integer complexity knob K. Tails clamped to identity outside the data box so the warp cannot drift where no data constrains it.

**Stationary velocity field (P2, primary).** Define

    v(x) = s · Σ_m w_m k(x, c_m)

with M kernel centers c_m, and take T = the time-1 flow of v, integrated by a few RK4 steps. Diffeomorphic for any bounded v.

Preferred over coupling INNs for three reasons. It works at any dimension including d = 1, whereas coupling flows need d ≥ 2 to have coordinates to split. Severity enters as a single multiplicative scalar s on the field, so warp strength is exactly decoupled from parameter count M. And integration time t ∈ [0,1] traces a *continuous path* from identity to full warp, which is a cleaner curriculum knob than any weight-scaling scheme: train with t ramped from 0, and every intermediate model is a valid diffeomorphism rather than an interpolation artifact.

**Coupling INN (P2, alternate).** RQ-spline coupling layers, zero-initialized final layer of each coupling net so the stack begins at the identity, with a per-draw severity scalar scaling the perturbation. Retained because it produces qualitatively different deformation structure from a kernel field (sharper, more axis-entangled), which is useful for testing that results are not an artifact of one family.

### 4.2 Two controls that matter more than the family choice

**Severity as a logged scalar.** Whether from s in the velocity field or from perturbation scaling in the INN, record it with every prior sample.

*Rationale:* the spending curve (§8.3) must be conditioned on true warp severity, not only on nominal DOF. Severity and complexity are separate axes and conflating them will muddy the central experiment.

**Distortion-statistic rejection.** Evaluate log|det J| on a grid over the data box; reject or rescale draws outside a target band.

*Rationale:* naive sampling of flow weights gives log-det variance growing with depth, producing draws that crush one region and blow up another. Such warps occur in no real experiment, and a prior dominated by them wastes capacity. A band also makes the prior statable in one interpretable sentence ("local stretching between 1/3x and 3x"), which matters when defending results.

### 4.3 Sampling ranges

Randomize n_A (small, spanning the range where the p/n law bites), n_B, support overlap fraction, noise scales, and the BNN hyperparameters. The model must handle scarcity as a condition it recognizes, not as a fixed regime.

**Sample designs, not just sizes.** Include uniform, clustered, boundary-biased, and acquisition-like sampling for A, with the acquisition-like case producing both a restricted x-subregion and a concentrated y-band. Per §1.6, a model trained only on uniform designs will implicitly inherit the marginal-matching bias, because under uniform sampling the marginal and conditional routes agree and nothing forces the model to prefer the design-invariant one. Log the design type alongside severity so results can be stratified by it.

---

## 5. Architecture

### 5.1 Two-stream encoder with a B-side bottleneck

Encode B into a small set of latent tokens (Perceiver-style: learned queries cross-attending to B's points, with no attention path from A). A-points and queries then cross-attend to that latent set.

**What this buys.**

- **Warp-independence, purchased explicitly.** Because the B-stream never attends to A, its latent set does not change when A is re-warped. This is the only way to get the caching benefit; §3.2(c) rules out getting it for free.
- **Structural separation beats tagging.** With an additive ID token, the cloud distinction sits in a cross-term of W_q(e_x + id)·W_k(e_x + id) that the model must learn to isolate. Two streams prevent cross-cloud information from flowing through raw coordinate matching by construction.
- **Correct equivariance.** The symmetry group is S_{n_A} × S_{n_B}, not S_{n_A+n_B}.

**What it costs.** B cannot adapt its summary to A. Since B's job is to represent f_B, and f_B is defined without reference to A, this is defensible — but it is a real restriction, and it puts all cross-cloud work on §5.2. The bottleneck width is a capacity knob that should be swept, not guessed.

**Fallback.** If the bottleneck proves too lossy, revert to a single bidirectional context with concatenated (not additive) cloud IDs and per-cloud input embedders. Then warp marginalization costs K full forward passes and K is capped around 5.

### 5.2 Iterative correspondence refinement

*This replaces the "GW-in-context" head from revision 1, which was wrong. Sorted intra-cloud kNN distances fail twice over: they are exactly the quantity a non-uniform stretch alters, so they are not warp-invariant; and A is sparse while B is dense, so they differ by roughly n^(-1/d) even under the identity warp. They confound sampling density with deformation. GW proper is a quadratic assignment on full distance matrices, not a local descriptor, and it remains a baseline (§8.4) rather than an architectural component.*

**Layer 1 matches on y alone.** y is undeformed and shared, and is the only genuinely frame-free anchor available. The first correspondence is therefore semantic, not geometric.

**Layer ℓ > 1 matches in the partially-registered frame.** Once layer ℓ−1 has produced a transport estimate, A's points have provisional B-coordinates, and raw coordinate similarity becomes legitimate:

    φ_ℓ(Aᵢ) = [ yᵢ^A ,  T̂^(ℓ-1)(xᵢ^A) ,  rank-features(xᵢ^A) ]
    ψ(Bⱼ)   = [ yⱼ^B ,  xⱼ^B ,  rank-features(xⱼ^B) ]

    a_ij^(ℓ) = ⟨ W_q^(ℓ) φ_ℓ(Aᵢ) ,  W_k^(ℓ) ψ(Bⱼ) ⟩ / sqrt(d_h)

    π_ij^(ℓ) = softmax_j( a_ij^(ℓ) / τᵢ )

    T̂^(ℓ)(xᵢ^A) = Σ_j π_ij^(ℓ) · xⱼ^B

W_q and W_k are separate projections; the roles are asymmetric and the descriptors are per-cloud, so a shared projection would be incorrect. The value projection is frozen to the identity, so outputs are convex combinations of B's raw coordinates: the head cannot memorize, and it cannot place a point outside B's convex hull.

Rank features (within-cloud quantile position per axis) are retained because they are exactly invariant under P1 and give layer 1 something beyond y in the axis-aligned case. They are not claimed to be invariant under P2.

**The stack is an unrolled registration algorithm**, coarse-to-fine, closer to unrolled CPD with a semantic anchor than to anything GW. This is what the depth is for, and it is the answer to why stacking helps: each layer re-matches in a better-aligned frame.

**Temperature is per-row and context-dependent.** A single learned global scalar is wrong, as it would have to serve every sampled (A, B) pair and every region within a pair. Use

    τᵢ = softplus( u^T g  +  w^T hᵢ )

where g is a pooled pair-level token and hᵢ the A-point's own representation. Sharpness then adapts both to how discriminative the descriptors are for this pair and to how identifiable the transport is at this point.

**Row entropy is inspection-only.** With τᵢ learned per row, entropy is directly controlled by τᵢ and is close to tautological as an uncertainty measure; across a stack it is also layer-dependent with no canonical choice. Revision 1 claimed it as an independent calibrated readout, which was overstated. Use the final-layer coupling as a *visualization* — it is the one component you can look at and see whether the model found the correspondence — and take calibrated uncertainty from §5.3 instead.

### 5.3 Transport head as a bar distribution

Predict the distribution of T(xᵢ) **per point**, using the PFN's own Riemann/bar output rather than a Gaussian or a Gaussian mixture.

**Why bar rather than Gaussian mixture.**

- Reuses existing machinery and the existing loss; no new head type, no new calibration story.
- Represents multimodality natively without committing to a component count. Registration is genuinely multimodal — symmetric f admits reflections, periodic structure admits shifts — and a unimodal Gaussian averages modes into a location with no support while reporting a large σ that looks like honest uncertainty but is not.
- No Gaussian tail assumption, and bounded support is natural: bins are defined over B's data box, which is exactly where T(x) must land.
- Sampling for warp marginalization (§7) is trivial and exact.

**Multivariate handling.** Factorized per-axis bar distributions discard cross-coordinate correlation in the transport posterior, which matters under P2 where the deformation couples axes. For small d, make it autoregressive over axes: predict axis 1's bars, condition axis 2's on the sampled value, and so on. Cost is d sequential head evaluations on an already-computed representation.

**Why per-point targets rather than warp parameters θ.** Family-agnostic, so a single model can train across a mixture of warp families and the targets remain well-defined. Dense: n_A + n_B constraints instead of p. And per-point transport is what everything downstream consumes.

**Why this is the amortized Fisher map.** The training target T(xᵢ) is known from the prior but not inferable from the context where the data is uninformative. The NLL-minimizing predictive distribution is therefore the posterior over T(xᵢ) given that context: the model cannot do better than report its irreducible uncertainty, and it is penalized for reporting less. This is the same quantity the analytic Fisher calculation approximates —

    Var(T(x)) ≈ σ² p / ( n_A · E|∇f|² )

— but without linearization or asymptotics, and automatically accounting for B's local density and |∇f|'s local magnitude, because both varied across the prior. Use the interquartile range of the bar distribution wherever the analysis calls for σ. Falsifiable: see §8.5.

### 5.4 Fold statistic as a relatedness test

**Do not architecturally forbid folding.** A diffeomorphism cannot fold. So if the free transport head wants to — if the induced map is non-injective, orientation-reversing, or has negative Jacobian determinant somewhere — that is direct evidence against the modelling assumption f_A = f_B ∘ T. Projecting onto a diffeomorphism family would destroy exactly the signal that tells you whether transfer is warranted at all. Revision 1 got this backwards by treating folding purely as a defect to suppress.

**Computing it.** Take the pointwise median (or mode) of the transport distribution as a map estimate, evaluate on a grid over A's support, form the Jacobian by finite differences, and report

    fold fraction  =  measure{ x : det J(x) ≤ 0 }  /  measure{ support }

In d = 1, or per-axis, the cheaper equivalent is the rank-crossing rate: the fraction of pairs (i, k) whose ordering along an axis reverses between A-coordinates and predicted B-coordinates.

**Calibrating it.** Prior samples satisfy the shared-f assumption by construction, so evaluating the fold fraction across held-in prior draws gives a null distribution, stratified by n_A and severity. An observed pair exceeding the null upper quantile fails the test.

**Using it.** Fold fraction becomes a pre-transfer decision statistic: compute it, compare against the null, and fall back to A-alone if it fails. This pairs directly with the bounds of §8.1 — the bounds tell you what transfer *could* be worth, the fold statistic tells you whether the premise holds. Diffeomorphism projection (fitting a P1/P2 family to the head's output with a folding penalty) happens only *after* the test, when a guaranteed-invertible T̂ is needed downstream.

An open question worth testing: whether fold fraction degrades gracefully with the degree of task mismatch, or only fires at gross violation. Generate deliberately mismatched pairs (f_A = f_B ∘ T + perturbation, with a perturbation-magnitude sweep) and plot the statistic against it.

### 5.5 Predictive head, with the registration bottleneck

Standard bar-distribution output for p(y | x*), with x* mapped through the transport representation and conditioned on the B-latent set plus A.

**Critical constraint: no residual path from B to the prediction that bypasses the transport.** The query's only access to B must be through T̂. This is what prevents the collapse discussed in §6.5 — if raw B-information can reach the predictive head directly, the model can improve predictive loss while abandoning registration, and it will.

---

## 6. Training objective

    L  =  L_pred^A  +  λ₁ ( L_A_inB + L_B_inA )  +  λ₂ L_coupling

### 6.1 Predictive loss: score A-queries only

L_pred^A is the bar-distribution NLL on query points in A-coordinates, with A ∪ B in context.

**Rationale.** Including B-queries-given-B-context adds a term that is easy, large, and gradient-dominant, swamping the A-query signal where registration actually pays. This is the most likely single cause of a null result in the earlier token-ID attempt and the cheapest thing to get right.

### 6.2 Registration loss: bar-distribution NLL on both pushforwards

    L_A_inB = − Σ_i log p̂( S_B(zᵢ^A) | · )
    L_B_inA = − Σ_j log p̂( S_A(zⱼ^B) | · )

Bar-distribution NLL, per axis, autoregressive over axes — matching §5.3, and matching the PFN's own loss. (Revision 1 specified Gaussian-mixture NLL here; that was an inconsistency with no justification behind it.)

**Rationale for having this term at all.** Registration is a serial computation: match, aggregate, warp, re-predict. Soft matching is one attention layer; composing the rest needs several, and a shallow PFN may lack the depth to discover it from predictive loss alone. Auxiliary targets do not merely add supervision — they force the intermediate representation to exist rather than hoping it emerges. This is the load-bearing argument for the whole latent-z construction.

Keeping L_B_inA (the direction not used at inference) is deliberate: it is free given the generative model, it regularizes the transport representation toward genuine invertibility, and it supplies T⁻¹ for the §1.4 use cases.

### 6.3 Coupling loss, defined

Revision 1 named this term without specifying it. Two forms, in increasing strength:

**(a) Barycentric consistency (default, no correspondence needed).**

    L_coupling = Σ_i || Σ_j π_ij xⱼ^B  −  S_B(zᵢ^A) ||²

The coupling's own barycentric projection is supervised against the known true pushforward. Uses the target already available, needs no ground-truth matching, and applies at every layer of the §5.2 stack if you want intermediate supervision.

*Role:* the barycentric path is strictly lower-capacity than the §5.3 head — a convex combination of B's coordinates, with an identity value projection — so this term forces the early layers to form a usable correspondence rather than letting the transport head compensate for a poor one downstream. It is a representation-shaping regularizer, not an accuracy objective.

**(b) Correspondence cross-entropy (optional, requires shared-z anchors from §2).** For anchor points with known match j*(i):

    L_coupling = − Σ_{i ∈ anchors} log π_{i, j*(i)}

Stronger signal, directly supervising the transport plan rather than its projection. Only available for the anchor subset, and it risks teaching the model to expect exact matches. Keep the anchor fraction small, randomized, and ramped down during training.

Start with (a). Add (b) only if correspondence quality is diagnosed as the bottleneck.

### 6.4 What to hold out

Held-out A-query log-likelihood, reported against both bounds of §8.1. Never report a raw predictive number without them.

### 6.5 Loss weighting: floor and bottleneck, not annealing

Revision 1 proposed annealing λ₁ toward zero. That risks exactly what you would expect: "ignore B and predict from A alone" is a safe, easily-reachable attractor, and removing the registration signal invites the model to fall into it while predictive loss looks fine.

Three mechanisms instead, in order of reliability:

**(a) Architectural, primary.** The §5.5 bottleneck means the predictive head reaches B only through T̂. Collapse to marginal prediction then costs predictive loss directly, so the attractor is removed rather than merely disincentivized. This is a guarantee; the other two are mitigations.

**(b) Floor, not zero.** Ramp λ₁ down from a high initial value to a floor around 10–20% of it, never to zero. Registration stays supervised for the whole run.

**(c) Explicit collapse monitor.** Track two quantities throughout training on held-in prior samples: registration error on A_inB, and the *transfer gap* — held-out A NLL with B in context minus the same with B removed. If registration error rises while predictive loss falls, or if the transfer gap closes toward zero, collapse is underway. Both are cheap and should be on the training dashboard, not computed post hoc.

Gradient-normalized weighting (GradNorm-style) is a reasonable substitute for (b) if the hand-tuned floor proves brittle, but it does not substitute for (a).

---

## 7. Inference

**Warp-marginalized prediction.** Draw K transport samples from the §5.3 bar distribution, push A and the queries through each, and mix:

    p(y | x*)  =  (1/K) Σ_k  p( y | T̂_k(x*),  B ∪ T̂_k#A )

**Rationale.** Conditioning on a point estimate tightens the PPD by more than the evidence justifies. The mixture widens most where the transport is least constrained, which is where extrapolation is most tempting.

**Cost, honestly.** With the §5.1 two-stream design, the B-latent set is warp-independent and each sample re-embeds only A plus the queries; K = 10–20 is routine. Without it, each sample is a full forward pass over the whole context and K is capped around 5. This is one of the main things the bottleneck is buying, and it should be weighed against the representational cost noted in §5.1.

**Regional transfer gating.** Compute PPD width and held-out score as a function of x, with and without transfer. Transfer typically helps in the interior where T is pinned by A's coverage and hurts near the edges where T extrapolates. Restricting transfer to the safe region beats an all-or-nothing decision, and the transport distribution's own spread is the natural gate.

**Pre-transfer check.** Run the §5.4 fold test first. If it fails, do not transfer.

---

## 8. Experiments

### 8.1 Experiment 0 — replicate the ID-token result and establish bounds

Run before anything else, on the existing PFN, with no retraining.

**Protocol.**

1. Generate paired clouds from the latent-z model at P0 (identity) and P1 (mild, then severe).
2. Measure four numbers per pair, all held-out A-query NLL:
   - **A alone** — the lower bound. Transfer that does not beat this is negative transfer.
   - **A ∪ B, naive pooling** (no cloud distinction) — the failure mode to beat.
   - **A ∪ B with an additive dataset-ID token** — the configuration that did not previously work.
   - **A ∪ T_true#B — the oracle**, i.e. what B contributes when registration is free.
3. Report ΔNLL_oracle = NLL(A alone) − NLL(A ∪ T_true#B). Everything between the two bounds is the registration cost, and it is the total budget available to every method in this project.

**What each outcome means.**

- **ΔNLL_oracle small even at P0.** No transferable signal under this BNN prior. No architecture change will help; the prior must be constrained (§9) before anything else is worth building.
- **ΔNLL_oracle large, ID-token ≈ A-alone.** Transfer exists and was not exploited. Proceed, and check calibration direction: *underconfident but honest* means the model learned "I cannot register, so I hedge," which is Bayes-correct and points back at the prior; *overconfident* means naive pooling ignoring T, which points at the representation.
- **ID-token ≈ naive pooling.** The token carried no information at all — consistent with the additive cross-term argument in §5.1, and a direct case for the two-stream design.

**Variations worth running in the same sweep**, since they are nearly free and each isolates one of the §6.1 / §5.1 hypotheses: scoring A-queries only versus scoring both; concatenated versus additive ID; separate per-cloud input embedders.

Track the two bounds on every subsequent experiment in the project.

### 8.2 Experiment 1 — non-amortized reference

Latent-z generator at P1. Two-stage estimator: fit ĝ ≈ f_B from B alone (one PFN forward), then estimate T by maximizing Σ_i log p̂(yᵢ | T_θ(xᵢ^A), context = B) over the warp parameters, initialized from quantile matching. Finite-difference BFGS or CMA-ES on 4–8 parameters.

Establishes the phenomenon and the spending law before any new model exists, and becomes the reference every amortized result is measured against.

### 8.3 The spending law (the central experiment)

Sweep warp complexity p (spline bins K, or velocity-field centers M) against n_A; measure held-out A log-likelihood.

**Predictions.** An interior optimum p*(n_A) growing sublinearly in n_A; excess risk ≈ σ²p/n_A, so curves plotted against p/n_A should collapse. The gradient magnitude cancels to first order — warp error propagates as δy ≈ ∇f·δT while Var(T) ∝ 1/|∇f|² — leaving the standard parametric rate.

**Stratify by severity.** Condition on the logged severity scalar (§4.2). Collapse should hold within severity strata; if it holds only marginally, severity and complexity are interacting and the p/n law is incomplete.

**Falsification.** Failure to collapse means either the warp family is misspecified (bias term dominating) or the shared-function assumption is failing. The §5.4 fold statistic distinguishes these.

### 8.4 Baseline ladder

| Baseline | Role |
|---|---|
| A alone | Lower bound |
| Oracle T_true | Upper bound |
| Naive pooling | Failure mode |
| Additive dataset-ID token | Experiment 0's subject; the result that motivates the project |
| Within-cloud rank features, no fitted warp | The p = 0 point on the spending curve. Exactly invariant to monotone axis-aligned warps, so under a *uniform* design it should win at very small n_A and lose once n_A funds a real warp; the crossover is a headline result. Under selective design it is biased, not merely weak (§1.6) — run it stratified by design type and expect it to fail outright in the BO regime. |
| 1-D quantile matching | Closed-form axis-aligned registration. Shares the rank baseline's design bias; exhibiting that failure against the conditional-likelihood estimator is one of the clearer results available. |
| Declared-box normalization only, no fitted warp | The design-immune p = 0 point (§1.6). Replaces the rank baseline wherever domain bounds exist, and is the correct floor in the BO regime. |
| Coherent Point Drift | Classical soft-correspondence EM, unequal cardinality, no one-to-one assumption. Needs ambient overlap, so expect P2 to break it. |
| Entropic Gromov-Wasserstein + barycentric projection | The genuinely coordinate-free classical method, and the honest home for the GW idea. Minimizes pairwise-distance distortion, so a non-uniform stretch admits no zero-distortion coupling and GW returns the best isometric approximation; the bias is structured (it spreads the stretch) and should be visible. |
| Two-stage non-amortized (§8.2) | Isolates what amortization buys |

### 8.5 Diagnostics with predicted relationships

- **Transport IQR vs |∇ĝ(T̂(xᵢ))|⁻¹.** The Fisher argument predicts linearity. Flat IQR means an undertrained head collapsed to the marginal. Structured but off-axis means the prior produces an identifiability pattern the analysis missed, which is worth knowing on its own.
- **Fold fraction vs task-mismatch magnitude** (§5.4). Does the statistic degrade gracefully or only fire at gross violation?
- **Fisher conditioning, not y-spread.** Revision 1 proposed checking the spread of A's y-values against ĝ's range. That is the wrong statistic. Each A-point supplies one codimension-1 constraint, and what identifiability requires is that the constraint directions ∇ĝ(T̂(xᵢ)) *span* the space — which is exactly the conditioning of I(θ) = σ⁻² Σᵢ Jᵢᵀ gᵢ gᵢᵀ Jᵢ. Report its eigenvalue spectrum and condition number.

  The distinction matters and is counterintuitive: nearly-parallel level sets are the bad case (early BO on a plateau, all gradients aligned), while nested closed contours near an optimum are geometrically diverse and identify well *despite* a narrow y-band. A converged run's clustered points can therefore register better than a diffuse early one. Under adaptive design, plot registration error against BO iteration and locate where transfer becomes viable.
- **Coupling plan visualization** (§5.2, final layer). Qualitative, not calibrated, but it is the one component you can inspect directly.

### 8.6 y-distortion (P3)

Estimate T from **y-ranks only** via a differentiable concordance objective (soft Kendall τ) between {yᵢ} and {ĝ(T_θ(xᵢ))}. Exactly invariant to any monotone h. Recover h afterwards by isotonic regression of yᵢ on ĝ(T̂(xᵢ)), a clean 1-D problem given T̂.

Cost is roughly the Spearman-vs-Pearson efficiency premium (~10% under Gaussian noise), which buys a test: **material disagreement between rank-based and likelihood-based T̂ indicates y-distortion.** The amortized route samples h in the prior and infers both, but keep the two-stage rank version as a baseline to isolate which distortion the amortized model is actually handling.

---

## 9. Risks

| Risk | Symptom | Response |
|---|---|---|
| Warp absorbed into f under BNN prior | Small ΔNLL_oracle in Experiment 0 | Constrain the BNN prior (narrower lengthscale range, hyperparameters shared across the pair) so "same f" is a distinguishable hypothesis. Note the trade: a prior tight enough to make transfer identifiable may be too tight to cover real clouds. Experiment 0 measures how tight that bind is. |
| Shared-function assumption false | Fold statistic exceeds null; transfer LL plateaus below B's internal held-out score | Report the residual as genuine task difference. Do not increase warp capacity to close it. |
| B-bottleneck too lossy | Two-stream underperforms single-stream at P0 | Widen the latent set; if that fails, revert to single-stream and accept K ≈ 5 for marginalization |
| Registration collapse | Transfer gap closes during training; registration error rises while predictive loss falls | §6.5(a) should prevent it structurally; if it still occurs, the bottleneck has a leak — audit for residual paths from B to the predictive head |
| Coupling never leaves layer-1 quality | Transport head accurate but coupling plan diffuse at all layers | Add per-layer coupling supervision (§6.3a at every layer); if still flat, try §6.3b anchors |
| Dilution | Predictions ignore A; overconfident near A's support boundary | Subsample B to ratio r·n_A; treat r as a hyperparameter. A sharp optimum at low r means the warp is not capturing much and B is mostly contaminating. |
| Multimodal registration | Transport bars bimodal but downstream uses the median | Sample from the bars for marginalization rather than taking a point estimate; flag high-entropy points for regional gating |
| Marginal-matching bias learned implicitly | Model matches x-marginals; fails on selectively-sampled A while looking fine on uniform holdouts | Ensure the design prior (§4.3) covers acquisition-like sampling; hold out an adaptive-design split specifically |
| Warp uncertainty dominates function uncertainty in BO | Acquisition explores regions chosen by registration ignorance rather than function ignorance | Decompose predictive variance into warp and function components — currently unsolved for a bar-distribution head; interim mitigation is regional gating on transport spread |
| Prior/test mismatch | Good in-prior results, poor on real clouds | Report all results stratified by severity; be explicit that the amortized model is calibrated only within its prior |

---

## 10. Phasing

**Phase 0 — Experiment 0.** ID-token replication plus both bounds, on the existing PFN, no new training. Decides whether the deficit is prior-side or representation-side, and whether the project proceeds as specified.

**Phase 1 — Non-amortized reference.** Latent-z generator at P1; two-stage estimator; spending sweep (§8.3). Establishes the phenomenon and the p/n law before any new model exists.

**Phase 2 — Amortized, P1.** Two-stream encoder, iterative correspondence stack, bar-distribution transport head, registration bottleneck, A-only predictive scoring. Full baseline ladder. Target: match or beat the Phase 1 estimator at lower inference cost.

**Phase 3 — P2 and the fold test.** Velocity-field warps with the t-ramp curriculum. The regime where rank features, quantile matching and GW should all degrade. Calibrate the fold statistic and run the mismatch sweep.

**Phase 4 — P3 and marginalization.** Joint x/y distortion; warp-marginalized predictive distributions; regional transfer gating.

---

## Open items

- Whether the two-cloud PFN construction has been published; I have not verified the current literature on amortized registration inside PFNs specifically, as distinct from multi-task PFNs generally.
- Whether the two-stream bottleneck's representational cost is acceptable, or whether warp-independence must be given up. Measurable at P0 in Phase 2.
- Whether autoregressive-over-axes bar distributions scale acceptably past small d, or whether a low-rank plus residual parameterization is needed.
- Whether severity should be an inference target in its own right — a cheap global "how deformed is this pair" readout, complementary to the fold statistic.
- What replaces rank features as a P2-appropriate frame-free descriptor for layer 1, beyond y. Declared-box normalization (§1.6) resolves this wherever bounds exist, which covers the BO case; the no-bounds variant remains open.
- How to decompose the predictive variance into warp-uncertainty and function-uncertainty components under a bar-distribution transport head. Needed to diagnose the BO over-exploration failure mode in §1.6.
- Connection to the transfer-HPO literature. Per-task quantile/copula transforms of y are standard there and correspond to this project's P3 h-invariance; the x-side is usually assumed shared because the hyperparameter space is common across tasks, which is what makes the deformed-x case here distinct. I have not verified the current state of that literature and am not confident in the attribution.
