# Decision log

Choices that were made, some reversed, some re-made. Recorded so they don't get
re-litigated by whoever reads only the current code.

---

**D1. Divergence framing dropped in favour of registration as an estimand.**
Relative entropy between two clouds is only defined once they share a measurable
space, so any divergence number jointly measures task difference and
misregistration. Divergences survive as diagnostics, not as the objective.

**D2. Direction: A → B, not B → A.**
`y` is not transformed, so `p(y | T(x*), B ∪ T#A)` *is* `p(y | x*)` for a query
named in A-coordinates — a change of the conditioning coordinate, not of the
predicted variable, needing no Jacobian. The warp then acts on the small set. The
B → A direction is still available via the `t_enc` head for use cases that need a
callable model in A's frame.

**D3. `f` moved from B's frame to the latent frame.** *(reversed)*
The roadmap originally had `y = f(S_B(z))`, making B's observations a plain BNN of
its own coordinates and A's a warped-input BNN. That asymmetry is detectable from
marginals and makes encoder/decoder role randomisation incoherent.

**D4. kNN-distance descriptors retracted.** *(reversed)*
Proposed as the "Gromov-Wasserstein content" for matching. Wrong twice:
intra-cloud distances are exactly what a non-uniform stretch alters, and A is
sparse while B is dense, so they differ by ~n^(-1/d) even under the identity warp.
Neither warp-invariant nor density-invariant. Replaced by iterative refinement
anchored on `y`.

**D5. KV caching claim withdrawn.** *(reversed)*
PFN context attention is bidirectional, so B's representations depend on A's
coordinates whenever both share a context. Warp-independence is *purchased* by the
encoder/decoder split, not inherited — with a real cost, since the encoder cannot
adapt its summary to A.

**D6. Transport head: bar distribution, not Gaussian mixture.** *(reversed)*
Registration is genuinely multimodal (reflections, periodic shifts). A Gaussian
averages modes into a location with no support while reporting a large σ that
looks like honest uncertainty. Bars represent multimodality without a component
count and reuse the PFN's own output machinery.

**D7. Folding promoted from defect to specification test.** *(reversed)*
Originally something to suppress by projecting onto a diffeomorphism family. That
destroys the signal: a fold is evidence against `f_A = f_B ∘ T`, and therefore the
cheapest available test of whether transfer is warranted at all.

**D8. λ_T annealing replaced by a floor plus an architectural bottleneck.**
"Ignore the encoder, predict from A alone" is a safe, reachable attractor.
Annealing the registration signal to zero walks into it while predictive loss
looks fine. The bottleneck (cross-attention query built from the transport) is the
guarantee; the floor is the mitigation.

**D9. Marginal-matching estimators reclassified as biased, not merely weak.**
Under selective sampling, rank features and quantile matching fail by producing
*confident misregistration*. Conditional-likelihood estimation is design-invariant
because an acquisition rule depends only on past observations. Consequence: the
prior must contain restricted-support draws, or nothing forces the model to learn
the design-invariant route.

**D10. Fisher conditioning replaces the y-spread diagnostic.** *(reversed)*
Wide y-range does not imply identifiability (a tilted plane has parallel level
sets and identical gradients) and narrow does not preclude it (a bowl has radial
gradients spanning everything). What matters is whether the constraint directions
span the space.

**D11. Encoder–decoder adopted over a Perceiver bottleneck.**
Makes both baselines masking operations on one set of weights: zero the gates for
A-alone, teacher-force the transport for the oracle. No separate models, no
capacity confound. The translation analogy holds for cross-attention from an
incomplete target to a complete source, and breaks where it matters — in NMT the
vocabularies are fixed across examples, so alignment is learned once globally,
whereas here the frame relation is resampled per pair and must be inferred in
context. That is why the transport needs explicit supervision.

**D12. ρ, the relative-warp coefficient, replaces the absolute-severity ramp.**
Ramping both clouds' warps toward identity entangles two things: how deformed each
cloud looks alone, and how far apart the frames are. Only the second matters. A
convex combination of velocity fields is a velocity field, so ρ traces genuine
diffeomorphisms, with ρ = 0 giving the identity-transport anchor.

**D13. GW kept as a baseline, never as a loss.**
We have exact targets; GW is for when you have neither correspondence nor labels.
It is biased under non-uniform stretch (no zero-distortion coupling exists) and
under density mismatch — worst on exactly the hard cases.
