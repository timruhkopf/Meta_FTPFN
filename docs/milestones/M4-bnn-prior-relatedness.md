# M4 — BNN prior + relatedness mechanism

## Goal

Extend `src/ppfn/prior/bnn/` (currently just a sampler over ground-truth
functions — `BNNPrior.sample()` returns an MLP, nothing about a *pair* of
related tasks) into a real prior on the M2 contract, by choosing and
implementing one relatedness mechanism between A and B.

## The relatedness mechanism — pick one, don't build both speculatively

Two options were on the table; pick one for this milestone, note the other as a
future variant rather than building both now:

1. **Fixed weights + monotonic warp**: sample one BNN, keep its weights fixed
   for the pair, apply a monotonic input and/or output warp to get B — closest
   in spirit to the toy prior's warp/shift/scale, and the simpler of the two to
   validate against the M2 contract.
2. **Two BNN instantiations + interpolation**: sample two independent BNNs,
   sample `alpha ~ [0,1]`, interpolate between them (weight-space or
   output-space — decide and document which) so that `alpha` near 0 makes B
   almost entirely the first BNN (highly related, little novel information)
   and `alpha` near 1 makes B mostly the second (unrelated). This is more
   flexible (relatedness is a continuous knob, useful for later ablations) but
   more to get right first.

Recommendation if you want one: start with (1) — it reuses the toy prior's warp
machinery from M2 almost directly, so this milestone is mostly "swap the
ground-truth function generator from a toy function to a sampled BNN," which is
a much smaller step than building interpolation semantics from scratch. But
this is your call, not a default to accept unreviewed.

## Deliverables

1. `BNNPrior` (kept, active — `src/ppfn/prior/bnn/bnn_prior.py`) extended or
   wrapped to produce A/B pairs under the chosen relatedness mechanism.
2. `configs/prior/bnn.yaml`'s `dataset_class`/`dataloader_class` placeholders
   (currently `???` — see `.claude/rules/hydra.md`) filled in for real.
3. The M2 visualizer running against this prior.

## Acceptance criteria

- [ ] `configs/prior/bnn.yaml` composes and instantiates fully — no `???`
      placeholders remain in it.
- [ ] A test confirms the chosen relatedness knob actually changes measurable
      relatedness (e.g. correlation between A's and B's underlying function
      values at the same transported location) in the expected direction —
      not just that the code runs.
- [ ] The demo/visualizer produces a plot matching the M2 convention, and
      additionally shows what "more related" vs. "less related" pairs look
      like side by side (varying whatever knob the chosen mechanism exposes —
      warp strength, or `alpha`).
- [ ] `pytest` passes.

## Non-goals

- Not implementing both relatedness mechanisms — one, chosen deliberately.
- Not the ECDF-based BNN output calibration beyond what already exists in
  `bnn_prior.py` (`ensure_ecdf_loaded`) — reuse it, don't redesign it here.
