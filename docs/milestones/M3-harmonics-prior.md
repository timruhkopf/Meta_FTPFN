# M3 — Harmonics prior on the new contract

## Goal

Port the harmonics prior from `archive/` onto the M2 prior contract, so it's a
second real data point (after the toy prior) confirming the contract actually
fits something less trivial, and so its known issues get fixed in the same
pass rather than inherited silently.

## Context

`archive/src/ppfn/prior/harmonics/` (+ `harmonics_fix/`, an unreconciled fork —
see `archive/README.md`) has the closest existing precedent for this repo's
A/B-related-domains idea, including a working visualizer
(`heatmap_callback.py`) M2 already drew on. Known issue to resolve as part of
this port, not after: `configs/prior/harmonics.yaml`'s old `_target_`
(`prototype.harmonic_restart.harmonic_prior.InfiniteHarmonicsStream`) didn't
resolve to anything under `src/` — tracked in `docs/milestones/M0-repo-hygiene.md`. Since the
port rewrites the config anyway, decide explicitly whether `harmonics` or
`harmonics_fix` (or a merge) is the one going forward — don't silently pick.

## Deliverables

1. `src/ppfn/prior/harmonics/` reimplemented (or ported+adapted) against the
   M2 contract: same `{"train": {...}, "test": {...}}` shape, same `A_inB`/
   `B_inA` semantics, same `n_A`/`n_B`/`p`/`share_unrelated` config surface.
2. `configs/prior/harmonics.yaml` pointing at a real, resolving `_target_`,
   using the "meta info baked in" pattern.
3. The M2 visualizer running against this prior (subplots, heatmap, dense
   grid) — this is the actual test of whether the contract generalizes past
   the toy case.

## Acceptance criteria

- [ ] No reference to `prototype.harmonic_restart` remains anywhere in
      `configs/` or `src/`.
- [ ] `harmonics` vs `harmonics_fix` divergence is resolved or explicitly
      documented as intentional (update `docs/milestones/M0-repo-hygiene.md` accordingly —
      this milestone effectively closes out M0's open question).
- [ ] `pytest tests/prior/` (or wherever harmonics tests land) passes.
- [ ] The demo/visualizer produces a plot matching the M2 convention.

## Non-goals

- Not adding new harmonics features beyond what's needed to fit the contract.
- Not reconciling `harmonics`/`harmonics_fix` if the answer turns out to be
  "keep both for a documented reason" — a decision either way satisfies this
  milestone, an undocumented one doesn't.
