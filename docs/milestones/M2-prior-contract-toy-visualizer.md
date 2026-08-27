# M2 — Prior contract + toy warp/shift/scale prior + visualizer

## Goal

Establish the one interface every prior in this repo implements (toy, harmonics,
BNN, eventually multi-fidelity/real-benchmark ones), and build the simplest
possible real prior against it end-to-end, including the visual diagnostic that
gives confidence the data-generating process is doing what it claims.

## The prior contract

Every prior is a stream/iterable `torch.utils.data.Dataset` (matches
`cfg.prior.dataset_class` in `configs/prior/*.yaml`, consumed by
`cfg.prior.dataloader_class` — see `configs/prior/bnn.yaml`'s current
placeholders) that produces two related datasets A and B:

- A and B share an underlying effect, but B's domain is a warped / shifted /
  scaled / otherwise distorted version of A's domain.
- Raw datapoints for A and B are sampled **independently** — no itemwise
  pairing is implied.
- Config controls: `n_A`, `n_B`, problem dimension `p`, and
  `share_unrelated` — the fraction of batch items where B is a fully
  resampled, unrelated draw (no shared effect at all; must support `0`).
- Because the prior *knows* the ground-truth transform for related pairs, it
  can additionally produce `A_inB` and `B_inA` — the transported data (A
  mapped through the transform into B's domain, and vice versa). These are
  only meaningful for related pairs (not the `share_unrelated` ones) — decide
  and document what an unrelated pair's `A_inB`/`B_inA` contains (e.g. `NaN`s,
  a zero-length placeholder, or simply not present) rather than leaving it
  implicit.
- `__getitem__`/iteration yields a dict shaped:
  ```python
  {
      "train": {"A": (X_A, y_A), "B": (X_B, y_B), "A_inB": (...), "B_inA": (...)},
      "test":  {"A": (X_A_test, y_A_test), "B": (X_B_test, y_B_test)},
  }
  ```
  Pin down exact key names and tensor shapes here — every downstream baseline
  (M5, M6) and the eventual model (M9) depends on this not changing later.
- **Ragged shapes — decided (2026-08-27): padding + mask, on the train/context
  section only, via one shared `collate_fn`.** Since A/B are independently
  sampled and `share_unrelated` varies per batch item, `n_A`/`n_B` differ
  across a batch. Rather than each prior producing its own already-padded
  fixed-shape tensors, a `collate_fn` sits between the prior's per-item
  (ragged) output and the `DataLoader`'s batched tensors — build it once (see
  `.claude/rules/model-prototyping.md`'s `collate.py`), reused by every
  consumer (M5's baseline, M9's model), not reimplemented per model. See
  `docs/milestones/M9-proposed-model.md`'s padding section for the
  attention-masking implications this has on `AlongColumnAttention`'s
  `single_eval_pos` masking — padding must not silently leak into attention.

## The toy prior (this milestone's concrete deliverable)

A regression prior with one shared underlying effect (e.g. a random function
from a simple family — polynomial, sum of sinusoids, GP draw), where A is
anchored (identity domain) and B's domain is produced by applying a
warp/shift/scale transform to A's domain before evaluating the same effect.
Simple by design — this is the template the harmonics (M3) and BNN (M4) priors
get adapted to match, not a permanent toy to build features on top of.

## The visualizer

Per `.claude/rules/research-demos.md`: subplots for A and B, sampled points +
ground-truth function, binned predictive density as a heatmap over a dense
regular grid spanning the full domain, for a marginal model's predictions on
each of A and B. Reference `archive/src/ppfn/prior/harmonics/heatmap_callback.py`
for prior art on this exact kind of plot (built for the harmonics prior
originally) — adapt the pattern, don't assume it drops in unchanged against the
new contract.

## Acceptance criteria

- [ ] The prior contract is written down (a short doc or a docstring on an
      abstract base class under `src/ppfn/prior/`) that M3/M4 can be checked
      against.
- [ ] Toy prior implements the contract, is a real `IterableDataset`, and
      `configs/prior/toy.yaml` (or similar) wires it in with the "meta info
      baked in" pattern from `.claude/rules/hydra.md` (`num_inputs`, `p`, etc.
      as siblings of the `_target_` block).
- [ ] `share_unrelated=0` and `share_unrelated=1` are both exercised by a test
      and behave as documented.
- [ ] `dataset_class`/`dataloader_class` in `configs/prior/toy.yaml` actually
      instantiate and a `DataLoader` over them yields batches of the documented
      shape — replaces the `???` placeholder pattern for at least this prior.
- [ ] The shared `collate_fn` exists, pads the train/context section only, and
      produces a mask that M5/M9 can use — a test constructs a batch with
      deliberately mismatched `n_A`/`n_B` per item and checks the padded output
      shape and mask are correct.
- [ ] The visualizer runs as a demo block and produces the described plot for
      the toy prior.
- [ ] `pytest` passes.

## Non-goals

- Not making the toy prior configurable in every dimension the harmonics/BNN
  priors will eventually need — it should be genuinely simple.
- Not wiring the visualizer into the trainer as a callback yet (that can come
  later, once a real model exists to visualize) — a standalone demo script is
  sufficient here.
- Not `configs/prior/bnn.yaml`'s `dataset_class`/`dataloader_class` — that's M4.
