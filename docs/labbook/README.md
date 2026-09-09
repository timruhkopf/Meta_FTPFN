# Lab book

Append-only log of findings and ideas followed up on. See
`.claude/rules/labbook.md` for the convention. New entries go at the bottom;
never edit or remove a past line — if something here turns out to be wrong,
add a new entry saying so and link back.

## Entries

- **2026-08-27** — [Harmonics prior: domain-drift warp leaked the shift through B's raw x-coordinates](2026-08-27-harmonics-prior-domain-leakage.md). Confirmed in both archived forks; fixed via a domain-preserving Kumaraswamy warp during the `harmonics_fix` port. `commit: pending`
- **2026-08-27** — [BNN prior: `init_std` sampled independently of network width caused most sampled functions to look flat](2026-08-27-bnn-prior-init-std-width-coupling.md). Includes a first fix attempt that was verified to make things *worse* before landing on the right calibration. `commit: pending`
- **2026-08-28** — [ifBO's `DatasetPrior` MLP: same shared lineage (same unfixed init_std/width bug), but used as a multi-channel correlated seed source for learning-curve shape parameters, not the function itself](2026-08-28-ifbo-mlp-lineage-and-seed-source-pattern.md). Not implemented — flags a possible alternative to M4's weight-interpolation relatedness idea. `commit: pending`
- **2026-09-09** — [First implementation of the prior, encoder-decoder architecture, and training pipeline](2026-09-09-first-prior-encoder-decoder-training-pipeline.md). `s_max=0.1` calibration for the warp rejection band; a caught-before-shipping bug where oracle teacher-forcing didn't actually reach the cross-attention queries; design decisions made where the spec was ambiguous. `commit: pending`
