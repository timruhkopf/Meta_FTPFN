# M5 — Marginal single-task PFN baseline

## Goal

Train a reference "how good can a single-task PFN get" baseline, using the
installed TabPFN v2.5 architecture, on whichever prior is ready first (toy at
minimum, ideally harmonics/BNN too) — this is the number everything else
(including your own model, M9) gets checked against for negative transfer.

## What "marginal" means here

Per your spec, the model is trained on three conditioning regimes, all against
the M1 bar distribution:

- `A_test | A` — predict A's held-out points from A's own context only.
- `B_test | B` — same, for B.
- `A_test | [A, B_inA]` — predict A's held-out points from A's own context
  *plus* B's data transported into A's domain (`B_inA`, from the M2 contract).

This third regime is the actual point: it's the upper bound on how much a
model *could* exploit related-task information if it perfectly knew the
transform (since `B_inA` is prior-computed using ground truth, not learned).
Comparing against it is what makes "information loss" measurable in M9/M6.

## Deliverables

1. A thin wrapper around `tabpfn.architectures.tabpfn_v2_5` (imported, not
   reimplemented — same reasoning as M1) configured as the marginal model,
   trained via `PPFNTrainer` against the M1 criterion.
2. A training run (or `configs/experiment/`) that trains this to convergence
   on at least the toy prior — pin down what "converged" means concretely
   (e.g. held-out NLL plateau over N epochs, or a fixed large budget with a
   convergence check) before calling this done; an undertrained baseline
   invalidates every comparison built on top of it.
3. The information-loss study: subset `B_inA` from 0%–100% of available
   samples (context masking at inference on the trained model — not retraining
   per subset percentage, unless the retrain-per-subset numbers turn out to
   diverge meaningfully from masking, in which case document why both exist)
   and report the resulting NLL degradation curve, expressed as "average
   number of A-context samples this many B_inA samples were worth."
4. A saved checkpoint, loadable per the contract in `.claude/rules/checkpoints.md`
   — this checkpoint is what M7's cross-model callback loads for comparison,
   so get the prior-provenance tagging right here rather than retrofitting it.

## Acceptance criteria

- [ ] Model trains and its held-out NLL on `A_test | A` improves over an
      untrained baseline (sanity: it's actually learning).
- [ ] All three conditioning regimes are exercised by the training/eval code
      and produce distinguishable NLL numbers.
- [ ] The information-loss-vs-`B_inA`-fraction curve is produced and plotted.
- [ ] Checkpoint saves and reloads to identical eval NLL.
- [ ] Uses the shared `collate_fn` from M2 (padding on the train/context
      section only) and a test confirms a padded, ragged-`n_A`/`n_B` batch
      produces the same per-item output as an unpadded equivalent — this
      baseline needs the same padding correctness M9's model needs, not a
      separate implementation (see `docs/milestones/M9-proposed-model.md`'s
      padding section).
- [ ] `pytest` passes; a demo per `.claude/rules/research-demos.md` shows a
      forward pass through the wrapped model.

## Non-goals

- Not MTPFN (M6) or the cross-model callback (M7) — this milestone produces
  the checkpoint those consume, nothing more.
- Not every prior from M2–M4 — get this working on one prior first (toy),
  extend to harmonics/BNN once the mechanics are right.
