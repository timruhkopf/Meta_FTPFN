# M7 — Cross-model NAT-comparison callback

## Goal

A trainer callback that, given a model-under-test and the M5 marginal baseline
(loaded from checkpoint), evaluates both on **identical context** and reports
their predictive difference in nats — the apples-to-apples comparison your own
model (M9) and MTPFN (M6) both get checked against.

## Why this needs to be a callback, not ad-hoc eval code

You want this available for *every* model trained after this point, not a
one-off script — and it needs to run during/after training without becoming
part of the model or the trainer's core loop (see `.claude/rules/mlflow.md`'s
callback-return-a-dict pattern in `trainer.py`/`abstract_callback.py` for the
existing convention to follow).

## What "same context, no complexity difference" requires

1. **Identical input**: both models see exactly the same context points, same
   order (if order matters to either architecture), same query positions —
   construct one batch, feed it to both, don't sample twice.
2. **Identical bar distribution borders** (see M1's non-goal note — this is
   exactly where it becomes load-bearing): if the model-under-test and the
   marginal baseline were trained with different bucket borders, their NLLs
   aren't comparable numbers at all, regardless of how careful the rest of
   this callback is.
3. **Checkpoint provenance check**: per `.claude/rules/checkpoints.md`'s
   not-yet-enforced invariant — before comparing, verify the loaded marginal
   checkpoint's recorded prior config matches the current run's prior. This
   callback is the first real consumer of that invariant; implement the check
   here rather than deferring again.
4. **No leakage**: the marginal baseline must not be updated (frozen, `eval()`
   + `no_grad`) during this comparison — it's a fixed reference, not something
   this training run touches.

## Deliverables

1. A callback (`src/ppfn/trainer/callbacks/`) that loads a marginal-baseline
   checkpoint once, and on a configurable schedule (e.g. every N epochs, not
   necessarily every step) constructs a shared-context batch, evaluates both
   models, and logs the NAT difference — per-context-condition (`A|A`, `B|B`,
   `A|[A,B_inA]` from M5) where applicable to the model-under-test.
2. The checkpoint-provenance check from `.claude/rules/checkpoints.md`,
   implemented for real (not just documented) as part of this callback's setup.
3. A config entry under `configs/callbacks/` following the existing
   `mlflow`/`clip` pattern.

## Acceptance criteria

- [ ] Given two known models (e.g. the same marginal baseline checkpoint
      compared against itself), the reported NAT difference is ~0 — this is
      the correctness check that the "same context" machinery actually is
      identical, not approximately so.
- [ ] Given a deliberately mismatched prior config, the provenance check
      raises rather than silently comparing incomparable models.
- [ ] The callback logs per-condition NAT differences distinguishably (not
      one aggregate number that hides which regime is failing).
- [ ] `pytest` passes.

## Non-goals

- Not a general-purpose model-comparison framework — scoped to this specific
  marginal-baseline-vs-model-under-test comparison.
- Not deciding M9's architecture — this callback just needs *a* model to plug
  in once M9 exists; MTPFN (M6) or even M5's own baseline against itself is
  sufficient to build and validate it now.
