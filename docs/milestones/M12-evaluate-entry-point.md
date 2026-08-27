# M12 — Evaluation entry point (`evaluate.py`)

## Status: sketch — blocked on M5–M7 producing checkpoints + the comparison callback

## Goal

A second Hydra entry point, alongside `src/train.py`, for loading a checkpoint
and running it against held-out priors/benchmarks — this doesn't exist yet, and
`src/train.py` isn't the right place for it (it's shaped around training a run
that logs to MLflow as it goes, not around loading something already-trained).

## Why this is a real gap, not just "call train.py differently"

1. **Checkpoint/config provenance must be checked, not assumed.** Per
   `.claude/rules/checkpoints.md`'s not-yet-enforced invariant (and M7's
   callback, which implements it for the training-time comparison case) —
   `evaluate.py` is the other place this matters: loading a `bnn`-trained
   checkpoint and evaluating it against `harmonics` data should fail loudly,
   not produce a plausible-looking wrong number.
2. **It isn't a training run.** `MLflowCallback` (`.claude/rules/mlflow.md`)
   owns a run's lifecycle assuming it's a training run — an eval pass needs
   its own MLflow run (or none at all, or logging as a child/tagged run under
   the training run it's evaluating) without dragging in the callback's
   training-specific setup (epoch/step counting, checkpoint-saving).
3. **`PPFNTrainer` isn't built for eval-only.** Loading a model, running
   forward passes, and computing metrics without a training loop, optimizer,
   or scheduler around it is a different code path — decide whether this
   reuses pieces of `PPFNTrainer` or is genuinely separate.

## Deliverables (once unblocked)

1. `src/evaluate.py` — Hydra entry point mirroring `src/train.py`'s structure
   where it makes sense (device setup, resolver registration) and diverging
   where it shouldn't (no trainer/callback-handler training loop).
2. The checkpoint-provenance check, shared with M7's callback rather than
   reimplemented — extract it once both consumers exist.
3. A `configs/evaluate.yaml` (or similar) root config, separate from
   `configs/config.yaml`, since the two entry points have genuinely different
   required config shape (a checkpoint path instead of a full model/optimizer/
   scheduler stack).

## Acceptance criteria (once unblocked)

- [ ] `python src/evaluate.py checkpoint=<path> prior=<name>` runs a trained
      checkpoint against a prior and reports bar-distribution NLL.
- [ ] Loading a checkpoint against a mismatched prior config raises, per the
      provenance check.
- [ ] Does not require `MLflowCallback`'s training-run assumptions to be
      satisfied (no fake epoch/step bookkeeping just to make the callback happy).

## Non-goals

- Not real-benchmark evaluation specifics (HPOBench/mfpbench harnesses) — that's
  M10's job to define; this milestone is the entry-point mechanism only.
