---
paths:
  - "src/ppfn/trainer/callbacks/checkpoint.py"
  - "src/ppfn/trainer/trainer.py"
  - "src/ppfn/model/**"
---

# Checkpoint conventions

## There are currently two different checkpoint formats — know which one you're touching

- `PPFNTrainer._save_checkpoint`/`load_checkpoint` (`trainer.py`): a minimal dict
  — `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`,
  `best_loss`. Saved synchronously to a single fixed filename.
- `CheckpointCallback` (`checkpoint.py`): a richer, async-saved snapshot — adds
  `eon`, `global_step`, `scaler_state_dict` (if AMP), a `metrics` dict, a JSON
  sidecar, and uploads to MLflow as artifacts on `on_train_end`. This is the one
  actually wired up via `configs/callbacks/`.
- These are **not the same schema** (`best_loss` vs `best_score`, no `eon`/
  `global_step`/sidecar in the trainer's own method) and a checkpoint written by
  one is not loadable by the other's `load_checkpoint`/`on_trainer_init`. If you're
  adding checkpoint-consuming code (e.g. loading a frozen PFN for the anamorphic or
  mtpfn baseline models), check which path actually produced the file you're
  loading — don't assume. This divergence is itself worth resolving at some point;
  it isn't currently tracked as a milestone, so flag it if it blocks you rather
  than picking one silently.

## Frozen-backbone / trainable-params convention

`PPFNTrainer.__init__` does:
```python
if hasattr(model, 'get_trainable_params'):
    trainable_params = model.get_trainable_params(optimizer.keywords['weight_decay'])
else:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
```
A model wrapping a frozen pretrained backbone (frozen PFN + trainable
cross-attention/adapter layers, as in the anamorphic/PPFN designs) should:
- Set `requires_grad = False` on the frozen backbone's parameters at construction
  time (not rely on the optimizer to skip them — `eval()` mode alone does not
  freeze weights against gradient updates).
- Either implement `get_trainable_params(weight_decay)` to return proper param
  groups (needed if the backbone and the new layers should get different weight
  decay treatment), or ensure plain `p.requires_grad` filtering is sufficient.
- When adding a test for a new such model, assert directly that no frozen
  parameter appears in the optimizer's param groups — this is the kind of bug
  that trains fine and reports plausible-looking numbers while being silently
  wrong (the "backbone" is quietly being finetuned, or vice versa).

## Invariant to establish, not yet enforced: prior/checkpoint provenance

Nothing currently records *which prior config* produced the data a checkpointed
PFN was trained on. Given the live `harmonics` vs `harmonics_fix` fork
(`CLAUDE.md` "Known messy state"), a checkpoint trained under one and then reused
downstream (e.g. as a frozen backbone for `anamorphic`/`mtpfn`, or for
fine-tuning) against the other is a real, silent-failure risk: it will load,
train, and produce plausible-looking metrics that mean nothing, because the
model's learned prior no longer matches what it's being evaluated/conditioned
against downstream. No existing metric catches this.

If you're building anything that consumes a checkpoint trained elsewhere in this
repo (not just resuming the same run), consider whether it needs a startup check
comparing the checkpoint's recorded prior config (once one exists — see
`docs/milestones/M0-repo-hygiene.md`'s open question on `harmonics`/`harmonics_fix`) against
the current run's `cfg.prior`, and raising loudly on mismatch rather than
training through it. This isn't implemented yet — don't assume a check like this
exists just because it should.
