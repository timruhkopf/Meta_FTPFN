# M11 — Config completion for real training runs

## Status: sketch — blocked on M2–M6 field names existing

## Goal

Once priors (M2–M4), the marginal baseline (M5), and MTPFN (M6) exist, give each
a real `configs/` entry following the conventions already established
(`configs/prior/bnn.yaml`'s "meta info baked in" pattern, `configs/model/README.md`'s
convention for when a model shows up) — so every checkpoint-producing run comes
from a named, versioned config, not ad-hoc CLI overrides that can't be
reproduced later.

## Deliverables (once unblocked)

1. `configs/prior/{toy,harmonics,bnn}.yaml` — replacing today's placeholders.
2. `configs/model/{marginal,mtpfn}.yaml` (and your own model, once M9 exists).
3. `configs/benchmark/*.yaml` once M10 is unblocked.
4. `configs/experiment/*.yaml` per distinct training run shape (not one config
   endlessly overridden via CLI flags — see `.claude/rules/hydra.md`'s
   `experiment_name` prefix convention for how these get named/tracked).

## Acceptance criteria (once unblocked)

- [ ] No `???` placeholders remain in `configs/config.yaml`'s defaults for any
      milestone that's actually done.
- [ ] Every config group member instantiates via `hydra.compose` +
      `instantiate` (extend `tests/test_configs.py`'s pattern).
- [ ] Each real training run this repo has produced a checkpoint for has a
      corresponding checked-in `configs/experiment/*.yaml` that reproduces it.

## Non-goals

- Not starting before M2–M6 produce real field names/classes to configure —
  writing these configs against interfaces that don't exist yet means
  rewriting them once those interfaces land for real.
