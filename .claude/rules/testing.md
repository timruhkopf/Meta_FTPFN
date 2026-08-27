---
paths:
  - "tests/**"
---

# Testing conventions

`tests/` was rebuilt from scratch on 2026-08-27 alongside `configs/` — currently
just `test_configs.py` (config-composition sanity) + `conftest.py`
(`register_resolvers()` + a `compose_cfg` fixture wrapping `hydra.compose`). The
previous, larger suite (unit tests mirroring `src/ppfn/model|prior/`, plus
`test_integration.py`) is parked at `archive/tests/` — real precedent for shape
and style once model/prior code comes back from `archive/`, not something to
blindly restore piecemeal.

No `[tool.pytest.ini_options]` section exists in `pyproject.toml` yet — pytest
runs with defaults (`pytest` from repo root discovers everything under `tests/`).
`ppfn` is editable-installed, so no `sys.path` hacks are needed in test files.

## Config-composition tests (the current pattern — see `tests/test_configs.py`)

- Compose through `hydra.initialize`/`compose`, not by hand-building a
  `DictConfig` — that's what actually exercises the defaults-list wiring
  (`configs/config.yaml`'s group selections, `# @package _global_` overrides).
- `model`/`benchmark` are mandatory-but-unset (`???`); every compose call needs
  `~model ~benchmark` to opt out (see `.claude/rules/hydra.md`).
- Test placeholders (`???` fields like `prior.dataset_class`,
  `trainer.trainer_class.criterion`) by asserting they raise
  `omegaconf.errors.MissingMandatoryValue` — that's the intended behavior right
  now, not a bug to work around. Once one is filled in, replace that test with a
  real instantiation test rather than deleting the coverage.
- For anything that actually instantiates cheaply (a prior, an optimizer/
  scheduler partial, a callback), call `hydra.utils.instantiate` on it for real
  and assert on the resulting object — see `test_prior_bnn_instantiates`,
  `test_scheduler_instantiates` for the pattern. `BNNPrior` loads a cached ECDF
  file (`src/ppfn/prior/bnn/prior_ecdf/`) rather than regenerating it, so this
  stays fast; don't assume every prior/model will be this cheap to instantiate.

## Unit / integration tests (once model/prior code returns)

Follow the archived precedent in `archive/tests/`: build the smallest real
object you can rather than mocking it (small tensors, tiny batches, a
`device` fixture that falls back to CPU so tests pass on CPU-only machines),
and assert on shapes/dtypes/specific values over "it didn't throw."
`archive/tests/conftest.py` and `archive/tests/test_integration.py` show the
shape this took last time (a `mybatch`/`ft_batch_factory` fixture, a
parametrized full-model integration test) — pull the pattern, not necessarily
the exact fixtures, since the model code they were built against may not come
back unchanged.

## Smoke-testing training runs

There's no pytest wrapper around a real Hydra run. The smoke test for
trainer/config wiring is a short real invocation:

```bash
python src/train.py experiment_name=00-debug-smoke ~model ~benchmark trainer.epochs=1 trainer.steps=5
```

Use this (or something similarly short) whenever a change touches `train.py`,
`configs/`, the trainer, or callbacks — a config that fails to instantiate, or a
callback that throws, often won't be caught by unit tests alone since those don't
exercise the full `instantiate(cfg...)` chain. Right now this will fail at
`cfg.prior.dataset_class` (expected — see `CLAUDE.md`); once that's filled in,
check the run actually reaches `on_train_end`/`log_on_train_end` (i.e. doesn't
crash mid-epoch), not just that the process exits 0 — `PPFNTrainer.fit` swallows
`KeyboardInterrupt`/`GracefulExit` in its `finally` block, so a hung or
SIGTERM'd run can still look clean on exit code.
