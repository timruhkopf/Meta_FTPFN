---
paths:
  - "configs/**"
  - "src/train.py"
  - "src/**/train.py"
---

# Hydra conventions

This repo uses file-based Hydra config groups (`configs/*/`), not structured
configs / a `ConfigStore` — don't introduce dataclass configs unless asked, it'd be
a second, inconsistent way of doing the same thing.

## The DictConfig boundary

`src/train.py::run()` is the only place that should touch `cfg` as a `DictConfig`.
Everything below it (`PPFNTrainer`, models, priors, callbacks) takes plain
objects/values, built via `hydra.utils.instantiate`. Concretely, the existing
pattern is:

```python
model = instantiate(cfg.model.model_class).to(device)
optimizer_partial = instantiate(cfg.optimizer)   # partial, called later with params
trainer = instantiate(cfg.trainer.trainer_class, model=model, optimizer=optimizer_partial, ...)
```

If you find yourself passing `cfg` (or a sub-tree of it) into a function under
`src/ppfn/`, stop — either resolve the values you need at the call site in
`train.py`/`run()`, or add a small typed argument to the callee. The cost of
violating this is that unit-testing that function then requires constructing a
full Hydra config tree instead of passing a dict/dataclass.

Exception: `trainer.config = OmegaConf.to_container(cfg, resolve=True)` in
`train.py` — that's deliberately a plain resolved dict stored for
logging/provenance (see `mlflow.md`), not a live `DictConfig` being threaded through.

## `_target_` / `instantiate` conventions

- Config groups mirror `src/ppfn/` roughly 1:1: `configs/model/`, `configs/prior/`,
  `configs/optimizer/`, `configs/scheduler/`, `configs/trainer/`, `configs/callbacks/`.
- Optimizer/scheduler configs use `instantiate` to produce a **partial** (the class
  isn't given `params`/`optimizer` yet — `PPFNTrainer.__init__` calls
  `optimizer(trainable_params)` itself). Keep that convention when adding new
  optimizers/schedulers; don't fully instantiate them in config.
- `configs/callbacks/*.yaml` entries become the `callbacks` dict passed into the
  trainer (`trainer.trainer_class.callbacks: ${callbacks}` in `config.yaml`) — keys
  there are the callback names used for lookup, values are `_target_`ed callback
  instances.
- Custom OmegaConf resolvers (`mod`, `div`, `add`, `mul`, `githash`,
  `get_git_branch`) live in `src/ppfn/utils/resolvers.py::register_resolvers()`
  — idempotent, safe to call from any entry point (train.py's `__main__`,
  `tests/conftest.py`, a future `evaluate.py`). Add new ones there, not inline
  in `train.py` or scattered across configs.

## Meta info baked into config, not just `_target_` + args

Where a config produces something whose shape other configs need to know about
(dimensionality, task count, ...), keep that as a plain sibling field next to the
`_target_` block, and have the `_target_` block's own args interpolate it —
don't just bury it inside the constructor args where nothing else can reach it.
`configs/prior/bnn.yaml` is the reference example:

```yaml
num_inputs: 8
num_outputs: 1

prior_class:
  _target_: ppfn.prior.bnn.bnn_prior.BNNPrior
  num_inputs: ${prior.num_inputs}
  num_outputs: ${prior.num_outputs}
```

A model config that needs the prior's output dimensionality should read
`${prior.num_outputs}`, never hardcode a second copy of the number.

## Deployment (local/slurm via `hydra-submitit-launcher`)

`configs/deployment/{local,slurm}.yaml` are `# @package _global_` files that
`override /hydra/launcher` to `submitit_local`/`submitit_slurm` (the plugin is
a real dependency — `hydra-submitit-launcher` in `pyproject.toml`). Selected via
`configs/experiment/*.yaml`'s own defaults list (`/deployment: local` or
`slurm`), not passed directly on the command line in normal use. Field names
must match the plugin's actual dataclasses
(`hydra_plugins.hydra_submitit_launcher.config.{LocalQueueConf,SlurmQueueConf}`,
installed in `.venv`) — don't invent a field name without checking there first;
Hydra's structured-config validation will reject an unknown one at compose time,
which is a good thing (fail at compose, not mid-job).

This replaces the old `configs/dispatch/` group name — `dispatch/local.yaml` was
previously empty (local runs bypassed submitit entirely); `deployment/local.yaml`
now goes through `submitit_local` too, so local and SLURM stay on the same code
path.

## `model` / `benchmark` are mandatory-but-unset (`???`)

No active implementation exists for either (see `configs/model/README.md`,
`configs/benchmark/README.md`) — every `compose()`/CLI invocation needs
`~model ~benchmark` (or a real config once one exists) or Hydra fails loudly at
compose time. Don't "fix" this by giving them a dummy default just to make a run
go further; that would hide the actual gap.

## Experiment naming

`experiment_name` follows a fixed prefix convention enforced by
`assert_clean_tree_for_real_runs` (`src/ppfn/utils/git_tools.py`):
`00-debug-*` (dirty tree OK, use while iterating), `01-pretraining-*`,
`02-baseline-*`, `03-sweep-*` (`02-*`/`03-*` require a clean git tree — the
git SHA logged to MLflow must be trustworthy). Don't add new prefixes without
updating that check.

## `harmonics` prior is parked, not gone

`configs/prior/harmonics.yaml` and the code it targeted are both in
`archive/` (see `archive/README.md`) — the dangling `_target_` bug tracked in
`docs/milestones/M0-repo-hygiene.md` still applies whenever that comes back, it just isn't
live right now. `prior: bnn` is the only real option today.
