# Meta-FTPFN

PPFN: batch-awareness in Prior-Fitted Networks (PFNs) for meta-task learning, built
on top of `ifBO`/PFNs4HPO (freeze-thaw PFN). Research code, actively changing shape.

**Current state:** reset to a bare skeleton on 2026-08-26 (models, other priors,
benchmarks, all tests, all Hydra configs moved to `archive/` — see
`archive/README.md`), then on 2026-08-27 `configs/` and `tests/` were rebuilt
fresh alongside it, matching the kept skeleton rather than restoring the archived
ones. **There is still no active model or training objective** —
`configs/config.yaml` declares `model`/`benchmark` as mandatory-but-unset (`???`)
and `configs/prior/bnn.yaml`/`configs/trainer/default.yaml` each have one
placeholder field (`prior.dataset_class`/`dataloader_class`, `trainer.criterion`)
for the same reason: real implementation exists (`BNNPrior`, `PPFNTrainer`), but
the glue code that would make training actually run doesn't yet. Verified
end-to-end: `python src/train.py experiment_name=00-debug-x ~model ~benchmark`
composes the full config and fails exactly at `cfg.prior.dataset_class` — that's
the expected, honest failure point, not a bug.

## Environment & commands

- Package/env manager: `uv` (see `pyproject.toml`). Python 3.10, editable-installed
  (`import ppfn` works from anywhere, no `sys.path` hacks needed).
- Tests: `pytest` from repo root (no `pytest.ini`/config section yet). Currently
  just `tests/test_configs.py` — config-composition sanity checks, see
  `.claude/rules/testing.md`. The old suite is parked in `archive/tests/`.
- Train: `python src/train.py experiment_name=00-debug-<name> ~model ~benchmark`
  — the `~model ~benchmark` opt-outs are required until those groups have a real
  config (see `configs/model/README.md`); even then it currently fails at
  `cfg.prior.dataset_class` (no stream/dataset wrapper around `BNNPrior` yet).
  Real (non-debug) runs must be from a clean git tree —
  `assert_clean_tree_for_real_runs` in `src/ppfn/utils/git_tools.py` raises
  otherwise for `experiment_name` starting with `02-baseline`/`03-sweep`.
- Custom OmegaConf resolvers (`${mul:...}`, `${githash:...}`, etc.) live in
  `src/ppfn/utils/resolvers.py::register_resolvers()` — call it before composing
  any config outside of `python src/train.py` (tests do this in `conftest.py`).
- `.env` holds `MLFLOW_TRACKING_URI` / `ROOT`, loaded by `src/ppfn/piplelines/train.py` — don't
  commit secrets there, and don't assume it's present in a fresh checkout.
- Lint/format: `ruff` is a dev dependency; a `.claude/settings.json` hook auto-runs
  `ruff format` on Python files Claude edits. No `ruff` config section in
  `pyproject.toml` yet, so it runs with defaults.

## Layout (active tree)

- `src/ppfn/trainer/` — `PPFNTrainer` + callback system (`trainer/callbacks/`:
  `abstract_callback`, `checkpoint`, `mlflow_cb`, `grad_clipping`,
  `early_stopping`, `meta_test`, `hefty_meta_test`). Kept intact, imports clean
  with nothing else in the active tree.
- `src/ppfn/prior/bnn/` — the one prior kept active (`bnn_prior.py`, `mlp.py`).
  `src/ppfn/prior/__init__.py` is now empty (it used to re-export
  `multi_fidelity` symbols — those moved to `archive/`, so the re-export would
  have broken the package on import; don't restore it without also restoring
  `multi_fidelity`).
- `src/ppfn/utils/git_tools.py`, `gracefull_exit.py` — cross-cutting, kept as-is.
  `mybatch.py` and `deprecate.py` moved to `archive/` (only used by archived code).
- `src/ppfn/piplelines/train.py` — Hydra entry point. Reads `cfg.prior.dataset_class`/
  `dataloader_class` (fixed 2026-08-27 — used to read `cfg.dataset.*`, a
  pre-existing mismatch with every `config.yaml` this repo has had).
- `configs/` — rebuilt 2026-08-27 to match the kept skeleton: `experiment/`,
  `deployment/` (local/slurm via `hydra-submitit-launcher`, replaces the old
  `dispatch/` group name), `prior/` (`bnn.yaml`, real), `trainer/`, `optimizer/`,
  `scheduler/`, `callbacks/` (real), `model/` + `benchmark/` (empty, `???`,
  README explains why and the convention to follow once filled in).
- `tests/` — rebuilt 2026-08-27, currently just `test_configs.py` + `conftest.py`.
- `archive/` — everything from before the 2026-08-26 reset, see `archive/README.md`.
- `docs/ROADMAP.md` + `docs/milestones/*.md` — active work, see below.

## Path-scoped rules

Loaded automatically when you touch matching files — see `.claude/rules/`:
- `hydra.md` — the DictConfig boundary, `_target_`/`instantiate` conventions, the
  "meta info baked in" pattern (`configs/prior/bnn.yaml`), and the
  `hydra-submitit-launcher` deployment setup.
- `mlflow.md` — run/tag/param conventions and current callback-owns-lifecycle
  debt in `trainer/callbacks/mlflow_cb.py` (kept, active).
- `checkpoints.md` — checkpoint dict contract (the trainer.py vs. CheckpointCallback
  schema mismatch — both still active) and the frozen-PFN/prior-match invariant
  (relevant again once archived model/prior code returns).
- `testing.md` — unit vs. integration vs. smoke-run expectations, and the current
  config-composition test pattern in `tests/test_configs.py`.

## Working agreements

- State goals and invariants, not verification checklists — don't add "always
  double check" style instructions; they cause over-verification on later steps.
- Prefer small, runnable checks (a `00-debug-*` run, a targeted `pytest`) over
  reasoning about correctness in the abstract — this is numerical/ML code where
  silent wrongness (e.g. a prior mismatch) doesn't show up as an exception.
- Don't touch `data/`, `external/`, `.venv/`, `outputs/`, `mlruns/` — enforced by a
  hook, but don't try to route around it either.
- Don't pull things out of `archive/` speculatively — wait to be asked, or for a
  milestone that names the specific piece needed.
- Real findings (non-obvious bugs, "tried X, it didn't work, here's why," a
  verified fix) get a `docs/labbook/` entry, not just a chat message — see
  `.claude/rules/labbook.md`. Append-only; a wrong first attempt gets recorded
  as wrong, not silently rewritten.
