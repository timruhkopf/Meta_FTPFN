# Archive

Parked work-in-progress, moved here in one pass on 2026-08-26 to reset the active
tree to a bare skeleton: `src/ppfn/trainer/`, `src/ppfn/prior/bnn/`, `src/ppfn/piplelines/train.py`,
and `src/ppfn/utils/{git_tools,gracefull_exit}.py` stayed in place; everything else
— models, other priors, benchmarks, all tests, all Hydra configs — moved here
verbatim. Nothing was deleted or rewritten during the move; layout mirrors the
original repo root (`archive/src/...`, `archive/tests/...`, `archive/configs/...`)
so pulling something back is a plain `git mv archive/<path> <path>` (plus whatever
config/import wiring it needs at the new location — check `CLAUDE.md`'s rules for
the current state of the kept skeleton before wiring it back in).

## What's in here

- `src/ppfn/model/` — `mymodel/` (the PPFN wrapper+stream design with
  `test_integration.py`-style coverage before the move), `anamorphic/` (block
  design exploration — see `anamorphic/ideas/*.md` for the competing proposals:
  BlockDesign, JEPA v1/v2, Perceiver v1/v2, VLA_JEPA — none settled as "the"
  design as of the move), `baselines/mtpfn/` (comparison model + BO harness).
- `src/ppfn/prior/harmonics/` and `harmonics_fix/` — an unreconciled fork. Treat
  a PFN trained against one as **not** interchangeable with the other; nothing
  enforces this automatically.
- `src/ppfn/prior/multi_fidelity/` — freeze-thaw/multi-fidelity prior variants
  (`mf_ftpfn`, `mf_ftpfn_refactor`, `meta_batch`).
- `src/ppfn/prior/warp.py` — used only by `harmonics/stream_dataset.py`.
- `src/ppfn/bench/` — `mfpbench`/`bo` benchmark harnesses, used by the baseline.
- `src/ppfn/utils/mybatch.py` — the `MyBatch` class; used throughout the archived
  model/prior code above and by most of the archived tests. Not used by anything
  in the kept `trainer/`.
- `src/ppfn/utils/deprecate.py` — used only by `multi_fidelity/mf_ftpfn_refactor`.
- `tests/` — the full previous suite, unpartitioned (including `tests/test_trainer/`,
  which exercises the kept `trainer/callbacks/checkpoint.py` and
  `model/mymodel/multistream_objective.py` — the latter is archived, so that test
  won't run standalone without pulling the model code back too).
- `configs/` — every Hydra config group, including `trainer/`, `optimizer/`,
  `scheduler/`, `callbacks/` (which *do* pair with the kept trainer code) and
  `config.yaml` itself. **`src/ppfn/piplelines/train.py` currently has no config to run against**
  — `config_path="../configs"` points at nothing until at least `config.yaml` +
  a `model` + a `prior` + `trainer`/`optimizer`/`scheduler`/`callbacks` config are
  copied back. That's intentional, not a bug from the move.

## Known issues at time of archiving (still true, just no longer "live")

- `configs/prior/harmonics.yaml`'s `_target_`
  (`prototype.harmonic_restart.harmonic_prior.InfiniteHarmonicsStream`) does not
  resolve against anything under `src/` — stale, likely from an earlier refactor.
  Fix this *before* wiring `harmonics` back in, not after.
- `trainer.py`'s own `_save_checkpoint`/`load_checkpoint` and `CheckpointCallback`
  in `trainer/callbacks/checkpoint.py` (kept, not archived) use two different,
  incompatible checkpoint dict schemas — relevant again the moment archived model
  code needs to load a checkpoint produced by either path.

## Reintroducing something

1. `git mv archive/<path> <path>` for the code itself.
2. Pull back the matching `archive/configs/...` group(s) it needs.
3. Check `.claude/rules/*.md` in the active tree — they describe the kept
   skeleton's conventions (DictConfig boundary, MLflow pattern, checkpoint
   contract); update them once real config groups exist again so `paths:` scoping
   still matches something.
4. Pull back the matching `archive/tests/...` only if you want that coverage
   restored now — it's fine to leave it parked until the code it tests is back.
