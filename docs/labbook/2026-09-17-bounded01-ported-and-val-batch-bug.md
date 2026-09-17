# Ported the [0,1]-normalized prior into this branch; found and fixed a real val-batch bug

`commit: pending`

## What this is

The user's `ppfn.prior.lupi` `[0,1]`-normalization work
(flagged in-progress in
[2026-09-17-prior-01-normalization-and-real-run-prep.md](2026-09-17-prior-01-normalization-and-real-run-prep.md))
was built and wrapped up in the sibling worktree (`Meta_FTPFN`,
`lupi-iterative-registration`), uncommitted there. Per the user's direction
("the other session is currently building it into its training pipeline;
and so should you"), ported the shared prior/calibration layer into this
branch and wired `bounded01` into `FlowMatchingRegistrationPFN`.

## What the prior change actually does (read from the sibling's diff)

`ppfn.prior.lupi.sampler.sample_pair(..., bounded01=True)` (now the
`configs/prior/lupi.yaml` default): every reported value lands in `[0,1]`
BY CONSTRUCTION, the way `BNNPrior`'s own output already does, while `T`/`h`
keep real, independently-sweepable variation:

- **New module `ppfn.prior.lupi.ecdf`**: `normalize_f_probe` self-normalizes
  `f`'s raw output (wild per-draw location/scale, measured: per-draw mean
  -3.96 to +6.05, std varying 193x across draws) using ONLY that draw's own
  probe (`zlat_b`) — no cross-draw or oracle information. `fit_global_f_ecdf`/
  `load_or_fit_f_ecdf` build (and disk-cache, `_ecdf_cache/`, analogous to
  `BNNPrior`'s own cache) a prior-design-time-fixed reference ECDF from
  MANY self-normalized draws pooled together, so a fresh draw's quantiles
  spread across most of `(0,1)` instead of collapsing (measured: naive
  pooling without self-normalization first collapsed a fresh draw's own
  quantile spread to `[0.67, 0.70]`; self-normalizing first gives spans of
  `0.80-0.98`).
- **`ppfn.prior.lupi.monotone.KumaraswamyMap`**: `h(u) = 1-(1-u^a)^b`,
  exactly `[0,1]->[0,1]` for any `a,b>0`, `a=b=1` is the identity.
  `sample_kumaraswamy_h(rng, severity)` draws `a,b` from a range that widens
  with `severity` (a new knob, `[0,1]`, deliberately independent of `rho` —
  `h_severity`/`h_severity_range` in `sample_pair`/`configs/prior/lupi.yaml`,
  same sweep-independently discipline as `d`/cardinality controls
  elsewhere in this repo). Supersedes the old `MonotoneMap` + a new
  `calibrate_amplitude` helper (kept, used only when `bounded01=False`) —
  the amplitude-calibration approach bounded `h`'s composed SCALE but not
  its LOCATION diversity, which still gave `FullSupportBarDistribution`'s
  one shared, pooled-quantile-fit bin geometry poor per-draw resolution
  (median coverage 4/64 bins for a typical individual draw's own range);
  `bounded01` fixes this at the root by making every draw's value land in
  the SAME `[0,1]` range instead of trying to fit one geometry to a
  heavy-tailed pooled sample.
- **`ppfn.model.baselines.calibration.sample_calibration_borders(...,
  bounded01=True)`** short-circuits to plain `uniform_bin_borders(n_bins,
  0, 1)` — closed-form, no calibration draws needed at all, since there's
  no per-draw diversity left for a data-driven quantile fit to correct for.

## What was ported into this branch

Verified via `git diff bf810d7 ddd17f0 -- <these files>` that the two
branches hadn't already diverged on any of them (empty diff), so the
sibling's uncommitted working-tree diff applied here with `git apply`
cleanly, no manual conflict resolution:

- `src/ppfn/prior/lupi/sampler.py`, `dataset.py`, `monotone.py` (diff-applied)
- `src/ppfn/prior/lupi/ecdf.py` (new file, copied)
- `src/ppfn/model/baselines/calibration.py`, `id_token_pfn.py`,
  `lupi_bounds_pfn.py`, `lupi_id_token_pfn.py` (diff-applied — `bounded01`
  threaded into `IDTokenPFN`/`BoundsPFN`/`LUPIIDTokenPFN`, needed here
  because `FlowMatchingRegistrationPFN` wraps `BoundsPFN` internally)
- `configs/prior/lupi.yaml`, `configs/model/lupi_bounds_pfn.yaml`,
  `configs/model/lupi_id_token_pfn.yaml` (diff-applied)

**My own model's part** (not in the sibling's diff, this branch-specific):
added `bounded01: bool = False` to `FlowMatchingRegistrationPFN.__init__`,
threaded to the internal `BoundsPFN(..., bounded01=bounded01)` call only
(`flow_field` has no bar-distribution head, nothing to thread there) — same
one-line pattern the sibling used for `LUPIIDTokenPFN`/`IDTokenPFN`. Added
`bounded01: ${prior.bounded01}` to
`configs/model/lupi_flow_matching_registration.yaml`.

## A real bug found while re-verifying: validation batch ignored `bounded01`

Re-ran the debug Hydra pipeline after porting
(`lupi_flow_matching_registration_debug`) and got a training/validation
split that made no sense: `loss/nll_student=0.18` (training) vs.
`val/loss/nll_student=51.36` (validation) — a 280x gap on the SAME tiny
debug config, one epoch apart.

**Root cause**: `IDTokenTrainer.__init__` (`src/ppfn/trainer/id_token_trainer.py`)
builds its own validation batch via `build_training_item(...)`, reading most
prior kwargs off the live training dataset via `getattr(train_dataset,
"n_a_range", ...)` etc. (an existing, already-documented pattern, added
2026-09-10 for a different OOM-prevention reason) — but `bounded01`,
`h_gain_range`, `h_severity`, `h_severity_range` were never added to that
`getattr` list, so the validation batch silently fell back to
`build_training_item`'s own default (`bounded01=False`, the OLD unbounded
scale) while every training batch came from the dataset's actual
`bounded01=True`. The model's bar distribution has FIXED `[0,1]` bins under
`bounded01=True` — scoring real (unbounded-scale) validation targets
against those bins is exactly what produced the 51-nat blowup (targets
landing far out in `FullSupportBarDistribution`'s tail).

This is the same *shape* of bug `.claude/rules/checkpoints.md` already
flags as an open risk ("prior/checkpoint provenance" — a checkpoint trained
under one prior config scored against another, silently, plausibly-looking
wrong), just one stage earlier: not checkpoint-reuse across runs, but
train/val split WITHIN one run.

**Fix**: added the same four `getattr(train_dataset, ..., <default>)` reads
for `bounded01`/`h_gain_range`/`h_severity`/`h_severity_range`, threaded
into the validation `build_training_item` call, mirroring the existing
`val_d`/`val_n_a_range`/etc. pattern exactly — no new config fields needed,
it automatically stays in sync with whatever the training dataset was
actually configured with. This fixes the bug for EVERY model using
`IDTokenTrainer` (`BoundsPFN`, `LUPIIDTokenPFN`, `FlowMatchingVelocityField`,
`FlowMatchingRegistrationPFN`), not just this one.

**Verified**: re-ran the same debug config after the fix —
`val/loss/nll_student=0.19` vs. training `0.13`, sane and consistent.

## Verification performed

- Module `__main__` diagnostic (`bounded01=False` default path, unchanged
  behavior) — still passes.
- Direct smoke test of `FlowMatchingRegistrationPFN(bounded01=True)` against
  a `LUPIStreamDataset(bounded01=True)` batch: confirmed `enc_z`/
  `enc_z_inA`/`dec_qry_z` all land in `(0,1)` (clipped to `[1e-4, 1-1e-4]`
  by the prior), `bar_dist.borders` are exactly `[0.0, 1.0]`, gradients
  reach every parameter, loss finite (`nll_student≈0.33` on an untrained
  16-bin model — sane for `[0,1]`-scale data, vs. the ~3.5-5.7 nats seen
  under the old unbounded scale in the previous entry).
- Hydra dry-run compose (`lupi_flow_matching_registration_debug`) confirms
  `prior.bounded01`/`model.bounded01`/`model.model_class.bounded01` all
  resolve to `True` by default now.
- Full debug Hydra run, post-fix: train and val losses consistent
  (`nll_student` 0.13 train / 0.19 val), all four loss terms finite and
  logged correctly.

## Still open

- The real-run config (`configs/experiment/lupi_flow_matching_registration.yaml`,
  prepped in the previous entry) has not been re-verified against
  `bounded01=True` beyond the debug-sized smoke run above — same "not
  launched yet, needs the user's go-ahead" status as before, now with the
  prior dependency actually satisfied rather than pending.
- The `getattr(train_dataset, "h_gain_range", (0.5, 2.0))` fallback default
  in the trainer fix above is dead in practice whenever `bounded01=True`
  (per `sample_pair`'s own docstring, `h_gain_range` is ignored in that
  mode) — kept for parity with the non-`bounded01` real runs
  (`lupi_bounds`, `lupi_id_token_baseline`) that still exercise that path.
- Did not audit whether `RegistrationTrainer` (the OTHER trainer class,
  `ppfn.trainer.registration_trainer`, per this same file's own comment
  referencing it) has an analogous validation-batch gap -- out of scope
  here (nothing in this branch uses it), flagging only in case it matters
  elsewhere.

commit: pending
