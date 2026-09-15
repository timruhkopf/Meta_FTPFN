# LUPI bounds PFN, query-source/rho stratification, and a val-batch bug

*Log, 2026-09-15. Follow-up to
[LUPI-distilled alignment-aware multi-task PFN: first build](2026-09-14-lupi-alignment-pfn-build.md).
User asked to (a) stratify the small oracle-student gap by query source and
rho rather than trust the aggregate, (b) add a lower/upper bound PFN
(reusing one checkpoint for both bounds, not training two separate models),
and (c) collect lower/student/oracle/upper together as a "standing view".*

## Stratified diagnostics against the existing checkpoint (no retraining)

`ppfn.prior.lupi.sampler.LUPIPair.qry_source` was already computed (spec
§6.2's 3-way query mixture) but dropped between `sampler.py` and
`dataset.py` -- threaded through to `LUPIBatch.dec_qry_source`. New
`ppfn.monitor.lupi` module (own registry, mirrors
`ppfn.monitor.registry`'s pattern) computes query-source- and
rho-stratified NLL/gap.

Run against `01-pretraining-lupi-baseline`'s best checkpoint (256-item
resample at the same val seed/ranges, `scripts/lupi_eval_checkpoint.py`):

| bucket | n | student NLL | oracle NLL | gap |
|---|---|---|---|---|
| ρ=0 | 906 | −2.272 | −2.272 | 0.000 |
| ρ∈(0,0.3) | 1659 | −2.601 | −2.616 | 0.014 |
| ρ∈[0.3,0.7) | 2139 | −2.378 | −2.406 | 0.029 |
| ρ≥0.7 | 1970 | −2.425 | −2.454 | 0.029 |
| query near B | 2662 | −2.563 | −2.577 | 0.013 |
| query uniform | 2662 | −2.481 | −2.512 | 0.031 |
| query near A-ctx | 1350 | −2.080 | −2.097 | 0.017 |

Findings: the gap grows from 0 at ρ=0 to ~0.029 by ρ≈0.3 then *plateaus* --
mid and high ρ give essentially the same gap, so unresolved registration at
high ρ specifically isn't where the model loses the most ground relative to
the oracle (a curriculum-dilution explanation alone doesn't fit). Near-B
queries get the BEST NLL of all three buckets (evidence the B-path is
genuinely being read, not ignored -- a B-blind model would do no better
there than on uniform queries) but the SMALLEST gap, while uniform queries
(no nearby exact value match) show the largest gap. Working hypothesis:
quantile-normalized value-matching (identical in both modes) is carrying
most of the prediction quality at every ρ; oracle position information adds
the most exactly where a value-only shortcut isn't available. Not confirmed
-- needs the attention-visualization notebook to actually check.

## Bug found and fixed: validation batch didn't match training ranges

`LUPITrainer.__init__` built its fixed validation batch via
`build_training_item(...)` without passing `n_a_range`/`n_b_range`/
`n_qry_range` -- so `01-pretraining-lupi-baseline`'s own logged
`val/loss/*` metrics were computed on `build_training_item`'s defaults
(n_b up to 1024) regardless of the run's actual `prior.n_b_range` (capped
at 100). A real train/val distribution mismatch, present for the entire
finished run. Fixed: `val_n_a_range`/`val_n_b_range`/`val_n_qry_range` now
threaded through, wired from `prior.n_a_range`/etc. in
`configs/trainer/lupi.yaml`. `scripts/lupi_eval_checkpoint.py`'s own
numbers (above) were unaffected -- it passed matching ranges explicitly.

## Bounds PFN: one checkpoint, two brackets

`ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN` -- a plain PFNBlock-based
model (reuses `IDTokenPFN`'s additive-domain-tag pattern) pooling
`[A_ctx ; B_inA]`, `severed` toggle at forward time (`severed=True` masks
`B_inA` out entirely -> lower bound; `severed=False` -> upper bound).
`B_inA` (new field, `LUPIPair.x_b_inA`/`LUPIBatch.enc_x_inA`): B's latent
transported through A's OWN warp, no inversion needed (same trick the
near-B query bucket already used, applied to the whole B cloud) --
registration-free by construction, since it's handed the correct
A-frame position directly rather than requiring the model to infer it.
`ppfn.loss.lupi_bounds_loss.BoundsLoss` trains both bounds every batch
(mirrors `LUPILoss`'s "both modes every batch"), so lower/upper always
come from one consistent set of weights -- "reusing the checkpoint" per
the user's own framing, not training two separate models.

`ppfn.monitor.lupi.compute_bounds_report` + `scripts/lupi_bounds_report.py`:
loads both checkpoints, evaluates all four (lower/student/oracle/upper) on
one shared batch, logs as its own small MLflow run
(`run_name=standing-view-bounds-report`) in `lupi-alignment-pfn` --
CLAUDE.md's "always report three gaps, never a bare NLL" extended across
two models. Not yet run for real (needs the bounds-PFN checkpoint, still
training).

## Launched on ulysses (chained, sequential -- one GPU)

1. `01-pretraining-lupi-bounds` (BoundsPFN, run_name=`lupi-bounds-pfn`)
2. `01-pretraining-lupi-no-ce-control` (LUPIPFN, `loss.lambda_d=0`,
   run_name=`lupi-baseline-no-ce-control`) -- the distillation-attribution
   control from spec's own build order §8 step 3.

Both report to `lupi-alignment-pfn` (same experiment as the main run), each
~7h at the same prior config -- chained via a background shell script so
(2) starts automatically when (1) finishes, no babysitting needed.

## Also: MLflow experiment descriptions

Set `mlflow.note.content` on all 8 experiments in the ulysses tracking
store (`lupi-alignment-pfn`, `bridge-pfn`, `id-token-baseline`,
`arch-verification`, `registration-step4-pathway`,
`pfn-variable-dim5-reference`, `pfn-variable-dim5-long`, `pfn-pretrain`,
`Default`) so the MLflow UI's Overview tab shows what each is for.

## Still open

- 1D visualization notebook (data, true function, posterior mean, bar-
  distribution heatmap), 4 columns: BoundsPFN-severed (lower) /
  LUPIPFN-student / LUPIPFN-oracle / BoundsPFN-pooled (upper) -- deferred,
  needs the bounds-PFN checkpoint to exist for columns 1 and 4.
- The value-matching-vs-position hypothesis above is unconfirmed.

commit: pending
