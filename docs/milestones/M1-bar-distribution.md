# M1 — Bar distribution as the first-class loss

## Goal

Make the bar distribution the one loss/objective every model in this repo trains
against, wired in as a real, importable, tested component before any baseline
needs it.

## Context

The `tabpfn` package (already a `pyproject.toml` dependency, installed at
`8.2.0`) ships the exact code linked from PriorLabs/TabPFN:
`tabpfn.architectures.shared.bar_distribution.{BarDistribution,
FullSupportBarDistribution}`. This is **not** a porting job — import it. Check
first whether the installed version matches the GitHub `main` branch closely
enough (same constructor signature, same `cdf`/`icdf`/`forward` behavior); if it
has drifted, that's a decision to surface, not silently work around.

## Deliverables

1. A thin wrapper/module under `src/ppfn/trainer/` (or wherever the "criterion"
   for `configs/trainer/default.yaml` should live — see `.claude/rules/hydra.md`
   on the DictConfig boundary) that constructs `FullSupportBarDistribution` from
   config (bucket borders / number of buckets, etc.) and exposes it the way
   `PPFNTrainer`'s `criterion` argument expects (see `trainer.py::_train_step`:
   `loss, step_metrics = self.criterion(output, batch=batch, **fwd_kwargs)`).
2. Bucket-border construction must be a **shared, reusable** function — see
   Non-goals — since M7 requires the marginal baseline and the model-under-test
   to use *identical* borders for their NAT comparison to mean anything.
3. Metrics: alongside NLL, log how far predictions fall outside the
   distribution's support (i.e. how often/how far the target lands beyond the
   lower/upper bucket edges) — you explicitly asked for this ("measures how far
   off we are from the lower and upper bounds, when we use appropriate inputs").
   Check what `BarDistribution` already exposes for this before writing new code.
4. A demo (per `.claude/rules/research-demos.md`, adapted since this isn't a
   model with a forward-pass-on-random-input shape): given a small set of
   synthetic (x, y) points and a toy target function, show the resulting bar
   distribution's predicted density as a 1D heatmap over a dense y-grid, with
   the true y overlaid — this is the visual sanity check that borders/binning
   are sane before any model trains against it.

## Acceptance criteria

- [ ] `FullSupportBarDistribution` importable and constructible from a
      `configs/`-driven config (even if `trainer=default`'s `criterion: ???`
      is what finally gets filled in — that's this milestone's job).
- [ ] A unit test constructs the distribution, feeds it a small batch of
      logits + targets, and checks: NLL is finite for in-support targets,
      and the reported out-of-bounds metric is nonzero for a deliberately
      out-of-range target.
- [ ] The demo script produces a plot and is runnable via `python
      path/to/module.py`.
- [ ] `pytest` passes.

## Non-goals

- Not reimplementing bar distribution math — only wrapping/configuring the
  installed `tabpfn` implementation.
- Not deciding per-prior bucket borders yet (M2–M4 priors may need different
  y-ranges) — this milestone establishes the *mechanism* for constructing
  borders from config/data statistics, not the final numbers for every prior.
- Not the cross-model NAT-comparison callback — that's M7, once two trained
  models actually exist to compare.
