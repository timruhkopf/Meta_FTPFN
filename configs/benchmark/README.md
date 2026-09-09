# `benchmark` config group — currently empty

Same situation as `configs/model/`: the only benchmark code that existed
(`src/ppfn/bench/mfpbench`, `bo`) moved to `archive/src/ppfn/bench/` in the
2026-08-26 reset, and no config for it was ever built even before that (the old
`configs/bench/` directory was already empty). `config.yaml` declares
`benchmark: ???` so this is a loud placeholder, not a silent gap.

Unlike `model`/`prior`/`trainer`, nothing in `../../src/ppfn/pipelines/train.py` reads `cfg.benchmark`
yet — before adding a real config here, first decide *what consumes it*: a
periodic-evaluation callback during training (see the kept but currently-unused
`trainer/callbacks/meta_test.py` / `hefty_meta_test.py` for a plausible existing
hook point), a separate `evaluate.py` entry point, or something else. That's a
milestone-level decision, not part of today's setup.

## Convention to follow once this is filled in

Same "meta info baked in" shape as `configs/prior/bnn.yaml` — descriptive fields
(e.g. which task suite, how many tasks) alongside the `_target_` block:

```yaml
# example shape, not a real file
benchmark_name: lcbench_tabular
num_tasks: 35

benchmark_class:
  _target_: ppfn.bench.<...>
  ...
```
