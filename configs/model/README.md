# `model` config group — currently empty

No model config exists here on purpose: every model implementation that was in
this repo (`mymodel/`, `anamorphic/`, `baselines/`) was moved to
`archive/src/ppfn/model/` in the 2026-08-26 reset (see `archive/README.md`) and
none has been pulled back yet. `configs/config.yaml` declares `model: ???` so
`src/train.py` fails loudly (`You must specify 'model'...`) instead of a
confusing `AttributeError` on `cfg.model` deep inside `run()`.

## Convention to follow once a model comes back

Mirror the "meta info baked in" shape used in `configs/prior/bnn.yaml`: keep
descriptive fields (whatever the model needs to know about its input, e.g.
dimensionality) as siblings of the `_target_` block, and interpolate rather than
duplicate where the value is already defined elsewhere — e.g. a model that needs
the prior's output dimensionality should read `${prior.num_outputs}`, not repeat
a hardcoded number:

```yaml
# example shape, not a real file
hidden_dim: 128
num_layers: 6

model_class:
  _target_: ppfn.model.<...>
  input_dim: ${prior.num_inputs}
  hidden_dim: ${model.hidden_dim}
  num_layers: ${model.num_layers}
```

Add the real file as `configs/model/<name>.yaml` and switch `config.yaml`'s
`model: ???` to `model: <name>`.
