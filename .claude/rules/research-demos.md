---
paths:
  - "src/ppfn/model/**"
  - "src/ppfn/prior/**"
---

# Demo-block convention for models and priors

Every model and prior module gets a runnable `if __name__ == "__main__":` demo —
not a docstring example, an actual block that runs when you `python
path/to/module.py`. This is how confidence in the forward pass / data-generating
process gets built, not just correctness-by-inspection. Keep the demo cheap
(small shapes, CPU-friendly) — it's a sanity check, not a benchmark.

## Model modules

The `__main__` block builds a small instance of the model and runs a forward
pass on synthetic input, printing input/output shapes (and, where relevant,
intermediate shapes at any point that's easy to get wrong — padding, masking,
stream splits). If the model has a non-trivial `get_trainable_params` or
freezing behavior (see `.claude/rules/checkpoints.md`), the demo should also
print which parameters are frozen vs. trainable.

## Prior modules

The demo samples a batch and produces a 1D diagnostic plot (`matplotlib`,
`plt.show()` or save to a scratch path — not a test, no pytest assertion
needed) with:

- **Subplots for A and B** (side by side or stacked, not overlaid in one axes)
  — this is a two-domain prior; conflating A and B in one plot hides exactly
  the thing you're trying to build confidence in.
- The **sampled raw data points** for that domain, plus its **ground-truth
  function** where the prior can expose one (it usually can, since priors here
  are simulators).
- The **(binned) predictive probability distribution as a heatmap** over the
  domain, wherever the demo has logits to show (a prior demo showing its own
  data-generating process may have no logits yet; a baseline/model demo run
  against a prior should show this for its predictions).
- Logits/predictions are evaluated on a **dense, regular grid spanning the
  entire domain** — not the (irregular, prior-sampled) query positions used at
  training time. Irregular query points produce plotting artifacts (gaps,
  aliasing) that look like model failures but are just sampling density
  artifacts. Build the grid once per plot, independent of how the prior itself
  samples training queries.

This convention is deliberately about *looking at the data*, not automated
testing — `.claude/rules/testing.md` covers pytest conventions separately.
