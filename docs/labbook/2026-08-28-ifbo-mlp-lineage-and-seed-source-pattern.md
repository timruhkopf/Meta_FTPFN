# ifBO's `DatasetPrior` MLP: same shared lineage, but used as a multi-channel correlated seed source, not the function itself

**commit:** pending
**Related:** [2026-08-27 — BNN prior: `init_std`/width coupling](2026-08-27-bnn-prior-init-std-width-coupling.md), `docs/milestones/M4-bnn-prior-relatedness.md`

## What was investigated

Whether `external/ifBO/ifbo/priors/ftpfn_prior.py`'s `MLP`/`DatasetPrior` share
lineage with our own `BNNPrior`/`MLP`, and if so, whether the
`init_std`/width decoupling bug from the previous entry rides along with it.

## What was found — verified against the actual file, not just asserted

`ftpfn_prior.py`'s `MLP` (lines 119–166) is essentially the same construction
as ours, pre-fix: `num_layers = randint(8,16)`, `num_hidden = randint(36,150)`,
`init_std = uniform(0.089, 0.193)` sampled **independently** of width, same
`sparseness = 0.145`, same preactivation/output noise ranges. Confirmed by
reading the file directly (not just the pasted summary) — this is shared,
evolving code across the AutoML-Freiburg lineage (this repo's `bnn_prior.py`,
ifBO), not independently re-derived each time. The same non-issue documented
in the 2026-08-27 entry above rides along into ifBO too — not fixed there
either, as of this repo's vendored copy.

One additional, directly relevant confirmation: ifBO's `DatasetPrior.
_output_for` **does** apply `self.normalizer(input)` and uses the result
(`input = self.normalizer(input)`, line ~185) — unlike our own `mlp.py`,
where the analogous call computes a value and discards it (see the
`FIXME: check if the original mlp also did not use the normalizer!` this repo
still carries). This is now the third confirmation (PFNs4BO, ifBO, and this
prior's own `Normalize(0.5, sqrt(1/12))` instantiation) that the normalizer
is meant to be applied — strengthens the case for resolving that FIXME in
favor of applying it, though that's still a deliberate decision to make, not
done here.

## The actual point of interest: the MLP isn't the function here

This is a genuinely different design pattern from ours, worth remembering
independent of the bug-lineage question. In `DatasetPrior` (verified: `class
DatasetPrior`, `new_dataset()`, `MyRNG`, `OUTPUT_SORTED` all present and
matching this description):

1. One MLP per synthetic task, resampled in `new_dataset()` — same cadence
   as our `BNNPrior`'s per-task resample, then queried many times across
   many configs within that task.
2. Built with `num_outputs=23` (`DatasetPrior(num_params, 23)`), not 1 — one
   forward pass on a config `x` produces 23 channels at once, not one `y`.
3. Each channel is independently rank-normalized against `OUTPUT_SORTED` — a
   static, pre-shipped `.npy` (`np.searchsorted`), the same core idea as our
   ECDF but computed once offline and checked in, rather than fit lazily per
   config the way `BNNPrior.ensure_ecdf_loaded` does.
4. `MyRNG` treats each channel's rank as a uniform sample and pushes it
   through a *different* inverse-CDF per channel (`norm.ppf`, `gamma.ppf`,
   `beta.ppf`, `expon.ppf` all present in the file) to produce one specific
   learning-curve-shape parameter each — verified via the inline `# 0`,
   `# 1, 2, 3, 4`, `# 5`, ... comments in `curves_for_configs`, which
   literally index which output channel feeds which parameter (`Yinf`, basis-
   curve mixture weights, shape/skew parameters, saturation scale/points,
   etc.).
5. Those parameters combine (a mixture of parametric basis curves) into the
   actual learning curve `y(epoch | config)` — not traced further here, not
   relevant to the pattern itself.

So the MLP's job is to be a smooth, multi-channel, correlated source of
pseudo-randomness, not the function being modeled. Because the network is
smooth in `x` (the config), nearby configs get similar-but-not-identical
values across all 23 channels, so every derived curve-shape parameter varies
smoothly and jointly with the config — nearby hyperparameter configs get
visibly similar learning curves, matching real HPO landscapes. Sampling 23
independent parameters per config instead would give every config an
uncorrelated, discontinuous curve shape.

## Why this might matter for M4 (not yet acted on)

M4's BNN relatedness mechanism has two options on the table (see
`docs/milestones/M4-bnn-prior-relatedness.md`); the "two BNN instantiations +
alpha interpolation" option was flagged in the 2026-08-27 milestone review as
likely mathematically unsound, because linearly interpolating two
independently-initialized networks' raw *weights* doesn't correspond to
smooth interpolation in function space (no mode connectivity between
independent inits). This ifBO pattern — one smooth network, multiple output
channels, per-channel rank-normalize + a chosen inverse-CDF — is a way to get
several correlated-but-distinct derived quantities *without* interpolating
weights at all: A and B could each read off different channels (or a shared
channel with different per-channel transforms) of the *same* underlying
smooth field, giving controllable relatedness through the field's own
smoothness rather than through weight-space arithmetic. Not implemented or
decided — flagged here as a concrete alternative worth considering when M4
actually gets worked, not a plan to act on now.
