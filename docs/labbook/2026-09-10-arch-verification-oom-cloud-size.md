# `arch_verification` OOM: uncapped cloud sizes, not batch size

commit: 6df8fe6

## What was investigated

The `arch_verification` experiment (a stripped-down version of the
`step4_pathway` go/no-go run — `ArchVerificationLoss` +
`ppfn.monitor.arch_verification`, computing three NLLs — lower/severed,
upper-2/pooled, and the real encoder-decoder — on the same held-out query
tokens) OOM'd on Ulysses's 24GB GPU immediately after launch, before
completing a single epoch.

## First two attempts didn't work — say so plainly

The first hypothesis was that `prior.batch_size` was too high (it had
worked at 8 for `step4_pathway`, which uses the same reference model size).

- Attempt 1: cut `val_size` 64 → 24. **Still OOM'd**, now inside the
  query-token cross-attention during the eval pass.
- Attempt 2: cut `prior.batch_size` 8 → 2, `val_size` 24 → 12, enabled
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Memory looked stable
  at ~3.4GB for a minute of training. **Then OOM'd anyway**, jumping
  straight from 3.4GB to 23.64GB — proof `batch_size` was never the actual
  lever, since a batch of 2 items shouldn't spike memory 7x on its own.

Both attempts were verified empirically (real launches on the GPU, not
guessed), and both were wrong. Stopped re-tuning blindly at that point and
brought the log traces to the user rather than trying a third batch-size
guess.

## Actual root cause

The registration prior (`ppfn.prior.registration.sampler.sample_pair`)
draws the encoder cloud `n_b` from a **log-uniform range up to 1024**
points, and batches pad every item to the batch's max cloud size. A single
item drawing a large `n_b` dominates the whole batch's memory regardless of
how small the other items are — reducing `batch_size` doesn't reduce that
single item's own cost.

`ArchVerificationLoss` and `RegistrationLoss`'s `L_pathway` branch both build
a **pooled context `[A ; B_inA]`** (`ppfn.prior.registration.dataset.
build_pooled_context`) and run it through the decoder's own **bidirectional
self-attention** — `O((n_a + n_b)^2)`. At `n_b` near its 1024 ceiling this
quadratic blows past 24GB even for a batch of 1-2 items. `step4_pathway`
never hit this because it OOM'd (on a *different*, batch-size-driven cause
at the time) before running long enough to draw an extreme `n_b`.

Capping only `n_b_range` (to `(8, 100)`) was not sufficient either: `
build_training_item`'s role randomization (35% chance) can swap which
cloud plays encoder vs decoder, so a role-swapped draw's "encoder" comes
from `n_a_range`, which was still uncapped. Verified empirically:

```
n_b_range=(8, 100), n_a_range left at default (8, 256):
  enc_x (n_b_range) capped at 100
  but role_swapped=True draws still produced enc_x up to 214 rows
```

## Fix

Plumbed `d`, `n_a_range`, `n_b_range` through `build_training_item` →
`RegistrationStreamDataset` → `configs/prior/{registration,p0_identity}.yaml`
(`sample_pair` already accepted them; they just weren't reachable from
config). `RegistrationTrainer` now reads these off `train_loader.dataset`
for its own held-out validation batch too, so validation never drifts from
whatever cloud-size/dimension regime training is actually using.

`configs/experiment/arch_verification.yaml` caps **both** `n_a_range` and
`n_b_range` to `(8, 100)` and fixes `d=1`. Verified: 200 draws now bound the
pooled context to <=194 tokens (was up to 1280).

Before/after, same reference model size (`d_model=256`, 6 encoder / 8
decoder layers):

| config | GPU memory | outcome |
|---|---|---|
| `n_b` up to 1024, `batch_size=8` | OOM before epoch 0 | crash |
| `n_b` up to 1024, `batch_size=2` | 3.4GB then spiked to 23.64GB | crash |
| `n_a`/`n_b` capped to `(8,100)`, `batch_size=16` | ~19.5GB, stable across 15+ epochs | training normally |

With the real cost driver capped, `batch_size` and `val_size` were restored
to 16 / 64 (from the panic-reduced 2 / 12) — there was headroom to spare.

## Also root-caused a low-severity secondary issue

The visualisation notebook built alongside this investigation
(`notebooks/prior_warp_visualization.ipynb`) independently surfaced that the
production-calibrated `s_max=0.1` warp severity produces a barely-visible
warp on a typical draw (severity ~ Uniform(0,1)*s_max, so a "typical" draw
at 0.1 warps very little) — not a bug, just a reminder that `s_max=0.1` is
tuned for training's rejection rate, not for eyeballing the warp's effect.
