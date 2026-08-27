# WarpAlignPFN

TabPFN v2.5 extended with a distorted auxiliary table B that gets aligned into
A's domain in-context, then concatenated into A's row axis.

Built against `tabpfn==8.3.0`. Every attention primitive, the bar distribution,
and the whole embedding stem are imported, not reimplemented.

```
pip install tabpfn==8.3.0
```

## Files

| file | contents |
|---|---|
| `blocks.py` | `AlignerBlock` — B queries, A_train keys |
| `model.py`  | `WarpAlignPFN`, `WarpAlignConfig`, `make_borders` |
| `losses.py` | objective: task NLL + teacher soft-CE + translation + anti-collapse |
| `train.py`  | `MetaBatch`, severity curriculum, two-phase loop, diagnostics |

## What is imported from TabPFN

| symbol | from | used for |
|---|---|---|
| `TabPFNBlock` | `tabpfn_v2_5` | all four stacks |
| `AlongRowAttention`, `AlongColumnAttention` | `tabpfn_v2_5` | aligner self-attention halves |
| `LowerPrecisionLayerNorm`, `ENCODING_SIZE_MULTIPLIER` | `tabpfn_v2_5` | norms, B's target embedder width |
| `TabPFNV2p5` (with `nlayers=0`) | `tabpfn_v2_5` | embedding stem + `AddThinkingRows` |
| `CrossAttention` | `tabpfn_v3` | both cross branches |
| `FullSupportBarDistribution` | `shared.bar_distribution` | loss, soft-CE, all heads |

`CrossAttention` is the module underneath v3's `ColumnAggregator` — the
cross-feature attention you flagged. We take the bare attention rather than v3's
`CrossAttentionBlock`, which is pre-norm, so the whole model stays on v2.5's
post-norm convention and remains warm-start compatible.

## Six decisions worth arguing about

**1. B is concatenated into the row axis, not attached as a side branch.**
`A_test` reads `[thinking, A_train, B_hat]` under a *single* softmax, so A_train
and B_hat compete for the same probability mass. Unrelated B is starved
automatically — the gate is intrinsic, per-query, per-column. A separate
cross-attention branch normalises over B alone and must spend its full mass
there, which is why it would need an external gate bolted on. Implementation is
free: stock `AlongColumnAttention` with `single_eval_pos = 64 + n_A_train + R_B`.

**2. No Sinkhorn, no OT — balanced or otherwise.** Attention is not a coupling.
Each A query gets its own softmax over the key set; there is no marginal
constraint on the B side, so `|B| >> |A_train|` is just a longer key sequence,
not a violated mass-balance. OT only becomes the right frame if you are
*supervising* an explicit correspondence, and there is no pairing to supervise
with. Where a distributional objective is genuinely wanted, `losses.energy_distance`
handles unequal sample sizes natively with no coupling to solve.

**3. Feature-cross before row-cross, and the first aligner block is
feature-cross only** (`aligner_row_cross_from=1`). Row scores are dot products
in cell-embedding space; computing them before the column spaces are
recalibrated is exactly the false-similarity problem. Caveat: this is an
inductive-bias argument, not an expressivity one — learned `W_q`/`W_k` can absorb
a column-wise affine warp either way. Ablate it against
`aligner_row_cross_from=0` and against a reversed order.

**4. Separate scaler fits per table.** `_preprocess_and_embed_features` is called
with `num_train_labels=n_A_train` for A and `num_train_labels=R_B` for B, so
`TorchStandardScaler` fits independently on each. That removes the affine part of
the warp before the network sees it and leaves the aligner only the residual
nonlinearity. Free, and it measurably shrinks what you are asking the model to
learn.

**5. Shared constant-column mask.** TabPFN drops constant columns *per table*.
Letting it do so independently for A and B would silently break the schema
alignment the whole design rests on. `_shared_column_mask` intersects the masks
and pre-selects, so the internal removal becomes a no-op. This is a real bug you
would otherwise hit only on tasks where some column happens to be constant in
one table.

**6. y-only translation, not full-row reconstruction.** You were right that
predicting every feature post-translation is a bad trade. `translation_head`
supervises just `B_hat`'s y cell against `B_inA`'s y, through the same bar
distribution with A's bins. That is the value vector actually carrying the
function's shape into A_test, at the cost of one extra head.

## Bins

`make_borders` is a **fixed** grid on standardised-y space, not A_train
quantiles. A_train is sparse, quantile borders would wobble across meta-tasks,
and wobbling borders make the teacher soft-CE meaningless — student and teacher
must share `borders` for `full_ce` to mean anything. Standardise y with A_train's
mean/std upstream and the fixed grid covers it.

## Teacher

Two-phase, teacher frozen during Phase 1. A co-trained teacher is a moving target
stacked on an already-hard alignment problem and the failure is hard to diagnose.
Warm-start the teacher from the released v2.5 checkpoint; it is stock TabPFN on
your A-prior with dense contexts and converges fast. If you later want
co-adaptation, use an EMA of the student's decoder rather than a second free
model — no moving-target instability, no extra parameter set. And if your prior
is conjugate, skip all of this: compute the exact posterior predictive and the
teacher is analytic, free, and genuinely optimal-in-prior.

## The failure mode to instrument first

The aligner is grounded on a sparse A_train, so the cheapest solution is to
memorise those few rows and map all of B onto their support. B_hat collapses onto
A_train, attention looks healthy, training NLL looks fine, and you have destroyed
precisely the extra shape information that motivated the design.

Three defenses, all wired in: cross-feature weighted early and heavily (pooled
column statistics survive small `n_A_train` far better than row matching);
`anti_collapse` (VICReg variance + covariance) on B_hat's rows; and
`effective_rank` logged as a first-class metric. Watch effective rank, not loss.

## Reporting

Never report raw NLL. `train.oracle_gap` gives floor (B ablated), student, and
ceiling (fed true `B_inA`) on identical tasks. The number that means something is
**fraction of the oracle-alignment gap recovered**.

`train.utility_of_B` is the headline diagnostic: per-example delta-NLL from
including B, plotted against true severity. If it does not decay as severity
rises, the model has not learned to detect distortion — it has learned to trust
B unconditionally, and you get sharp, confident, wrong predictions on shifted
tasks while averaged metrics look fine. This measures what B actually buys rather
than where the softmax happened to point, and needs no attention hooks.

## Known gaps

- **Relatedness gate is an embedding shift, not a logit bias.** `severity_token`
  is added to B_hat scaled by the predicted severity. A true additive bias on
  B_hat's attention logits would need a ~15-line fork of `AlongColumnAttention`
  to thread a key-bias argument through. The shift is more expressive but less
  directly interpretable; fork it if you want a clean scalar knob to read off.
- **MQA on test rows.** `AlongColumnAttention` routes test-row queries through
  `k[:, :, :1]`. That is a KV-cache optimisation, and squeezing a *heterogeneous*
  context (`A_train` + `B_hat`) through one head is a real bottleneck. It is
  active in the decoder as written. Fork the class to disable it during training
  and re-enable only for cached inference.
- **Private-API dependency.** `_preprocess_and_embed_features` and
  `_preprocess_and_embed_targets` are private and pinned to 8.3.0. Pin the
  version; they will move.
- **Aligner memory.** `col_summary.expand(...).reshape(...)` in the cross-feature
  branch materialises one tensor the size of the B stream. Acceptable, but it is
  the first thing to fuse if you hit memory limits.
- **`load_tabpfn_weights` is untested** against a real checkpoint — the block
  tiling and the AlignerBlock key remapping are written from the module layout,
  not verified against downloaded weights. Check `strict=False` return values.
