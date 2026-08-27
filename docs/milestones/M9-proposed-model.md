# M9 — Transport PFN: encoder→decoder domain translation via thinking-row cross-attention

## Status: spec'd (2026-08-27) — and a working starting point already exists

Not a from-scratch design: `src/ppfn/model/unwarp.py` (`ThinkingWarpTabPFN`)
already implements a version of this idea and already trains (its own
`__main__` block's logged loss curve shows NLL going from ~1.4 to ~-0.99).
This milestone is about restructuring and extending it — checkpoint-initializing
the decoder from M5, replacing its bridge/broadcast attention with a
TabPFN-native cross-attention module, changing the training objective from
"reconstruct oracle `B_inA`" to "predict `A_test` through the decoder," and
adding the optional KL-distillation branch — not throwing it away. See
`.claude/rules/model-prototyping.md` for the package structure to grow it into.

## Grounding: verified against the actual installed TabPFN v2.5 source

Read in full before touching this file further:
`.venv/lib/python3.10/site-packages/tabpfn/architectures/tabpfn_v2_5.py`.
Key facts, checked against that source, not assumed:

- **Naming convention**: every tensor variable name carries a suffix spelling
  out its axis order — `B`=batch, `R`=rows (thinking+train+test), `Ri`=input
  rows before thinking rows are added, `C`=feature-groups/columns,
  `E`/`D`=embedding size, `T`=number of thinking rows, `N`=train(+thinking)
  rows, `M`=test rows, `H`=heads, `F`=`head_dim*num_heads`. e.g. `x_BRCE`,
  `x_BcRE` (batch and columns folded together for column-attention, one
  sequence per (batch, column) pair, attending along rows), `x_BrCE`/`x_BrSE`
  (batch and rows folded together for row-attention, attending along
  columns/features). **Match this exactly in new code** — it's how you keep
  the (Row, Column, D) math checkable at a glance instead of by re-deriving it.

- **The row-vs-column naming is not what it sounds like — this is the exact
  gotcha you flagged.** `AlongRowAttention` = "attention between features of a
  single row" = fixes one row, attends across the feature/column axis (this is
  what you'd informally call "feature attention"). `AlongColumnAttention` =
  "attention between cells of a single column" = fixes one feature-column,
  attends across the row axis (this is what you'd informally call
  "row"/in-context/sample attention — the actual ICL mechanism, with the
  train/test masking). The class is named for the axis held **fixed**, not the
  axis attended **over**. `TabPFNBlock` runs one `AlongRowAttention`
  (`per_sample_attention_between_features`) then one `AlongColumnAttention`
  (`per_column_attention_between_cells`) then an MLP, each block, transposing
  `x_BRCE ↔ x_BCRE` in between.

- **Thinking rows are already an inducing-point mechanism — this is your
  reframing confirmed, not a new thing to build.** `AddThinkingRows` prepends
  `T` learned rows (broadcast identically across whatever the feature-group
  axis `C` happens to be, since `C` varies per dataset) to the row axis, and
  shifts `single_eval_pos` by `T` so the thinking rows count as part of the
  train-like context (`N = T + n_train`). In `AlongColumnAttention`, all of
  `N` (thinking+train) mutually attends within itself, and every test row
  attends only to that `N`-block. So after a few `TabPFNBlock`s, the `T`
  thinking-row embeddings have been contextualized by the entire train set —
  a **fixed-size** (`T`, config-set, independent of `n_train`) summary of a
  variable-size context. That's exactly the "fixed tensor shape comparison
  between A and B despite their `N` differing" property you want, and it's
  already implemented, not something this milestone adds.

- **The `_decode` hook**: `TabPFNV2p5._decode(..., only_return_standard_out=False)`
  already returns a dict with `train_embeddings`/`test_embeddings` alongside
  the standard output — the existing seam for pulling contextualized row (and,
  by slicing, thinking-row) embeddings out of a forward pass, rather than
  needing to copy `TabPFNV2p5.forward`'s ~370-line body. **Verify this actually
  gives you the contextualized thinking-row embeddings specifically** (they
  live at row indices `0:T`, before `train_start`) before assuming it's
  sufficient — first concrete task of this milestone, not an assumption to
  build on unchecked.

## What `unwarp.py` already gets right, and what changes

Already correct / reusable as-is:
- Uses real `TabPFNBlock`s (`stream_A_blocks`, `stream_B_blocks`) for the
  per-domain row+column attention — not a reimplementation.
- Already frames the problem as: contextualize A and B independently with
  shared/identical thinking-token initialization, then bridge the two
  thinking-row sets, then let B's data rows pull from the bridged
  representation. This is structurally the right shape.
- Trains against `FullSupportBarDistribution` (M1) already.
- Already gets a real prior (`HarmonicMixturePrior`/`InfiniteHarmonicsStream`,
  currently the archived harmonics prior — will become whichever M2–M4 prior
  is ready) producing a `Y_B_in_A` field to check against.

What this milestone changes:

1. **The bridge/broadcast attention (`bridge_cross_attn`, `broadcast_cross_attn`)
   currently flattens `(thinking_rows, features)` into one sequence dimension
   and runs a generic `nn.MultiheadAttention` over it.** This discards the
   Row/Column axis distinction that the rest of the architecture (and TabPFN
   itself) carefully preserves — exactly the kind of thing to "verify you're
   doing" rather than assume is fine because it runs. Replace with a
   TabPFN-native cross-attention module (see below) that keeps the
   feature-group axis `C` as a batched dimension (like `AlongColumnAttention`
   does) instead of flattening it away.
2. **`stream_A_blocks` is currently a freshly-initialized `TabPFNBlock` stack.**
   Replace with the **checkpoint-initialized M5 decoder** — this is "the
   decoder-only PFN... use this checkpoint as the initialization" from your
   description. Whether it stays frozen or gets fine-tuned during this
   milestone's training is an open question to decide empirically, not assumed
   either way going in.
3. **The training objective currently is: predict `Y_B_in_A` directly
   (oracle target) from `data_rows_B_final`.** New objective: the decoder
   predicts `A_test` from `[A_train, encoder(B)]`, trained against the M1 bar
   distribution on `A_test`'s actual targets — the encoder's job is to make
   `encoder(B)` useful to the decoder for that, not to reconstruct
   `Y_B_in_A`'s raw values. The optional KL-distillation branch (below) is
   how oracle `B_inA` still gets used as a training signal, just not as a
   direct reconstruction target.
4. **No padding/masking currently** — `n_A`/`n_B` are fixed per prior config
   (`n_A=20, n_B=50`). Needs real per-batch-item length handling (see below).

## Architecture (target shape)

1. **Decoder**: a `TabPFNV2p5`-equivalent stack (cell/target encoding,
   `AddThinkingRows`, `TabPFNBlock`s) — this *is* M5's marginal baseline,
   loaded from its checkpoint (per `.claude/rules/checkpoints.md`'s contract —
   this is also the first real consumer of that contract for a
   non-M5/M6/M7 purpose, so the prior-provenance check needs to hold here too).
2. **Encoder**: a second row+column-attention stack processing B, plus a new
   cross-attention module where B's representations query into the decoder's
   **contextualized thinking rows** (i.e. run the decoder on A first — or
   reuse a cached forward pass — to get thinking-row embeddings that already
   summarize A, then let the encoder's B-processing cross-attend to those).
   This is the "B can modulate itself to translate into domain A" step.
3. **Final decode**: the same decoder, now given `[A_train, encoder_output]`
   as context (in place of oracle `B_inA`) and `A_test` as query — reusing
   exactly the conditioning regime M5's checkpoint was already trained to
   handle (`A_test | [A, B_inA]`), just with a learned `B_inA` instead of the
   prior's ground-truth one.

## Optional side branch: KL-distillation from the oracle-fed decoder

Since we're still training on a synthetic prior, the ground-truth `B_inA` is
available even though it won't be at real deployment. Use it as a **second,
parallel decoder pass** — feed the checkpoint decoder `[A_train, B_inA(oracle)]`
(exactly M5's third conditioning regime) alongside the main
`[A_train, encoder_output]` pass. Two things fall out of this:

1. The oracle-fed pass's own NLL should stay good (a sanity check that the
   decoder "still produces a valid target" under its original training
   regime — if it doesn't, something about how the decoder was loaded/adapted
   broke it, and that needs fixing before trusting anything about the encoder).
2. The oracle-fed pass's **output distribution** (bar-distribution logits) can
   serve as a soft target for the encoder-fed pass's output distribution, via
   KL divergence, as an auxiliary loss term. This is a behavioral/outcome-level
   distillation signal (match what the decoder *predicts*, not what the
   encoder's intermediate representation *is*) — useful precisely because
   there's no requirement that `encoder_output` look like `B_inA` in raw
   value-space, only that it produces the same downstream predictions. This
   guides the encoder even where direct value-space supervision would be
   ill-defined (the encoder isn't required to reconstruct `B_inA` exactly).

Investigate and validate this as a genuine optional branch (a config flag
turning it on/off), not a mandatory part of the training loop — the primary
NLL-on-`A_test` loss must work on its own first.

## Padding / collate_fn (applies to M5's baseline too)

`n_A`/`n_B` differ per batch item (per the M2 prior contract) — both this
model and M5's baseline need this handled correctly, not just the new one:

- Padding goes on the **train/context section only** (per your instruction) —
  decide and verify what happens to the corresponding attention masks in
  `AlongColumnAttention`'s `single_eval_pos`-based masking (padded context rows
  must not be attended to as if they were real data; `single_eval_pos` alone
  doesn't know about padding within the train block).
- A **shared `collate_fn`** sits between the prior's per-item output (ragged
  `n_A`/`n_B`) and the `DataLoader`'s batched tensors — build it once (per
  `.claude/rules/model-prototyping.md`'s `collate.py`), used by both M5's
  baseline and this model, not two divergent implementations.
- This closes M2's previously-open "ragged shapes" question — see
  `docs/milestones/M2-prior-contract-toy-visualizer.md`, updated to reference
  this decision.

## Prototyping structure

See `.claude/rules/model-prototyping.md` — `src/ppfn/model/unwarp.py` gets
promoted into `src/ppfn/model/thinking_warp/` as it grows, decomposed by
architectural role (encoder input, cross-attention, decoder, encoder,
top-level model, collate), with ablations as config flags on one class rather
than new files.

## Deliverables

1. Verified extension-point analysis of `TabPFNV2p5.forward`/`_decode` for
   pulling out contextualized thinking-row embeddings (this milestone's first
   concrete task, per the "Grounding" section above).
2. The TabPFN-native cross-attention module (replacing `bridge_cross_attn`/
   `broadcast_cross_attn`), following the `Attention` base class and naming
   convention.
3. Decoder checkpoint-loading from M5, with the provenance check.
4. The restructured training loop: primary loss = decoder NLL on `A_test`
   given `[A_train, encoder_output]`; optional KL-distillation branch as a
   config flag.
5. The padding/masking fix + shared `collate_fn`, applied to both this model
   and M5's baseline.
6. `if __name__ == "__main__":` forward-pass demo per
   `.claude/rules/research-demos.md`, plus a tensor-shape verification test
   (asserting exact shapes/axis order at each architectural stage, named per
   the same axis-suffix convention) — this is the "make sure you verify what
   you are doing here" requirement made concrete.

## Acceptance criteria

- [ ] A unit test exercises the extension-point analysis's conclusion (either
      `_decode`'s embeddings dict is used directly, or a documented, minimal
      seam into `TabPFNV2p5` is added — not a copy-paste of `forward`).
- [ ] The new cross-attention module has a shape-verification test covering
      differing `n_A`/`n_B` (including the padded case).
- [ ] Decoder loaded from an M5 checkpoint reproduces that checkpoint's
      standalone eval NLL on `A_test | [A, B_inA(oracle)]` before any
      encoder/cross-attention training happens (this is the "still produces a
      valid target" sanity check from the KL side-branch, and doubles as a
      regression test that loading didn't silently corrupt anything).
- [ ] End-to-end training run (short, `00-debug-*`) shows the primary NLL
      loss on `A_test` decreasing.
- [ ] Padding is verified to not leak into attention (a test with a
      deliberately padded, short-context batch item produces the same output
      as an unpadded equivalent).
- [ ] Compared against M8's task-token baseline and M5's oracle via the M7
      callback, once training is stable enough to produce a meaningful number.
- [ ] `pytest` passes.

## Non-goals

- Not deciding frozen-vs-fine-tuned decoder weights in advance — determine
  empirically, document the choice and why.
- Not building the KL-distillation branch as mandatory — optional, validated
  independently of the primary loss path.
- Not wiring this into `configs/model/` yet if it's still changing shape
  week to week — that's M11's job once this stabilizes.
