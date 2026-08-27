---
paths:
  - "src/ppfn/model/thinking_warp/**"
  - "src/ppfn/model/unwarp.py"
---

# Prototyping structure for the transport-PFN model (M9)

This architecture is under active design (see
`docs/milestones/M9-proposed-model.md`) — the goal here is letting it change
shape repeatedly without becoming another `archive/src/ppfn/model/anamorphic/`
(a dozen competing whole-file prototypes with no shared structure, eventually
all archived at once because none of them could be told apart or built on top
of each other). One canonical package, decomposed by architectural role.

## Package layout

`src/ppfn/model/unwarp.py` (the current single-file prototype,
`ThinkingWarpTabPFN`) is the starting point, not a throwaway — it already
trains (see its own `__main__` block's logged loss curve) and already uses
real `TabPFNBlock`s for the per-stream row+column attention. Promote it into a
package as the architecture grows, split by role, not by experiment attempt:

```
src/ppfn/model/thinking_warp/
    __init__.py
    encoder_input.py   # cell/target encoding (currently unwarp.py's TabPFNEncoder
                        # + TorchStandardScaler) — hand-reimplemented from
                        # tabpfn_v2_5's own encoding path; periodically diff
                        # against tabpfn.architectures.tabpfn_v2_5's
                        # _add_column_embeddings/feature_group_embedder for drift
                        # rather than letting the two silently diverge.
    cross_attention.py # the TabPFN-native replacement for unwarp.py's
                        # bridge_cross_attn/broadcast_cross_attn (see M9 —
                        # these currently flatten thinking-rows*features into one
                        # sequence for a generic nn.MultiheadAttention, which
                        # throws away the Row/Column axis structure that's
                        # otherwise carefully preserved everywhere else).
    decoder.py          # the checkpoint-initialized decoder (M5's marginal
                         # baseline, loaded, not a fresh TabPFNBlock stack).
    encoder.py           # the row+column-attention encoder + its cross-attention
                          # into the decoder's contextualized thinking rows.
    model.py              # top-level module composing encoder+decoder(+optional
                           # KL-distillation head, see M9). This is what
                           # unwarp.py's ThinkingWarpTabPFN.forward becomes.
    collate.py             # the padding/masking collate_fn (see M9's padding
                            # requirement) — shared with M5's baseline, not
                            # duplicated.
```

## Rules while this is in flux

- **One architecture, config-flag ablations.** A variant (e.g. "with vs.
  without the KL-distillation branch," "1 vs. N cross-attention layers") is a
  constructor/config parameter on `model.py`'s class, not a new file. If an
  ablation needs genuinely different code paths, an `if` inside the one class
  is still better than a sibling file — the anamorphic mess happened one
  "just copy it and tweak" file at a time.
- **Reuse `tabpfn.architectures.tabpfn_v2_5`'s actual modules
  (`TabPFNBlock`, `AlongRowAttention`, `AlongColumnAttention`,
  `AddThinkingRows`, the `Attention` base class) — don't reimplement attention
  math that already exists there.** Where something genuinely new is needed
  (the cross-attention module doesn't exist upstream), build it by
  subclassing/mirroring `Attention`'s structure, not `nn.MultiheadAttention`
  from scratch — see `.claude/rules/hydra.md`-adjacent reasoning: staying
  close to existing, already-correct code is what "change only where
  necessary" means in practice.
- **Follow `tabpfn_v2_5.py`'s tensor-naming convention exactly**: every tensor
  variable name carries a suffix spelling out its axis order in single letters
  — `B`=batch, `R`=rows (thinking+train+test), `C`=feature-groups/columns,
  `E`/`D`=embedding or head dim, `T`=thinking rows, `N`=train(+thinking) rows,
  `M`=test rows, `H`=heads, `F`=`head_dim*num_heads`. e.g. `x_BRCE`, `x_BcRE`
  (batch*columns folded, attention along rows), `x_BrCE` (batch*rows folded,
  attention along columns/features). This isn't cosmetic — getting an axis
  wrong in this code is a silent shape-broadcast bug, not a crash, given how
  much of TabPFN's own code relies on `.view`/`.reshape`/`.transpose` without
  further validation. Name tensors so a wrong axis is visible on sight.
- **Verify tensor shapes at each stage explicitly** (asserts or a shape-check
  test) rather than trusting that concatenation/reshape "probably" did the
  right thing — required by M9's acceptance criteria, not optional polish.
