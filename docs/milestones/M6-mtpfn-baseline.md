# M6 — MTPFN baseline: reintegrate + retrain on the new prior

## Goal

Get MTPFN (Li, Daulton, Müller, Wilson, Bakshy — NeurIPS 2025 SPIGM Workshop,
`external/92_Robust_Transfer_for_Bayesia.pdf`) working as a comparison baseline
under this repo's trainer/prior contract, and show it fails to recover the warp
effectively when trained on the new A/B-warp prior — as the paper's own
architecture would predict, since it wasn't designed for this.

## Important: this already mostly exists — don't rebuild it

`archive/src/ppfn/model/baselines/mtpfn/` contains a faithful reproduction
built directly from the PDF (not a guess), with a detailed README documenting
exactly where an earlier attempt got corrected against the real paper (output
head is Gaussian not bar-distribution, one shared `[TASK]` token not per-task
embeddings, 12 intra + 11 inter interleaved attention layers, Algorithm A.1's
Dirichlet-task-proportion prior, the specific resampling mechanism for
negative transfer, and the paper's exact training hyperparameters). It was
smoke-tested end-to-end at the paper's reported scale (~72.5M params). Read
that README in full before writing anything — re-deriving this from the paper
again would be redoing work that's already correct.

There are genuinely **two separate things** here, both needed:

1. **The faithful reproduction, on the paper's own prior** (`prior.py`'s
   Algorithm A.1 — ICM kernel + Dirichlet task proportions). This is what
   "precise to the letter" refers to and it's *already built* in `archive/`.
   Reintegrating it (`git mv` back per `archive/README.md`) and confirming it
   still trains is this milestone's first deliverable.
2. **MTPFN retrained on the M2 prior contract** (toy/harmonics/BNN's A/B-warp
   structure, not the paper's own multi-task ICM prior) — this is the actual
   comparison you want: does an architecture built for "many tasks, some
   related, learned via a shared task token" recover a *specific, known,
   invertible* two-domain warp as well as a model that's told about the
   transform directly? The paper's own prior doesn't have `A_inB`/`B_inA` at
   all, so this requires adapting MTPFN's input handling to consume the M2
   contract's shape (2 tasks: A and B, `share_unrelated` playing a role
   analogous to the paper's own `p`), not its own prior.

## The "more prepended tokens" variant

Your own extension — more than one learned prepended token, in place of
MTPFN's single shared `[TASK]` token — should be built as a clearly-separated
variant (e.g. `n_task_tokens` config knob defaulting to the paper's `1`), not a
silent modification of the faithful reproduction. Keep both reachable: the
`n_task_tokens=1` config must still reproduce the paper's exact architecture,
since that's the fair-comparison anchor for whatever you build in M9.

## Deliverables

1. `mtpfn/` reintegrated from `archive/`, importable, trainable via
   `PPFNTrainer`, config group under `configs/model/` (or wherever the
   eventual `model` group convention lands — see `configs/model/README.md`).
2. MTPFN trained on its own paper-faithful prior (regression test that the
   reintegration didn't break anything) and, separately, on the M2 prior
   contract (the actual new comparison).
3. The `n_task_tokens` variant, defaulting to `1` (paper-faithful).
4. A report/plot showing MTPFN's warp-recovery failure mode on the new prior
   (e.g. its predictions on `A_test` given `[A, B]` vs. the marginal
   baseline's `A_test | [A, B_inA]` from M5 — the gap is the point).

## Acceptance criteria

- [ ] `archive/src/ppfn/model/baselines/mtpfn/` reintegrated with no
      regressions versus its own README's claims (parameter count, smoke test
      behavior).
- [ ] MTPFN trains on the M2 prior contract without architecture changes
      beyond input adaptation (i.e. the comparison is fair — same trainer,
      same bar-distribution-or-documented-deviation, same prior).
- [ ] `n_task_tokens=1` reproduces the original architecture's parameter count
      and layer structure exactly.
- [ ] The failure-mode comparison against M5's baseline is produced.
- [ ] `pytest` passes; a demo per `.claude/rules/research-demos.md` shows a
      forward pass.

## Non-goals

- Not re-deriving MTPFN from the paper from scratch — reuse and validate the
  existing `archive/` implementation.
- Not fixing MTPFN's warp-recovery failure — demonstrating and measuring it is
  the point of this milestone, not solving it (that's what M9's model is for).
