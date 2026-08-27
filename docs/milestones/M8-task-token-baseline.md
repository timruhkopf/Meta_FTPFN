# M8 — Task-token TabPFN baseline (can plain attention undo the warp?)

## Goal

Before committing to whatever specialized architecture M9 turns out to need,
find out whether the *simplest possible* change to the existing marginal
baseline already solves the problem: take the exact same TabPFN v2.5 backbone
from M5, add nothing but a learnable per-task token added into each task's
context, concatenate A and B, and see how far plain self-attention gets on its
own at inferring `A_test`. If this already gets close to M5's oracle
(`A_test | [A, B_inA]`, which uses the *known* ground-truth transform), the
warp is something plain attention can recover implicitly and M9 may not need
much beyond this. If it doesn't, that's a real, load-bearing result — not
"architecture is presumably needed," but "we checked, and unaided attention
provably fails at this specific thing."

## The mechanism — deliberately minimal, and deliberately not MTPFN's mechanism

This is **not** MTPFN (M6). Keep the distinction sharp, since both involve
"task tokens" and it's easy to blur them:

- **This baseline**: one learnable embedding vector per task (`A`, `B`),
  **added** (element-wise) to every context token belonging to that task —
  applied once, at/after the existing input feature encoding, before whatever
  TabPFN v2.5's backbone already does. No new sequence positions, no attention
  masking, no separate intra-/inter-task attention paths. A and B's
  (now task-tagged) context tokens are simply concatenated into one sequence
  and handed to the **unmodified** TabPFN v2.5 self-attention stack, exactly
  as M5's baseline already does for a single task's context.
- **MTPFN (M6)**: a shared `[TASK]` token **prepended** as an extra sequence
  position, plus a specific interleaved intra-/inter-task attention topology
  (12 intra + 11 inter layers) — a real architectural change, not just an
  input-encoding change.

The whole point of keeping this baseline's change minimal is isolating one
variable: does the model need to be *told* which points come from which task
(this milestone), or does it additionally need *specialized attention
structure* to make use of that (M6's territory)? Conflating the two would
answer neither question.

## Setup

- Same TabPFN v2.5 wrapper as M5, same `configs/model/` entry plus a task-token
  addition (a small learnable `nn.Embedding(num_tasks=2, d_model)` or
  equivalent, added post-encoding).
- Trained on the M2 prior contract (start with the toy warp prior — it's the
  cleanest yes/no case; extend to harmonics/BNN once M3/M4 exist).
- Context: A's train points (tagged `A`) concatenated with B's **raw** train
  points (tagged `B`, *not* `B_inA` — no ground-truth transport given here,
  that's exactly what M5's oracle regime gets and this one doesn't). Query:
  `A_test`.
- Evaluated against the M1 bar distribution, compared via the M7 callback
  machinery (reuse it — this is exactly the apples-to-apples comparison it was
  built for) against:
  1. M5's `A_test | A` (lower bound — no B information at all).
  2. M5's `A_test | [A, B_inA]` (upper bound / oracle — perfect known
     transform).
  3. This milestone's `A_test | [A_tagged, B_tagged]` (raw B + task identity
     only, no known transform — plain attention has to do the work).
  4. M6's MTPFN on the equivalent conditioning, once M6 exists — same
     question, specialized-attention version.

## Acceptance criteria

- [ ] Model trains and its `A_test` NLL is reported alongside M5's two
      reference numbers (1) and (2) above, via M7's callback, in the same
      nats units, on the same bar-distribution borders.
- [ ] A written conclusion, not just numbers: does (3) land close to (2)
      (attention alone recovers the warp) or close to (1) (task tokens alone
      aren't enough)? This conclusion is the actual deliverable — it's what
      determines whether M9 needs to exist in whatever form was originally
      planned, or can be simplified.
- [ ] A demo per `.claude/rules/research-demos.md` showing the forward pass
      with task tokens applied.
- [ ] `pytest` passes.

## Non-goals

- Not building any specialized attention structure — that would defeat the
  point of isolating "just the task signal" as one variable. If this baseline
  needs more than an additive embedding to be fairly evaluated, that's a
  finding to report, not a reason to quietly add structure and blur the result.
- Not the M6-vs-this comparison if M6 isn't done yet — (1)–(3) above are
  sufficient to answer the core question; (4) is a nice-to-have once available,
  not a blocker.
- Not deciding M9's architecture — this milestone informs that decision, it
  doesn't replace your own description of the model.
