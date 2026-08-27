---
name: milestone
description: Work a roadmap milestone end-to-end (read spec, plan, implement, verify against acceptance criteria, update status). Use when the user says "/milestone M1" or "go for M2" etc.
---

# Working a milestone

Argument: a milestone id (e.g. `M0`, `M1`). If none is given, read
`docs/ROADMAP.md` and ask which one, rather than guessing.

## Steps

1. **Read the spec.** Files are named `docs/milestones/<id>-<slug>.md` (e.g.
   `M8-task-token-baseline.md`) — match on the `<id>-` prefix, don't assume the
   exact slug. If nothing matches, stop and say so — don't improvise a
   milestone that hasn't been written. Read `docs/ROADMAP.md` for the exact
   filename/link and surrounding context (what's already done, what this
   depends on).

2. **Read the invariants that apply.** Root `CLAUDE.md`, plus any nested
   `CLAUDE.md` in directories this milestone's deliverables touch. Don't skip
   this even if the milestone looks simple — the "Known messy state" and
   invariant sections exist specifically to prevent silent-wrong ML bugs (prior
   mismatch, DictConfig leaking into library code, etc.).

3. **Plan in vertical slices.** Each deliverable in the spec should map to a
   checkable unit of work. If a deliverable is vague, that's a spec problem —
   flag it back to the user rather than resolving the ambiguity silently,
   especially for anything touching prior/model semantics (getting this wrong is
   invisible, not a crash).

4. **Implement**, respecting existing conventions over introducing new ones
   (this is a research repo mid-refactor — match the surrounding pattern unless
   the milestone explicitly asks to change it).

5. **Verify against the acceptance criteria section**, item by item — actually
   run the commands listed there (tests, a debug training run), don't reason
   about whether they'd pass. For anything Hydra/training-related, prefer a
   short `experiment_name=00-debug-*` run over trusting static analysis.
   Whenever possible and sensible, add a test to tests/ for any new behaviour.
   remove tests, when they are no longer supported. Notify the user beforehand!

6. **Report.** For each acceptance-criteria checkbox: done, or blocked-and-why.
   Update the checkboxes in the milestone file and the status column in
   `docs/ROADMAP.md`. If new open questions surfaced, add them to the milestone's
   "Open questions" section rather than silently deciding.

## What this skill is not for

- Not for open-ended "improve X" requests with no milestone file — just do the
  task normally.
- Not for deciding what the next milestone should be — that's the user's call;
  at most, note in your report that M0 (or whichever) surfaced work worth turning
  into a new milestone.
