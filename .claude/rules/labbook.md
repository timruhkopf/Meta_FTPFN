# Lab book — `docs/labbook/`

A running, append-only log of things learned and ideas followed up on —
distinct from `docs/milestones/` (which describes *planned* work) and
`docs/reviews/` (point-in-time audits). This is for empirical findings as
they happen: "we investigated X and found Y," "tried Z, it didn't work,
here's why," "here's a fix and the evidence it actually helped." Adopted
2026-08-27 at the user's request, mirroring a convention from another of
their repos.

## Structure

- `docs/labbook/README.md` — the index. **Append-only**: add a new line per
  entry, in an "Entries" table/list, in chronological order. Never edit or
  remove a past line to "correct" it — if a finding turns out to be wrong or
  superseded, add a *new* entry that says so and links back to the old one.
  The log is the record of what was believed and when, not just what's
  currently true.
- One file per entry, `docs/labbook/YYYY-MM-DD-descriptive-slug.md` —
  descriptive enough to identify the finding from the filename alone in a
  directory listing, matching the convention already used for
  `docs/milestones/M<n>-<slug>.md` and `docs/reviews/YYYY-MM-DD-<slug>.md`.

## What goes in an entry

- What was investigated and why (link the milestone/conversation that
  prompted it, if any).
- What was actually found — with the evidence (numbers, a quoted repro, a
  plot reference), not just the conclusion. "Verified empirically" needs the
  verification shown, not asserted — same bar as everywhere else in this repo.
- If a fix was applied: what changed, and the before/after evidence that it
  actually helped. **If a first attempt at a fix turned out to be wrong,
  say so explicitly** — don't quietly rewrite history to look like the right
  answer was obvious from the start (see the BNN `init_std` entry for the
  precedent: the first calibration attempt was verified and found to make
  things *worse*, and the entry says so before describing the corrected one).
- A `commit:` field with the short git hash of the commit that introduced the
  finding/fix, once committed. Leave as `pending` if the corresponding change
  hasn't been committed yet — don't guess or backfill a hash, and don't block
  writing the entry on committing first. Whoever commits the change should
  update the entry's `commit:` field (or ask to have it updated) once the
  hash exists.

## When to add an entry

Any time a real investigation produces a finding worth remembering — a bug
with a non-obvious root cause, a design decision made for a specific reason,
a "we tried X, here's what actually happened" result. Not every code change
needs one; routine implementation work covered by a milestone's own
deliverables/acceptance criteria doesn't need a separate lab-book entry
unless something *unexpected* was learned along the way.
