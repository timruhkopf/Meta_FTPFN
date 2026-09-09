---
name: references
description: Add or update an entry in docs/REFERENCES.md, the running collection of papers relevant to this project. Use when the user says "/references <paper/link>", pastes a paper URL and asks to track it, or asks what we know about a paper/method already in the file.
---

# Maintaining `docs/REFERENCES.md`

A running collection of papers worth remembering *in relation to this
project* — not a general-purpose bibliography. Every entry earns its place
by connecting back to something in `docs/ROADMAP.md` / `docs/ARCHITECTURE.md`
(a mechanism we're using, a baseline we owe, a method we're positioning
against), not just "this is relevant to PFNs/BO in general."

## Adding an entry

1. **Get the actual paper**, don't work from the title alone. Fetch the link
   (or read the local PDF if the user says they've downloaded one — check
   `external/` first, e.g. `external/*.pdf`) and read enough to state the
   idea accurately.
2. **File it under an existing section** of `docs/REFERENCES.md` by topic
   (e.g. "PFNs as surrogates", "Privileged information / auxiliary
   supervision"). Add a new section heading only if nothing existing fits —
   check with the user before inventing a taxonomy from scratch if it's not
   obvious.
3. **Write the entry** in this exact shape:

   ```markdown
   ### <Title> ([link](<url>))[ — [repo](<gh-url>)]

   **Idea:** 2-4 sentences on the actual mechanism/method, specific enough
   that someone could tell it apart from a neighboring paper in the same
   section. Not a one-line label.

   **Relevance / limitations:** How this connects to *our* problem or
   approach specifically — what we'd reuse, what result of theirs bears on a
   design choice we made, or what baseline it is. Then say where it falls
   short for us or where it's merely adjacent (different setting, different
   assumption we don't get to make, etc.) — a reference with no stated
   limitation is a lead you haven't finished chasing.
   ```

   - Add the `— [repo](...)` link only when a reference implementation
     exists and matters for reproducing the method (we'd actually look at
     the code) — not reflexively for every paper.
   - If the user flags a paper as a **baseline we need to reconstruct**
     (not just read), say so explicitly in **Relevance / limitations** with
     that exact phrase, and cross-reference it from `docs/ROADMAP.md` too
     (its baseline ladder table, e.g. §8.4, or the "Open items" section if
     no table fits) — a reference-only note in `docs/REFERENCES.md` is easy
     to lose track of; the roadmap is what actually gets worked from.
   - Note version/lineage caveats inline where they matter (e.g. "TabPFN"
     covers multiple, meaningfully different architectures across major
     versions — say which version a claim applies to rather than citing
     "TabPFN" as if it were one fixed method).

4. **Don't rewrite past entries to relitigate them.** If a later paper
   changes how we read an earlier entry, add a note to the later entry
   pointing back, rather than editing the old one — same append-only spirit
   as `docs/labbook/`, though this file isn't a strict chronological log so
   reorganizing section structure (not entry content) is fine.

## What this skill is not for

- Not a general literature-search tool — don't go fetch "related work" the
  user hasn't pointed at. Add what's requested; suggest more only if an
  obvious, closely-related omission jumps out while you're already reading
  something.
- Not for `docs/labbook/` findings (empirical results from *our* experiments)
  or `docs/ARCHITECTURE.md`/`docs/ROADMAP.md` design decisions — those are
  different files with different conventions.
