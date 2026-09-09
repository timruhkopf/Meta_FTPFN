# Notebooks

Ad hoc `.ipynb` investigations — a specific question, run interactively, with
the plots that answered it kept inline. Distinct from the other two places
that look similar:

- `.claude/rules/research-demos.md` demo blocks (`if __name__ == "__main__":`
  in a model/prior module) are a permanent, cheap sanity check that ships
  with the module and stays correct as the module changes.
- `docs/labbook/` is the write-up once an investigation produces a finding
  worth remembering — the conclusion and the evidence, not the exploration
  that got there.

A notebook here is the exploration itself: allowed to be messy, dead ends and
all. If it produces a real finding, write it up in `docs/labbook/` per
`.claude/rules/labbook.md` and link back to the notebook rather than
duplicating the analysis there.

## Convention

- One notebook per question, named `YYYY-MM-DD-descriptive-slug.ipynb`.
- Clear outputs before committing only if they're large/binary-heavy
  (big arrays, big plots that bloat diffs); keep small plots inline since
  that's the point of using a notebook here.
