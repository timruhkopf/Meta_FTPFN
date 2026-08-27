# M0 — Repo hygiene: resolve known config/prior drift

> **Parked as of 2026-08-26.** The repo was reset to a bare skeleton and
> everything this milestone touches (`configs/prior/harmonics.yaml`,
> `src/ppfn/prior/harmonics*`) moved to `archive/`. Don't work this milestone
> until those are pulled back — see `archive/README.md`. Left as-is otherwise so
> the reasoning isn't lost.

## Goal

Close the gap between what `configs/` claims and what `src/ppfn/` actually
contains, so the next milestone doesn't inherit silent footguns. This is cleanup,
not new capability.

## Deliverables

1. `configs/prior/harmonics.yaml` either repointed to a real `_target_` under
   `src/ppfn/prior/`, or removed if superseded — resolve the
   `prototype.harmonic_restart.harmonic_prior.InfiniteHarmonicsStream` dangling
   reference (see `CLAUDE.md` "Known messy state").
2. A decision (recorded, not just made in someone's head) on `src/ppfn/prior/
   harmonics/` vs. `src/ppfn/prior/harmonics_fix/`: are they merged, is one
   deprecated, or do both stay with a documented reason? Whatever the answer,
   `configs/prior/*.yaml` should unambiguously point at the intended one(s).
3. A one-line note per surviving `anamorphic/ideas/*.md` proposal on its current
   status (active candidate / superseded / parked) so the directory reads as a
   decision log rather than an undifferentiated pile.

## Acceptance criteria

- [ ] `python src/train.py experiment_name=00-debug-m0 prior=harmonics` runs past
      dataset instantiation without an import/target error.
- [ ] No remaining reference to `prototype.harmonic_restart` anywhere in `configs/`.
- [ ] `pytest tests/prior/` passes.
- [ ] `docs/milestones/M0-repo-hygiene.md` (this file) has every checkbox above checked, or the
      remaining ones explained under "Open questions" below.

## Non-goals

- Not fixing the trainer/callback design debt noted in `.claude/rules/mlflow.md`
  (callback owns run lifecycle) — that's a separate, larger change with its own
  tradeoffs, not repo hygiene.
- Not resolving which anamorphic design direction to pursue — only labeling status.
- Not touching `external/ifBO` — vendored, out of scope here.

## Open questions

- Is `harmonics_fix` a strict superset/replacement of `harmonics`, or do they model
  meaningfully different priors that both need to stay reachable via config? This
  determines whether M0's deliverable 2 is a merge or a rename+dual-support.
