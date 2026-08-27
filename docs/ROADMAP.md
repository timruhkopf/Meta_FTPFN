# Roadmap

Index of active/planned milestones. Each has its own file in `docs/milestones/`
with a goal, deliverables, acceptance criteria, and explicit non-goals. To work a
milestone, use `/milestone <id>` (see `.claude/skills/milestone/`) rather than just
saying "go do M2" — the skill reads the spec, plans against it, and verifies the
acceptance criteria before reporting done.

Status legend: `todo` / `in-progress` / `blocked` / `done`.

| ID | Title | Status |
|----|-------|--------|
| M0 | [Repo hygiene: resolve known config/prior drift](milestones/M0-repo-hygiene.md) | parked |
| M1 | [Bar distribution as the first-class loss](milestones/M1-bar-distribution.md) | todo |
| M2 | [Prior contract + toy warp/shift/scale prior + visualizer](milestones/M2-prior-contract-toy-visualizer.md) | todo |
| M3 | [Harmonics prior on the new contract](milestones/M3-harmonics-prior.md) | todo |
| M4 | [BNN prior + relatedness mechanism](milestones/M4-bnn-prior-relatedness.md) | todo |
| M5 | [Marginal single-task PFN baseline](milestones/M5-marginal-baseline.md) | todo |
| M6 | [MTPFN baseline: reintegrate + retrain on the new prior](milestones/M6-mtpfn-baseline.md) | todo |
| M7 | [Cross-model NAT-comparison callback](milestones/M7-nat-comparison-callback.md) | todo |
| M8 | [Task-token TabPFN baseline (can plain attention undo the warp?)](milestones/M8-task-token-baseline.md) | todo |
| M9 | [Transport PFN: encoder→decoder via thinking-row cross-attention](milestones/M9-proposed-model.md) | todo — spec done, prototype exists |
| M10 | [Real benchmarks: mfpbench / HPOBench + multi-fidelity budget prior](milestones/M10-real-benchmarks.md) | deferred |
| M11 | [Config completion for real training runs](milestones/M11-config-completion.md) | sketch — blocked on M2–M6 |
| M12 | [Evaluation entry point (`evaluate.py`)](milestones/M12-evaluate-entry-point.md) | sketch — blocked on M5–M7 |
| M13 | [MLflow tracking backend for concurrent SLURM jobs](milestones/M13-mlflow-tracking-backend.md) | sketch — design can start now |
| M14 | [Cluster orchestration (sweeps)](milestones/M14-cluster-orchestration.md) | sketch — blocked on M13 |

M1–M8 form one arc: get the shared loss right, get one prior + its visualizer
right as the template for the other two, get both baselines trained and
comparable on equal footing, and answer the "does plain attention already
solve this" question (M8) before committing to the actual model idea (M9) or
touching real benchmarks (M10). Don't skip ahead to M10 "to see if it's
promising" — the whole point of M1–M8 is that real-benchmark numbers are
uninterpretable without a validated marginal baseline, a known-faithful MTPFN
comparison, and a clear answer on whether M9 needs to exist at all in its
planned form.

M11–M14 are the infrastructure this arc will eventually need at cluster scale
(real configs per run, an eval-only entry point, a tracking backend that
survives 72h-bounded SLURM jobs without a persistent DB server, and sweep
orchestration on top of it) — cross-cutting, not part of the M1→M10 dependency
chain itself, and mostly written as forward-looking sketches since M11/M12/M14
need interfaces M2–M7 haven't produced yet. **M13 is the exception**: the
tracking-backend problem is fully specified by a real, already-hit constraint
(no persistent Postgres, SQLite empirically fails under this cluster's
concurrency, the filesystem-store workaround is no longer viable) and doesn't
need to wait on the model/prior work — see `docs/milestones/M13-mlflow-tracking-backend.md`.

**2026-08-26:** the active tree was reset to a bare skeleton (trainer/callbacks,
entry point, `bnn` prior, two cross-cutting utils) — everything M0 concerns
(the `harmonics` config, the `harmonics`/`harmonics_fix` fork) moved to
`archive/`. M0 is parked, not done or dropped: revisit it if/when `harmonics` is
pulled back in (see `archive/README.md`).

**2026-08-27:** `configs/` and `tests/` rebuilt to match the kept skeleton
(`prior=bnn`, `trainer`, `optimizer`, `scheduler`, `deployment` via
`hydra-submitit-launcher`); `model`/`benchmark` config groups intentionally left
mandatory-and-unset pending M9/M10.

## Non-goals for this roadmap file itself

- Not a design doc — milestone files hold the "why", this file is just an index.
- Not a backlog of every idea in `archive/src/ppfn/model/anamorphic/ideas/` —
  those are proposals under evaluation, parked, not committed work items.

## Problem Setting

I want to write research code for PFNs, that enable them to use meta-knowledge
from previous experiments to improve their performance on new tasks. Critically,
i have a task A, which is the target and has sparse data, and a task B, which is
the source/related and is dense, but unfortunately distorted by e.g. warping.
