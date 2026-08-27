# M10 — Real benchmarks: mfpbench / HPOBench + multi-fidelity budget prior

## Status: deferred

Explicitly not to be started until M1–M7 (and ideally M9) are validated on the
synthetic priors — real-benchmark numbers are uninterpretable without a known
marginal baseline and a known-faithful MTPFN comparison already in hand. This
file is a sketch of what's known so far, not a ready-to-work spec.

## Corrections to check before relying on prior assumptions

- **HPOBench is not currently present anywhere under `external/`** — only
  `mf-prior-bench` is (`external/ifbo_icml2024/src/mf-prior-bench`), and it's
  currently **commented out** of `pyproject.toml`'s dependencies/workspace
  members, i.e. not installed editable right now. Confirm HPOBench is still
  wanted (clone it) before this milestone assumes it's available.
- `mfpbench`'s prior implementation attempt already in
  `archive/src/ppfn/prior/multi_fidelity/` (`mf_ftpfn`, `mf_ftpfn_refactor`)
  is explicitly **not validated** per your own note — treat it as a starting
  point to check, not working code to build on top of unchecked.

## Known requirements

1. **Meta-train/test split over tasks** — held-out tasks, not held-out points
   within a task; needs an explicit split policy (random, by task family, by
   some difficulty/similarity stratification) decided and documented before
   any benchmark number is reported.
2. **Search-space compatibility check** — "sufficiently compatible" search
   spaces across tasks needs an actual criterion (e.g. matching
   dimensionality/type after a documented canonicalization, or a measured
   compatibility score with a threshold), not an eyeballed judgment per task
   pair.
3. **Dirichlet multi-fidelity budget-allocation prior** — allocates budget
   across curves in a sequence. `archive/src/ppfn/prior/multi_fidelity/
   mf_ftpfn_refactor/allocation_prior.py`'s `AllocationPrior` already does
   something in this spirit (Gamma-draws normalized to weights, which *is* a
   valid Dirichlet-sampling construction: `Gamma(α,α)` draws normalized sum to
   `Dirichlet(α,...,α)`) — check whether it already satisfies what you have in
   mind or needs rebuilding; it predates the M2 prior contract and isn't
   necessarily consistent with it.

## Non-goals (for now)

- Not starting implementation before M1–M7 are done.
- Not assuming HPOBench is available without checking.
- Not treating the archived `mf_ftpfn`/`mf_ftpfn_refactor` code as validated.
