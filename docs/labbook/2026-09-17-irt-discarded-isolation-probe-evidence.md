# IRT discarded: isolation-probe evidence, and where the idea goes next

*Log, 2026-09-17. Supersedes the design and build described in
[2026-09-16-lupi-registration-mechanism-and-architecture-survey.md](2026-09-16-lupi-registration-mechanism-and-architecture-survey.md)
and `docs/labbook/2026-09-16-LUPI-ICP-Transformer.md` (the user's own copy
of the IRT design conversation, committed only on the branch below, not
ported here). Branch `lupi-iterative-registration` (off this branch at
`f85193a`) is kept on origin as a record; nothing further is built on it.*

## What was built

Following the architecture survey's own recommended incremental path, an
"Iterative Registration Transformer" (IRT) was implemented on branch
`lupi-iterative-registration`: `RegistrationIterationBlock` (RIB) —
alternating value-step (position-correspondence → h-slot update → RQS
readout) and position-step (value-correspondence + A-anchor pull + B-
smoothness pull, each with its own learnable weight/temperature) —
stacked `n_iters=4` times, an ICP-style alternating-refinement design.
Commits `c92e775` (model) and `bf810d7` (component-isolation probes).

Per the "verify components in isolation before trusting them coupled"
principle, two dedicated probe models were built and trained on ulysses at
`d=1`, each pinning one of the two unknowns to ground truth so the other's
recovery mechanism could be judged on its own:

- `ValueStepProbePFN` — `T` pinned to `enc_x_inA` every iteration (trained
  at `force_rho_zero=true`, so the pin is exact), isolating the h-slot
  update + RQS readout mechanism.
- `PositionStepProbePFN` — `h` pinned to `enc_z_inA` every iteration
  (trained at the newly-added `force_h_identity=true`), isolating the
  position-refinement (A-anchor + B-smoothness pull) mechanism.

Plotted in `notebooks/irt_component_isolation_1d.ipynb`.

## What was found

**Value-step mechanism: a real win.** RMSE 0.8085 vs. a 3.0706 do-nothing
baseline over the held-out draw plotted in the notebook — the h-slot
update + monotone RQS readout genuinely recovers the value distortion from
B's value-correspondence alone.

**Position-step mechanism: weak, and the learned parameters explain why.**
RMSE 0.1359 vs. a 0.1444 baseline — about a 6% improvement, not the clear
win the value-step probe gave. The learned step-size parameter
`alpha≈0.0948` (of the RIB's position-update pull, sigmoid-mapped from
`alpha_raw`) explains this directly: the position step learned to barely
move points at all. Each of the 4 iterations applies a near-vestigial
update, so 4 iterations end up only marginally better than 0.

This is not, on its own, evidence that ICP-style alternating refinement is
architecturally wrong for this problem — it could equally be an
optimization/initialization issue with this specific parameterization
(e.g. `alpha`'s sigmoid parameterization biasing it toward small values
early, with no pressure to grow once nearby local optima are reached), and
the coupled model (T and h refined jointly, not each pinned to ground
truth) was never itself trained to convergence before this decision.

**Also fixed en route** (kept for the record, scoped to the IRT branch
only): the value-step probe's log-sigma head exhibited the Seitzer et al.
2022 heteroscedastic-NLL pathology (mean sigma 0.87→15.2 over 40 steps on
a fixed batch, loss spike to 131) under plain Gaussian NLL, fixed with
β-NLL reweighting (detached `sigma²` as a per-sample loss scale). Not
ported back here — no head with this pathology exists on this branch —
but worth remembering if a future Gaussian-NLL head appears anywhere in
this codebase.

## Why discard rather than iterate on the position step

Discussed with the user directly rather than treated as a closed
experimental verdict: given the weak, ambiguous position-step result, and
a separate, more basic open question raised in the same conversation —
**the coupled IRT model produces a point estimate for `T`/`h` at the end of
its iteration stack, with no principled account of registration's genuine,
often multimodal uncertainty over the iterations themselves** (ICP-shaped
refinement has no obvious per-iteration uncertainty semantics to fall back
on) — the user chose to redirect toward a formulation where transport *and*
its uncertainty are native to the same mechanism, rather than debug the
current position step's `alpha` parameterization or bolt an uncertainty
head onto point-estimate iterations after the fact.

That redirection — conditional flow matching / stochastic-interpolant-style
registration, where uncertainty comes from stochastic sampling of the same
learned dynamics used to transport points, not a separate head — is logged
next in
[2026-09-17-lupi-registration-sde-flow-matching-direction.md](2026-09-17-lupi-registration-sde-flow-matching-direction.md).

## What's ported forward, what's left behind

Ported back to this branch (commit `135bdd1`): `force_h_identity`
(`ppfn.prior.lupi.sampler.sample_pair` and its threading through
`build_training_item`/`LUPIStreamDataset`/`configs/prior/lupi.yaml`/
`IDTokenTrainer`'s validation batch) — a general-purpose isolation tool,
useful independent of IRT specifically (e.g. for verifying the SDE
direction's own components the same way).

Left on `lupi-iterative-registration`, not ported: `RegistrationIterationBlock`/
`IterativeRegistrationPFN`, `RQSValueHead`/`rational_quadratic_spline_forward`,
`ValueStepProbePFN`/`PositionStepProbePFN`, their losses, their Hydra
configs, and `notebooks/irt_component_isolation_1d.ipynb`.

commit: pending
