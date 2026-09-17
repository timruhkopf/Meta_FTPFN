# Registration as a learned stochastic transport: flow matching / SDE direction

*Log, 2026-09-17. Design log — nothing implemented yet. Follow-up to
[2026-09-16-lupi-registration-mechanism-and-architecture-survey.md](2026-09-16-lupi-registration-mechanism-and-architecture-survey.md)
and the reason for discarding the ICP-shaped alternative:
[2026-09-17-irt-discarded-isolation-probe-evidence.md](2026-09-17-irt-discarded-isolation-probe-evidence.md).
Prompted directly by the user, who asked (1) whether formulating registration
as an ODE/PDE makes sense, after pointing at ICON (Yang, Liu, Osher et al.,
PNAS 2023, arXiv:2304.07993 — verified via ar5iv, see the discarded-IRT
entry's context: single forward pass, no iterative refinement, no
uncertainty, ruled out as a direct template for the reasons below), and (2)
for the precise mechanics of the SDE idea before anything gets built:
how points from B are transported into `hat{B_inA}`, and how uncertainty on
the result is conveyed.*

## The core idea

Treat registration itself as a **stochastic transport process**, not a
point-estimate function computed once (bare attention) or refined by a
fixed number of discrete correction steps (IRT/ICP). Concretely: define a
continuous path `z_t`, `t ∈ [0,1]`, in the *joint* position-value space,
starting at B's raw observation and ending at its fully registered image:

```
z_0 = (x_j^B, y_j^B)                    # B's own frame, raw observed value
z_1 = (x_j^{B→A}, y_j^{B→A})            # exactly enc_x_inA, enc_z_inA -- ground truth from the prior
```

A transformer `v_φ(z_t, t, context)` is trained to predict the *velocity*
`dz_t/dt` along this path, conditioned on the same pooled context the
current architecture already builds (A's context cloud, B's raw cloud).
At inference, integrating `v_φ` from `t=0` to `t=1` — by an ODE solver for
a deterministic mean estimate, or by adding a diffusion term and solving
the corresponding SDE for stochastic samples — produces `hat{z_1}`, i.e.
`hat{x_j^{B→A}}` and `hat{y_j^{B→A}}` **jointly, from one mechanism**.

This directly answers the open question that motivated dropping IRT: IRT
produced a point estimate for `T`/`h` with no native account of the
correlated uncertainty between them. Here, uncertainty is not a separate
head bolted onto a point estimate — it **is** the spread of the stochastic
process's own endpoint distribution, sampled by re-running the SDE with
different noise realizations. `T` and `h` are recovered jointly (one
`z=(x,y)` state, one learned drift), so the correlation between position
and value uncertainty is represented automatically, not assembled from two
independently-calibrated heads.

## Training mechanics: why this avoids Neural-ODE instability

The key design point, and the direct answer to "is there an SDE
transformer already": you do **not** train by backpropagating through an
ODE/SDE solver (that family — Neural ODEs, Chen et al. 2018 — is exactly
the fragile, slow-to-train approach this avoids). Instead, use **conditional
flow matching** (Lipman et al. 2023, "Flow Matching for Generative
Modeling") or the closely related **stochastic interpolants** framework
(Albergo & Vanden-Eijnden 2023): pick a fixed, simple reference path
between the known endpoints `z_0` and `z_1` — e.g. the straight line
`z_t = (1-t) z_0 + t z_1` — whose velocity is known in closed form
(`dz_t/dt = z_1 - z_0`, a constant for the straight-line choice). Train
`v_φ` by plain regression:

```
L_flow = E_t~Unif(0,1), (z_0,z_1)~prior [ || v_φ(z_t, t, context) - (z_1 - z_0) ||^2 ]
```

No solver in the loop at training time at all — `z_t` is just an
interpolation, computed directly from the LUPI prior's own known `z_0`,
`z_1` for every draw. This is *simulation-free* training, the entire point
of the flow-matching reformulation over classical Neural-ODE training, and
it is the reason this is tractable at the scale this project already
trains at (large batches of short, cheap regression targets, exactly like
every other loss in this codebase — nothing about this needs a slower
per-step solver call during training).

The solver only appears at **inference** time, when actually computing
`hat{z_1}` from `z_0` — and only there does step count become a real,
principled knob: more integration steps trade compute for lower
discretization error, a property with actual theory behind it (an ODE
integrator's local truncation error), unlike IRT's `n_iters=4`, which was
an architectural hyperparameter with no comparable justification.

## Where uncertainty comes from, concretely

Two options, from simplest to richest:

1. **Deterministic flow (ODE), stochastic only via the model's own
   epistemic ensemble** — cheapest, but doesn't give calibrated per-draw
   aleatoric uncertainty; not the target design, mentioned only for
   completeness.
2. **A genuine SDE**: `dz_t = v_φ(z_t, t, context) dt + σ(t) dW_t`, with a
   learned or scheduled noise scale `σ(t)`. Sampling the *same* trained
   `v_φ` multiple times with independent Brownian paths produces a genuine
   empirical distribution over `hat{z_1}` per query point — multimodal by
   construction if the underlying registration genuinely has more than one
   plausible resolution (e.g. reflection ambiguity), which is exactly the
   kind of uncertainty CLAUDE.md's "settled against Gaussian/GMM transport
   heads" entry already flags as a hard requirement. This is the
   **Schrödinger Bridge Matching** picture (Liu et al. 2023, "I2SB: Image-
   to-Image Schrödinger Bridge") — the closest existing framing to "a
   learned stochastic bridge between two known point distributions," since
   I2SB already handles exactly the paired-endpoints setting this project
   has (not the harder unpaired-marginals problem Schrödinger bridges are
   more commonly posed for).

At inference, the practical recipe is: draw `K` independent SDE
trajectories per query point, report the empirical mean as the point
estimate and the empirical spread (or fit a light output distribution to
the `K` samples) as the uncertainty — a Monte Carlo cost genuinely new to
this project (nothing currently trained requires multiple stochastic
forward integrations per query), and the main new engineering complexity
this direction introduces.

## Precedent for the backbone: is there an "SDE transformer"?

Not exactly a drop-in one, but composable precedent exists for every piece:

- **DiT** (Peebles & Xie 2023) — a plain transformer backbone (not a U-Net)
  used as the score/velocity network for diffusion/flow-matching models,
  with the timestep `t` and any conditioning injected via adaptive layer
  norm. Establishes that a transformer is a perfectly good architecture for
  `v_φ` — nothing about flow matching requires a convolutional backbone.
- **FoldFlow** (Bose et al. 2023) — SE(3) flow matching specifically over
  **point clouds** (protein backbones), i.e. exactly the "predict a
  per-point velocity, conditioned on the rest of the set, via attention"
  shape this would need, just in a different domain.
- **LDDMM** (Beg et al. 2005) is worth naming explicitly as the classical
  anchor: diffeomorphic image/shape registration *already* works by
  integrating a time-varying velocity field, which is **literally the same
  mechanism the LUPI prior itself uses to generate `T`** (`flow_rk4`
  integration of a sampled velocity field — see `ppfn.prior.registration.warp`).
  This is the strongest inductive-bias argument for this direction over
  attention or ICP: the model would be learning to approximate the same
  *class* of process that generated its own training data, rather than an
  unrelated mechanism (bare attention, alternating correspondence/update)
  being asked to reproduce a velocity-field integral's output indirectly.

No single paper combines "transformer backbone + point-cloud conditioning +
paired-endpoint (Schrödinger-bridge-style) flow matching + explicit
uncertainty via SDE sampling" for a registration-shaped problem — this
would be a genuine synthesis across FoldFlow's set-attention conditioning,
I2SB's paired-endpoint bridge framing, and DiT's transformer-as-backbone
choice, not an off-the-shelf architecture to import.

## Open questions, deliberately not settled here

- **Reference path choice**: straight-line interpolation is simplest but
  assumes a "reasonable" path between `z_0` and `z_1` exists in raw
  position-value space; the stochastic-interpolants framework generalizes
  this (e.g. paths with their own built-in noise schedule) if the
  straight-line choice turns out to train poorly.
  This is the free
  parameter to isolate first once this is actually built, mirroring how
  `force_rho_zero`/`force_h_identity` isolate `T`/`h` for the existing
  architecture.
- **Coupling to the predictive stack**: does `v_φ`'s context conditioning
  reuse the existing pooled-attention backbone (A_ctx + raw B) wholesale,
  or is it a smaller dedicated module whose *output* (`hat{z_1}` samples)
  then feeds a separate downstream PFN readout? The former keeps one
  backbone; the latter isolates flow-matching training from the
  predictive-loss curriculum, closer to how the transport head is already
  a separate module today.
- **Inference cost**: `K` SDE samples × `n_b` points × (solver steps) per
  forward pass is new, real inference-time cost this project hasn't had to
  budget for yet.
- **Verification plan, following this branch's own "isolate before
  trusting coupled" convention**: at `rho=0`, `force_h_identity=true`, the
  flow's target degenerates to the identity map (`z_0 == z_1`) — the same
  cheap sanity check already used for `TransportHead` and the IRT probes,
  now checking that `v_φ` learns to predict near-zero velocity everywhere
  along that degenerate path before trusting it on real (`rho>0`, real
  `h`) draws.

## Status

Nothing implemented. New branch to carry this work: created off this
branch (`pfn-baseline-reset`) as `lupi-registration-sde`.

commit: pending
