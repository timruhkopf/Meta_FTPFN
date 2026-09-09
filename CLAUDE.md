# CLAUDE.md

Context for an agent working in this repository. Read this first.

Companion documents:
- `docs/ARCHITECTURE.md` — the buildable spec: prior, architecture, losses,
  training regime, evaluation. Section numbers referenced below point here.
- `docs/ROADMAP.md` — the argument behind the spec, and the experiment programme.
- `docs/decisions.md` — choices made, several of them reversed. Read before
  proposing a change to anything listed under "Settled against" below.
- `docs/REFERENCES.md` — papers relevant to this project, each tied back to a
  specific mechanism, baseline, or open question above; see the `/references`
  skill for the convention.

---

## The problem

Transfer information from an abundant labelled point cloud **B** to a scarce one
**A** when the two live in non-uniformly deformed input coordinates. No point
correspondence, no shared frame, different cardinalities. A PFN infers the
registration in context. Training is entirely on a synthetic prior where the warp
is known, so every registration target is supervised.

Identifying assumption (the "gauge"): `f_A = f_B ∘ T`. One latent function, two
coordinate systems. Without it `T` and `f` are confounded, since warping the input
of a flexible function class just produces another member of that class.

Motivating application is transfer between related black-box optimisation or
HPO runs, but nothing in the build depends on that and no optimiser is simulated.

---

## Conventions that are easy to get backwards

| Symbol | Meaning |
|---|---|
| `z` | latent frame. The function lives here: `y = f(z) + ε` |
| A / decoder cloud | scarce, coordinates `Φ_0(z_A)` |
| E / encoder cloud | abundant, coordinates `Φ_ρ(z_E)` |
| `T` | maps **A → B**, never B → A |
| `A_inB` | transport target, `Φ_ρ(z_A)` |
| `ρ` | relative-warp coefficient; `ρ=0` ⇒ transport is exactly the identity |
| `γ_ℓ` | cross-attention gate; all zero ⇒ exact A-only PFN |
| upper-1 | transport teacher-forced |
| upper-1b | same pair regenerated at `ρ=0` |
| upper-2 | pooled oracle, decoder-only on `[A ∪ B_inA]` |

`f` is defined on **z**, not on `S_B(z)`. Putting it in B's frame makes B's
observations a plain BNN of B's own coordinates while A's are a warped-input BNN —
detectable from marginals, and it breaks encoder/decoder role randomisation.

Both clouds are generated from a shared latent `z` via two independently sampled
warps, so every transport target is a **forward** evaluation. Nothing is inverted
anywhere in the prior. Preserve that property.

---

## Invariants — do not "simplify" these away

Each was arrived at by ruling out an alternative. Breaking one gives a model that
trains fine and measures the wrong thing.

1. **Cross-attention queries are built from `t_i^(ℓ)`, the current transport
   estimate** — not from raw `x̃_i^A`. This is what makes registration necessary
   rather than merely encouraged.
2. **Transport is supervised on query tokens, not only context tokens.** Query
   tokens are mask-isolated, so mapping a *new* `x*` into B's frame is exactly the
   inference-time operation. Context-only supervision gives a model that registers
   its own data and cannot map anything new.
3. **Layer 1 cross-attends on `y` only, no coordinates.** Before any
   cross-attention A has never seen B, so a coordinate match at layer 1 compares
   unaligned frames and every later layer refines from noise. `y` is the only
   shared undeformed channel.
4. **The `ρ=0` point mass in the curriculum is permanent**, never annealed. It
   stops the cross-attention pathway specialising on warped pairs and degrading on
   aligned ones — a failure with no other signal.
5. **`λ_T` has a floor and never reaches zero.** "Ignore the encoder, predict from
   A alone" is a safe, reachable attractor.
6. **Predictive loss scores decoder-cloud queries only.** Scoring encoder-cloud
   targets adds an easy, large, gradient-dominant term that swamps the signal from
   the regime where registration pays. Role-swapped passes contribute the
   transport and coupling losses only, never the predictive loss.
7. **Support restriction in the prior is not optional.** Under uniform sampling,
   marginal-matching and conditional-likelihood registration agree, so nothing
   forces the model to learn the design-invariant one. It will match x-marginals
   and then fail on real scarce clouds while looking fine in validation.
8. **Coordinate normalisation uses the declared domain box** — the image of the
   domain under the warp — never the empirical bounding box of sampled points. The
   empirical box is design-dependent and reintroduces exactly the bias this project
   exists to avoid. Pad it: the box is grid-estimated and the extremes of a
   diffeomorphic image of a cube need not lie on the cube's boundary, so points can
   land outside and get silently clipped into the transport head's tail bins.
9. **Deep supervision with `w_ℓ ∝ ℓ`.** Without it the stack is not coarse-to-fine
   and the depth is wasted.
10. **Distil against upper-1 at `ρ>0`; upper-2 only at `ρ=0`.** At `ρ>0` the pooled
    oracle mixes the registration gap with the architectural gap and aims the
    student at a possibly unreachable target.

---

## Settled against — don't reintroduce

- **Sorted kNN-distance descriptors for matching.** Fail twice over: intra-cloud
  distances are exactly what a non-uniform stretch alters, and A is sparse while B
  is dense, so they differ by ~n^(-1/d) even under the identity warp. Neither
  warp-invariant nor density-invariant.
- **A Gromov-Wasserstein quadratic term as a loss.** We have exact targets from
  the prior; GW is for when you have neither correspondence nor labels. Biased
  under non-uniform stretch (no zero-distortion coupling exists) and under density
  mismatch — worst on exactly the hard cases. GW stays an *evaluation baseline*.
- **Sinkhorn / balanced transport.** A covers a strict subregion of B by
  construction, so balancing imposes that A's mass spreads over all of B and drags
  the transport outward. Row-stochastic softmax is correct here.
- **Forbidding folds architecturally.** A fold is evidence the shared-`f`
  assumption is violated, and therefore the cheapest test of whether transfer is
  warranted at all. Projecting onto a diffeomorphism family destroys that signal.
  Measure first, project afterwards if an invertible map is needed downstream.
- **KV caching across warp samples.** PFN context attention is bidirectional, so
  B's representations depend on A's coordinates whenever both share a context.
  Warp-independence is *purchased* by the encoder/decoder split, not inherited.
- **Gaussian or Gaussian-mixture transport heads.** Registration is genuinely
  multimodal (reflections, periodic shifts); a Gaussian averages modes into a
  location with no support while reporting a σ that looks like honest uncertainty.
- **An unconstrained learned map from A's representation into B's space.** It lets
  the model manufacture an alignment that makes cross-attention look successful
  without corresponding to real transport. Affine, then supervised residual.
- **A global scalar attention temperature.** It would have to serve every sampled
  pair and every region within a pair. Per-row and per-pair.

---

## Always report three gaps, never a bare NLL

| Gap | Isolates |
|---|---|
| upper-2 → upper-1b | encoder/decoder split vs pooling. No frames involved. |
| upper-1b → upper-1 | cost of reading a memory in a different frame, transport known |
| upper-1 → model | registration inference — the actual research problem |

All configurations are masking or teacher-forcing changes on one set of weights,
so differences cannot be attributed to capacity or optimisation. The first two are
measurable before the model registers anything. Headline number:
`(NLL_lower − NLL_model) / (NLL_lower − NLL_upper1)`.

---

## Repo conventions

- **Config: Hydra.** Groups under `conf/`: `prior/`, `model/`, `train/`, `loss/`.
  No argparse, no hyperparameters hardcoded in `src/`. Every run reproducible from
  its resolved config. Keep a `prior=p0_identity` variant (ρ=0 always) and a
  `train=step4_pathway` variant for the go/no-go run below.
- **Tracking: MLflow.** Log the flattened resolved config as params and the monitor
  set (spec §4.5) as metrics. Keep the monitor set declared in one module; nothing
  gets logged that isn't declared there, or the dashboard stops being readable.
- **Prior in numpy, model in torch.** The prior runs in dataloader workers.
- **Tests**: the `ρ=0` invariants are the ones that matter — at `ρ=0` the transport
  target must equal the input coordinates exactly, and the pooled context must
  equal `[A ; B_inA]`. These catch frame and normalisation bugs that otherwise
  surface much later as "registration mysteriously doesn't work."
- No notebooks in the repo. Type hints on public functions. Docstrings say *why*.

---

## Build order — work top down, don't skip

1. **Prior + invariant tests.** Velocity-field warps (spec §1.3), BNN function on
   `z`, support restriction, the ρ mixture.
2. **Prior visualiser.** Plot `z` / `x_A` / `x_E` / `y` for d=1,2 over ~50 draws.
   Check the rejection rate and the realised `log|det J|` band. Nothing else is
   worth writing until the prior looks right.
3. **Severed decoder only** — a plain A-only PFN on this prior. Establishes the
   lower bound and sanity-checks the whole data path.
4. **Go/no-go.** Encoder + gated cross-attention, trained at `ρ=0` only, transport
   residual disabled (`t_i = x̃_i^A`, exactly correct there). Add the pathway
   distillation loss. Target: match upper-2. Registration contributes nothing at
   `ρ=0`, so failure here is unambiguously architecture or plumbing rather than a
   hard inference problem.
5. **Transport head + transport loss**; open the ρ curriculum. Watch transport NLL
   fall across layers and the `ρ=0` anchor hold.
6. **Coupling head 0 + coupling loss.** Visualise the attention plan at d=1 — it
   should look like a soft monotone correspondence.
7. **Global affine head + coordinate-free layer 1.**
8. **Oracle distillation, role randomisation, full monitors.**
9. **Bounds, fold calibration, baselines, spending law.**

---

## When unsure

Ask rather than guess on anything touching frames, directions, or which tokens a
loss applies to. Those are the errors that produce a plausible-looking model
measuring the wrong thing, and they are not visible in the training curves.
