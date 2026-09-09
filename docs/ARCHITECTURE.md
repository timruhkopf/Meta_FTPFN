# Concrete Specification: Encoder–Decoder PFN for In-Context Registration and Transfer

Companion to `ROADMAP.md`. That document argues the design; this one is buildable. Everything here is prior-only. No BO runs, no BO emulation.

---

## 0. Summary of the design

A transformer encoder–decoder. The **encoder** ingests the abundant cloud B and produces a memory. The **decoder** is a PFN over the scarce cloud A plus its query points, with gated cross-attention to that memory. The decoder's cross-attention queries are built from the model's **current estimate of where each A-point sits in B's frame**, so registration is on the critical path to using B at all.

Three properties this buys, each of which was a separate mechanism in the roadmap and is now structural:

| Property | Mechanism |
|---|---|
| A-alone baseline, same weights | Zero the cross-attention gates |
| Oracle baseline, same weights | Teacher-force the transport with ground truth |
| Collapse cannot be free | Cross-attn query depends on transport; abandoning registration forfeits B |
| Oracle is a *teacher*, not just a bound | Same weights with transport teacher-forced; student distilled against its PPD (§3.5) |

---

## 1. Generative prior

One sample = one dataset pair. Nothing here requires simulating an optimizer.

### 1.1 Draw

```
d          ~ Uniform{1, 2, 3, 5}
n_B        ~ LogUniform[256, 1024]        (integer)
n_A        ~ LogUniform[8, 256]           (integer)

# latent points
z^B        ~ p_z over [0,1]^d              (n_B points)
z^A        ~ p_z restricted to R_A         (n_A points)

# two warps, both forward-evaluated only
S_A, S_B   ~ p_S
x^A = S_A(z^A)        x^B = S_B(z^B)

# function in the LATENT frame
f          ~ BNN prior
y = f(z) + ε,   ε ~ N(0, σ_obs²)
```

**Why f is defined on z and not on B's frame.** The roadmap wrote `y = f(S_B(z))`, which makes B's observations a plain BNN of B's own coordinates while A's are a warped-input BNN. That asymmetry is detectable from marginal statistics and is incompatible with the role-swapping in §4.3. Putting f on z makes both clouds warped-input BNNs of their own coordinates, so neither is distinguishable as "the undistorted one." The gauge assumption still holds, with f_B(u) := f(S_B⁻¹(u)) existing implicitly and never needing evaluation.

### 1.2 Support mismatch without simulating an optimizer

`R_A` is a random restriction of the latent domain:

```
R_A ~ { full domain              w.p. 0.40
        axis-aligned sub-box     w.p. 0.30   (volume fraction ~ U[0.15, 0.7])
        ball around a point      w.p. 0.20   (radius ~ U[0.2, 0.6])
        union of 2-3 blobs       w.p. 0.10 }
```

Within `R_A`, draw z^A from a mixture of uniform and clustered (2–5 Gaussian blobs, bandwidth ~ U[0.03, 0.15]).

This produces the phenomenon that matters — A covering a strict subregion of B's support, with a non-uniform density inside it — using two lines of sampling code. A y-band concentration arises for free at whatever rate a restricted region happens to land on a monotone part of f, which is the honest rate rather than one hand-tuned to mimic an acquisition trajectory. **Log the realized region type, volume fraction, and A's y-range as a fraction of B's**, so results can be stratified by mismatch severity after the fact.

Deliberately out of scope: sequential/adaptive designs. Deferred, see §8.

### 1.3 Warp family

Stationary velocity field, integrated by RK4 with 5 steps:

```
M          ~ Uniform{4, ..., 32}           kernel centers
c_m        ~ Uniform[0,1]^d
w_m        ~ N(0, I_d)
ℓ_kern     ~ LogUniform[0.1, 0.5]          RBF bandwidth
s          ~ Uniform[0, 1] · s_max         severity
v(u) = s · Σ_m w_m exp(−‖u − c_m‖² / 2ℓ_kern²)
S = flow of v from t=0 to t=1
```

Draw S_A and S_B independently. Rejection: evaluate `log|det J|` on a 5^d grid over the unit cube; reject the draw if `max − min > log(9)`, i.e. outside roughly a 3× local stretch band either way. Expect a few percent rejection at s_max ≈ 1; tune s_max to hit that.

**Relative warp coefficient ρ.** A convex combination of velocity fields is a velocity field, so its flow is a valid diffeomorphism. Define

```
Φ_ρ = flow of ( (1−ρ)·v_A + ρ·v_B ),        ρ ∈ [0, 1]
encoder-cloud coordinates := Φ_ρ(z^B)
decoder-cloud coordinates := S_A(z^A) = Φ_0(z^A)
```

ρ = 1 recovers `x^B = S_B(z^B)`. ρ = 0 gives `Φ_0(z^B) = S_A(z^B) = B_inA`, so both clouds sit in A's frame and the required transport is exactly the identity. Every intermediate ρ is a genuine diffeomorphism, never an interpolation artifact.

**Why ρ rather than an absolute-severity ramp.** Ramping `s_A` and `s_B` together pulls both frames toward the latent frame, which entangles two distinct quantities: how deformed each cloud looks in isolation, and how far apart the two frames are. Only the second is what registration must overcome. ρ isolates it, so absolute severity can stay at full strength — keeping each cloud's marginal geometry realistic — while the relative warp is annealed independently.

**Declared boxes.** Record `box_A = bbox(S_A([0,1]^d))` and `box_B = bbox(S_B([0,1]^d))`, computed from the *domain image*, not from sampled points. These stand in for the declared search space available at inference. All coordinate normalization uses them, which is what makes normalization immune to A's support restriction.

### 1.4 Function prior

```
depth      ~ Uniform{1, 2, 3}
width      ~ LogUniform[16, 128]
act        ~ Uniform{tanh, relu, gelu}
scale      ~ LogUniform[0.5, 2.0]           weight std multiplier
σ_obs      ~ LogUniform[0.01, 0.3]          (relative to std of f over the domain)
```

### 1.5 Derived targets

```
x^E = Φ_ρ(z^B)                                    # encoder cloud, at coefficient ρ
box_E = bbox( Φ_ρ([0,1]^d) )                      # its declared box

x̃^A = normalize(x^A, box_A)       x̃^E = normalize(x^E, box_E)     # → [0,1]^d
A_inB_target(i) = normalize( Φ_ρ(z_i^A), box_E )
B_inA_target(j) = normalize( Φ_0(z_j^B), box_A )   # = normalize(S_A(z_j^B), box_A)
pooled_context   = [ (x̃^A, ỹ^A) ∪ (B_inA_target, ỹ^E) ]   # both clouds in A's frame
ỹ = (y − mean(y^A ∪ y^E)) / std(y^A ∪ y^E)
```

At ρ = 0, `A_inB_target = x̃^A` and the encoder cloud coincides with `pooled_context`'s second half. All targets remain forward evaluations at every ρ; nothing is inverted.

Both transport targets are forward evaluations. Nothing is ever inverted.

---

## 2. Architecture

Reference sizes in brackets. Tune later; these are what to build first.

```
d_model = 256,  n_heads = 8,  d_ff = 1024,  L_enc = 6,  L_dec = 8
pre-LN, GELU, dropout 0.0
```

### 2.1 Input embeddings

```
enc_tok_j = Linear_B([ x̃_j^B , ỹ_j^B ])                    → d_model
dec_ctx_i = Linear_A([ x̃_i^A , ỹ_i^A , 1 ])                → d_model   (labelled)
dec_qry_q = Linear_A([ x̃_q^A , 0     , 0 ])                → d_model   (query)
```

Separate `Linear_A` and `Linear_B` projections. Cloud identity is carried by *which projection and which stack* a token goes through, never by an added tag — the additive-tag cross-term problem is avoided by construction.

### 2.2 Encoder (cloud B)

Six standard pre-LN blocks, full bidirectional self-attention over B's tokens. Output:

```
M ∈ R^{n_B × d_model}
```

The encoder never sees A. This is what makes M independent of A's warp and A's design, and it is what makes the severed-encoder baseline exact.

Also produce a pooled summary `g_B = mean(M)` for temperature prediction.

### 2.3 Decoder block

Each of the `L_dec` blocks, for token i at layer ℓ:

**(a) PFN self-attention** over decoder tokens, with the standard mask: context tokens attend to all context tokens; query tokens attend to context tokens only, never to other queries or themselves.

**(b) Transport update.** A residual readout on top of a predicted global affine:

```
t_i^(ℓ) = clamp01( A_g x̃_i^A + b_g + Δ_ℓ(h_i^(ℓ)) )      Δ_ℓ : d_model → d,  zero-init
```

`Δ_ℓ` is zero-init so every layer begins at whatever the global affine says, matching the perturbation-of-identity structure of the warp prior, and the per-point head only ever fits a residual.

`(A_g, b_g)` comes from the global affine head (§2.5). `t_i^(0) = A_g x̃_i^A + b_g`, computed *after* the layer-1 cross-attention (§2.3c), so the first transport estimate is already informed by B rather than being a pure A-side guess.

**(c) Gated cross-attention to M**, with queries built from the transport estimate:

```
q_i = W_q^(ℓ) [ h_i^(ℓ) ,  φ(t_i^(ℓ)) ,  ỹ_i ]
k_j = W_k^(ℓ) [ M_j     ,  φ(x̃_j^B)   ,  ỹ_j^B ]

a_ij = ⟨q_i, k_j⟩ / sqrt(d_head)
π_i· = softmax_j( a_ij / τ_i )
```

`φ` is a Fourier feature map on coordinates (8 frequencies per axis), so that proximity in the warped frame is expressible at multiple scales.

**Layer 1 is coordinate-free.** At `ℓ = 1` drop the coordinate terms entirely:

```
q_i = W_q^(1) [ h_i , ỹ_i ]          k_j = W_k^(1) [ M_j , ỹ_j^B ]
```

Reason: before any cross-attention, A's representation has never seen B, so a coordinate-based match at layer 1 compares two unaligned frames and is close to random under a strong warp — and every later layer then refines from noise. y is the only shared, undeformed channel available, so the first read is anchored on it. Layers 2+ include coordinates, because by then `t^(ℓ-1)` supplies a provisional shared frame.

**Head 0 is the coupling head.** Its value projection is frozen to the identity on `x̃^B`, so its output is literally a barycentric transport estimate:

```
T_bary,i^(ℓ) = Σ_j π_ij^(ℓ,head0) · x̃_j^B
```

which lies in B's convex hull by construction and is directly supervisable (§3.3). Heads 1..7 are ordinary learned-value heads.

**Row temperature**, per-point and per-pair:

```
τ_i = softplus( u^T g_B  +  w^T h_i^(ℓ)  +  b )
```

A single global scalar would have to serve every sampled pair and every region within a pair, which is wrong on both counts.

**Row-stochastic, not balanced.** No Sinkhorn. A covers a strict subregion of B by construction in this prior, so mass balance is a false constraint that would drag the transport outward.

**Gate:**

```
h_i ← h_i + γ_ℓ · CrossAttn_out,        γ_ℓ scalar, init 1.0, logged every step
```

Setting all `γ_ℓ = 0` yields an exact A-only PFN. `γ` is also the collapse monitor (§4.5).

**(d) FFN**, standard.

### 2.4 Global affine head

After the layer-1 (coordinate-free) cross-attention, predict a global map from pooled representations:

```
A_g = I + reshape( MLP_aff([ g_B , mean_i h_i^(1) ]) )     → d×d, zero-init so it starts at I
b_g =     MLP_aff_b([ g_B , mean_i h_i^(1) ])              → d,   zero-init
```

Declared-box normalization (§1.5) already removes per-axis translation and scale. What it does not remove is residual rotation and shear, which velocity-field warps generate freely. This is the classical registration hierarchy — global first, deformable second — and it gives the per-point head a far better starting point than the identity.

**Keep it structured.** An unconstrained learned map from A's representation into B's space would let the model manufacture an alignment that makes cross-attention *look* successful without corresponding to any real transport. Affine, then supervised per-point residual, forecloses that.

Supervise directly: `L_affine = ‖ A_g x̃_i^A + b_g − A_inB_target(i) ‖²` averaged over context points, weight 0.3, held constant. It is a coarse target that the residual head is expected to improve on, not an accuracy objective.

### 2.5 Output heads

All heads read the final decoder representation.

**Predictive head.** Bar distribution over ỹ: 64 bins over [−4, 4] plus two half-open tail bins. Applied to query tokens.

**Transport head.** Autoregressive over axes, bar distribution per axis, 64 bins over [0,1]:

```
for k = 1..d:
    logits_k = MLP_k([ h_i , emb(t̂_1..k-1) ])       → 64
```

Applied to **both** context and query tokens. Query-token transport supervision matters: mapping a *new* x* into B's frame is exactly the inference-time operation, and query tokens are mask-isolated from each other, so they must learn to infer their own transport by attending to A's context. Supervising it directly is the difference between a model that registers the context and one that registers arbitrary points.

Autoregressive over axes rather than factorized, so cross-coordinate correlation in the transport posterior survives — which matters under a velocity-field warp that couples axes. Cost is d sequential MLP evaluations on an already-computed representation.

---

## 3. Objectives

```
L = L_pred
  + λ_T   · L_transport      (§3.2, deep-supervised, both context and query tokens)
  + λ_C   · L_coupling       (§3.3, barycentric head-0 projection, every layer)
  + λ_aff · L_affine         (§2.4, global affine, context tokens)
  + λ_D   · L_distil         (§3.4, vs transport-forced oracle, ρ > 0)
  + λ_P   · L_pathway        (§3.5, vs pooled oracle, ρ = 0 slice only)
```

### 3.1 Predictive loss

Bar-distribution NLL on **query tokens of the decoder cloud only**. Never on encoder-cloud targets: that term is easy, large, gradient-dominant, and it swamps the signal from the regime where registration actually pays.

### 3.2 Transport loss

Bar NLL of the autoregressive head against `A_inB_target`, deep-supervised across layers:

```
L_transport = Σ_ℓ w_ℓ · NLL_bar( T^(ℓ) , A_inB_target ),     w_ℓ ∝ ℓ,  Σ w_ℓ = 1
```

Layer-weighted so late layers dominate but early layers still receive signal, which is what makes the coarse-to-fine refinement in §2.3(b–c) actually train rather than being nominal.

Applied to context and query tokens both.

### 3.3 Coupling loss

The barycentric projection of head 0, at every layer, against the same target:

```
L_coupling = Σ_ℓ w_ℓ · ‖ T_bary^(ℓ) − A_inB_target ‖²
```

The barycentric path is strictly lower-capacity than the transport head — a convex combination of B's coordinates through a frozen identity value projection. It cannot memorize and cannot leave B's hull. Supervising it forces the *attention pattern itself* to be a usable correspondence, rather than allowing a diffuse pattern that a downstream MLP compensates for.

No ground-truth point correspondence is needed; the known pushforward target does the work.

### 3.4 Oracle distillation

Run a second forward pass with the transport **teacher-forced** to ground truth at every layer, for context and query tokens alike, and with the global affine bypassed:

```
teacher:  t_i^(ℓ) := A_inB_target(i)          all ℓ, all tokens, gates active
student:  normal forward pass
L_distil = mean_over_queries  KL( P_teacher(ỹ) ‖ P_student(ỹ) )
```

Both distributions are 64-bin categoricals, so the KL is exact and costs nothing.

**Why this is stronger than `L_pred` alone.** `L_pred` supplies one scalar per query, the NLL of a single sampled y. The teacher supplies all 64 bin probabilities. The difference matters more here than in ordinary distillation because the teacher's PPD *shape* encodes the epistemic state — where the response surface is pinned down and where it is not — and that shape is exactly what misregistration corrupts. A hard label cannot carry it.

**And the loss is the metric.** KL(teacher ‖ student) on the test points is the registration cost in nats, which is the project's headline quantity (§5.1). The training objective and the evaluation number become the same thing rather than proxies for each other.

**Forward KL, deliberately.** KL(teacher ‖ student) penalizes the student placing low mass where the teacher places high mass, i.e. it punishes overconfidence. That is the failure mode to guard against: a query token cross-attends to B directly (it must, being mask-isolated from other queries), so a bad transport estimate lets it pull wrong information from B with no A-mediated correction, confidently. Forward KL is the counterweight.

**Implementation.**

- **Stop-gradient on the teacher.** Without it the model can lower the loss by degrading the teacher, since teacher and student share all weights.
- **Ramp in from 20% of training** (`λ_D : 0 → 1.0` over the next 20%, then held). Before that the teacher is not good enough to be worth imitating and the term is noise.
- Cost is one extra decoder forward pass per step. The encoder pass is shared.

### 3.5 Pathway distillation (ρ = 0 slice only)

At ρ = 0 both clouds are in A's frame, so the pooled oracle `decoder([A ∪ B_inA])` is available *and* the transport is exactly the identity. The gap between it and the two-stream configuration is therefore purely architectural, with no registration component at all.

```
if ρ == 0:
    L_pathway = mean_over_queries  KL( P_pooled(ỹ) ‖ P_two-stream(ỹ) )
```

Stop-gradient on the pooled pass, same as §3.4.

**This is the one place where distilling against the pooled oracle is correct.** At ρ > 0 that target mixes the registration gap with the architectural gap and aims the student at something it may be structurally unable to reach (§5.1). At ρ = 0 the registration term is identically zero, so the loss trains exactly one thing: the encoder-decoder pathway's ability to match full pooling.

### 3.6 Weights

```
λ_T   : 3.0 → 0.5 over the first 30% of training, then held      (floor, never 0)
λ_C   : 1.0 → 0.2 over the first 30%, then held
λ_aff : 0.3 constant
λ_D   : 0.0 until 20%, then → 1.0 over the next 20%, then held
λ_P   : 1.0 constant, applied only on the ρ = 0 slice
```

Never annealed to zero. "Ignore the encoder and predict from A alone" is a safe, reachable attractor, and removing the registration signal is an invitation to fall into it while predictive loss looks fine. The floor is the second line of defence; the §2.3(c) query construction is the first.

---

## 4. Training regime

### 4.1 Optimization

```
batch          32 dataset pairs
optimizer      AdamW, lr 3e-4, wd 0.01, β=(0.9, 0.98)
schedule       linear warmup 5k steps → cosine to 3e-5
grad clip      1.0
steps          ~500k, datasets sampled fresh each step (no epochs)
precision      bf16 with fp32 master weights
```

Datasets are generated on the fly. Pre-generate a fixed validation set of 2048 pairs, stratified over d, severity s, region type, and n_A, and never train on it.

### 4.2 Relative-warp curriculum, with a permanent identity anchor

Sample ρ per dataset from a mixture:

```
ρ = 0                    w.p. 0.15          (permanent anchor, all through training)
ρ ~ Beta(a_k, b_k)       otherwise
```

with `(a_k, b_k)` shifting mass upward over training: start near Beta(1, 5) (relative warps mostly mild), reach Beta(1, 1) — uniform — by 30% of training, and hold.

**The point mass at ρ = 0 is kept for the whole run, not annealed away.** It is what prevents the cross-attention pathway from drifting into a pattern that works only for warped pairs and degrades on the trivial case. It also supplies clean gradient to the memory-reading mechanism with zero registration noise, since the transport is exactly right there.

**Continuous coverage over ρ matters as much as the anchor.** A discrete identity-versus-warped split is detectable from box alignment alone, and the model would be free to learn a binary mode switch. Dense coverage of intermediate ρ forces one continuous mechanism instead. Detectability itself is not a problem — a model *should* exploit near-aligned frames when it sees them, and real pairs are sometimes nearly aligned.

Every ρ yields a genuine diffeomorphism (§1.3), so there is never a phase of training on invalid targets. This is the specific advantage of the velocity-field family over weight-scaled coupling INNs.

### 4.3 Role randomization

With probability 0.35, swap which cloud goes to the encoder and which to the decoder.

- Gives `B_inA` supervision through the *same* head with no extra machinery: under the swap, "A_inB" is B_inA.
- Regularizes the transport representation toward genuine invertibility.
- Forces size-robustness, since the decoder must handle both 8 and 1024 context points.

**Swapped passes contribute `L_transport` and `L_coupling` only, never `L_pred`.** Predicting the abundant cloud from the scarce one is the easy direction and would reintroduce exactly the gradient imbalance §3.1 exists to prevent.

### 4.4 Two masking modes, sampled during training

- **Full** (p = 0.85): gates active, normal operation.
- **Severed** (p = 0.15): all `γ_ℓ = 0`, so the decoder is an A-only PFN, and only `L_pred` applies.

Keeping the severed mode in training means the A-alone baseline is a genuinely well-fit model rather than an out-of-distribution configuration, so the lower bound in §5 is honest rather than artificially weak.

### 4.5 Live monitors

Log every 500 steps on the fixed validation set:

| Monitor | Meaning | Failure signature |
|---|---|---|
| mean \|γ_ℓ\| per layer | how much B is being used | drifting to 0 → collapse |
| transfer gap: NLL(severed) − NLL(full) | value extracted from B | closing toward 0 → collapse |
| transport NLL, per layer | is coarse-to-fine happening | flat across ℓ → refinement is nominal |
| coupling MSE, per layer | is attention a real correspondence | flat/high → head 0 is diffuse |
| fold fraction | spec violation rate | rising → transport leaving the diffeo manifold |
| KL to upper-1 (ρ>0) | registration gap | plateauing high → registration not improving |
| KL(upper-2 ‖ upper-1b) | architectural cost of the bottleneck | large → memory path too tight |
| NLL at ρ=0 vs ρ~1 | is the anchor holding | ρ=0 degrading over training → pathway drift |

The first two together are the collapse alarm: registration error rising while predictive loss falls is the diagnostic signature.

---

## 5. Evaluation

### 5.1 Bounds, from the same weights

```
lower   (A alone)          : all γ_ℓ = 0
model                      : normal forward pass
upper-1 (transport-forced) : t_i^(ℓ) := A_inB_target(i), all layers, all tokens
upper-1b (identity config) : same pair re-generated at ρ = 0; normal forward pass
upper-2 (pooled oracle)    : encoder severed; decoder context = pooled_context
```

`ΔNLL_oracle = NLL_lower − NLL_upper1` is the registration budget, and `(NLL_lower − NLL_model) / ΔNLL_oracle` is the fraction of available transfer actually captured. That is the headline number. All four configurations come from one set of weights, so differences cannot be attributed to capacity or optimization.

**A three-way decomposition.** `upper-1b` — the same pair regenerated at ρ = 0, so the encoder holds B_inA and the transport is the identity — is what makes the decomposition clean:

| Gap | Isolates | Response if large |
|---|---|---|
| `upper-2` → `upper-1b` | encoder/decoder split vs full pooling. No frames involved, transport is the identity. | widen the memory path; add B-side refinement conditioned on A |
| `upper-1b` → `upper-1` | cost of reading a memory encoded in a *different* frame, with the transport known exactly | change how cross-attention keys and queries are built |
| `upper-1` → `model` | registration inference | the actual research problem |

Without `upper-1b`, the first two terms are conflated: `upper-1` teacher-forces the transport but still has the encoder holding the cloud in a different frame, so comparing it directly to `upper-2` mixes "two-stream vs pooled" with "cross-frame vs same-frame."

The first two gaps are measurable before the model registers anything, so they are available from the earliest runs.

This split also determines the distillation targets. `L_distil` (§3.4) uses `upper-1`, matched pathway, registration signal only. `L_pathway` (§3.5) uses `upper-2` but *only at ρ = 0*, where the registration term vanishes. Distilling against `upper-2` at ρ > 0 would aim the student at a target it may be structurally unable to reach, with no way to attribute the residual.

### 5.2 Fold statistic

At eval, take the per-axis transport median on a 20^d grid over A's normalized box (subsample for d ≥ 3), form the Jacobian by finite differences, and report

```
fold_fraction = |{ u : det J(u) ≤ 0 }| / |grid|
```

Every prior sample satisfies the shared-f assumption by construction, so the validation distribution of `fold_fraction`, stratified by n_A and s, is the null. Store its quantiles; they are the calibration table used at deployment to decide whether a real pair is registrable at all.

### 5.3 Identifiability diagnostic

For each validation pair, form `I = Σ_i g_i g_iᵀ` with `g_i = ∇_u ĝ(u)|_{t̂_i}` by finite differences through the predictive head. Report `λ_min(I)` and `cond(I)`, and regress the transport head's per-point IQR against `λ_min`-derived predictions. The Fisher argument predicts a monotone relationship; a flat IQR means the head has collapsed to the marginal and is not reading identifiability at all.

This replaces the y-range check, which measures the wrong thing: a tilted-plane f has wide y-range and degenerate constraint geometry, while a bowl has narrow y-range and excellent geometry.

### 5.4 Spending law

Sweep transport capacity against n_A. Since the transport head is nonparametric, vary capacity by the number of decoder layers `L_dec ∈ {2, 4, 8, 12}` and, in a separate parametric arm, by fitting a velocity field with `M ∈ {2, 4, 8, 16, 32}` centers to the head's output. Plot held-out A NLL against `p / n_A` and check for collapse, stratified by severity `s` and region type.

### 5.5 Baselines

All evaluated on the same fixed validation set:

| Baseline | Note |
|---|---|
| A alone (severed) | Lower bound |
| Oracle (teacher-forced) | Upper bound |
| Declared-box normalization, no fitted warp | Design-immune p=0 point; the honest floor |
| Within-cloud rank features | Report stratified by region type; expect it to be *biased*, not merely weak, on restricted-support pairs |
| 1-D quantile matching | Same expectation |
| CPD | Needs ambient overlap; box normalization gives it a fair start |
| Entropic GW + barycentric projection | The coordinate-free classical method; expect structured bias since a non-uniform stretch admits no zero-distortion coupling |
| Two-stage: fit ĝ from B, optimize T by conditional likelihood | The non-amortized reference; isolates what amortization buys |
| Additive-ID-token single-stream PFN | The configuration that previously failed; same prior, same budget |

---

## 6. Build order

1. **Prior + visualizer.** Generate pairs, plot z / x^A / x^B / y for d = 1, 2. Eyeball 50 draws. Check the rejection rate and the realized `log|det J|` band. Nothing else is worth writing until the prior looks right.
2. **Severed decoder only.** Train a plain A-only PFN on this prior. It must match a standard PFN's behaviour. This is the lower bound and the sanity check on the whole data path.
3. **Add encoder + gated cross-attention, trained at ρ = 0 only** (`Δ_ℓ` disabled, `t_i = x̃_i^A`, which is exactly correct at ρ = 0). Add `L_pathway`. Target: match `upper-2` — the pooled oracle — since registration contributes nothing here and the only question is whether the two-stream pathway can carry what pooling carries. If it cannot, the memory path is too tight and no amount of registration machinery will help.
4. **Enable the transport head and `L_transport`, open the ρ curriculum.** Watch transport NLL fall across layers, and watch the ρ = 0 anchor hold as ρ mass shifts upward.
5. **Enable head 0 and `L_coupling`.** Visualize `π` for d = 1. It should look like a soft monotone correspondence.
6. **Global affine head + coordinate-free layer 1.** Check that `t^(0)` alone already beats identity init at moderate severity; this isolates the coarse stage before the residual stage is asked to do anything.
7. **Oracle distillation.** Add `L_distil` once the model is competent. Log both upper bounds from the outset so the registration gap and the architectural gap are separated from the first run.
8. **Role randomization, severity curriculum, full monitors.**
9. **Bounds, fold calibration, baselines, spending law.**

Step 3 is the real go/no-go, and it is now a training regime rather than a spot check. At ρ = 0 registration is exactly the identity, so a failure there is unambiguously an architecture or plumbing problem rather than a hard inference problem — and the pooled oracle gives a precise target to hit rather than a vague "should be better than severed."

---

## 7. What is load-bearing

If any of these is wrong, the design is wrong, and each is cheap to test early:

- **Cross-attention queries built from `t_i^(ℓ)`.** This is what makes registration necessary rather than merely encouraged. Ablate by building queries from `x̃_i^A` instead; if performance is unchanged, the model was never registering and the whole premise fails.
- **Transport supervision on query tokens.** Without it the model registers its context and cannot map new points.
- **f defined on z.** Without it the two clouds are statistically distinguishable and role randomization is incoherent.
- **Support restriction in the prior.** Without it, marginal-matching and conditional-likelihood registration agree, nothing forces the model to learn the design-invariant one, and it will fail on real scarce clouds while looking fine in validation.
- **Deep supervision with `w_ℓ ∝ ℓ`.** Without it the layer stack is not doing coarse-to-fine and the depth is wasted.
- **Coordinate-free layer 1.** Without it the first transport estimate is formed before A has seen B, so the first cross-attention matches unaligned frames and every subsequent layer refines from noise. Ablate by enabling coordinates at layer 1 and comparing transport NLL at layer 2.
- **Distilling against `upper-1` at ρ > 0, and `upper-2` only at ρ = 0.** The wrong choice aims the student at an unreachable target and silently conflates the registration gap with the architectural one.
- **The permanent ρ = 0 anchor.** Drop it and the cross-attention pathway is free to specialize on warped pairs and degrade on aligned ones, with no signal that it has happened. Ablate by annealing the anchor away and watching NLL at ρ = 0.

---

## 8. Deferred

- **Cycle consistency, as the correspondence-free constraint GW was supposed to provide.** Role randomization (§4.3) already gives both directions from one set of weights, so `T̂_{B→A}( T̂_{A→B}(x_i) ) ≈ x_i` is computable with two forward passes — the second direction evaluated at query tokens, since the composed points are not B's data points. Unlike GW this references no intra-cloud geometry, so it inherits neither the stretch bias nor the density bias, and it needs no targets. Two possible roles: a weak auxiliary loss during training, and — more valuable — a **test-time diagnostic**, since it is one of only two quantities in this design computable on a real pair with no ground truth (the other being the fold statistic, §5.2). Worth adding once the base model trains, mainly for the diagnostic.

- **Adaptive/sequential designs.** Emulating acquisition-driven sampling — Sobol prefixes of varying length, then y-quantile-and-timestep-conditioned selection from the BNN surface — would test the §1.6 ignorability argument directly. Deferred as expensive, and the restricted-support mechanism of §1.2 covers the support-mismatch half of the phenomenon at negligible cost. Worth revisiting once the base model works, at which point the question is narrow: does a model trained on restricted-support-but-non-adaptive designs transfer to adaptive ones, or does adaptivity introduce a distinct bias?
- **P3, y-distortion.** Sample a monotone h per cloud, apply to y. Adds an h-inference head and the rank-based invariance test.
- **Warp-marginalized prediction.** Sample K transports from the head, re-run the decoder per sample, mix. The encoder memory is warp-independent so only the decoder repeats, but the decoder is the expensive part here, so K ≈ 10 is realistic rather than the K ≈ 20 the roadmap suggested.
- **Variance decomposition** into warp and function components, needed for the BO over-exploration diagnostic. No clean method under a bar-distribution transport head.
