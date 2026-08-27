# MTPFN — Multi-Task Prior-Data Fitted Networks

A faithful reproduction of **"Robust Transfer for Bayesian Optimization with Prior-Data Fitted Networks"** (Yucen Lily
Li, Sam Daulton, Samuel Müller, Andrew Gordon Wilson, Eytan Bakshy — NeurIPS 2025 SPIGM Workshop), built directly from
the uploaded PDF (not the earlier search-fragment reconstruction).

## What changed from my first attempt

My first pass (before I had the actual PDF) got the *spirit* right but guessed wrong on several specifics. Corrected
against the real paper:

|                             | First guess                                                   | Actual paper                                                                                                                                                     |
|-----------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Output head                 | Discretized "bar"/Riemann distribution (PFNs4BO-style)        | **Gaussian** `(μ, σ²)` output, per Figure 2                                                                                                                      |
| Task encoding               | Per-task-index learned embedding                              | **One shared** learnable `[TASK]` token, prepended to every task's sequence (Section 3.2) — this is what lets it generalize to unseen numbers of tasks           |
| Attention topology          | A pooling-based two-level scheme I invented                   | Exact **12 intra-task + 11 inter-task layers, interleaved**, ending on an intra layer (Section 5, Figure 2)                                                      |
| Prior                       | Fixed N points/task, shared input grid across tasks           | **Algorithm A.1**: `n` total points, each independently assigned to a task via `Categorical(π)`, `π ~ Dirichlet(α)` — genuinely variable, ragged per-task counts |
| Lengthscale prior           | log-uniform                                                   | **Gamma(3, 6)**, "BoTorch v1.11 default lengthscale prior"                                                                                                       |
| Negative-transfer mechanism | Zero out entries of a sampled correlation matrix pre-sampling | **Resample** an unrelated source task's `y` entirely from a **fresh, independent single-task GP** with a new lengthscale, *after* the joint ICM draw             |
| Training hyperparameters    | My own defaults                                               | Paper's actual numbers: batch size 16, AdamW lr 1e-4, cosine annealing, ~50M synthetic datasets, 4 heads, hidden size 512                                        |

## Repo layout

```
mtpfn/
  prior.py           Algorithm A.1 (isotropic) / A.2 (ARD) data-generating process
  gaussian_head.py    (mu, sigma^2) output parameterization + Gaussian NLL
  model.py            MTPFN: shared [TASK] token + interleaved intra-/inter-task encoders
  train.py            meta-training loop w/ the paper's reported hyperparameters
  bo.py               wrap a trained model as a BO surrogate + closed-form EI
```

All four files were smoke-tested end-to-end: the prior produces correctly ragged per-task point counts and exactly-one
held-out query point on the target task; the model runs a forward pass at the paper's *exact* reported scale (23 layers,
hidden size 512, 4 heads — **~72.5M parameters**); the training loop's loss decreases; the BO wrapper produces sane
mean/std/closed-form-EI outputs.

## The prior (`prior.py`) — Algorithm A.1, "Robust Isotropic Full-Rank ICM"

```
Require: sequence length n, number of tasks T, unrelated-task probability p
 1: sample inputs {x_i}_{i=1}^n ~ Uniform([0,1]^d)
 2: sample task proportions π ~ Dirichlet(α)
 3: for i=1..n: sample task id t_i ~ Categorical(π)
 4: sample task covariance matrix K_T ~ LKJ(η=1)
 5: sample input lengthscale ℓ ~ Gamma(3, 6)
 6: K_X = RBF(x_i, x_j; ℓ) over ALL n points
 7: K = K_T[t_i, t_j] * K_X[i, j]                       (ICM kernel, gathered by task id)
 8: sample y ~ N(0, K)                                   (one joint draw over all n points)
 9: for each SOURCE task j (j != target task 0):
10:   with probability p:
11:     sample a fresh lengthscale ℓ_j ~ Gamma(3, 6)
12:     K_X^(j) = RBF kernel on {x_i : t_i = j} using ℓ_j
13:     resample y^(j) ~ N(0, K_X^(j))                    (overwrite; now an unrelated single-task GP draw)
```

`p` is the single most important knob in the paper: it is "the probability that any given source task is drawn
independently from the target task" (Section 3.1) and is exactly what teaches MTPFN to ignore irrelevant auxiliary tasks
at BO time (Figure 3's `p ∈ {0, 0.1, 0.2}`
sweep). The Appendix A.2 **ARD** variant (independent lengthscale per input dimension rather than one shared scalar) is
available via
`MTPFNPriorConfig(ard=True)`.

Two practical (non-paper) additions needed to make this batchable, called out explicitly in the code: (a) task 0 is
exempt from the independent-resample loop since it's the target by definition, and (b) a minimum-point-count rejection
guard so a task can't randomly get zero points.

Because task assignment is genuinely random per point, per-task counts vary within a batch — the implementation returns
task-padded tensors
`[B, T, L, ...]` with a `valid_mask` and a `query_mask` (marking task 0's held-out test point (s)) rather than assuming
a fixed count per task.

## The model (`model.py`) — Section 3.2 / Figure 2

- **Feature Encoder**: each `(x, y)` pair is embedded via `x_encoder(x) +
  y_encoder(y)`; query points (unknown `y`) get a learned "missing value"
  embedding instead.
- **`[TASK]` token**: a single shared learnable vector, prepended to every task's sequence. It is *not* indexed by task
  identity — its representation is shaped purely by what it attends to, which is exactly why (per Section 3.2) the model
  "naturally handles inputs of varying lengths" and "can still meaningfully integrate new task representations"
  at test time even with more tasks than seen during training.
- **Intra-Task Encoder**: a standard transformer layer, self-attention over one task's own sequence (including its
  `[TASK]` token), applied independently per task — `O(T·D²)` total.
- **Inter-Task Encoder**: a standard transformer layer, self-attention only over the `T` tasks' `[TASK]`-token
  summaries — `O(T²)`. The updated summary is scattered back into position 0 of each task's sequence before the next
  intra-task layer.
- These interleave as **Intra, Inter, Intra, Inter, ..., Intra** — one more intra- than inter-task layer, matching the
  paper's 12-intra / 11-inter split (Section 5) — reducing attention cost from the naive
  `O(D²T²)` to `O(D²T + T²)` (Section 3.2).
- **Output Layer**: linear head at the query position (s) → `(μ, σ²)`
  (Figure 2's explicit output), trained with Gaussian NLL.

```python
from mtpfn.model import MTPFN
from mtpfn.train import full_paper_model_kwargs

model = MTPFN(input_dim=1, **full_paper_model_kwargs())
# -> 4 heads, hidden size 512, 12 intra + 11 inter layers, ~72.5M params
```

## Training (`train.py`) — Section 5

```python
from mtpfn.train import train_mtpfn, TrainConfig, full_paper_model_kwargs
from mtpfn.prior import MTPFNPriorConfig

model = train_mtpfn(
    prior_cfg=MTPFNPriorConfig(num_tasks=4, input_dim=1, p_independent=0.2),
    train_cfg=TrainConfig(steps=3_125_000, batch_size=16),  # ~50M datasets @ batch 16
    model_kwargs=full_paper_model_kwargs(),
)
```

Loss is the Gaussian NLL evaluated only at the target task's held-out query point, matching the paper's stated objective
`L_NLL = E_{D~p(D|h)}[-log f_θ(y_test | x_test, D_train)]`. `TrainConfig`
defaults to the paper's exact reported numbers (batch 16, AdamW lr 1e-4, cosine annealing, ~3.125M steps ≈ 50M
datasets / batch 16); override
`steps`/`batch_size`/`device` for quick local testing (see the
`if __name__ == "__main__"` block for a tiny CPU-sized smoke test).

## Using it for BO (`bo.py`)

```python
from mtpfn.bo import MTPFNSurrogate
import torch

surrogate = MTPFNSurrogate(model, device="cpu")
task_data = [
    {"x": x_target_obs, "y": y_target_obs},  # task 0: the live optimization
    {"x": x_aux1, "y": y_aux1},  # task 1: e.g. a related HPO run
    {"x": x_aux2, "y": y_aux2},  # task 2: possibly unrelated
]
x_candidates = torch.linspace(0, 1, 100).unsqueeze(-1)
ei = surrogate.expected_improvement(x_candidates, task_data, best_f=current_best)
next_x = x_candidates[ei.argmax()]
```

Because the output head is Gaussian, Expected Improvement has the usual closed form (Jones et al., 1998) — no Monte
Carlo needed.

## Not (yet) reproduced here

- The exact HPOBench / Tabular FC-Net benchmark harness (Section 5.1) and the ScaML / ICM / single-task-GP baselines
  it's compared against — this package gives you the model, prior, and training loop; wiring it into BoTorch/GPyTorch
  benchmark loops is a separate integration task.
- The fine-tuning experiments (Section 4.4 / Appendix E.2) and the single-task PFN baseline (Appendix E.1, an 8-layer
  non-hierarchical transformer trained on GP draws) are described but not implemented here, since they're
  baselines/ablations rather than the core method.
- The fully-Bayesian-inference-vs-NUTS comparison (Section 4.3) is an evaluation, not something to "build."

## Key knobs to experiment with

- `p_independent` in `MTPFNPriorConfig` — the paper's central negative-transfer robustness dial (their Figure 3 sweeps `p ∈ {0, 0.1,
  0.2}`).
- `ard` — switch between Algorithm A.1 (isotropic, one shared lengthscale)
  and Algorithm A.2 (ARD, per-dimension lengthscales); the paper's Appendix B.2 finds ARD generally outperforms
  isotropic on real HPO benchmarks but with more variance.
- `dirichlet_alpha` — controls how balanced/imbalanced per-task point counts are within an episode (not swept in the
  paper; their evaluation setup fixes 5 target + 20-per-auxiliary points directly rather than via the Dirichlet draw,
  which is more of a training-time detail).

## Notes & critical assessment & Quick-fix ideas

Does MTPFN still work effectively if the related tasks are warped? The flat attention would require the tokens to mean
the same thing across tasks. Does hierarchical pooling & attention allow for more flexibility in the representation of
tasks?

Output-warping (task functions agree at the same x, but one is a monotonic nonlinear transform of the other, e.g.
y_aux = h (f_shared (x))) — this is not a spatial correspondence problem, it's a task-level nuisance parameter problem.
Since h is constant across the whole domain, the information needed to correct for it is a handful of numbers describing
"what h is," which fits comfortably in a single pooled summary vector. So the [TASK] token bottleneck isn't really a
bottleneck here: if you trained on a prior that included such warps, I'd expect the model to learn to infer h's rough
shape from context and bake a correction into the summary token it exchanges. The real gap for this case is the prior,
not the architecture — Algorithm A.1/A.2 only ever generates linear-Gaussian ICM correlation (a single scalar
correlation coefficient) or full independence. There's no mechanism in the DGP for a nonlinear output link, so MTPFN as
trained here has zero training signal for it, regardless of what the network could represent in principle.

Input-warping (f_aux (x) actually corresponds to f_target (g (x)) for some unknown monotonic g, i.e. the correspondence
is position-dependent) — this is where I think the "flat" pooling really hurts, independent of the prior question. Walk
through what actually gets exchanged between tasks:

Intra-task attention pools each task's own points into its [TASK] token — this pooling is permutation-invariant and,
critically, discards the individual point positions once compressed into that one vector. Inter-task attention only ever
exchanges these single pooled vectors, at O (T²) cost. A target query at some specific x* never gets to attend to
specific auxiliary points near g (x*) — it only ever sees its own task's [TASK] token, which carries one static,
query-independent summary of the auxiliary task.

To actually exploit an input warp, the model would need something like "for this query at x*, look up what the auxiliary
task did near g (x*)" — a query-conditioned retrieval from another task's raw points. The architecture as described
structurally can't do that: the auxiliary task's contribution is fixed before the model even knows which x* is being
queried. This is exactly the tradeoff the paper is explicit about — they buy O (D²T + T²) scalability by giving up the
point-to-point interactions that naive O (D²T²) full attention would allow, and it's precisely those point-to-point
interactions that a warp-correction mechanism would need. So I'd expect this case to fail to learn efficiently even with
an appropriately augmented prior, not just be undertrained.

## Annotations:

### A. Is MTPFN a Mixed effects model for same schema?

MTPFN uses intra-task attention and inter-task attention. From a conceptual standpoint, this is similar to having a
mixed effects model where the intra-task attention captures task-specific effects and the inter-task attention captures
shared effects across tasks. The [TASK] token acts as a summary of each task's information, allowing the model to learn
how to combine information from different tasks effectively. It however hinges on the aligned table schemata between tasks and 
probably (verify ?) will require that the coordinate systems they live in mean the same thing. 

## Quick fix ideas

* A plausible architectural fix without giving up all the scalability: replace the single [TASK] token with a small set
  of landmark tokens per task (à la Set Transformer's ISAB / Perceiver inducing points), so cross-task exchange carries
  some coarse localized structure rather than one fully global vector. Still not full point-level attention, but
  strictly more capacity to represent spatially-varying relatedness, at a modest O (m²T²) cost for m landmarks instead
  of O (T²).
* A cheaper non-architectural mitigation for the "known warp family" case: pre-normalize each task's x (e.g.
  quantile-normalize) into a shared frame before feeding it in. This can absorb some monotonic input warps as a
  preprocessing step rather than something the network has to learn, but it's fragile (needs enough points to estimate
  the transform, and you'd need to invert it to get predictions back in the target's native coordinates), and it's
  exactly the kind of hand-designed assumption the paper is trying to avoid needing.

