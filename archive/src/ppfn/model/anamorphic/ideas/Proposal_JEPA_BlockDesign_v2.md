### 1. How TabPFN Handles Variable Column Widths (Cell Tokenization)

In standard TabPFN and TabPFN v2, the model does not hardcode the number of features $d$. Instead, it uses **Cell
Tokenization**:

* Every individual tabular cell $(i, j)$ is projected into a uniform hidden token embedding of
  dimension $d_{\text{model}}$.
* A single tabular row $i$ with $d$ features is treated as a sequence of $d$
  tokens: $[h_{i, 1}, h_{i, 2}, \dots, h_{i, d}]$, where $h_{i, j} \in \mathbb{R}^{d_{\text{model}}}$.
* **Feature Attention** operates along this variable sequence length $d$. Because self-attention is natively agnostic to
  sequence length, TabPFN can ingest a 4-feature table or a 40-feature table using the exact same weights.

This is the architectural foundation that makes your schema-agnostic alignment possible: because Domain A ($d_A$
columns) and Domain B ($d_B$ columns) share the exact same hidden channel dimension $d_{\text{model}}$, **their features
can trade information and align in latent space regardless of schema width differences.**

---

### 2. Complete Architectural Blueprint

```
=== TEACHER ORACLE (PRE-TRAINED, FROZEN) ===
[Input: A_train, B_in_A (Privileged GT Prior), A_test]
                         │
                         ▼
            ┌─────────────────────────┐
            │   Frozen TabPFN v2      │
            └────────────┬────────────┘
                         │
                         ▼
         P_teacher(y_test | A, B_in_A) ──────────┐
                         ▲                       │  KL-Divergence Loss
                         │                       │  (y-Distribution Alignment)
                         │                       ▼
         P_student(y_test | A, B_in_A) ◄─────────┘
                         ▲
                         │
             ┌───────────┴───────────┐
             │  Predictor (TabPFN)   │ ◄── Phase 1: FROZEN (Enforces Canonical Manifold)
             └───────────▲───────────┘     Phase 2: UNFROZEN (Adapts residual uncertainty)
                         │  
                         ├── Latent Context: Z = [Z_A | Z_{B->A}]
                         │   ▲
                         │   │  Latent Relational GW Loss
                         │   ▼  (Gram Matrix Distance in Embedding Space)
                         │  E(B_in_A) ◄── Native Embeddings of Privileged GT
                         │
             ┌───────────┴───────────┐
             │  Student Aligner      │ ◄── Trainable (ECDF + Additive y + Staged QKV)
             └───────────▲───────────┘
                         │
[Input: A_train, B_train (Block Diagonal ECDF Design)] ── (A_test bypasses Aligner straight to Predictor)
=== STUDENT (TRAINABLE ALIGNER + PREDICTOR) ===

```

---

### 3. Step-by-Step Computational Pipeline

#### Phase 1: ECDF & Additive-Target Normalization (Pre-processing)

1. Transform all continuous feature columns in Domain A and Domain B into domain-relative Empirical Cumulative
   Distribution Functions (ECDFs), mapping values into the invariant rank space $[0, 1]$.
2. Transform observed targets $y$ into domain-relative ECDFs: $\tilde{y}_i \in [0, 1]$.
3. Apply standard TabPFN cell tokenization to build the orthogonal Block-Diagonal matrix. For every cell $(i, j)$,
   additively inject the row's normalized target embedding into the feature token:

$$h_{i, j}^{ (0)} = \text{Linear} (\tilde{X}_{i, j}) + \text{Embed} (\tilde{y}_i) + \text{Pos} (j)$$

This ensures every token natively carries its **ECDF feature rank** plus its **target outcome signature**, establishing
a universal coordinate-free language across both domains.

#### Phase 2: Direct Cross-Subspace Overwrite (Layer 1 Alignment)

Instead of querying static null tokens, Layer 1 immediately breaks orthogonality via **Feature-Wise Cross-Attention
between active blocks**:

* To initialize the top-right cross-block ($\tilde{X}_B^C$), perform cross-attention where Domain A's columns query
  Domain B's columns:

$$Q = \tilde{X}_A, \quad K = \tilde{X}_B, \quad V = \tilde{X}_B$$

* Because every cell token additively contains its $\tilde{y}_i$ embedding, the attention weights match columns by
  comparing their correlation profiles with the target outcome. This directly overwrites the initial off-diagonal empty
  slots with geometrically informative imputations.

#### Phase 3: Interleaved Grounding & Refinement (Mid-Layers)

For subsequent layers $2 \dots K$ in the Student Aligner, alternate between two processing modes:

1. **Intra-Domain Self-Attention (Row & Feature):** Allow Domain A and Domain B to update their internal relational
   covariance independently.
2. **Cross-Block Grounding (Row-Wise):** Use the imputed cross-blocks as Queries against Domain A's active support rows
   ($Q = \tilde{X}_B^C, \; K = V = \tilde{X}_A$) to anchor the transported tokens strictly to the geometric manifold of
   Domain A.

#### Phase 4: Zero-Shot Predictor Handoff (Inference)

* The Student Aligner outputs a canonical latent support context: $Z_{\text{context}} = [Z_A \mid Z_{B \to A}]$.
* The target query rows (`A_test`) **completely bypass the Student Aligner** to prevent distortion, entering directly at
  the Predictor stage.
* The frozen TabPFN v2 Predictor performs pure Row Attention between `A_test` and $Z_{\text{context}}$ to emit binned
  logits for $P_{\text{student}} (y \mid \text{A\_test})$.

---

### 4. Mathematical Objective & Training Dynamics

The system is trained end-to-end on synthetic pairs generated from your Data Generation Process (DGP) prior using a
two-part objective that completely eliminates NLL vs. MSE objective collision:

$$\mathcal{L}_{\text{total}} = \mathcal{D}_{\text{KL}}\left (P_{\text{teacher}} \parallel P_{\text{student}}\right) + \lambda_{\text{GW}} \left\Vert{} G (Z_{B \to A}) - G (E (B_{\text{in } A})) \right\Vert{}_F^2$$

| Loss Term                                             | Mechanism & Purpose                                                                                                                                                                                                                                                                                                              |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Semantic Distillation** ($\mathcal{D}_{\text{KL}}$) | Computes KL-Divergence between the Student's binned logits and the frozen Teacher Oracle's posterior. This transfers the full Bayesian uncertainty and penalizes distributional mismatch.                                                                                                                                        |
| **Latent Relational GW** ($\mathcal{L}_{\text{GW}}$)  | Computes the Frobenius distance between inner-product Gram matrices ($G(M) = \frac{1}{d_{\text{model}}} M M^T$) of the Student's aligned tokens ($Z_{B \to A}$) and the native embeddings of the privileged ground truth ($E(B_{\text{in } A})$). This enforces Riemannian isometry without coordinate-space MSE scaling issues. |

---

### 5. Critical Review: Theoretical Strengths vs. Remaining Risks

#### A. Why This Architecture Succeeds (The Theoretical Edge)

* **Solves Unsupervised OT Degeneracy:** By using privileged ground-truth pairs ($B_{\text{in } A}$) in the DGP prior,
  it turns an ill-posed unsupervised manifold alignment problem into a supervised relational distillation task.
* **Eliminates Objective Collision:** The alignment loss ($\mathcal{L}_{\text{GW}}$) operates exclusively on latent
  inner products at the Aligner-to-Predictor interface, leaving the downstream Predictor 100% dedicated to
  distributional $y$-inference.
* **Invariance to Monotonic Warping:** Converting continuous columns to ECDF rank representations ($[0, 1]$) makes any
  arbitrary monotonic scaling or power-law distortion mathematically invisible to the model.
* **Prevents Dialect Drift:** Freezing a pre-trained TabPFN v2 as the Predictor during warmup forces the Student Aligner
  to emit tokens that strictly abide by standard tabular geometry.

#### B. Honest Critique: Hidden Failure Modes & Architectural Edge Cases

* **1. Categorical & Non-Monotonic Feature Breaking:** ECDF normalization is a silver bullet for *continuous,
  monotonically warped* features. However, if Domain B contains categorical variables with permuted labels, or if a
  continuous feature undergoes a non-monotonic quadratic warp (for example, $X_B = (X_A - 0.5)^2$), ECDF ranking
  destroys the bijection. The Aligner will struggle to map folded distributions without explicit piecewise-monotonic
  tokenization.
* **2. The "No-Correlation" Column Trap:** If Domain A has an informative feature $X_{A, 1}$ that has a strong
  non-linear relationship with $y$ but a near-zero *linear* correlation
  ($\langle \tilde{X}_{A, 1}, \tilde{y} \rangle \approx 0$), and Domain B has an analogous feature $X_{B, 2}$,
  additive $\tilde{y}$ cell injection alone may fail to disambiguate it from pure noise columns during Layer 1
  cross-attention.
* **3. Attention Sink / Null-Token Domination:** In Layer 1, when computing
  cross-attention $Q = \tilde{X}_A, \; K = V = \tilde{X}_B$ across different column widths ($d_A \neq d_B$), attention
  Softmax denominators can become dominated by uninformative or redundant columns if Domain B has a high proportion of
  noise features. An explicit **top-k sparsity mask** or temperature scaling is required to prevent Softmax entropy
  collapse.
* **4. Latent Gram Matrix Quadratic Scaling:** While $O (N^2)$ relational Gram matrices are computationally trivial for
  standard 50-row HPO support sets, if you scale this architecture to large tabular meta-learning tasks ($N > 1000$
  rows), calculating $\mathcal{L}_{\text{GW}}$ across batches will become a GPU memory bottleneck.