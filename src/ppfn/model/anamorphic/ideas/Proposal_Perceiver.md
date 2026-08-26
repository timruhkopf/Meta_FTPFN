# Research Proposal: Dual-Perceiver Cross-Domain Imputation PFN

**Title:** Zero-Shot Non-Linear Feature-Space Alignment and Cross-Domain Transfer via Dual-Perceiver Prior-Data Fitted
Networks

**Target Domain:** Tabular Meta-Learning, In-Context Bayesian Optimization, & Cross-Task Transfer

---

## 1. Executive Summary

In-context tabular foundation models, such as Prior-Data Fitted Networks (PFNs), have transformed few-shot learning and
Bayesian optimization by approximating Bayesian inference in a single forward pass. However, extending PFNs to
multi-task or multi-source settings remains fundamentally limited by existing cross-task assumptions. Current multi-task
PFN frameworks (like MTPFN) rely on priors built upon the Intrinsic Coregionalization Model (ICM), which assumes linear
coregionalization across task covariances. Consequently, when tabular datasets share a target task but operate under
**non-linear feature warps, scale shifts, or shuffled measurement spaces**, traditional multi-task models suffer from
severe misalignment or negative transfer.

We propose the **Dual-Perceiver Cross-Domain Imputation PFN**, a novel architecture that formulates cross-domain
feature-space translation as an **in-context block-matrix imputation problem**. By leveraging $y$-conditioned cell
embeddings and Perceiver IO column bottlenecks, our model extracts distribution-level **Column Signatures** from
unpaired tabular sources. A cross-attention bridge then dynamically aligns the feature coordinate bases and imputes the
cross-domain observations without requiring paired rows or Euclidean coordinate losses. Supervised entirely via a
4-quadrant compositional Bar-Distribution Negative Log-Likelihood (NLL) within a multi-task synthetic DGP, this approach
enables **zero-shot, non-linear domain adaptation and doubles the effective context size** for downstream Bayesian
inference.

---

## 2. Problem Statement & Motivation

Multi-task tabular learning frequently encounters scenarios where two datasets—Table $A$ and Table $B$—share identical
or related target concepts $Y$, but their input feature spaces $X$ differ structurally due to:

1. **Non-Linear Measurement Warps:** Sensor drift, non-linear saturation curves ($\log (X)$, power laws), or differing
   diagnostic assay scales.
2. **Feature Re-ordering & Scrambling:** Unaligned column ordering or latent linear combinations across data collection
   pipelines.
3. **Low-Data Constraints:** Few-shot regimes where Table $B$ has only 5–10 observed context rows, making independent
   single-task inference poorly calibrated.

### Why Existing Solutions Fail

* **Multi-Task Gaussian Processes & MTPFNs:** Standard multi-task surrogates assume linear task coregionalization
  ($K = K_T \otimes K_X$). When $X$ undergoes non-linear warping, the linear covariance assumption breaks down, forcing
  the model to either fail to transfer or degrade due to negative transfer.


* **Unsupervised Optimal Transport (OT / Gromov-Wasserstein):** Unsupervised distribution alignment methods require
  large sample sizes to estimate marginal densities accurately and often fail in low-data, few-shot regimes.
* **Pure Euclidean Reconstruction (MSE on $X$):** Direct coordinate reconstruction forces models to optimize over noisy,
  ill-posed feature space geometries rather than focusing on task-relevant predictive semantics.

---

## 3. Proposed Methodology

We convert domain adaptation into a unified **4-quadrant block imputation matrix**:

$$\mathbf{M}_{\text{context}} = \begin{bmatrix} \mathbf{A}_{\text{train}} & \hat{\mathbf{A}}_{in\_B} \\ \hat{\mathbf{B}}_{in\_A} & \mathbf{B}_{\text{train}} \end{bmatrix}$$

where the off-diagonal blocks $\hat{\mathbf{A}}_{in\_B}$ and $\hat{\mathbf{B}}_{in\_A}$ represent Table $A$ and
Table $B$ imputed into each other’s feature coordinate space.

```
       [Sub-Table A: (X_A, y_A)]                     [Sub-Table B: (X_B, y_B)]
                │                                             │
                ▼                                             ▼
    [TabularCellEmbedder + PE_j]                 [TabularCellEmbedder + PE_j]
                │                                             │
                ▼                                             ▼
    [Axial Attention (Row/Col)]                   [Axial Attention (Row/Col)]
                │                                             │
                ▼                                             ▼
     [Perceiver IO Summarizer]                     [Perceiver IO Summarizer]
                │                                             │
                ▼                                             ▼
     Column Signatures Z_cols_A                    Column Signatures Z_cols_B
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                     ┌──────────────────────────────────┐
                     │  Cross-Domain Imputation Bridge  │
                     │  (Softmax Attention Transition)  │
                     └─────────────────┬────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     Imputed Block: \hat{A}_{in_B}                 Imputed Block: \hat{B}_{in_A}
                │                                             │
                ▼                                             ▼
   Context B: [True_B || \hat{A}_{in_B}]         Context A: [True_A || \hat{B}_{in_A}]
                │                                             │
                ▼                                             ▼
      [Inference Cross-Attn]                        [Inference Cross-Attn]
     Evaluates Test Queries B                      Evaluates Test Queries A
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                  ┌─────────────────────────────────────────────────┐
                  │    Compositional Bar-Distribution NLL           │
                  │ NLL(A_test) + NLL(B_test) +                     │
                  │ λ [ NLL(\hat{A}_{in_B}) + NLL(\hat{B}_{in_A}) ] │
                  └─────────────────────────────────────────────────┘

```

### Core Architectural Pillars

#### A. Target-Conditioned Cell Embedding with Deterministic $\text{PE}_j$

Each raw cell entry is mapped into a vector space incorporating target information and spatial column identity:

$$\mathbf{h}_{i,j} = \text{MLP}_x (x_{i,j}) + \text{MLP}_y (y_i) + \mathbf{p}_j^{\text{col}}$$

* **Conditioning on $Y$:** Including $y_i$ anchors each cell to its predictive outcome, ensuring column representations
  reflect joint mutual information $P (X_c, Y)$ rather than isolated marginal distributions.
* **Deterministic Column PE ($\mathbf{p}_j^{\text{col}}$):** Adopting TabPFN v2.5’s pre-generated random-seeded column
  embeddings provides spatial identity and prevents feature collapse without breaking set-based processing.
* **Masked $Y$ Token:** Unlabelled test points and unimputed target entries replace $\text{MLP}_y (y_i)$ with a
  learnable `[NO_LABEL]` embedding to prevent target leakage.

#### B. Independent Axial Attention

Local cell representations within Table $A$ and Table $B$ are refined independently using alternating row-wise
(cross-feature) and column-wise (cross-sample) self-attention. This step binds local feature correlations before
dimensionality reduction.

#### C. Perceiver IO Column Summarization

To decouple computational complexity from sample size ($N$), a Perceiver IO bottleneck compresses the cell grid:

1. **Read:** $L$ latent queries cross-attend to flattened cells in $O (N \cdot C)$ time.
2. **Process:** Deep self-attention layers process the latent array in $O (L^2)$ time.
3. **Write:** $C$ learned column queries cross-attend to the latents, outputting a deterministic **Column Signature
   Matrix**:

$$\mathbf{Z}_{\text{cols}} \in \mathbb{R}^{C \times D}$$

#### D. Cross-Domain Off-Diagonal Imputation Bridge

Using column signatures $\mathbf{Z}_A$ and $\mathbf{Z}_B$, the model derives a $C \times C$ cross-space transition
operator via scaled dot-product attention:

$$\mathbf{A}_{B \to A} = \text{Softmax}\left (\frac{\mathbf{Z}_B \mathbf{Z}_A^T}{\sqrt{D}}\right)$$

Multiplying this transition operator against Table $A$'s cell representations warps its rows directly into Domain $B$'s
coordinate basis, generating $\hat{\mathbf{A}}_{in\_B}$ (and symmetrically, $\hat{\mathbf{B}}_{in\_A}$).

#### E. Four-Quadrant Compositional Bar NLL Loss

Rather than optimizing over noisy Euclidean coordinates ($\text{MSE} (X)$), supervision is driven strictly by predictive
likelihood using TabPFN's continuous `FullSupportBarDistribution`:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}} (A_{\text{test}}) + \mathcal{L}_{\text{NLL}} (B_{\text{test}}) + \lambda \left[ \mathcal{L}_{\text{NLL}} (\hat{A}_{in\_B}) + \mathcal{L}_{\text{NLL}} (\hat{B}_{in\_A}) \right]$$

---

## 4. Expected Impact & Key Advantages

1. **Zero-Shot Transfer Across Non-Linear Warps:** Enables robust transfer learning between datasets operating under
   non-linear feature transformations where traditional multi-task Gaussian processes fail.


2. **Doubled Effective Sample Size:** By accurately imputing $\hat{\mathbf{A}}_{in\_B}$, context sizes for few-shot
   Bayesian optimization double from $\vert{}B\vert{}$ to $\vert{}B\vert{} + \vert{}A\vert{}$, significantly improving
   calibration and posterior uncertainty.
3. **No Paired Observations Required:** Alignment is performed purely at the schema/column-signature level, making the
   architecture applicable to unpaired real-world datasets.
4. **End-to-End PFN Integration:** The entire pipeline executes as a single, deterministic transformer forward pass,
   eliminating iterative optimization or post-hoc alignment heuristics.

---
