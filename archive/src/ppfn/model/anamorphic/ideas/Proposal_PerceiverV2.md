# Cross-Table Relational Attention Network (CT-RAN)

**An Asymmetric Block Architecture with Perceiver-Latent Bottlenecks for Multi-Table Tabular Processing**

This architecture extends **TabPFN** to simultaneously process two tabular datasets, $A \in \mathbb{R}^{B \times (F_A + 1) \times R_A \times D}$ and $B \in \mathbb{R}^{B \times (F_B + 1) \times R_B \times D}$, with arbitrary and mismatched row counts ($R_A \neq R_B$) and feature counts ($F_A \neq F_B$). Crucially, **targets $y$ are appended as a dedicated column** on equal footing with input features, mirroring the real TabPFN design.

It enables **zero-shot cross-table data augmentation** at inference time while preserving row permutation equivariance, feature-count invariance, per-target-column structural identity, and strict test-set isolation.

---

## 1. Dedicated Target Columns & Native Intra-Table Attention

In authentic TabPFN, target labels $y$ are **not** broadcast as an additive bias across feature cells ($x_{ij} + y_i$). Instead, the target is concatenated as its own dedicated column along the feature axis:

$$X_A = \text{Concat}\big([\text{Embed}(X_A),\; \text{Embed}(Y_A)],\; \text{dim}=\text{feature}\big) \in \mathbb{R}^{B \times (F_A + 1) \times R_A \times D}$$

This allows the target column to act as a full participant in feature-attention—competing, querying, and exchanging information with other columns on equal footing.

Before any cross-table interaction occurs, each table is processed independently through standard TabPFN row and feature attention layers to build local relational context by subsuming the non-attending sequence dimension into the batch dimension:

```
Row Attention:      [B, F_A+1, R_A, D]  ──>  [(B · F_A+1), R_A, D]  ──SelfAttn──>  [B, F_A+1, R_A, D]
Feature Attention:  [B, F_A+1, R_A, D]  ──>  [(B · R_A), F_A+1, D]  ──SelfAttn──>  [B, F_A+1, R_A, D]

```

### Row-Wise Self-Attention

For Table $A$, the feature dimension $(F_A + 1)$ is subsumed into the batch dimension so attention runs strictly across rows:


$$A_{\text{row}} = \text{Unstack}\Big(\text{SelfAttn}\big(\text{Stack}_{B, F_A+1}(A)\big)\Big) \in \mathbb{R}^{B \times (F_A + 1) \times R_A \times D}$$

### Feature-Wise Self-Attention

Next, the row dimension $R_A$ is subsumed into the batch dimension so attention runs across features (including the dedicated $y$-column):


$$A_{\text{feat}} = \text{Unstack}\Big(\text{SelfAttn}\big(\text{Stack}_{B, R_A}(A_{\text{row}})\big)\Big) \in \mathbb{R}^{B \times (F_A + 1) \times R_A \times D}$$

*(Table $B$ undergoes the identical sequence independently to yield $B_{\text{feat}} \in \mathbb{R}^{B \times (F_B + 1) \times R_B \times D}$.)*

---

## 2. Perceiver-IO Latent Bottleneck & Cross-Feature Attention

Because $R_A \neq R_B$, we cannot directly concatenate tables along the feature dimension. We use a **Perceiver-IO Encoder-Decoder bottleneck** to compress variable row counts into a shared latent row size $R_L$, perform cross-table feature attention, and decode back.

```
Encoder (Row Compression):  [(B · F_A+1), R_A, D]  ──CrossAttn(Q=L)──>  [(B · F_A+1), R_L, D]
Cross-Table Feature Attn:   [(B · R_L), (F_A+1) + (F_B+1), D]            (Shared R_L dimension)
Decoder (Row Expansion):    [(B · F_A+1), R_L, D]  ──CrossAttn(Q=s)──>  [(B · F_A+1), R_B, D]

```

### A. The Encoder (Compressing $R_A, R_B \to R_L$)

We initialize a learnable latent query seed $L \in \mathbb{R}^{R_L \times D}$ (purely in embedding space, invariant to feature counts). For every feature independently (including the target column), $L$ queries the rows of $A$ and $B$:

1. **Subsume Feature Dimension:** $A_{\text{feat}} \to [(B \cdot (F_A + 1)), R_A, D]$
2. **Broadcast Latents:** $Q_{\text{enc}} = \text{Repeat}(L) \in \mathbb{R}^{(B \cdot (F_A + 1)) \times R_L \times D}$
3. **Cross-Attention Compression:**

$$A_{\text{lat}} = \text{CrossAttn}\Big(Q = Q_{\text{enc}},\; K = A_{\text{feat}},\; V = A_{\text{feat}}\Big) \in \mathbb{R}^{(B \cdot (F_A + 1)) \times R_L \times D}$$


$$B_{\text{lat}} = \text{CrossAttn}\Big(Q = Q_{\text{enc}},\; K = B_{\text{feat}},\; V = B_{\text{feat}}\Big) \in \mathbb{R}^{(B \cdot (F_B + 1)) \times R_L \times D}$$



> **Key Bottleneck Property:** By sharing the learnable weights of $L$ across both tables, "latent slot $k$" specializes consistently across $A$ and $B$, providing an aligned row-basis for cross-table exchange.

### B. Cross-Table Feature Attention ($F_A \leftrightarrow F_B$)

With both tables aligned to exactly $R_L$ latent rows, we unstack and restack to subsume $(B, R_L)$ into the batch dimension, then concatenate along the feature dimension:

$$H_{\text{lat}} = \big[ A_{\text{lat}} \mathbin{\Vert} B_{\text{lat}} \big] \in \mathbb{R}^{(B \cdot R_L) \times ((F_A + 1) + (F_B + 1)) \times D}$$

We apply Multi-Head Self-Attention across $H_{\text{lat}}$ so that every feature in Table $A$ (including its $y$-column) attends to every feature in Table $B$ (including its $y$-column):


$$\tilde{H}_{\text{lat}} = \text{SelfAttn}(H_{\text{lat}}) \in \mathbb{R}^{(B \cdot R_L) \times ((F_A + 1) + (F_B + 1)) \times D}$$


We then slice $\tilde{H}_{\text{lat}}$ back into $\tilde{A}_{\text{lat}} \in \mathbb{R}^{(B \cdot (F_A + 1)) \times R_L \times D}$ and $\tilde{B}_{\text{lat}} \in \mathbb{R}^{(B \cdot (F_B + 1)) \times R_L \times D}$.

---

## 3. Per-Target-Column Querying & Zero-Shot Imputation ($A_{\text{in } B}$ and $B_{\text{in } A}$)

To construct the off-diagonal blocks of our matrix without collapsing column identity via mean-pooling, we use a **learned per-target-column query** to attention-pool over the source table's actual feature tokens.

### A. Attention-Pooling per Target Column

For each target column $f \in [F_A + 1]$, we introduce a learnable query vector $q_f \in \mathbb{R}^{1 \times D}$ (shared across rows). For a given row $r \in R_B$, $q_f$ attends over Table $B$'s actual $(F_B + 1)$ feature tokens for that row:

$$s[r, f] = \text{CrossAttn}\Big(Q = q_f,\; K = B_{\text{feat}}[:, :, r, :],\; V = B_{\text{feat}}[:, :, r, :]\Big) \in \mathbb{R}^{1 \times D}$$

This yields a tensor of specialized queries $S_{B \to A} \in \mathbb{R}^{B \times (F_A + 1) \times R_B \times D}$ where entry $(r, f)$ encodes: *"What signal in row $r$ of Table $B$ is most relevant for target column $f$ of Table $A$?"*

### B. Conditioned Decoding

We decode $\tilde{A}_{\text{lat}}$ conditioned directly on $S_{B \to A}$ rather than a single pooled row vector:

$$B_{\text{in } A}[r, f] = \text{CrossAttn}\Big(Q = s[r, f],\; K = \tilde{A}_{\text{lat}}[:, f, :, :],\; V = \tilde{A}_{\text{lat}}[:, f, :, :]\Big)$$

In batch tensor form:


$$B_{\text{in } A} \in \mathbb{R}^{B \times (F_A + 1) \times R_B \times D}, \quad A_{\text{in } B} \in \mathbb{R}^{B \times (F_B + 1) \times R_A \times D}$$

* **Computational Cost:** $\mathcal{O}\big(R_B \cdot (F_A+1) \cdot (F_B+1)\big)$ for attention-pooling plus $\mathcal{O}\big(R_B \cdot (F_A+1) \cdot R_L\big)$ for decoding. No $\mathcal{O}(N^2)$-scale bottleneck is introduced.

### C. Source Indicator Embeddings

To prevent cardinality imbalance from allowing synthetic rows to overwhelm real rows during softmax attention, we inject learnable **Source Indicator Embeddings** ($e_{\text{real}}, e_{\text{synth}} \in \mathbb{R}^D$):

$$\hat{A} = A_{\text{feat}} + e_{\text{real}}, \quad \hat{B}_{\text{in } A} = B_{\text{in } A} + e_{\text{synth}}$$

$$\hat{B} = B_{\text{feat}} + e_{\text{real}}, \quad \hat{A}_{\text{in } B} = A_{\text{in } B} + e_{\text{synth}}$$

---

## 4. The Block Matrix Layout & Context Phase

We assemble the support rows into a unified $(R_A + R_B) \times ((F_A + 1) + (F_B + 1))$ Block Matrix:

$$\mathcal{M}_{\text{support}} = \begin{bmatrix}  \hat{A} & \hat{A}_{\text{in } B} \\  \hat{B}_{\text{in } A} & \hat{B}  \end{bmatrix} \in \mathbb{R}^{B \times ((F_A + 1) + (F_B + 1)) \times (R_A + R_B) \times D}$$

Within $\mathcal{M}_{\text{support}}$, we execute:

* **Full Horizontal Feature Attention:** Runs across size $((F_A+1) + (F_B+1))$ for every row $r \in [R_A + R_B]$, allowing native features, translated features, and target columns to harmonize.
* **Full Vertical Row Attention:** Runs across size $(R_A + R_B)$ for every feature column $f \in [(F_A+1) + (F_B+1)]$, allowing real rows and translated auxiliary rows to exchange statistical strength.

---

## 5. Asymmetric Vertical Split & Dedicated Target Column Decoding

To prevent zero-padding contamination ($0$-tokens in the off-diagonal test blocks distorting softmax values) and maintain strict test-set independence, **test rows never participate in horizontal feature attention across tables**.

```
           F_A + 1 (Table A Schema)           F_B + 1 (Table B Schema)
     +------------------------------+------------------------------+
     |      A_test + e_real         |         0 (IGNORED)          |  <── Test Row Attn (Vertical Only)
     +------------------------------+------------------------------+
     |          A + e_real          |        A_in_B + e_synth      |
     |                              |                              |  <── Support Block (Full Horizontal
     |      B_in_A + e_synth        |             B + e_real       |      & Vertical Attention)
     +------------------------------+------------------------------+
     |         0 (IGNORED)          |        B_test + e_real       |  <── Test Row Attn (Vertical Only)
     +------------------------------+------------------------------+

```

### A. Test Row Attention for Table $A$

1. We take $A_{\text{test}} \in \mathbb{R}^{B \times (F_A + 1) \times R_{\text{test\_A}} \times D}$ and add $e_{\text{real}}$. Note that during test inference, the target column slice $A_{\text{test}}[:, -1, :, :]$ is initialized with a learned query/placeholder token.
2. We subsume feature dimension $(F_A + 1)$ into the batch dimension: $[(B \cdot (F_A + 1)), R_{\text{test\_A}}, D]$.
3. **Vertical Row Cross-Attention:** $A_{\text{test}}$ queries the concatenated column block $[ \hat{A} \mathbin{\Vert} \hat{B}_{\text{in } A} ]$:

$$A_{\text{out}} = \text{CrossAttn}\Big(Q = A_{\text{test}},\; K = \big[ \hat{A} \mathbin{\Vert} \hat{B}_{\text{in } A} \big],\; V = \big[ \hat{A} \mathbin{\Vert} \hat{B}_{\text{in } A} \big]\Big)$$



### B. Test Row Attention for Table $B$

Symmetrically, $B_{\text{test}} \in \mathbb{R}^{B \times (F_B + 1) \times R_{\text{test\_B}} \times D}$ queries the right-hand column block vertically:


$$B_{\text{out}} = \text{CrossAttn}\Big(Q = B_{\text{test}},\; K = \big[ \hat{A}_{\text{in } B} \mathbin{\Vert} \hat{B} \big],\; V = \big[ \hat{A}_{\text{in } B} \mathbin{\Vert} \hat{B} \big]\Big)$$

### C. Target Extraction & NLL Calculation

In true TabPFN style, **only the dedicated target column's final embedding is decoded to logits**. We slice the last column index (`-1`) from the updated test representations:

$$\text{Logits}_A = \text{DecoderLinear}\big(A_{\text{out}}[:, -1, :, :]\big) \in \mathbb{R}^{B \times R_{\text{test\_A}} \times C}$$

$$\text{Logits}_B = \text{DecoderLinear}\big(B_{\text{out}}[:, -1, :, :]\big) \in \mathbb{R}^{B \times R_{\text{test\_B}} \times C}$$

Because $A_{\text{out}}$ and $B_{\text{out}}$ are computed through disjoint vertical splits, their predictions are mathematically independent:


$$\mathcal{L}_{\text{total}} = \text{NLL}(\text{Logits}_A,\; Y_{A_{\text{test}}}) + \lambda \cdot \text{NLL}(\text{Logits}_B,\; Y_{B_{\text{test}}})$$

---

## Complete Tensor Transformation Reference

| Stage | Input Shape | Operation | Output Shape |
| --- | --- | --- | --- |
| **1. Intra-Table Attention** | $[B, F_A+1, R_A, D]$ | Row Attn $\to$ Feature Attn across columns (incl. dedicated $y$) | $[B, F_A+1, R_A, D]$ |
| **2. Latent Encoder** | $[(B \cdot (F_A+1)), R_A, D]$ | CrossAttn via Seed $L \in \mathbb{R}^{R_L \times D}$ | $[(B \cdot (F_A+1)), R_L, D]$ |
| **3. Cross-Feature Attn** | $[(B \cdot R_L), (F_A+1) + (F_B+1), D]$ | SelfAttn across joined feature + target space | $[(B \cdot R_L), (F_A+1) + (F_B+1), D]$ |
| **4. Latent Decoder** | $[(B \cdot (F_A+1)), R_L, D]$ | CrossAttn via Per-Target-Column queries $s[r, f]$ | $[B, F_A+1, R_B, D]$ ($B_{\text{in } A}$) |
| **5. Context Matrix** | Two $[R, F+1, D]$ pairs | 2D Block Assembly + Source Tokens | $[B, (F_A+F_B+2), R_A + R_B, D]$ |
| **6. Test Inference ($A$)** | $[(B \cdot (F_A+1)), R_{\text{test\_A}}, D]$ | Vertical CrossAttn over $[A \mathbin{\Vert} B_{\text{in } A}]$ | $[B, F_A+1, R_{\text{test\_A}}, D]$ |
| **7. Target Extraction** | $[B, F_A+1, R_{\text{test\_A}}, D]$ | Slice dedicated `-1` target column $\to$ Linear Decoder | $[B, R_{\text{test\_A}}, C_{\text{classes}}]$ |

---
