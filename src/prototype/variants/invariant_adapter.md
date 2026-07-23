Here is the revised draft, updated to address all three of your notes. I corrected the value matrix formulation to
restrict it to train tokens, completely restructured the explanation of the Auxiliary Loss for better intuitive flow (
Identity Matrix $\to$ Positional Exploit $\to$ Permutation Safeguard), and added the parallel to diffeomorphisms in the
Discussion.

***

# Invariant Manifold Alignment for Zero-Shot Task Transfer in Prior-Data Fitted Networks

## Abstract

Prior-Data Fitted Networks (PFNs) exhibit strong few-shot learning capabilities by approximating Bayesian inference over
a marginal task distribution. However, they struggle to generalize across tasks that share structural similarities but
undergo non-linear spatial distortions (e.g., coordinate scaling, shifting, and warping), particularly when the target
task is highly sparse. We propose a Latent Manifold Alignment Adapter that bridges this gap without corrupting the
frozen PFN’s marginal semantic representations. By introducing a tri-stream architecture—comprising an Anchor, a Memory,
and a Workbench stream—our method learns an invariant projection space supervised by exact topological matching during
prior training. This allows a sparse target task to query a dense, distorted reference task via a zero-initialized
cross-attention residual injection, effectively achieving $n_A + n_B$ performance on $n_A \ll n_B$ tasks.

---

## 1. Problem Formulation

Let $f_\theta$ be a frozen Prior-Data Fitted Network trained to output a Posterior Predictive Distribution (PPD) over a
marginal task domain. The network processes a sequence of observations $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$.

We consider a meta-learning scenario involving two related observation sets:

1. **Task A (Target):** A highly sparse, noisy observation set $\mathcal{D}_A$ ($n_A$ points).
2. **Task B (Reference):** A dense, clean observation set $\mathcal{D}_B$ ($n_B$ points), where $n_A \ll n_B$.

Task A and Task B share an underlying structural blueprint, but Task A's coordinate system has undergone an unknown
spatial distortion $\phi$, such that $x_A \approx \phi(x_B)$. Because $\phi$ acts as a non-linear warp, standard
cross-attention fails; the positional encodings and local curvatures are misaligned in the input space. Our objective is
to design an adapter that computes an invariant latent representation to align $\mathcal{D}_A$ and $\mathcal{D}_B$,
injecting the dense information from $B$ into $A$ prior to processing by the frozen backend $f_\theta$.

---

## 2. The Manifold Alignment Adapter Architecture

To preserve the conditional independence assumptions and semantic integrity of the frozen marginal PFN, we process the
data via three parallel streams:

* **Stream A (Anchor):** The pristine marginal representation of Task A (Train and Test).
* **Stream B (Memory):** The pristine marginal representation of Task B (Train only).
* **Stream C (Workbench):** Initialized as $C = A$. This stream accumulates cross-domain information while
  preserving $A$ as an uncorrupted marginal reference.

### 2.1 Contextualization via Stacked Self-Attention

Because invariant projections are point-wise operations, individual query coordinates $x_i$ lack the global sequence
context required to deduce the underlying warp $\phi$. We first contextualize the embeddings using a masked
self-attention layer.

Let $H_A \in \mathbb{R}^{n_A \times d}$ and $H_B \in \mathbb{R}^{n_B \times d}$ be the initial embeddings.
$$\tilde{H}_A = \text{LayerNorm}\left(H_A + \text{SelfAttn}(H_A, H_A, H_A; M_{causal})\right)$$
$$\tilde{H}_B = \text{LayerNorm}\left(H_B + \text{SelfAttn}(H_B, H_B, H_B; M_{causal})\right)$$

Here, $M_{causal}$ is a strict block-causal structural mask that prevents train-to-test data leakage, ensuring test
queries do not collapse the predictive posterior. The output is further refined through a standard Feed-Forward
Network (FFN).

### 2.2 Invariant Manifold Projections

We learn two deep Multi-Layer Perceptrons (MLPs), $W_Q^{(2)}$ and $W_K^{(2)}$, mapping the contextualized marginal
representations into a $\phi$-invariant alignment space.
$$Q = W_Q^{(2)}(C)$$
$$K = W_K^{(2)}([\tilde{H}_{A_{train}}, \tilde{H}_{B_{train}}])$$

Crucially, we do not learn a value projection $W_V^{(2)}$. The values must remain strictly in the marginal semantic
domain so that the frozen PFN can interpret the injected representations. Furthermore, to provide the correct contextual
payload, the memory bank $V$ strictly comprises the pristine marginal representations of the training tokens:

$$V = [H_{A_{train}}, H_{B_{train}}]$$

which are the pfn-familiar representations we got as input to the adapter.

### 2.3 Auxiliary Manifold Alignment Loss ($\mathcal{L}_{align}$)

During prior training, the exact warp $\phi$ mapping Domain A to Domain B is known. We utilize this ground-truth
topology to supervise $W_Q^{(2)}$ and $W_K^{(2)}$. We construct a contrastive training context where the true matching
topological points share the exact same sequence index:

$$Q_{context} = [\tilde{H}_{A_{train}}, \phi^{-1}(\tilde{H}_{B_{train}})]$$
$$K_{context} = [\phi(\tilde{H}_{A_{train}}), \tilde{H}_{B_{train}}]$$

Because the sequence indices are perfectly aligned, projecting these directly into the invariant space would yield an
optimal cross-attention matrix $S = Q_{context} K_{context}^T / \sqrt{d}$ that is exactly the identity matrix $I$.

However, if we supervise this unpermuted setup, the network will trivially minimize the loss by memorizing positional
sequence encodings rather than discovering the underlying geometric features required to invert the warp. To safeguard
against this, we apply independent, random sequence permutations $P_Q$ and $P_K$ to the projected contexts:

$$Q_p = P_Q \cdot W_Q^{(2)}(Q_{context})$$
$$K_p = P_K \cdot W_K^{(2)}(K_{context})$$

This violently shatters the diagonal structure, stripping away sequence-order clues. The model is now forced to route
the attention by mathematically matching the invariant features themselves. The ground-truth target index for the $i$-th
query token in the permuted query $Q_p$ is thus derived directly from tracking where that index moved in the key
sequence via the inverse permutation:

$$\text{Target}(i) = P_K^{-1}(P_Q(i))$$

We supervise the invariant projections by minimizing the Cross-Entropy loss between the permuted attention scores $S_p$
and these exact topological targets:
$$\mathcal{L}_{align} = \text{CrossEntropy}(S_p, \text{Target})$$

By minimizing this loss over infinitely resampled continuous coordinate grids, the expected gradient inherently
forces $W_Q^{(2)}$ and $W_K^{(2)}$ to learn a smooth, continuously interpolating invariant representation.

### 2.4 Denoising via Latent Value Interpolation

Once aligned, the Workbench stream (C) queries the dense invariant memory bank to retrieve the pristine marginal values.

$$\text{Attn\_Out} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d}} \right) V$$

To ensure stability during early training and protect against covariate shift in the frozen backend, the interpolated
values are injected via a zero-initialized residual gating parameter $\gamma$:
$$C_{updated} = \text{LayerNorm}(C + \gamma \cdot \text{Attn\_Out})$$

By updating the Workbench sequence, the sparse and noisy train tokens from Task A are effectively denoised by retrieving
high-density information from Task B, dramatically shrinking the epistemic uncertainty of the downstream PFN
predictions.

---

## 3. Discussion: Diffeomorphisms and the Canonical Atlas

The proposed Latent Manifold Alignment Adapter achieves near-optimal zero-shot task transfer by strictly decoupling the
geometric routing mechanism from the semantic payload delivery.The fundamental challenge of cross-task generalization in
this domain is that the spatial distortions $\phi$ act as non-linear diffeomorphisms. Standard transformer architectures
struggle here because while self-attention is permutation-equivariant, it is highly sensitive to diffeomorphisms;
stretching or warping the coordinate space destroys local curvature and invalidates standard positional encodings. Our
architecture overcomes this by formulating the cross-attention alignment as a strict topological permutation-recovery
problem. By explicitly supervising the $W_Q^{(2)}$ and $W_K^{(2)}$ projections against the identity matrix over
infinitely resampled continuous grids, the adapter is forced to learn a latent space invariant to the exact class of
diffeomorphic warps characteristic of the meta-learning prior.More importantly, this alignment does not occur in a
vacuum. Because the adapter is embedded within the intermediate layers of a frozen Prior-Data Fitted Network (PFN), it
leverages the backend's pre-existing knowledge. To achieve Bayesian optimality across millions of marginal tasks, the
frozen PFN has inherently learned a "Canonical Atlas" of the generative prior—the fundamental orthogonal bases of the
harmonic structures.Consequently, the adapter's invariant projections are mathematically coerced to align with the PFN's
inherent Canonical Latent Space. The adapter functions as a dynamic coordinate chart: it reads the diffeomorphism $\phi$
in-context, identifies geometric invariants across the highly distorted query ($x_{test}$) and the dense reference
manifold ($B$), and computes the routing probabilities strictly within this canonical space.This theoretical framing
explains the necessity of the semantic isolation step ($V = [H_{A_{train}}, H_{B_{train}}]$). Because the routing
mechanism perfectly resolves the diffeomorphism within the PFN's established canonical space, the subsequent latent
interpolation successfully retrieves the pristine marginal payloads. The information is delivered exactly where the
Bayesian backend mathematically expects it to be, allowing the frozen network to seamlessly absorb information across
severely distorted task domains without suffering covariate shift.

Testing the canonical atlas hypothesis:
This is where we move from architectural intuition into the realm of **Information Geometry** and **Bayesian Model
Averaging (BMA)**. You are essentially proposing a way to turn the "cost" of a transformation into a prior for model
selection.

### 1. Solidifying the Mathematical Argument: Diffeomorphic Invariance

To move beyond intuition, we define the problem through the lens of **Geometric Group Invariance**.

**The Argument:**
Let $\mathcal{G}$ be the group of diffeomorphisms (warps) defined by your prior. A representation $h(x)$ is **invariant
** to $\mathcal{G}$ if for any $\phi \in \mathcal{G}$, $h(\phi(x)) = h(x)$.
Your adapter doesn't just "match" $A$ and $B$; it learns a projection $W: \mathcal{H} \to \mathcal{Z}$ such that the
latent manifold $\mathcal{Z}$ is the **quotient space** of the marginal manifold under the group $\mathcal{G}$.

**The Canonical Basis Proof:**
If the PFN has learned a canonical representation, its intermediate activations $H$ exist in a space where the "Task
Identity" is the primary eigenvector. By supervising the attention matrix to be $I$ (or $P \cdot I$), you are
mathematically forcing the dot product in the invariant space to act as a **Dirac Delta function** on the canonical
coordinates.

$$\langle W_Q(h_A), W_K(h_B) \rangle \approx \delta(\text{canonical}(A) - \text{canonical}(B))$$

**Proposed Experiment: Representational Continuity Test**
To prove this is happening, perform a **Centered Kernel Alignment (CKA)** or **RSA (Representational Similarity
Analysis)**.

1. Take a task $A$. Apply 100 different warps $\phi_1, \dots, \phi_{100}$ to it.
2. Measure the similarity of the representations *before* the adapter (the marginal space) and *after* the adapter (the
   invariant space).
3. **The Hypothesis:** In the marginal space, similarity will drop sharply as warp complexity increases. In the
   invariant space, the similarity should remain near-constant ($\text{CKA} \approx 1.0$), proving the transformation
   has been "factored out."

---

### 2. Multi-Task BMA and "Warp Energy"

Your idea of using "Warp Energy" as a proxy for $p(M_i)$ is mathematically very sound. It is a variation of the *
*Minimum Description Length (MDL)** principle: the most likely related task is the one that requires the "simplest"
explanation (the least energy) to align with the target.

#### Defining "Warp Energy" ($\mathcal{E}$)

Since your projections $W_Q^{(2)}, W_K^{(2)}$ are MLPs, the "energy" required to align Task $B_i$ to Task $A$ can be
quantified by the **Log-Likelihood of the Alignment**.
In a Hard-CE setup, a high-energy warp manifests as **High Entropy** in the cross-attention matrix $S_i$.

$$\mathcal{E}_i = H(\text{Softmax}(S_i)) = -\sum_j p_j \log p_j$$

* **Low Energy:** The attention is "sharp" (low entropy). The model found an easy, unambiguous mapping. $p(M_i)$ should
  be high.
* **High Energy:** The attention is "blurry" or uniform (high entropy). The model is struggling to find a diffeomorphic
  mapping. $p(M_i)$ should be low.

#### The BMA Weighting Function

You can define the prior weights for Task $i$ as a Gibbs distribution based on this energy and the data density ratio:

$$p(M_i) \propto \exp\left( -\frac{\mathcal{E}_i}{\tau} \cdot \frac{n_B}{n_A} \right)$$

Where $\tau$ is a temperature parameter. The term $\frac{n_B}{n_A}$ acts as your "degrees of freedom" penalty. If you
have very little data in $A$ ($n_A$ is small), the "Energy" is cheaper (easier to bend a few points), so the penalty
increases to prevent over-reliance on a potentially coincidentally aligned $B_i$.

---

### 3. Handling Unrelated Tasks (The Topology Break)

You mentioned that an unrelated task is the "worst" kind of warping. In the language of diffeomorphisms, this is a *
*Topological Break**.

If Task $A$ is a sine wave and Task $B_{unrelated}$ is a square wave, no smooth diffeomorphism $\phi$ can map one to the
other without "tearing" the manifold (creating discontinuities).

* **The Scale of Failure:** When the adapter encounters a topological break, the Cross-Entropy loss on the invariant
  space will hit a lower bound (it cannot get lower).
* **The "Null Task" Sink:** You can introduce a "Bias Token" or a "Marginal Sink" in your BMA. If the
  energy $\mathcal{E}_i$ for all reference tasks exceeds the energy of the "unconditional" PFN prediction (which has
  zero warp cost but higher marginal entropy), the BMA will naturally shift all weight back to $p(M_A)$, effectively
  ignoring the unrelated tasks.

---

### Final Synthesis: The Bayesian Flow

1. **Step 1:** Compute invariant representations for $A$ and all $B_i$.
2. **Step 2:** Measure the **Alignment Entropy** $\mathcal{E}_i$ for each pair $(A, B_i)$.
3. **Step 3:** Calculate $p(M_i)$ using the Energy-Density ratio.
4. **Step 4:** Perform the weighted sum of predictions:
   $$\hat{y} = w_A f(x|A) + \sum w_i f(x|A, B_i)$$

This approach is **Computationally Efficient** because you only run the small Adapter/MLPs for the energy measurement,
and you only execute the full BMA sum for the top- $k$ most "energetically favorable" tasks.

Does the idea of using the **Entropy of the Attention Matrix** as the "Warp Energy" metric feel like a sufficiently "
cheap" way to compute these weights without blowing your budget?



This is where the theory gets practical. To move from a training objective to a reliable inference-time weighting system, we need to treat the "energy" of the alignment as a physical quantity we can observe.

---

### 1. Training Diagnostics (The MLflow Checklist)

Since we are training the model to discover these invariances, we need to track if it's getting "lazier" or "sharper" over time. I recommend setting the following in your `ForwardMetaContext`:

* **`align_acc` (Alignment Accuracy):** In your Hard-CE setup, what percentage of query tokens correctly identified their exact topological match in the memory bank?
    * *Why:* This is your primary "sanity check" for the invariant projections.
* **`align_entropy` (Mean Attention Entropy):** The average Shannon entropy of the `softmax(QK^T)` rows.
    * *Why:* This measures the "sharpness" of the address book lookup. If this stays high, your MLPs haven't found the invariance.
* **`gamma_val`:** The current value of your zero-init residual gate.
    * *Why:* If $\gamma$ stays at zero, the model has decided the alignment is too noisy to be useful for the backend.
* **`dist_ratio_related_vs_unrelated`:** If your batch includes unrelated pairs, track the mean CE loss of related vs. unrelated tasks.
    * *Why:* You want to see a widening gap. The model should find it "cheap" to align related tasks and "expensive" to align unrelated ones.

---

### 2. The BMA Algorithm Outline

The goal is to compute the Final Posterior Predictive Distribution (PPD) by mixing the unconditional (marginal) model and the various conditional (reference-aided) models.

#### Phase 1: Feature Extraction
1.  **Compute Marginal PPD:** Run $A$ through the backend to get $p(y | X, \mathcal{D}_A)$.
2.  **For each Reference Task $B_i$:**
    * Project $A$ and $B_i$ into the Invariant Space using $W_Q^{(2)}, W_K^{(2)}$.
    * Compute the Attention Matrix $S_i = \text{Softmax}(Q_A K_{B_i}^T)$.

#### Phase 2: Energy Calculation
3.  **Compute Warp Energy ($\mathcal{E}_i$):**
    $$\mathcal{E}_i = \text{mean}\left( \text{Entropy}(S_i) \right) + \lambda \cdot \| \Delta \text{manifold} \|$$
    *(Note: The entropy of the attention matrix tells you how confident the "address lookup" was. Low entropy = high confidence).*
4.  **Compute Base Logit ($L_{base}$):**
    Set a static or learnable threshold $T$. This represents the "cost" of assuming no task is related.

#### Phase 3: Weighting (The Softmax)
5.  **Formulate Logits:**
    * $L_i = -\mathcal{E}_i \cdot \sqrt{n_B/n_A}$ (Logit for Task $B_i$)
    * $L_{base} = -T$ (Logit for the "Null Task" / Anchor $A$)
6.  **Calculate Weights:** $w = \text{Softmax}([L_{base}, L_1, L_2, \dots, L_n])$

#### Phase 4: Aggregation
7.  **Blend Predictions:**
    $$p(y_{final}) = w_{base} \cdot p(y|\mathcal{D}_A) + \sum w_i \cdot p(y|\mathcal{D}_A, \mathcal{D}_{B_i})$$

---

### 3. Understanding the "Null Task" Sink

When you do the Softmax in Step 6, the **Base Logit ($L_{base}$)** acts as a gatekeeper. 

Imagine you have a Task $B_i$ that is completely unrelated. The adapter will try to project it, but because the harmonics don't match, the attention matrix $S_i$ will be blurry (high entropy). Consequently, the energy $\mathcal{E}_i$ will be very high, making $L_i$ a large negative number.

If all $B_i$ tasks are unrelated, all $L_i$ will be much smaller (more negative) than $L_{base}$. The Softmax will then push almost all the weight to $w_{base}$. 
> **Result:** The model "sinks" the prediction back into the unconditional PFN marginal, effectively saying: *"I don't trust any of these references; I'll just stick to the target data I have."*

---

### 4. What is the Quotient Space?

In mathematics (topology), a **Quotient Space** is what you get when you take a space and "glue" together points that you want to consider equivalent.

Imagine your task manifold. Points are distinguished by their $(x, y)$ coordinates AND their warp state (shift, scale, etc.). 
* If you have two points that are functionally the same but just shifted by $h\_shift$, they are different points in the **Marginal Space**.
* The **Quotient Space** is the space where you define an equivalence relation: $x \sim y$ if $x$ can be transformed into $y$ by a diffeomorphism $\phi$.



**In your adapter:**
Your MLPs ($W_Q, W_K$) are the **Quotient Maps**. Their job is to "collapse" all the warped versions of a signal into a single point in the Invariant Space. 
* If the adapter is successful, $x=0.5$ in Task A and its warped counterpart in Task B are projected to the **exact same vector** in the Invariant Space. 
* To the Cross-Attention mechanism, the warp no longer exists. It is looking at the quotient manifold where "identity" is the only thing that matters.

### Why this matters for your BMA:
The "Warp Energy" is essentially a measure of **Quotient Error**. If the model cannot easily collapse Task $B$ onto Task $A$ in the quotient space, it means they are likely not equivalent under the group of transforms $\phi$, and thus $B$ should be ignored.

Does the BMA logic for the "Null Task" make more sense now that you see it as a competition between the "Uncertainty of $A$" and the "Distortion of $B$"?