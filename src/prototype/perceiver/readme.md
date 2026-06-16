There is a crucial mathematical distinction between standard multi-modal attention and your feature-wise
attention that makes the DeepSets/Perceiver approach necessary. Let's untangle the multi-modal connection first, and
then dive into exactly how the set-attention bottlenecks work.

### **The Multi-Modal Connection: Sequence Length vs. Embedding Dimension**

In standard multi-modal models (like CLIP, BLIP, or Flamingo), the varying sizes ($n_{text}$ and $n_{image}$) represent
the **sequence length**.

* Text matrix: $X_{text} \in \mathbb{R}^{n_{text} \times d_{model}}$
* Image matrix: $X_{image} \in \mathbb{R}^{n_{image} \times d_{model}}$

Because the embedding dimension ($d_{model}$) is identical, you can easily compute cross-attention between them. The
resulting attention matrix simply becomes a non-square $n_{text} \times n_{image}$ matrix. The attention mechanism
naturally handles differing sequence lengths.

**Your Feature-Wise Problem:**
When you transpose your data to attend over *features* instead of *instances*, the roles swap. Your sequence length
becomes the number of features ($d$), and your embedding dimension becomes the number of samples ($n_A$ or $n_B$).

* Task A transposed: $X_A^T \in \mathbb{R}^{d \times n_A}$
* Task B transposed: $X_B^T \in \mathbb{R}^{d \times n_B}$

You cannot compute a dot product (cross-attention) between a Query vector of size $n_B$ and a Key vector of size $n_A$.
The linear algebra simply fails.

To fix this, we must map both $n_A$ and $n_B$ into a shared, fixed-size latent embedding dimension $h$. This is where
DeepSets and Perceivers come in.

---

### **1. The DeepSets Approach (Permutation-Invariant Projection)**

A DeepSet is the simplest mathematically sound way to map a set of varying size into a fixed-size vector, regardless of
ordering.

For a single feature $j$, you have a set of scalar values from Task A: $F_{A,j} = \{x_1, x_2, \dots, x_{n_A}\}$.
You want to map this to a fixed vector $v_{A,j} \in \mathbb{R}^h$.

**The Process:**

1. **Element-wise Projection:** Pass every scalar $x_i$ through a small shared MLP ($\phi$). This projects each 1D value
   into an $h$-dimensional vector.
2. **Pooling:** Apply a symmetric pooling operation (sum, mean, or max) across the $n_A$ dimension.
3. **Set Projection:** Pass the aggregated vector through a second MLP ($\rho$).

$$v_{A,j} = \rho \left( \frac{1}{n_A} \sum_{i=1}^{n_A} \phi(x_i) \right)$$

You repeat this for Task B. Now, regardless of $n_A \ll n_B$, both tasks are represented by matrices of
size $\mathbb{R}^{d \times h}$. You can now perform standard feature-wise cross-attention to translate $B$'s coordinates
into $A$'s coordinates.

---

### **2. The Perceiver Approach (Cross-Attention Bottleneck)**

While DeepSets work well, mean/max pooling can destroy complex distributional information. The **Perceiver architecture
** (specifically the Perceiver Resampler) was designed specifically to solve this exact bottleneck problem in a more
expressive way.

Interestingly, this is *exactly* how multi-modal models like DeepMind's Flamingo handle massive, variable-length video
and image inputs before feeding them to a text decoder.

Instead of mathematically pooling the set, you use a fixed number of **Learnable Latent Queries** to "read" the set.

**The Process:**

1. Define a fixed set of learnable latent vectors for your features: $Z \in \mathbb{R}^{k \times h}$ (where $k$ is how
   many "summary tokens" you want per feature; $k=1$ is often enough).
2. Project your feature values $F_{A,j} \in \mathbb{R}^{n_A \times 1}$ into $\mathbb{R}^{n_A \times h}$ to serve as Keys
   and Values.
3. Compute cross-attention where the fixed latents $Z$ query the variable-sized set $F_{A,j}$:

$$Z_{out} = \text{Softmax}\left( \frac{(Z W_Q)(F_{A,j} W_K)^T}{\sqrt{d_k}} \right) (F_{A,j} W_V)$$

**Why this is brilliant for your use case:**

* The output $Z_{out}$ is *always* strictly $\mathbb{R}^{k \times h}$, completely independent of $n_A$.
* It is permutation invariant (treating the data as a true set).
* Unlike a rigid DeepSet mean/max pool, the attention mechanism dynamically learns *what* parts of the empirical
  distribution to extract based on the task gradients.

### **Bringing it Together for Your Architecture**

If you use the Perceiver cross-attention block, your forward pass structure looks like this:

1. **Bottleneck:** Pass $A_{train}$ features and $B_{train}$ features through a Perceiver block. Both are reduced to
   robust, fixed-size representations $\in \mathbb{R}^{d \times h}$.
2. **Translation:** Feature-wise cross-attention where $B_{bottleneck}$ queries $A_{bottleneck}$. The model learns how
   the features of $B$ relate to the features of $A$.
3. **Reconstruction (Optional/Auxiliary):** Project the translated features back into the sample space of $B_{train}$ to
   calculate your $\mathcal{L}_{aux}$ against $B_{inA}$.
4. **Final Inference:** Now that $B_{train}$ is aligned with $A$'s coordinate system, your standard row-based PFN
   attention uses $[A_{train}, \tilde{B}_{train}]$ to infer $A_{test}$.

So let's recap the architecture:

A_train and B_train are passed through a shared, but respective featurewise cross attention, then B_train is allowed to
perceive A_train via the perceiver cross attention, then we apply row attention for A_train and B_train, where A_test is
allowed to cross attend on both A_train and B_train.