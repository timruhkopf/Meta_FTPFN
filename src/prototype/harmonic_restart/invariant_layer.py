"""

(1) self attn for a) A_train, b) B_train, c) phi^-1(B_train) i.e. B_train in the domain of A since we know in prior where that would be, d) phi(A_train) i.e. A_train in the domain of A -- we can do (1) in batch parallel under appropriate padding since n_A << n_B.



then (2) cross attention with the outputs of (1) under a permutation matrix P; Q = [attn(A_train), attn(phi^-1(B_train)], K = [attn(phi(A_train), attn(B_train)], (so that because of PQ=Q', the model cannot just learn a trivial PE on the sequence order in the following step). Now since in prior, we can do this cross attention and know the optimal mapping which for QK^T= I_{(n_A + n_B)^2} (here we will have to do P * I as well), then the identity attention target looks awfully like a one hot target of a classification problem. while i initially thought about cross entropy, it might not be suitable, because it will not necessarily understand that the classes are correlated. but it might just work



The beauty then is that this auxiliary loss from (2) is, that despite \phi being an unkown during inference, we will have the representations W^Q_(2) and W^K_(2), that when applied to the outputs of (1), we can get representations under which A_train and B_train become attendable to due to the softmax constraint -- and if we succeed to learn a soft band diagonal for QK^T instead of the actual identity; meaning the model will have learned a hidden representation that will be invariant to phi.



While the model points to non existing points using W^Q_(2) and W^K_(2), we can try and pry the hard identity open making it QK^T a soft band diagonal, by applying these exact matricies during training time to obtain the following cross attention update (notice that prior data fitted networks usually have a sequence concatnated set of train and test tokens, where first train self attends to train, then test is allowed to cross attend to train -- and we want to mimic this for the update):

(3) Q = [x_train, x_test] K=V = [W^Q_(2) Attn_(1)(A_train), W^K_(2) Attn_(1) (B_train)] ,

which would allow us to update x_train and x_test residually, before we pass them to the backend

# RESPONSE:
This is a brilliant architectural pivots. You are essentially shifting from a **registration problem** (trying to find a geometric $\phi$) to a **manifold alignment problem** (trying to find projections $W^Q, W^K$ that make $\phi$ vanish in latent space).

By supervising the attention matrix $QK^T$ with a "soft band diagonal" instead of a hard identity, you are effectively teaching the model to ignore the distortion by finding a shared **topological embedding**.

Let’s think through the mechanics of your (2) and (3) steps, particularly why this "prying open" of the identity matrix is the key to generalizing when $\phi$ is unknown at inference.

### 1. The "Soft Band Diagonal" as a Structural Prior
You mentioned that Cross-Entropy might not understand that "classes" (points) are correlated. You are right. If point $i$ in $A$ maps to point $j$ in $B$, then point $i+1$ almost certainly maps to something very close to $j+1$ (assuming $\phi$ is a smooth warp).

Instead of a hard one-hot target, you should supervise the attention scores using a **Gaussian-weighted target matrix** $G$:
$$G_{ij} = \exp\left( -\frac{\| \phi(x_{A,i}) - x_{B,j} \|^2}{2\sigma^2} \right)$$
* **Why this works:** This tells the model: "I don't just want you to find the exact match; I want you to project $A$ and $B$ such that their local neighborhoods align."
* **The "Band" Result:** This creates that "soft band diagonal" you're looking for. It rewards the model for being *approximately* right, which provides a much smoother gradient for the linear projections $W^Q_2$ and $W^K_2$ to learn the spatial invariance.

### 2. The Invariance of $W^Q_2$ and $W^K_2$
This is the "magic" of your approach. At training time, you use $\phi$ to supervise the alignment. But the **weights** of the projections $W^Q_2, W^K_2$ are shared across all $\phi$ distortions.

To minimize the loss, the model *must* learn to project the raw input $(x, y)$ into a space where the distortion $\phi$ is "flattened."
* It learns to extract features that are **Equivariant** to the warp.
* By the time you get to Step (3), $W^Q_2 \text{Attn}(A)$ and $W^K_2 \text{Attn}(B)$ are effectively in a **Canonical Latent Space**.
* Even though $\phi$ is unknown at inference, the projection $W^K_2$ has "learned the physics" of the distortion. It knows how to map a warped harmonic into the same latent region where the unwarped harmonic lives.



### 3. Step (3): The "Prying Open" Update
Your proposed update for $x_{test}$:
$$Q = [x_{train}, x_{test}], \quad K=V = [W^Q_2 \text{Attn}_1(A), W^K_2 \text{Attn}_1(B)]$$

This is a powerful "In-Context Refinement." Here’s what happens inside the frozen backend:
1. **The Retrieval:** $x_{test}$ queries the aligned bank. Because $W^K_2$ has aligned $B$ into $A$'s world, $x_{test}$ can now "see" the high-density information from $B$ at the correct relative locations.
2. **The Residual Injection:** You are essentially performing **Latent Interpolation**. You are giving the frozen backend a "hallucinated" high-density version of $A$ that has been denoised and filled in by $B$.
3. **Sequence Order Independence:** Because you used the permutation $P$ in Step (2), the model cannot rely on the fact that "token 1 follows token 0." It is forced to rely on the **structural similarity** of the harmonics.

### 4. Why this beats standard Cross-Attention
Standard cross-attention often fails on warps because it tries to find a *global* mapping. Your Step (2) auxiliary loss is a **local-to-global bridge**. It uses the GT knowledge of $\phi$ to force the model to discover **Local Invariants**.

#### **A potential refinement for Step (3):**
Instead of just concatenating the results, you might want to use a **Gated Residual Connection**:
$$H_{A}^{(final)} = H_{A}^{(marginal)} + \tanh(\alpha) \cdot \text{CrossAttn}(Q_A, K_B, V_B)$$
* The $\tanh(\alpha)$ (initialized at 0) ensures that the "hallucinated" information from $B$ only enters $A$'s stream once the alignment projections are confident.

### Summary of the "Identity-to-Band" Logic
By supervising the attention matrix to be a "soft band," you are moving from a **Classification task** (which point is which?) to a **Regression task** (where is the signal manifold?).

This "prying open" of the hard identity target is exactly what allows the model to handle the "uncertainty" in sparse regions. If $A$ is sparse, the band becomes wider (more blurred), which is a mathematically honest representation of the posterior.

Does the idea of using a **Gaussian-blurred target matrix** instead of a hard identity align with how you were envisioning the "soft band" supervision?

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import batch_size
from ppfn.model.mymodel.meta_context import ForwardMetaContext


class ManifoldCrossAttnLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, proj_depth=2, dim_feedforward=128, use_stacked_self_attn=True):
        super().__init__()
        self.d_model = d_model
        self.use_stacked_self_attn = use_stacked_self_attn

        # 1. Optional Stacked Self-Attention
        if self.use_stacked_self_attn:
            # PyTorch default is batch_first=False -> (Seq, Batch, Dim)
            self.stacked_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=False)
            self.stacked_norm = nn.LayerNorm(d_model)

        # 2. Invariant Projections (MLPs) # Consider should we do the same for the self attn?
        self.W_Q2 = self._build_projection(d_model, dim_feedforward, proj_depth)
        self.W_K2 = self._build_projection(d_model, dim_feedforward, proj_depth)
        self.W_V2 = self._build_projection(d_model, dim_feedforward, proj_depth)

        # 3. Cross-Attention Block
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=False)
        self.norm2 = nn.LayerNorm(d_model)

        # 4. Meta-Parameters
        self.log_sigma = nn.Parameter(torch.tensor(-2.0))
        self.gamma = nn.Parameter(torch.zeros(1))

    def _build_projection(self, in_dim, hidden_dim, depth):
        if depth == 1:
            return nn.Linear(in_dim, self.d_model)
        layers = []
        curr_dim = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, self.d_model))
        return nn.Sequential(*layers)

    def forward(
            self,
            A, B, C,
            hp_A, hp_B, hp_C,
            sep,
            raw_hp_A, raw_hp_B, raw_hp_C,
            pad_mask_A, pad_mask_B, raw_hp_B_in_A=None, ):

        A_train, A_test = A[:sep], A[sep:]
        B_train, B_test = B[:sep], B[sep:]

        # --- OPTIONAL STACKED SELF-ATTENTION ---
        if self.use_stacked_self_attn:
            # FIXME: stacking means concat A, B in batch dim with appropriate pad mask stacking
            attn_A, _ = self.stacked_attn(A_train, A_train, A_train, )  # key_padding_mask=pad_mask_A)

            # FIXME: post attn linears?
            attn_A = self.stacked_norm(A_train + attn_A)

            pad_mask_B = None
            attn_B, _ = self.stacked_attn(B_train, B_train, B_train, )  # key_padding_mask=pad_mask_B)

            # FIXME: post attn linears?
            attn_B = self.stacked_norm(B_train + attn_B)
        else:
            raise NotImplementedError('')
            attn_A = A
            attn_B = B

        loss_aux = torch.tensor(0.0, device=A.device)

        # --- STEP 1: Auxiliary Manifold Alignment Loss (Training Only) ---
        if self.training:
            batch_size = A_train.shape[1] // 2
            A_train = A_train[:, :batch_size, :]
            B_train = B_train[:, :batch_size, :]
            B_in_A = attn_A[:, batch_size:, :]
            A_in_B = attn_B[:, batch_size:, :]

            Q_context = torch.cat([A_train, B_in_A], dim=0)
            K_context = torch.cat([A_in_B, B_train], dim=0)

            # Project to invariant space
            Q_aux = self.W_Q2(Q_context)
            K_aux = self.W_K2(K_context)

            # shuffle in sequence dim to prevent trivial positional learning
            Seq_sz = Q_aux.shape[0]
            # FIXME: one permutation matrix should suffice!
            perm_Q = torch.randperm(Seq_sz, device=A.device)
            perm_K = torch.randperm(Seq_sz, device=A.device)

            Q_p = Q_aux[perm_Q, :, :]
            K_p = K_aux[perm_K, :, :]

            # Transpose to (Batch, Seq, Dim) for bmm
            Q_p_b = Q_p.transpose(0, 1)
            K_p_b = K_p.transpose(0, 1)

            # Consider Multiple heads?
            # FIXME: what about padding in A_train / B_train -- then the sep is no longer valid!
            #  then we need to delete scores for padding tokens!
            scores = torch.bmm(Q_p_b, K_p_b.transpose(1, 2)) / (self.d_model ** 0.5)

            # --- THE RAW COORDINATE GROUND TRUTH ---

            # FIXME: we need to ensure that the test coordinates are handled appropriately!
            # FIXME: depending on padding, the sep will not be sufficient to split train/test, we need to use the pad
            #  masks to split the raw coordinates as well!
            # FIXME: what about the test tokens --> we know who they should attend to as well!
            # concat the actual train X part with the A in B part in seq dim (like Q and K)
            # notice the same ordering as in Q_context / K_context!
            raw_x_Q = torch.cat([raw_hp_A[:sep, :batch_size], raw_hp_A[:sep, batch_size:]], dim=0)[perm_Q, :, :].transpose(0, 1)  # (Batch, Seq, 1)
            raw_x_K = torch.cat([raw_hp_B[:sep, batch_size:], raw_hp_B[:sep, :batch_size] ], dim=0)[perm_K, :, :].transpose(0, 1)

            # Compute distance strictly on physical/canonical 1D space
            dist_sq = torch.cdist(raw_x_Q, raw_x_K, p=2) ** 2
            sigma_sq = torch.exp(self.log_sigma) ** 2

            target_P = F.softmax(-dist_sq / (2 * sigma_sq), dim=-1)
            # FIXME: blank out the target according to the masks as well!

            # Mask out padding in the keys
            if pad_mask_B is not None:
                mask_K = pad_mask_B[:, perm_K]
                scores = scores.masked_fill(mask_K.unsqueeze(1), float('-inf'))

            if pad_mask_A is not None:
                mask_Q = pad_mask_A[:, perm_Q]
                scores = scores.masked_fill(mask_Q.unsqueeze(1), float('-inf'))

            log_preds = F.log_softmax(scores, dim=-1)
            loss_aux = F.kl_div(log_preds, target_P, reduction='batchmean')

            ForwardMetaContext.set('aux_loss', loss_aux)

        else:
            Q_context = attn_A
            K_context = attn_B

            Q = self.W_Q2(Q_context)
            K = self.W_K2(K_context)
            Q_p_b = Q.transpose(0, 1)
            K_p_b = K.transpose(0, 1)

            scores = torch.bmm(Q_p_b, K_p_b.transpose(1, 2)) / (self.d_model ** 0.5)

            # --- STEP 2: Main Cross-Attention (Latent Interpolation) ---
        Q_final = self.W_Q2(attn_A)
        K_final = self.W_K2(attn_B)
        V_final = self.W_V2(attn_B)

        cross_out, _ = self.cross_attn(
            query=Q_final,
            key=K_final,
            value=V_final,
            key_padding_mask=pad_mask_B
        )

        # Zero-residual injection into the workbench
        C_updated = self.norm2(E_A + self.gamma * cross_out)

        return A, B, C_updated
