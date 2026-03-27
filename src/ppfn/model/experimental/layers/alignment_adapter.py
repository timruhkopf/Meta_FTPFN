import torch
import torch.nn as nn


# TODO:
#  0. add B2 as another stream in the batch
#  1. add stream transforms to get A_train as query for B2, padding the loss for all others (or B2, if A_train is shorter)
#  2. add a new multistream objective that knows about B2's objective.


class AlignmentStreamAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_task_pe=True, use_C_as_A=False, modify_B=False):
        """
        A 4-stream adapter (A, B, B2, C) that aligns B into A's domain (creating B2),
        filters it for negative transfer, and updates the contextual scratchpad (C).
        """
        super().__init__()
        self.d_model = d_model
        self.use_task_pe = use_task_pe
        self.use_C_as_A = use_C_as_A
        self.modify_B = modify_B

        if use_task_pe:
            # Index 0: Task A, Index 1: Task B/B2
            self.task_embedding = nn.Embedding(2, d_model)

        # -----------------------------------------------------------------
        # 1. Alignment Components (B -> B2)
        # -----------------------------------------------------------------
        self.norm_q_align = nn.LayerNorm(d_model)
        self.norm_k_align = nn.LayerNorm(d_model)

        # The Attention Sink: A learnable key, but its value will be forced to 0
        self.sink_key = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.align_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)


        # -----------------------------------------------------------------
        # 2. Contextual Integration Components (C -> A + B2)
        # -----------------------------------------------------------------
        self.norm_q_c = nn.LayerNorm(d_model)
        self.norm_k_c = nn.LayerNorm(d_model)
        self.norm_v_c = nn.LayerNorm(d_model)
        self.cross_attn_c = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.init_as_identity()

    def init_as_identity(self):
        """
        Forces the adapter to output zeros initially to maintain the frozen prior.
        """
        # Zero out C's MHA output projection
        nn.init.zeros_(self.cross_attn_c.out_proj.weight)
        if self.cross_attn_c.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn_c.out_proj.bias)

        # Safely zero out the final Linear layer in the FFN
        # We look specifically for the last nn.Linear module
        for layer in reversed(self.ffn):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                break  # Stop after initializing the final linear layer

        if self.use_task_pe:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)


    def align_B_to_A(self, B_train, A_train):
        """
        With the "Attention Sink" mechanism, we give the model a safe escape hatch for tokens in B that
        can't find a good match in A.

        The "Attention Sink" or Null Token (Implicit Routing)
        You append a single, learnable [SINK] token to $A_{train}$'s sequence before cross-attention.
            * The Rule: The Value ($V$) associated with the [SINK] token is strictly fixed to a zero vector.
            * The Mechanism: When $B_{train}$ generates its Queries ($Q$) to look at $A_{train}$'s Keys ($K$),
            it has the option to attend to the [SINK] key.
            * The Result: If a token in $B_{train}$ finds nothing geometrically or semantically similar in $A_{train}$
            (i.e., it can't confidently map itself), the model learns to route all its attention probability mass to
            the [SINK] token. Because the sink's value is zero, that specific $B$ token becomes a zero-vector in
            $B'_{train}$, effectively erasing itself from the context.


        """
        batch_size = B_train.shape[1]
        device = B_train.device

        # Expand sink key for the batch and create a zero-tensor for the sink value
        sink_k = self.sink_key.expand(1, batch_size, -1)
        sink_v = torch.zeros_like(sink_k, requires_grad=False).to(device)

        # Prepend sink to A_train
        Q_align = self.norm_q_align(B_train)
        K_align = torch.cat([sink_k, self.norm_k_align(A_train)], dim=0)
        V_align = torch.cat([sink_v, A_train], dim=0)  # V doesn't need norm here

        # B tries to find its structural equivalents in A
        B_aligned, align_weights = self.align_attn(Q_align, K_align, V_align)

        return B_aligned, align_weights

    def forward(self, A, B, C, sep,  **kwargs):
        """
        A, B, C: Latent representations of shape (T, Batch, d_model)
        B2: The aligned stream. If None, it is generated.
        sep: The sequence index separating train from test context.
        """
        batch_size = A.shape[1]
        device = A.device

        A_train = A[:sep]
        B_train = B[:sep]

        # Evaluate use_C_as_A before alignment
        if self.use_C_as_A:
            target_for_alignment = C[:sep]
        else:
            target_for_alignment = A_train


        # =================================================================
        # STEP 1: Domain Alignment (Generate B2_train)
        # =================================================================

        B_aligned, align_weights = self.align_B_to_A(B_train, target_for_alignment)
        """
        Main intent here is to align B's train context; and project it into the domain of A, creating B2.
        This allows us to place an explicit loss on the alignment quality, by asking 
        Q = A_train, K = B2_train, V = B2_train --> NLL[Bar() , y^A_train], we have a side constraint of alignment quality.
        
        Why this perfectly captures your requirement:
            * It is flexible: It does not dictate where $B'_{train}$ must sit in the latent space. 
            It only dictates that wherever it sits, the frozen PFN must be able to read it and successfully compute $A$'s outputs.
            * It preserves rich information: If $B$ has extra information outside the support of $A$, the PFN will 
            simply ignore it for $\mathcal{L}_{cross}$ (since it's only queried at $X_{A_{train}}$), but that extra 
            information survives and can be utilized in $\mathcal{L}_{task}$.
            * It prevents lazy attention: Without $\mathcal{L}_{cross}$, the cross-attention adapter might just learn
            to output zeros or copy $A_{train}$ exactly, collapsing the information in $B$. By forcing $B'_{train}$ 
            to act as a standalone context for $A$'s domain, you guarantee it carries meaningful weight.
            
        The final loss to train your adapter layers would simply be a weighted sum:
        $$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{cross}$$
        
        # ------------
        """

        # =================================================================
        # STEP 2: Contextual Integration (Update Scratchpad C)
        # =================================================================
        C_query = C

        if self.use_C_as_A:
            A_train = C[:sep]  # Optionally use C as the query for alignment, instead of A_train. This allows C to evolve more freely.


        # Apply task embeddings to distinguish the source domains
        if self.use_task_pe:
            pe_A = self.task_embedding(torch.tensor(0, device=device))
            pe_B = self.task_embedding(torch.tensor(1, device=device))
            A_context = A_train + pe_A
            B2_context = B_aligned + pe_B
        else:
            A_context = A_train
            B2_context = B_aligned

        # C queries the joint, safely aligned context
        context = torch.cat([A_context, B2_context], dim=0)

        attn_out, _ = self.cross_attn_c(
            self.norm_q_c(C_query),
            self.norm_k_c(context),
            self.norm_v_c(context)
        )

        # First Residual Update (Identity preserved by initialization)
        C = C + attn_out

        # Second Residual Update (FFN)
        C = C + self.ffn(self.norm_ffn(C))

        if self.modify_B:
            # if we want to add an alignment penalty, where B_test is e.g. A_test or A_train,
            # then rather than having a new stream B2, we can directly modify B with the aligned version for the
            # train context, and keep the test context as is.
            B = torch.cat([B_aligned, B[sep:]], dim=0)

        return A, B, C

