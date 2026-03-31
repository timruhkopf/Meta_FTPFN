import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """A standard 2-layer MLP to add non-linear depth."""

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class GatedLatentTransferLayer(nn.Module):
    """
    Gated Latent Transfer (GLT) Layer

    An asymmetric transfer learning layer designed to extract features from a source task (B)
    and inject them into a target task (A/C) while defending against negative transfer.

    Architecture Highlights:
    1. Shared Self-Attention: Contextualizes A, B, and C independently to extract
       relational features (e.g., local gradients, extrema).
    2. Spatial Interpolation: Uses cross-attention to estimate (inter-/exterpolate) features for query
       locations (C_test) based on known context (C_train).
    3. Cross-Attention for Transfer: C attends to both A and B's features, allowing it to
         selectively integrate information from B. It uses dummy tokens as an "escape valve" to ignore B when it's unhelpful.
         This avoids the necessity of a softmax constraint
    5. Gating and residual update:

        - Query-Aware Gate: A post-retrieval sigmoid gate that compares the
          requested query against the retrieved feature to scale the residual update.
          loosely inspired by Qwen3Attention
          Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. https://arxiv.org/abs/2505.06708
          https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py#L237

    Inspired by: Perceiver IO (Jaegle et al., 2021) and Neural Processes.
    """
    def __init__(self, dmodel=128, use_gate=False, use_valve=False, hp_mode="concat", compute_own_self_ABC_attn=True, use_struct_gate=True, use_spatial_interpolation=False):
        super().__init__()
        self.hp_mode = hp_mode
        self.dmodel = dmodel

        in_dim = dmodel * 2 if hp_mode == "concat" else dmodel

        self.linear_AB1 = MLP(in_dim, dmodel, in_dim)
        self.compute_own_self_ABC_attn = compute_own_self_ABC_attn
        if self.compute_own_self_ABC_attn:
            self.self_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4)

        self.use_struct_gate = use_struct_gate
        if self.use_struct_gate:
            self.alignment_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4)
            self.trust_proj = nn.Sequential(
                nn.Linear(2 * dmodel, dmodel),
                nn.Sigmoid()
            )
        self.cross_attention = nn.MultiheadAttention(embed_dim=in_dim, vdim=dmodel, num_heads=4)

        self.use_spatial_interpolation = use_spatial_interpolation
        if use_spatial_interpolation:
            self.C_test_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4, vdim=dmodel)

        self.out_proj = MLP(in_dim, dmodel, in_dim)

        self.use_gate = use_gate
        if use_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(4 * dmodel, dmodel),
                nn.Sigmoid()
            )
        self.use_valve = use_valve
        if use_valve:
            # Learnable dummy key and value (Shape: [1, 1, feature_dim])
            self.dummy_key = nn.Parameter(torch.randn(1, 1, in_dim))
            # Note: Adjust dummy_value dimension based on whether you concatenated hp into value earlier
            self.dummy_value = nn.Parameter(torch.randn(1, 1, dmodel))

        self.init_weights()

    def init_weights(self):
        """Neutral Initialization to fade-in the adapters layers """

    def _merge_hp(self, hp_tensor, data_tensor):
        """Helper to cleanly fuse coordinate embeddings and data."""
        if self.hp_mode == "concat":
            return torch.cat([hp_tensor, data_tensor], dim=-1)

        elif self.hp_mode == "ignore_hp":
            return data_tensor

        return hp_tensor + data_tensor  # "add" mode

    def forward(self, A, B, C, sep, hp, mask_A=None, mask_B=None, *args, **kwargs):
        """
            Args:
                A, B, C: Tensors of shape [T_total, Batch, D] (Data payloads)
                sep: Integer indicating the split point between train (context) and test (query)
                hp: Tuple of (hp_A, hp_B, hp_C) coordinate embeddings [T_total, Batch, D]
                mask_A, mask_B: Boolean masks [Batch, T_total] (True = padding)
        """
        device = A.device
        hp_A, hp_B, hp_C = hp

        # 1. Parse Context Components ----------------------------------------------------------------------------------
        A_train, B_train = A[:sep], B[:sep]
        C_train, C_test = C[:sep], C[sep:]
        hp_A_train, hp_B_train = hp_A[:sep], hp_B[:sep]
        hp_C_train, hp_C_test = hp_C[:sep], hp_C[sep:]

        # 2.0 self attend for A and B to extract relative features -----------------------------------------------------
        # To transform independent ($x, y$) coordinate points into context-aware shape descriptors. Instead of a point
        # just knowing "I am at $x=2.5, y=1.0$", self-attention allows it to realize "I am a local maximum in a
        # linearly increasing trend."
        # Matching functions/series, cross-task transfer cannot happen on raw coordinates alone.
        # Using shared weights for A, B, and C forces the network to develop a universal "shape vocabulary"
        # across all streams, which is a prerequisite for the later structural gating.
        # Consider: ideally, this should be already done by the pre-trained backbone, with the exception of
        #  the hp embeddings not being part of the attn.

        ABC = torch.cat([
            self._merge_hp(hp_A_train, A_train),
            self._merge_hp(hp_B_train, B_train),
            self._merge_hp(hp_C_train, C_train),
            # if we do this prior to any cross attention C is redundant (C=A)

        ], dim=1)

        # down project to dmodel
        ABC = self.linear_AB1(ABC)
        # if self.compute_own_self_ABC_attn:
        #     # 2.1. Deal with padding in self attention
        #     # since A and B have different sequence lengths, we need to build an attention mask
        #     # to prevent attending to the padded points.
        #     mask_A_train = mask_A[:, :sep] if mask_A is not None else None
        #     mask_B_train = mask_B[:, :sep] if mask_B is not None else None
        #
        #     if mask_A_train is not None and mask_B_train is not None:
        #         # C doesn't have a mask (it uses all its points), so it's all False
        #         mask_C_train = mask_A_train.clone()
        #
        #         # Concat along batch dimension (dim=0) because ABC has 3*Batch size
        #         self_attn_mask = torch.cat([mask_A_train, mask_B_train, mask_C_train], dim=0)
        #     else:
        #         mask_C_train = None
        #         self_attn_mask = None
        #
        #     # 2.2  A,B,C Shared self attention to extract relational features within A, B, C respectively.
        #     ABC, _ = self.self_attention(ABC, ABC, ABC, key_padding_mask=self_attn_mask)

        # Extract the feature descriptors for each task after self-attention
        a_dim, b_dim = A_train.shape[1], B_train.shape[1]
        A_feat, B_feat, C_feat = ABC[:, :a_dim], ABC[:, a_dim:a_dim + b_dim], ABC[:, a_dim + b_dim:]

        if self.use_struct_gate: # -------------------------------------------------------------------------------------
            mask_B_train = mask_B[:, :sep] if mask_B is not None else None
            # 3a. A interrogates B based on SHAPE, not location.
            # Query: A_feat, Keys: B_feat, Values: B_feat
            # (Notice: No hp_A or hp_B included here!)
            A_retrieved_from_B, _ = self.alignment_attention(
                query=A_feat,
                key=B_feat,
                value=B_feat,
                key_padding_mask=mask_B_train
            )

            # 3b. Evaluate the structural mismatch
            # If B is just a shifted A, A_retrieved_from_B will closely match A_feat
            # because A successfully found its structural twins in B.
            # If B is noise, this difference will be massive.
            structural_diff = torch.cat([A_feat, A_retrieved_from_B], dim=-1)

            # Project to a single score per point, then average across the A sequence
            # Shape goes from [T_A, Batch, D] -> [T_A, Batch, 1] -> [1, Batch, 1]
            pointwise_trust = self.trust_proj(structural_diff)  # e.g., Linear(D*2, 1) + Sigmoid
            task_trust_score = pointwise_trust.mean(dim=0, keepdim=True)

            # 3c. Gate Task B globally
            # If the structures don't align, task_trust_score drops to ~0.
            # Consider, multiplying the pointwise trust score on B_feat, AND the overall task trust score
            B_feat = B_feat * task_trust_score
            # B_feat *= pointwise_trust.expand_as(B_feat)

        # 3. Expand C_feat to C_test_feat by hp attention across coordinates. ------------------------------------------
        # spatial interpolation to get C_test_feat for the query points
        # Consider: that this could also be already be done in the backbone representation, so we might be able to just
        #  take C_test directly from the input without the need to attend
        # if self.use_spatial_interpolation:
        #     C_test_feat, _ = self.C_test_attention(hp_C_test, hp_C_train, C_feat, key_padding_mask=mask_C_train)
        # else:
        #     C_test_feat = self.linear_AB1(self._merge_hp(hp_C_test, C_test))

        # 4. Cross Attention from C to A's and B's features ------------------------------------------------------------
        # throw in A_feat and B_feat into one context for C to cross attend to.
        batch_size = A.shape[1]

        # Here we don't want the softmax constraint, because it will have a sum-to-one constraint across A and B,
        # but we want the model to be able to choose to attend to B
        # Expand dummy tokens to match batch size: [1, Batch, D]
        # Consider: In prior experiments, this section did not seem to contribute, because it more or less gave a fixed value
        if self.use_valve:
            d_key = self.dummy_key.expand(1, batch_size, -1)
            d_val = self.dummy_value.expand(1, batch_size, -1)
        else:
            d_key = torch.tensor([], device=device)
            d_val = torch.tensor([], device=device)

        query = torch.cat([
            self._merge_hp(hp_C_train, C_train),
            self._merge_hp(hp_C_test, C_test),
        ], dim=0)
        key = torch.cat([
            self._merge_hp(hp_A_train, A_train),
            self._merge_hp(hp_B_train, B_train),
            d_key  # <--- The Escape Valve
        ], dim=0)
        value = torch.cat([  # Value payload is without the hp, because only then it is in the
            A_feat, B_feat,
            d_val  # <--- The Escape Valve
        ], dim=0)  # we only want the features to be the value
        # we cannot attend to the B features, because they are not grounded in the same domain as C
        # TODO ideally, we'd know how B looks like in C's domain (B_train'), then we could make the value the raw B_train' payload (without hp).

        # Build the cross-attention mask
        # C is looking at A and B, so the keys sequence length is 2*T_max.
        # Mask shape needs to be [Batch, 2*T_max]
        # Build the cross-attention mask (Shape: [Batch, 2*sep])
        mask_A_train = mask_A[:, :sep] if mask_A is not None else None
        if mask_A_train is not None and mask_B_train is not None:
            # Concat along the sequence dimension (dim=1) for the mask
            cross_attn_mask = torch.cat([mask_A_train, mask_B_train], dim=1)
        else:
            cross_attn_mask = None

        # We must also append a 'False' to the cross_attn_mask so the dummy token is never masked out
        if cross_attn_mask is not None:
            if self.use_valve:
                dummy_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

                cross_attn_mask = torch.cat([cross_attn_mask, dummy_mask], dim=1)

        # Pass the mask to cross-attention
        # Consider: this is the key operation here!
        C_cross, _ = self.cross_attention(query, key, value, key_padding_mask=cross_attn_mask)

        # 5. Residual update to C with gating---------------------------------------------------------------------------
        # C is originally (T, B, D). We project C_cross down to match.
        # Evaluate the query AGAINST the retrieved context
        # This is heavily inspired by Highway Networks and GRUs.
        # It is local, point-by-point, and fully aware of the relationship between $C$ and $B$.
        # This allows the network to say: "I asked for a local maximum (Query),
        # but the feature vector I got back from B looks like a steep drop (C_cross).
        # This is useless to me. Close the gate."
        # Consider: this gating seemed to be uneffective.
        if self.use_gate:
            gate_input = torch.cat([query, C_cross], dim=-1)
            gate = self.gate_proj(gate_input)
        else:
            gate = 1.0

        # Gated Residual update
        C = C + gate * self.out_proj(C_cross)

        return A, B, C
