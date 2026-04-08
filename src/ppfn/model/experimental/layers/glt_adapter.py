import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext


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



class DistributionalGlobalGate(nn.Module):
    """Core idea: Instead of mean pooling or softmin/softmax pooling, which are just single summary statistics,
    we can learn a more expressive pooling function that looks at the entire distribution of trust scores."""
    def __init__(self, dmodel, num_quantiles=5):
        super().__init__()

        # We will extract 'num_quantiles' + 2 moments (mean, variance)
        self.dist_features_dim = num_quantiles + 4  # + mean, var, skewness, kurtosis

        # The Quantiles we want to sample (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]) # limited by n(task_B)
        self.register_buffer(
            "quantiles",
            torch.linspace(0.1, 0.9, num_quantiles)
        )

        # The MLP that looks at the distribution and outputs the global gate
        self.distribution_analyzer = nn.Sequential(
            nn.Linear(self.dist_features_dim, dmodel // 2),
            nn.GELU(),
            nn.Linear(dmodel // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, raw_trust_A):
        """
        raw_trust_A: [T_A, Batch, 1] - Pointwise trust scores
        """
        # 1. Cast and prepare
        raw_trust_A = raw_trust_A.float()
        n_A = raw_trust_A.shape[0]
        eps = 1e-6

        # 2. Calculate Raw Moments (Efficiently reduces dim=0 immediately)
        # These are E[X], E[X^2], E[X^3], E[X^4]
        # Memory footprint remains O(Batch) rather than O(T_A * Batch)
        mean_trust = raw_trust_A.mean(dim=0)  # E[X]
        raw_2 = torch.pow(raw_trust_A, 2).mean(dim=0)  # E[X^2]
        raw_3 = torch.pow(raw_trust_A, 3).mean(dim=0)  # E[X^3]
        raw_4 = torch.pow(raw_trust_A, 4).mean(dim=0)  # E[X^4]

        # 3. Calculate Variance and Central Moments
        # Var = E[X^2] - E[X]^2
        var_trust = raw_2 - (mean_trust ** 2)
        std_trust = torch.sqrt(var_trust + eps)

        # Central 3rd Moment: E[(X-mu)^3] = E[X^3] - 3*E[X^2]*mu + 2*mu^3
        central_3 = raw_3 - 3 * raw_2 * mean_trust + 2 * (mean_trust ** 3)
        skew_trust = central_3 / (std_trust ** 3 + eps)

        # Central 4th Moment: E[(X-mu)^4] = E[X^4] - 4*E[X^3]*mu + 6*E[X^2]*mu^2 - 3*mu^4
        central_4 = raw_4 - 4 * raw_3 * mean_trust + 6 * raw_2 * (mean_trust ** 2) - 3 * (mean_trust ** 4)
        kurt_trust = central_4 / (std_trust ** 4 + eps)

        # 4. Calculate Quantiles (Already relatively memory efficient)
        # Shape: [num_quantiles, Batch, 1]
        q_vals = torch.quantile(raw_trust_A, self.quantiles, dim=0)

        # 5. Assemble the Distribution Vector
        # Squeeze all to [Batch, N]
        q_vals = q_vals.transpose(0, 1).squeeze(-1)  # [Batch, num_quantiles]

        # Concatenate: All inputs should be [Batch, 1] before cat
        dist_vector = torch.cat([
            q_vals,
            mean_trust,
            var_trust,
            skew_trust,
            kurt_trust,
            # add in n_A, because it'll tell us how much we might already trust A on its own.
                torch.tensor([n_A], device=raw_trust_A.device).expand_as(mean_trust)
        ], dim=-1)

        # 6. Global Decision
        global_gate_logits = self.distribution_analyzer(dist_vector)

        return global_gate_logits.unsqueeze(0).to(raw_trust_A.dtype)


class StructuralGatingModule(nn.Module):
    """
    Evaluates the relationship between a Source (B) and Target (A).

    Supports different pooling strategies to aggregate structural alignment:
    - "mean": Average alignment across the sequence (baseline).
    - "softmin": Focuses on maximum discrepancy (the weakest link).
    - "softmax": Focuses on maximum alignment (the strongest link).
    """

    def __init__(self, dmodel, pool_mode="mean", temp=0.1, use_pointwise=True, use_global=True, zero_init=True):
        super().__init__()

        valid_modes = ["mean", "softmin", "softmax", "distributional", "gumbel"]
        if pool_mode not in valid_modes:
            raise ValueError(f"pool_mode must be one of {valid_modes}")

        self.pool_mode = pool_mode
        self.temp = temp
        self.use_pointwise = use_pointwise
        self.use_global = use_global

        # Shared alignment tool: A and B use this to "look" at each other
        self.alignment_attn = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4, batch_first=False)

        # Shared "Trust" evaluator: Projects [Feat, Retrieved_Feat] -> Score [0, 1]
        self.trust_proj = nn.Sequential(
            nn.Linear(2 * dmodel, dmodel),
            nn.GELU(),
        )

        if self.pool_mode == "mean":
            self.trust_proj.add_module("final_linear", nn.Linear(dmodel, 1))
            self.trust_proj.add_module("final_activation", nn.Sigmoid())
        elif self.pool_mode == "gumbel":
            # 1. ADD LAYERNORM: Prevents the features from exploding into massive logits
            self.trust_proj.add_module("norm", nn.LayerNorm(dmodel))

            # 2. CREATE FINAL LINEAR LAYER
            final_linear = nn.Linear(dmodel, 2)

            # 3. FIX BIASES: Force the gate to start "Closed" at step 0
            # Index 0 is "Drop" (We want this high)
            # Index 1 is "Keep" (We want this low)
            nn.init.constant_(final_linear.bias[0], 4.0)
            nn.init.constant_(final_linear.bias[1], -4.0)

            self.trust_proj.add_module("final_linear", final_linear)

        # if zero_init:
        #     nn.init.constant_(self.trust_proj[2].bias, -4.0)

        if pool_mode == "distributional":
            self.distributional_gate = DistributionalGlobalGate(dmodel)

    def forward(self, A_feat, B_feat, mask_A=None, mask_B=None):
        """
        Returns a scaling tensor of shape [T_B, Batch, 1]
        to be multiplied with B_feat.
        """
        batch_size = B_feat.shape[1]
        device = B_feat.device

        # Identity defaults (no filtering)
        pointwise_gate = torch.ones(B_feat.shape[0], batch_size, 1, device=device)
        global_gate = torch.ones(1, batch_size, 1, device=device)

        # 1. Pointwise: B finds its 'twins' in A to see which specific B-points are valid
        if self.use_pointwise:
            A_ret_for_B, _ = self.alignment_attn(
                query=B_feat, key=A_feat, value=A_feat, key_padding_mask=mask_A
            )
            pointwise_gate = self.trust_proj(torch.cat([B_feat, A_ret_for_B], dim=-1))

        # 2. Global: A checks if B as a whole matches its expected structure
        if self.use_global:
            B_ret_for_A, _ = self.alignment_attn(
                query=A_feat, key=B_feat, value=B_feat, key_padding_mask=mask_B
            )
            # Raw point-by-point trust from A's perspective: Shape [T_A, Batch, 1]
            raw_trust_A = self.trust_proj(torch.cat([A_feat, B_ret_for_A], dim=-1))

            # --- Pooling Strategies ---
            if self.pool_mode == "distributional":
                # FIXME: padding???
                global_gate = self.distributional_gate(raw_trust_A)

            elif self.pool_mode == "mean":
                # --- FIX: MASKED MEAN ---
                if mask_A is not None:
                    # mask_A is [Batch, T_A], True means padding. We want active tokens (False).
                    # Convert to [T_A, Batch, 1] to match raw_trust_A
                    active_A = (~mask_A).transpose(0, 1).unsqueeze(-1).float()

                    # Sum only the active tokens
                    sum_trust = (raw_trust_A * active_A).sum(dim=0, keepdim=True)
                    # Count only the active tokens (clamp to avoid division by zero)
                    count_active = active_A.sum(dim=0, keepdim=True).clamp(min=1.0)

                    global_gate = sum_trust / count_active
                else:
                    global_gate = raw_trust_A.mean(dim=0, keepdim=True)

            elif self.pool_mode == "gumbel":

                if mask_A is not None:
                    active_A = (~mask_A).transpose(0, 1).unsqueeze(-1).float()
                    sum_trust = (raw_trust_A * active_A).sum(dim=0, keepdim=True)
                    count_active = active_A.sum(dim=0, keepdim=True).clamp(min=1.0)
                    # pooled_logits shape: [1, Batch, 2]
                    pooled_logits = sum_trust / count_active
                else:
                    pooled_logits = raw_trust_A.mean(dim=0, keepdim=True)

                # --- THE GUMBEL MAGIC ---
                # hard=True forces the output to be strictly one-hot (e.g., [0, 1] or [1, 0])
                # during the forward pass, but uses soft gradients backward.
                if self.training:
                    safe_logits = torch.clamp(pooled_logits, min=-3.0, max=3.0)
                    gate_onehot = F.gumbel_softmax(safe_logits, tau=self.temp, hard=True, dim=-1)
                else:
                    # During inference, just take the greedy argmax for pure binary routing
                    gate_onehot = F.one_hot(pooled_logits.argmax(dim=-1), num_classes=2).float()

                # Extract the "Keep" channel (Index 1) and restore the shape to [1, Batch, 1]
                global_gate = gate_onehot[..., 1].unsqueeze(-1)

        detached_global = global_gate.detach()

        # 2. global_gate is already shape [1, Batch, 1] (sequence dim is pooled),
        # so we can just take the simple mean across the batch.
        ForwardMetaContext.set(global_gate=detached_global.mean())

        if self.use_pointwise:
            detached_pointwise = pointwise_gate.detach()

            if mask_B is not None:
                # Mask out the padded tokens in B before averaging
                active_B = (~mask_B).transpose(0, 1).unsqueeze(-1).float()
                true_pw_mean = (detached_pointwise * active_B).sum() / active_B.sum().clamp(min=1.0)

                # Do NOT use .item() so it passes the isinstance(v, torch.Tensor) filter in your training loop!
                ForwardMetaContext.set(pointwise_gate=true_pw_mean)
            else:
                ForwardMetaContext.set(pointwise_gate=detached_pointwise.mean())

        return pointwise_gate * global_gate


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

    def __init__(self, dmodel=128, use_gate=False, use_valve=False, hp_mode="concat", compute_own_self_ABC_attn=True,
                 use_struct_gate=True, use_spatial_interpolation=False, gate_params={"use_pointwise": True, "use_global": True}):
        super().__init__()
        self.hp_mode = hp_mode
        self.dmodel = dmodel

        in_dim = dmodel * 2 if hp_mode == "concat" else dmodel

        self.linear_AB1 = MLP(in_dim, dmodel, in_dim)
        self.compute_own_self_ABC_attn = compute_own_self_ABC_attn
        if self.compute_own_self_ABC_attn:
            self.self_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4)
            self.internal_norm_mha = nn.LayerNorm(dmodel)
            self.internal_norm_mlp = nn.LayerNorm(dmodel)
            self.internal_mlp = MLP(dmodel, dmodel, dmodel)


        self.use_struct_gate = use_struct_gate
        if use_struct_gate:
            # Pass your ablation flags here (e.g., use_pointwise=False)
            self.struct_gate_module = StructuralGatingModule(dmodel, **gate_params)
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

        self.norm_q = nn.LayerNorm(in_dim)
        self.norm_k = nn.LayerNorm(in_dim)
        self.norm_v = nn.LayerNorm(dmodel)  # Value is dmodel in your setup

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
        if self.compute_own_self_ABC_attn:
            # 2.1. Deal with padding in self attention
            # since A and B have different sequence lengths, we need to build an attention mask
            # to prevent attending to the padded points.
            mask_A_train = mask_A[:, :sep] if mask_A is not None else None
            mask_B_train = mask_B[:, :sep] if mask_B is not None else None

            if mask_A_train is not None and mask_B_train is not None:
                # C doesn't have a mask (it uses all its points), so it's all False
                mask_C_train = mask_A_train.clone()

                # Concat along batch dimension (dim=0) because ABC has 3*Batch size
                self_attn_mask = torch.cat([mask_A_train, mask_B_train, mask_C_train], dim=0)
            else:
                mask_C_train = None
                self_attn_mask = None

            # 2.2  A,B,C Shared self attention to extract relational features within A, B, C respectively.
            # ABC, _ = self.self_attention(ABC, ABC, ABC, key_padding_mask=self_attn_mask)
            ABC_norm = self.internal_norm_mha(ABC)
            ABC_attn, _ = self.self_attention(ABC_norm, ABC_norm, ABC_norm, key_padding_mask=self_attn_mask)
            ABC = ABC + ABC_attn  # Residual!

            # 2. Translator MLP (with Pre-Norm)
            ABC_norm2 = self.internal_norm_mlp(ABC)
            ABC_mlp = self.internal_mlp(ABC_norm2)
            ABC = ABC + ABC_mlp  # Residual!



        # Extract the feature descriptors for each task after self-attention
        a_dim, b_dim = A_train.shape[1], B_train.shape[1]
        A_feat, B_feat, C_feat = ABC[:, :a_dim], ABC[:, a_dim:a_dim + b_dim], ABC[:, a_dim + b_dim:]

        A_feat = self.norm_v(A_feat)
        B_feat = self.norm_v(B_feat)

        if self.use_struct_gate:
            m_A = mask_A[:, :sep] if mask_A is not None else None
            m_B = mask_B[:, :sep] if mask_B is not None else None

            # Get the unified gate
            gate = self.struct_gate_module(A_feat, B_feat, mask_A=m_A, mask_B=m_B)

            # Apply to Source
            B_feat = B_feat * gate

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

        query = self.norm_q(query)
        key = self.norm_k(key)
        # value = self.norm_v(value)

        # Build the cross-attention mask
        # C is looking at A and B, so the keys sequence length is 2*T_max.
        # Mask shape needs to be [Batch, 2*T_max]
        # Build the cross-attention mask (Shape: [Batch, 2*sep])
        mask_A_train = mask_A[:, :sep] if mask_A is not None else None
        mask_B_train = mask_B[:, :sep] if mask_B is not None else None
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
