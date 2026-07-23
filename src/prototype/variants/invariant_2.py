import math
import torch
from torch.nn import functional as F
from torch import nn

from ppfn.model.mymodel.meta_context import ForwardMetaContext
from prototype.harmonic_restart.invariant_layer import ManifoldAlignmentCriterion


class InvariantProjector(nn.Module):
    """Projects marginal features into a shared, invariant metric space."""

    def __init__(self, in_dim, hidden_dim, out_dim, depth, bias=False):
        super().__init__()
        layers = []

        if depth == 1:
            layers.extend([
                nn.Linear(in_dim, out_dim, bias=bias),
                # nn.LayerNorm(out_dim)
            ])
        else:
            curr_dim = in_dim
            for _ in range(depth - 1):
                layers.extend([
                    nn.Linear(curr_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                ])
                # curr_dim = hidden_dim

            # Final projection onto the metric space
            layers.extend([
                nn.Linear(curr_dim, out_dim, bias=bias),
                # nn.LayerNorm(out_dim)
            ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StackedFeatureExtractor(nn.Module):
    """Discovers local geometric features before cross-attention routing."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, pad_mask=None, struct_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=struct_mask, key_padding_mask=pad_mask)
        res = self.norm1(x + attn_out)
        out = self.norm2(res + self.ffn(res))
        return out


class PFNDecoderBlock(nn.Module):
    """Handles the injection of the retrieved memory back into the main workflow."""

    def __init__(self, d_model, dim_feedforward, dropout, post_op='resid', pre_norm=True):
        super().__init__()
        self.post_op = post_op
        self.pre_norm = pre_norm

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Zero-initialize the final projection
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.ffn[-2].bias)

    def forward(self, src, retrieved_signal, gamma):
        injected = gamma * retrieved_signal

        if self.post_op == 'copy_only':
            src = self.dropout1(injected)
            if self.pre_norm:
                src_ = self.norm2(src)
                src = src + self.ffn(src_)
            else:
                src = self.norm1(src)
                src = self.norm2(src + self.ffn(src))

        elif self.post_op == 'resid':
            if self.pre_norm:
                src = src + self.dropout1(injected)
                src_ = self.norm2(src)
                src = src + self.ffn(src_)
            else:
                src = src + self.dropout1(injected)
                src = self.norm1(src)
                src = src + self.ffn(src)
                src = self.norm2(src)

        return src


class ManifoldCrossAttnLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, proj_depth=2, dim_feedforward=128, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # self.log_tau = nn.Parameter(torch.ones(1) * math.log(1 / 0.07))
        self.gamma = 1.0

        # 1. Subclass Instantiation
        self.feature_extractor = StackedFeatureExtractor(d_model, nhead, dim_feedforward, dropout)

        self.W_Q2 = InvariantProjector(d_model, dim_feedforward, d_model, proj_depth)
        self.W_K2 = InvariantProjector(d_model, dim_feedforward, d_model, proj_depth)

        self.decoder = PFNDecoderBlock(d_model, dim_feedforward, dropout, post_op='resid')
        self.use_stacked_self_attn = True
        self.use_aux_loss = True
        self.update_C_train=True
        self.drop_prob = dropout
        self.manifold_alignment_criterion = kwargs.get('aux_loss', None)

    def _get_structural_mask(self, seq_len, sep, device):
        """
        Creates a block-causal mask with independent test evaluations.
        Rows = Queries, Cols = Keys. True means 'Do not attend'.
        """
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)

        # 1. Block Train from looking at Test (Top-Right quadrant)
        mask[:sep, sep:] = True

        # 2. Block Test from looking at other Test tokens (Bottom-Right quadrant)
        # (following PFN literature and keeping the test tokens conditionally independent!)
        if seq_len > sep:
            test_len = seq_len - sep
            # Create a block of all True (block everything)
            test_block = torch.ones((test_len, test_len), device=device, dtype=torch.bool)
            # Unblock the diagonal so test token `i` can look at test token `i`
            test_block.fill_diagonal_(False)

            # Apply this block to the bottom-right quadrant
            mask[sep:, sep:] = test_block

        return mask

    def aux_fwd(
            self,
            attn_A, attn_B,
            pad_mask_A, pad_mask_B,
            batch_size, sep, seq_len,
            X_A_in_B_test, X_B_train
    ):
        aux_data = {}
        device = attn_A.device

        # --- 1A: TRAIN TOKENS (Identity Dictionary Lookup) ---
        # STRICT DECOUPLING: W_Q2 is the Domain A expert, W_K2 is the Domain B expert.
        A_train_real = attn_A[:sep, :batch_size, :]
        B_train_real = attn_B[:sep, :batch_size, :]

        Q_train_aux = self.W_Q2(A_train_real)
        K_train_aux = self.W_K2(B_train_real)

        # Permute Train Queries to prevent absolute positional bias
        perm_Q_train = torch.randperm(sep, device=device)
        Q_train_p = Q_train_aux[perm_Q_train].transpose(0, 1)  # [Batch, Seq, Dim]
        K_train_t = K_train_aux.transpose(0, 1)  # [Batch, Seq, Dim]

        # STRICT L2 NORMALIZATION
        Q_train_norm = F.normalize(Q_train_p, p=2, dim=-1)
        K_train_norm = F.normalize(K_train_t, p=2, dim=-1)

        # tau = torch.exp(self.log_tau).clamp(max=100.0)
        scores_train = torch.bmm(Q_train_norm, K_train_norm.transpose(1, 2)) #* tau

        # Masking out padded keys
        pad_mask_B_train = pad_mask_B[:batch_size, :sep]
        pad_mask_K = pad_mask_B_train.unsqueeze(1)  # [Batch, 1, Seq_K]
        scores_train = scores_train.masked_fill(pad_mask_K, -1e4)

        # Permute the query pad mask so the loss masks out the right tokens
        pad_mask_A_train = pad_mask_A[:batch_size, :sep]
        query_pad_mask = pad_mask_A_train[:, perm_Q_train]

        aux_data.update({
            'scores_train': scores_train,
            'perm_Q_train': perm_Q_train,
            'query_pad_mask': query_pad_mask,
            'pad_mask_B_train': pad_mask_B_train,
            'sep': sep  # Pass this explicitly as seq_train is now just sep
        })

        # --- 1B: TEST TOKENS (Continuous Geometric Lookup) ---
        A_test_real = attn_A[sep:, :batch_size, :]

        Q_test_aux = self.W_Q2(A_test_real)
        K_test_aux = self.W_K2(B_train_real)  # Keys are still the pristine B_train

        n_test_A = A_test_real.shape[0]
        perm_Q_test = torch.randperm(n_test_A, device=device)
        Q_test_p = Q_test_aux[perm_Q_test].transpose(0, 1)
        K_test_t = K_test_aux.transpose(0, 1)

        Q_test_norm = F.normalize(Q_test_p, p=2, dim=-1)
        K_test_norm = F.normalize(K_test_t, p=2, dim=-1)

        scores_test = torch.bmm(Q_test_norm, K_test_norm.transpose(1, 2)) # * tau

        aux_data.update({
            'scores_test': scores_test,
            'X_A_in_B_test': X_A_in_B_test[perm_Q_test],
            'X_B_train': X_B_train,
        })

        if self.manifold_alignment_criterion is not None:
            self.manifold_alignment_criterion(batch_size=batch_size, **aux_data)

    def forward(
            self,
            A, B, C,
            sep,
            raw_hp_A, raw_hp_B, raw_hp_C,
            pad_mask_A, pad_mask_B,
    ):
        """
        Executes the manifold alignment forward pass, delegating feature extraction,
        invariant projection, and residual decoding to dedicated subclasses.
        """
        seq_len, b, _ = A.shape
        batch_size = A.shape[1] // 2 if self.training else A.shape[1]

        # =====================================================================
        # 1. DECOUPLED STACKED SELF-ATTENTION (Via Feature Extractor Subclass)
        # =====================================================================
        if self.use_stacked_self_attn:
            struct_mask = self._get_structural_mask(seq_len, sep, A.device)
            attn_A = self.feature_extractor(A, pad_mask_A, struct_mask)

            # B optionally shares the structural mask
            attn_B = self.feature_extractor(B, pad_mask_B if pad_mask_B.sum() > 0 else None, struct_mask)

            attn_C = self.feature_extractor(C, pad_mask_A, struct_mask)
        else:
            attn_A, attn_B, attn_C = A, B, C

        # =====================================================================
        # 2. AUXILIARY DATA EXTRACTION (TRAINING ONLY)
        # =====================================================================
        if self.training and self.use_aux_loss:
            # Unchanged call to your existing aux_fwd logic
            self.aux_fwd(
                attn_A, attn_B,
                pad_mask_A, pad_mask_B,
                batch_size, sep, seq_len,
                X_A_in_B_test=raw_hp_A[sep:, batch_size:, :],
                X_B_train=raw_hp_B[:sep, :batch_size, :]
            )

        # =====================================================================
        # 3. MEMORY BANK & INVARIANT CROSS-ATTENTION PROJECTIONS
        # =====================================================================
        pad_mask_mem = torch.cat([pad_mask_A[:batch_size, :sep], pad_mask_B[:batch_size, :sep]], dim=1)

        # Values are the pristine, untouched marginal embeddings
        V_final = torch.cat([A[:sep, :batch_size, :], B[:sep, :batch_size, :]], dim=0)

        # Keys (Anchor + Memory) projected into the invariant metric space
        K_A = self.W_K2(attn_A[:sep, :batch_size, :])
        K_B = self.W_K2(attn_B[:sep, :batch_size, :])
        K_final = torch.cat([K_A, K_B], dim=0)

        # Queries (Workbench)
        if self.update_C_train:
            Q_final = self.W_Q2(attn_C[:, :batch_size, :])
            src = C[:, :batch_size, :]
        else:
            Q_final = self.W_Q2(attn_C[sep:, :batch_size, :])
            src = C[sep:, :batch_size, :]

        # CRITICAL FIX: L2 Normalize the main stream to match the aux_fwd prior constraints!
        Q_norm = F.normalize(Q_final, p=2, dim=-1)
        K_norm = F.normalize(K_final, p=2, dim=-1)

        # Extract clamped Temperature
        # tau = torch.exp(self.log_tau).clamp(max=100.0)

        # =====================================================================
        # 4. TELEMETRY
        # =====================================================================
        if self.training:
            with torch.no_grad():
                wq_max = max(p.abs().max() for p in self.W_Q2.parameters())
                wk_max = max(p.abs().max() for p in self.W_K2.parameters())
                # ForwardMetaContext.set('Telemetry/attn_temperature', tau.item())
                ForwardMetaContext.set('Telemetry/act_max_Q_norm', Q_norm.abs().max().item())
                ForwardMetaContext.set('Telemetry/weight_max_W_Q2', wq_max.item())
                ForwardMetaContext.set('Telemetry/weight_max_W_K2', wk_max.item())
                ForwardMetaContext.set('Telemetry/act_max_Q_scaled', Q_final.abs().max().item())
                ForwardMetaContext.set('Telemetry/act_max_K', K_final.abs().max().item())

        # =====================================================================
        # 5. SCALED DOT PRODUCT ATTENTION
        # =====================================================================
        b_sz = Q_norm.size(1)
        seq_q = Q_norm.size(0)
        seq_k = K_norm.size(0)

        # SDPA automatically computes: (Q @ K.T) / sqrt(head_dim)
        # We want: (Q_norm @ K_norm.T) * tau
        # Therefore, we scale Q_norm by (tau * sqrt(head_dim)) before passing it in.
        scale_correction = (self.head_dim ** 0.5)  #*  tau
        Q_scaled = Q_norm * scale_correction

        # Reshape for multi-head attention
        Q = Q_scaled.transpose(0, 1).view(b_sz, seq_q, self.nhead, self.head_dim).transpose(1, 2)
        K = K_norm.transpose(0, 1).view(b_sz, seq_k, self.nhead, self.head_dim).transpose(1, 2)
        V = V_final.transpose(0, 1).view(b_sz, seq_k, self.nhead, self.head_dim).transpose(1, 2)

        # SDPA Mask: True for valid, False for padding
        valid_mask = (~pad_mask_mem).unsqueeze(1).unsqueeze(2)

        cross_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=valid_mask,
            dropout_p=self.drop_prob if self.training else 0.0
        )

        cross_out = cross_out.transpose(1, 2).contiguous().view(b_sz, seq_q, self.d_model).transpose(0, 1)

        # =====================================================================
        # 6. EXACT PFN RESIDUAL & FFN BLOCK (Via Decoder Subclass)
        # =====================================================================
        # Delegation handles 'copy_only' vs 'resid' and pre/post-norm logic
        src_decoded = self.decoder(src=src, retrieved_signal=cross_out, gamma=self.gamma)

        # =====================================================================
        # 7. TELEMETRY: ACTIVE REJECTION DETECTION
        # =====================================================================
        if self.training:
            with torch.no_grad():
                # Reconstruct the raw injected signal approximation for telemetry tracking
                injected_signal = self.gamma * cross_out

                if self.update_C_train:
                    valid_query_mask = ~pad_mask_A[:batch_size, :]
                    base_slice = C[:, :batch_size, :]
                else:
                    valid_query_mask = ~pad_mask_A[:batch_size, sep:]
                    base_slice = C[sep:, :batch_size, :]

                # base_slice is [Seq, Batch, Dim]. valid_query_mask is [Batch, Seq].
                valid_mask_seq_first = valid_query_mask.transpose(0, 1)

                if valid_mask_seq_first.any():
                    base_magnitude = torch.norm(base_slice, dim=-1)[valid_mask_seq_first].mean()
                    injection_magnitude = torch.norm(injected_signal, dim=-1)[valid_mask_seq_first].mean()

                    payload_ratio = (injection_magnitude / (base_magnitude + 1e-9)).item()
                    cos_sim = F.cosine_similarity(base_slice, src_decoded, dim=-1)[valid_mask_seq_first].mean().item()
                else:
                    payload_ratio = 0.0
                    cos_sim = 0.0

                ForwardMetaContext.set('Telemetry/adapter_payload_ratio', payload_ratio)
                ForwardMetaContext.set('Telemetry/adapter_cosine_drift', cos_sim)

        # =====================================================================
        # 8. SEQUENCE SPLICING
        # =====================================================================
        if self.update_C_train:
            C_updated = src_decoded
        else:
            # Prepend the mathematically untouched original C_train
            C_train_original = C[:sep, :batch_size, :]
            C_updated = torch.cat([C_train_original, src_decoded], dim=0)

        # Safety catch for gradient collapse
        if any([torch.any(torch.isnan(t)) for t in (A, B, C_updated)]):
            import pdb;
            pdb.set_trace()

        return A[:, :batch_size, :], B[:, :batch_size, :], C_updated


class ManifoldAlignmentCriterionV2(ManifoldAlignmentCriterion):
    def forward(
            self,
            scores_train,
            perm_Q_train,
            scores_test,
            X_A_in_B_test,
            X_B_train,
            query_pad_mask,
            pad_mask_B_train,
            sep,
            batch_size
    ):
        seq_train = sep  # Sequence length is now purely `sep`

        # =========================================================
        # 1. TRAIN TOKENS: Symmetric Hard Identity Loss (Doubly Stochastic)
        # =========================================================

        # --- A2B Loss (Row-wise) ---
        # The correct Key index for Query `i` is `perm_Q_train[i]`
        targets_A2B = perm_Q_train.unsqueeze(0).expand(batch_size, -1)

        flat_scores_A2B = scores_train.reshape(-1, seq_train)
        flat_targets_A2B = targets_A2B.reshape(-1)

        loss_A2B_unreduced = F.cross_entropy(flat_scores_A2B, flat_targets_A2B, reduction='none', label_smoothing=0.1)
        loss_A2B_unreduced = loss_A2B_unreduced.view(batch_size, seq_train)

        if query_pad_mask is not None:
            valid_queries = ~query_pad_mask
            loss_A2B = loss_A2B_unreduced[valid_queries].mean() if valid_queries.any() else torch.tensor(0.0,
                                                                                                         device=scores_train.device)
        else:
            loss_A2B = loss_A2B_unreduced.mean()
            valid_queries = torch.ones_like(loss_A2B_unreduced, dtype=torch.bool)

        # --- B2A Loss (Column-wise) ---
        # Transpose scores to apply Softmax across the A dimension
        scores_train_T = scores_train.transpose(1, 2)
        flat_scores_B2A = scores_train_T.reshape(-1, seq_train)

        # The correct Query index for Key `j` requires the inverse permutation
        perm_K_train = torch.argsort(perm_Q_train)
        targets_B2A = perm_K_train.unsqueeze(0).expand(batch_size, -1)
        flat_targets_B2A = targets_B2A.reshape(-1)

        loss_B2A_unreduced = F.cross_entropy(flat_scores_B2A, flat_targets_B2A, reduction='none', label_smoothing=0.1)
        loss_B2A_unreduced = loss_B2A_unreduced.view(batch_size, seq_train)

        if pad_mask_B_train is not None:
            valid_keys = ~pad_mask_B_train
            loss_B2A = loss_B2A_unreduced[valid_keys].mean() if valid_keys.any() else torch.tensor(0.0,
                                                                                                   device=scores_train.device)
        else:
            loss_B2A = loss_B2A_unreduced.mean()

        # Combine for Doubly Stochastic constraint
        loss_train = (loss_A2B + loss_B2A) / 2.0

        # Telemetry (Calculated purely on the forward A2B mapping)
        ForwardMetaContext.set('Telemetry/ce_aux_loss_train', loss_train.item())

        probs_train = F.softmax(scores_train, dim=-1)
        entropy_train = -(probs_train * torch.log(probs_train + 1e-9)).sum(dim=-1)
        ForwardMetaContext.set('Telemetry/adapter_align_entropy_train', entropy_train[valid_queries].mean().item())

        with torch.no_grad():
            preds_train = torch.argmax(scores_train, dim=-1)
            train_acc = (preds_train == targets_A2B)[valid_queries].float().mean()
            ForwardMetaContext.set('Telemetry/adapter_align_acc_train', train_acc.item())

            _, top3_preds_train = torch.topk(scores_train, k=3, dim=-1)
            train_top3_acc = (top3_preds_train == targets_A2B.unsqueeze(-1)).any(dim=-1)[valid_queries].float().mean()
            ForwardMetaContext.set('Telemetry/adapter_align_top3_acc_train', train_top3_acc.item())

        # =========================================================
        # 2. TEST TOKENS: Adaptive Gaussian Continuous Geometric Loss
        # =========================================================
        # (This section remains exactly the same as you previously defined it)
        targets_test = self._get_adaptive_gaussian_targets(
            X_A_in_B_test, X_B_train,
            key_pad_mask=pad_mask_B_train
        )

        if pad_mask_B_train is not None:
            mask_K = pad_mask_B_train.unsqueeze(1)
            scores_test = scores_test.masked_fill(mask_K, -1e4)
            targets_test = targets_test.masked_fill(mask_K, 0.0)
            targets_test = targets_test / (targets_test.sum(dim=-1, keepdim=True) + 1e-9)

        flat_scores_test = scores_test.reshape(-1, scores_test.size(-1))
        flat_targets_test = targets_test.reshape(-1, targets_test.size(-1))

        loss_test = F.cross_entropy(flat_scores_test, flat_targets_test)

        # Telemetry
        ForwardMetaContext.set('Telemetry/ce_aux_loss_test', loss_test.item())

        probs_test = F.softmax(scores_test, dim=-1)
        entropy_test = -(probs_test * torch.log(probs_test + 1e-9)).sum(dim=-1).mean()
        ForwardMetaContext.set('Telemetry/adapter_align_entropy_test', entropy_test.item())

        with torch.no_grad():
            _, true_target_indices = torch.topk(targets_test, k=self.k, dim=-1)

            preds_test = torch.argmax(scores_test, dim=-1, keepdim=True)
            test_acc = (preds_test == true_target_indices).any(dim=-1).float().mean()
            ForwardMetaContext.set('Telemetry/adapter_align_acc_test', test_acc.item())

            # Top 3 Accuracy (The Bug Fix)
            _, top3_preds_test = torch.topk(scores_test, k=3, dim=-1)  # [B, S, 3]

            # Align shapes to [B, S, 3, 1] == [B, S, 1, k] -> Result: [B, S, 3, k]
            # This ensures we strictly compare predictions to targets for the SAME token
            match_matrix = top3_preds_test.unsqueeze(-1) == true_target_indices.unsqueeze(-2)

            # Collapse the k dimension (Did a prediction match ANY valid target?)
            # Collapse the 3 dimension (Were ANY of the 3 predictions correct?)
            test_top3_acc = match_matrix.any(dim=-1).any(dim=-1).float().mean()
            ForwardMetaContext.set('Telemetry/adapter_align_top3_acc_test', test_top3_acc.item())
            
        penalty = ForwardMetaContext.get('loss_adapter_reg', default=0.0)
        total_loss = loss_train + loss_test + penalty

        ForwardMetaContext.set('ce_aux_loss', total_loss * self.loss_scale)

        return total_loss