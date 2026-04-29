import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class ManifoldCrossAttnLayer(nn.Module):
    def __init__(
            self,
            d_model=128,
            nhead=4,
            proj_depth=2,
            dim_feedforward=128,
            use_stacked_self_attn=True,
            update_C_train=True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_stacked_self_attn = use_stacked_self_attn
        self.update_C_train = update_C_train  # Toggle to benchmark updating A_train

        # 1. Stacked Self-Attention (Requires FFN to discover features)
        if self.use_stacked_self_attn:
            self.stacked_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=False
            )
            self.stacked_norm1 = nn.LayerNorm(d_model)
            self.stacked_ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, d_model)
            )
            self.stacked_norm2 = nn.LayerNorm(d_model)

        # 2. Invariant Projections (MLPs)
        self.W_Q2 = self._build_projection(d_model, dim_feedforward, proj_depth)
        self.W_K2 = self._build_projection(d_model, dim_feedforward, proj_depth)
        # self.W_V2 is explicitly omitted to preserve the marginal semantic basis

        # 3. Main Cross-Attention Block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=False
        )
        self.norm_final = nn.LayerNorm(d_model)

        # Optional FFN for the end of the block
        self.out_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        self.gamma = 1. # nn.Parameter(torch.zeros(1))

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
    def forward(
            self,
            A, B, C,
            hp_A, hp_B, hp_C,
            sep,
            raw_hp_A, raw_hp_B, raw_hp_C,
            pad_mask_A, pad_mask_B,
    ):

        seq_len = A.shape[0]
        batch_size = A.shape[1] // 2 if self.training else A.shape[1]

        # =====================================================================
        # UNIVERSAL MASK SANITIZATION
        # If a mask is None, we create an all-False tensor (meaning "no padding").
        # This completely eliminates the need for "is not None" checks later and
        # safely handles any future combination of padded A's and padded B's.
        # =====================================================================
        if pad_mask_A is None:
            pad_mask_A = torch.zeros((A.shape[1], seq_len), dtype=torch.bool, device=A.device)
        if pad_mask_B is None:
            pad_mask_B = torch.zeros((B.shape[1], seq_len), dtype=torch.bool, device=B.device)

        # --- OPTIONAL STACKED SELF-ATTENTION ---
        if self.use_stacked_self_attn:
            struct_mask = self._get_structural_mask(seq_len, sep, A.device)

            attn_A, _ = self.stacked_attn(A, A, A, attn_mask=struct_mask, key_padding_mask=pad_mask_A)
            A_res = self.stacked_norm1(A + attn_A)
            attn_A = self.stacked_norm2(A_res + self.stacked_ffn(A_res))

            attn_B, _ = self.stacked_attn(B, B, B, attn_mask=struct_mask, key_padding_mask=pad_mask_B)
            B_res = self.stacked_norm1(B + attn_B)
            attn_B = self.stacked_norm2(B_res + self.stacked_ffn(B_res))
        else:
            attn_A = A
            attn_B = B

        # --- STEP 1: Auxiliary Manifold Alignment Loss (Training Only) ---
        if self.training:
            A_train_real = attn_A[:sep, :batch_size, :]
            B_in_A = attn_B[:sep, batch_size:, :]
            A_in_B = attn_A[:sep, batch_size:, :]
            B_train_real = attn_B[:sep, :batch_size, :]

            Q_context = torch.cat([A_train_real, B_in_A], dim=0)  # Shape: (2*sep, batch, dim)
            K_context = torch.cat([A_in_B, B_train_real], dim=0)

            Q_aux = self.W_Q2(Q_context)
            K_aux = self.W_K2(K_context)

            Seq_sz = Q_aux.shape[0]  # 2 * sep
            perm_Q = torch.randperm(Seq_sz, device=A.device)
            perm_K = torch.randperm(Seq_sz, device=A.device)

            Q_p_b = Q_aux[perm_Q].transpose(0, 1)  # (batch, seq, dim)
            K_p_b = K_aux[perm_K].transpose(0, 1)

            scores = torch.bmm(Q_p_b, K_p_b.transpose(1, 2)) / (self.d_model ** 0.5)

            # --- The PI Shortcut for Hard CE Targets ---
            # Index i in Q maps perfectly to index i in K BEFORE permutation.
            target_indices = torch.argsort(perm_K)[perm_Q]
            target_indices = target_indices.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_Q)

            # Masking Padding
            if pad_mask_A is not None and pad_mask_B is not None:
                mask_A_real = pad_mask_A[:batch_size, :sep]
                mask_A_in_B = pad_mask_A[batch_size:, :sep]

                # B_real and B_in_A both derive from B, so they use pad_mask_B
                mask_B_real = pad_mask_B[:batch_size, :sep]
                mask_B_in_A = pad_mask_B[batch_size:, :sep]

                # Now we concatenate them exactly matching how Q_context and K_context were built:
                # Q_context = [A_train_real, B_in_A]
                mask_Q = torch.cat([mask_A_real, mask_B_in_A], dim=1)[:, perm_Q]

                # K_context = [A_in_B, B_train_real]
                mask_K = torch.cat([mask_A_in_B, mask_B_real], dim=1)[:, perm_K]

                # Block padded keys
                scores = scores.masked_fill(mask_K.unsqueeze(1), float('-inf'))
                # Block padded queries from contributing to loss
                target_indices = target_indices.masked_fill(mask_Q, -100)

            # Flatten for CE
            # Use .reshape() instead of .view() to handle the non-contiguous memory from .expand()
            flat_scores = scores.reshape(-1, scores.size(-1))
            flat_targets = target_indices.reshape(-1)

            ce_aux_loss = F.cross_entropy(flat_scores, flat_targets, ignore_index=-100, reduction='mean')

            ForwardMetaContext.set('ce_aux_loss', ce_aux_loss)

            # =====================================================================
            # TELEMETRY BLOCK 1: INVARIANT ALIGNMENT HEALTH
            # =====================================================================
            with torch.no_grad():
                # 1. Alignment Accuracy (Is the argmax actually hitting the target?)
                valid_mask = flat_targets != -100
                if valid_mask.sum() > 0:
                    preds = torch.argmax(flat_scores[valid_mask], dim=-1)
                    align_acc = (preds == flat_targets[valid_mask]).float().mean()
                    """
                    adapter_align_acc (Top-1 Alignment Accuracy): Cross-Entropy loss can sometimes go down just because
                    the model makes the wrong classes slightly less probable, without actually flipping the argmax. 
                    This metric strictly tracks whether the diffeomorphism mapping is succeeding. You want to see this
                    climb from near 0% to a healthy plateau.
                    """
                    ForwardMetaContext.set('adapter_align_acc', align_acc.item())


                    # Get top 3 predictions
                    """
                    The reasoning is, that while the model will certaintly fail on a continuous grid with resampled coordinates
                    to identify the exact match, smooth interpolation will still make it possible, that the correct one is in the 
                    top3, despite n_A + n_B tokens
                    """
                    _, top3_preds = torch.topk(flat_scores[valid_mask], k=3, dim=-1)
                    targets_expanded = flat_targets[valid_mask].unsqueeze(-1)

                    # Check if the true target is IN the top 3
                    top3_acc = (top3_preds == targets_expanded).any(dim=-1).float().mean()
                    ForwardMetaContext.set('adapter_align_top3_acc', top3_acc.item())

                # 2. Alignment Entropy (How blurry is the address book lookup?)
                probs = F.softmax(scores, dim=-1)
                safe_probs = probs + 1e-9  # Prevent log(0)
                entropy = -(probs * torch.log(safe_probs)).sum(dim=-1)  # (Batch, Seq_Q)

                if 'mask_Q' in locals():
                    entropy = entropy.masked_fill(mask_Q, 0.0)
                    mean_entropy = entropy.sum() / ((~mask_Q).sum() + 1e-9)
                else:
                    mean_entropy = entropy.mean()

                """
                adapter_align_entropy (Invariance Sharpness): As we discussed with the Bayesian Model Averaging (BMA) 
                concept, entropy represents "Warp Energy." Tracking this during training confirms whether the model is 
                learning a sharp, confident invariant space, or if it is just loosely smearing probabilities across the 
                sequence.
                """
                ForwardMetaContext.set('adapter_align_entropy', mean_entropy.item())
            # =====================================================================

            # --- STEP 2: Main Cross-Attention for Stream C (Latent Interpolation) ---

            # 1. The Keys (Routing Space): Built from transformed contextual embeddings
            A_train_key = attn_A[:sep, :batch_size, :]
            B_train_key = attn_B[:sep, :batch_size, :]
            key_bank = torch.cat([A_train_key, B_train_key], dim=0)

            # 2. The Values (Semantic Payload): Built from RAW pristine marginal inputs
            A_train_val = A[:sep, :batch_size, :]
            B_train_val = B[:sep, :batch_size, :]
            val_bank = torch.cat([A_train_val, B_train_val], dim=0)

            # 3. Concatenate Padding Masks for the Memory Bank
            if pad_mask_A is not None and pad_mask_B is not None:
                # pad_mask expects (Batch, Seq). We concatenated along Seq, so we concat masks along dim=1
                pad_mask_mem = torch.cat([
                    pad_mask_A[:batch_size, :sep],
                    pad_mask_B[:batch_size, :sep]
                ], dim=1)
            else:
                pad_mask_mem = None

            # 4. Project Keys into Invariant Space. Values remain pure marginals.
            K_final = self.W_K2(key_bank)
            V_final = val_bank

            # 5. Determine the Query scope based on update_C_train flag
            query_start_idx = 0 if self.update_C_train else sep

            # C is the workbench. We project it to query the invariant space.
            Q_input = C[query_start_idx:, :batch_size, :]
            Q_final = self.W_Q2(Q_input)

            # 6. Execute Cross Attention
            # PyTorch MultiheadAttention expects key_padding_mask of shape (Batch, Seq_K)
            cross_out, attn_weights = self.cross_attn(
                query=Q_final,
                key=K_final,
                value=V_final,
                key_padding_mask=pad_mask_mem
            )

            # =====================================================================
            # TELEMETRY BLOCK 2: PAYLOAD & GATING METRICS
            # =====================================================================
            if self.training:
                with torch.no_grad():
                    # # 3. Gamma Magnitude (Is the backend accepting the payload?)
                    # """
                    # adapter_gamma_val (The Valve): Because gamma is initialized to $0$, it acts as a valve protecting
                    # the frozen backend. If gamma stays at $0.0$, the gradients are telling you that the interpolated
                    # values ($V$) are too destructive, and the backend refuses to use them. If it steadily grows
                    # (e.g., to $0.1$ or higher), it proves the backend recognizes the semantic integrity of your payload.
                    # """
                    # ForwardMetaContext.set('adapter_gamma_val', self.gamma.item())

                    # 4. Main Interpolation Entropy (How confident is the final merge?)
                    # attn_weights shape: (Batch, Seq_Q, Seq_K)
                    main_probs = attn_weights + 1e-9  # Prevent log(0)
                    main_entropy = -(main_probs * torch.log(main_probs)).sum(dim=-1).mean()
                    """
                    adapter_main_entropy: This tells you what $x_{test}$ is doing. If this entropy is very low, it means 
                    $x_{test}$ is successfully doing Nearest-Neighbor lookups in the Address Book. If it is high, it 
                    means $x_{test}$ is interpolating its payload from a wide neighborhood of $B$ tokens.
                    """
                    ForwardMetaContext.set('adapter_main_entropy', main_entropy.item())
            # =====================================================================

            # 7. Zero-Residual Injection into C
            C_slice = C[query_start_idx:, :batch_size, :]
            C_updated_slice = self.norm_final(C_slice + self.gamma * cross_out)

            # 8. Reconstruct full C stream if we only updated the test tokens
            if not self.update_C_train:
                C_updated = torch.cat([C[:sep, :batch_size, :], C_updated_slice], dim=0)
            else:
                C_updated = C_updated_slice

            # 9. Optional FFN skip connection for final polish
            C_updated = self.norm_ffn(C_updated + self.out_ffn(C_updated))

            return A[:, :batch_size, :], B[:, :batch_size, :], C_updated
