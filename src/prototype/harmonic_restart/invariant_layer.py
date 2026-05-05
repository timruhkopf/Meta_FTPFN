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
            update_C_train=False,
            test_target_k=3,
            dropout=0.1
    ):
        """
        Latent Manifold Alignment Adapter for zero-shot task transfer in PFNs.

        This layer decouples geometric routing from semantic payload delivery. It uses
        parallel self-attention streams to deduce local manifold geometry, projects these
        features into a warp-invariant quotient space for cross-attention routing, and
        retrieves pristine marginal values to denoise a sparse target task using a
        dense reference task.

        Attributes:
            update_C_train (bool): If True, both train and test tokens of Stream C are
                updated. If False, only test tokens are updated, and the original
                C_train sequence is prepended to the output to strictly prevent
                covariate shift in the anchor distribution.
            gamma (nn.Parameter): Zero-initialized learnable scalar gating the residual
                cross-attention injection.
            """
        super().__init__()
        self.d_model = d_model
        self.use_stacked_self_attn = use_stacked_self_attn
        self.update_C_train = update_C_train  # Toggle to benchmark updating A_train

        # 1. Stacked Self-Attention (Requires FFN to discover features)
        if self.use_stacked_self_attn:
            self.stacked_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
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
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )
        self.norm_final = nn.LayerNorm(d_model)

        # Optional FFN for the end of the block
        self.out_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        self.gamma = 1.  # nn.Parameter(torch.zeros(1))

        self.manifold_alignment_criterion = ManifoldAlignmentCriterion(test_target_k)

        self.pre_norm = True  # Standard PFN usually uses pre-norm

        # === PFN-Style Residual Block Components ===
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()  # or F.relu, depending on your architecture
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Zero-initialize the final projection of the adapter block
        # so it begins as a perfect Identity mapping.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def _build_projection(self, in_dim, hidden_dim, depth):
        if depth == 1:
            return nn.Sequential(
                nn.Linear(in_dim, self.d_model),
                nn.LayerNorm(self.d_model)
            )
        layers = []
        curr_dim = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, self.d_model))
        layers.append(nn.LayerNorm(self.d_model))
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
        """
        Executes the manifold alignment forward pass and extracts auxiliary routing
        data for prior-training supervision.

        Args:
            A (Tensor): Marginal latent embeddings for Domain A (Anchor).
                Shape: `[Seq_A, Batch, d_model]`. Contains [A_train, x_test].
            B (Tensor): Marginal latent embeddings for Domain B (Memory).
                Shape: `[Seq_B, Batch, d_model]`. Contains [B_train].
                During training, Batch dimension contains shadow batches [B_real, B_in_A].
            C (Tensor): Marginal latent embeddings for the Workbench stream.
                Shape: `[Seq_A, Batch, d_model]`. Initialized identical to A.
            hp_A (Tensor): Processed hyper-parameters/coordinates for Domain A.
            hp_B (Tensor): Processed hyper-parameters/coordinates for Domain B.
            hp_C (Tensor): Processed hyper-parameters/coordinates for Domain C.
            sep (int): The sequence index separating train tokens from test tokens.
            raw_hp_A (Tensor): Unprocessed, physical coordinates for Domain A.
                Shape: `[Seq_A, Batch, d_coord]`. Used for exact Euclidean distance tracking.
            raw_hp_B (Tensor): Unprocessed, physical coordinates for Domain B.
            raw_hp_C (Tensor): Unprocessed, physical coordinates for Domain C.
            pad_mask_A (Tensor, optional): Boolean padding mask for Domain A.
                Shape: `[Batch, Seq_A]`. True indicates padded elements.
            pad_mask_B (Tensor, optional): Boolean padding mask for Domain B.
                Shape: `[Batch, Seq_B]`. True indicates padded elements.

        Returns:
            tuple:
                - C_updated (Tensor): The denoised Workbench sequence ready for the backend.
                  Shape: `[Seq_A, Batch, d_model]`.
                - aux_data (dict): A dictionary of routing matrices, permutations, and
                  coordinate states required by `ManifoldAlignmentCriterion` to compute
                  the geometric losses. Empty during inference.
        """
        seq_len = A.shape[0]
        batch_size = A.shape[1] // 2 if self.training else A.shape[1]

        # =====================================================================
        # UNIVERSAL MASK SANITIZATION
        # =====================================================================
        if pad_mask_A is None:
            pad_mask_A = torch.zeros((A.shape[1], seq_len), dtype=torch.bool, device=A.device)
        if pad_mask_B is None:
            pad_mask_B = torch.zeros((B.shape[1], seq_len), dtype=torch.bool, device=B.device)

        # =====================================================================
        # 1. DECOUPLED STACKED SELF-ATTENTION
        # =====================================================================
        if self.use_stacked_self_attn:
            # Domain A Pass: Uses Structural Mask to keep x_test conditionally independent
            struct_mask = self._get_structural_mask(seq_len, sep, A.device)
            attn_A, _ = self.stacked_attn(A, A, A, attn_mask=struct_mask, key_padding_mask=pad_mask_A)
            A_res = self.stacked_norm1(A + attn_A)
            attn_A = self.stacked_norm2(A_res + self.stacked_ffn(A_res))

            # Domain B Pass: Entirely separate
            # CONSIDER: we might want to also have x_Btest looking for its k-nearest in A_train
            #  to use the symmetry of the problem, but notice, that if n_A << n_B, this won't be
            #  a dense signal, because many points in B will have the same mapping!
            attn_B, _ = self.stacked_attn(B, B, B, key_padding_mask=pad_mask_B)
            B_res = self.stacked_norm1(B + attn_B)
            attn_B = self.stacked_norm2(B_res + self.stacked_ffn(B_res))

        else:
            attn_A, attn_B = A, B

        # =====================================================================
        # 2. AUXILIARY DATA EXTRACTION (TRAINING ONLY)
        # =====================================================================
        aux_data = {}
        if self.training:
            # --- 1A: TRAIN TOKENS (Identity Dictionary Lookup) ---
            A_train_real = attn_A[:sep, :batch_size, :]
            B_in_A = attn_B[:sep, batch_size:, :]
            A_in_B = attn_A[:sep, batch_size:, :]
            B_train_real = attn_B[:sep, :batch_size, :]

            Q_train_context = torch.cat([A_train_real, B_in_A], dim=0)
            K_train_context = torch.cat([A_in_B, B_train_real], dim=0)

            Q_train_aux = self.W_Q2(Q_train_context)
            K_train_aux = self.W_K2(K_train_context)

            # Permute Train Queries
            seq_train = Q_train_aux.shape[0]
            perm_Q_train = torch.randperm(seq_train, device=A.device)
            Q_train_p = Q_train_aux[perm_Q_train].transpose(0, 1)  # [Batch, Seq, Dim]
            K_train_t = K_train_aux.transpose(0, 1)  # [Batch, Seq, Dim]

            # this might crash with amp (float16)
            # scores_train = torch.bmm(Q_train_p, K_train_t.transpose(1, 2)) / (self.d_model ** 0.5)
            scale_factor = self.d_model ** 0.25

            # 1. TRAIN BMM
            Q_train_safe = Q_train_p / scale_factor
            K_train_safe = K_train_t / scale_factor
            scores_train = torch.bmm(Q_train_safe, K_train_safe.transpose(1, 2))


            aux_data.update({
                'sep': sep,
                'scores_train': scores_train,
                'perm_Q_train': perm_Q_train,
                'seq_train': seq_train,
                'batch_size': batch_size
            })

            # --- 1B: TEST TOKENS (Continuous Geometric Lookup) ---
            if seq_len > sep:
                A_test_real = attn_A[sep:, :batch_size, :]

                # Queries (x_test) and Keys (B_train)
                Q_test_aux = self.W_Q2(A_test_real)
                K_test_aux = self.W_K2(B_train_real)

                # Permute Test Queries
                seq_test = Q_test_aux.shape[0]
                perm_Q_test = torch.randperm(seq_test, device=A.device)
                Q_test_p = Q_test_aux[perm_Q_test].transpose(0, 1)
                K_test_t = K_test_aux.transpose(0, 1)

                # scores_test = torch.bmm(Q_test_p, K_test_t.transpose(1, 2)) / (self.d_model ** 0.5)
                Q_test_safe = Q_test_p / scale_factor
                K_test_safe = K_test_t / scale_factor
                scores_test = torch.bmm(Q_test_safe, K_test_safe.transpose(1, 2))

                # Extract coordinate vectors
                # raw_hp_A at [sep:, batch_size:, :] is phi(x_test_A) evaluated in B's domain
                X_test_warped = raw_hp_A[sep:, batch_size:, :]
                X_B_train = raw_hp_B[:sep, :batch_size, :]

                # Apply identical permutation to the test coordinates
                X_test_warped_p = X_test_warped[perm_Q_test]

                aux_data.update({
                    'scores_test': scores_test,
                    'X_test_warped_p': X_test_warped_p,
                    'X_B_train': X_B_train,
                    # Pass padding masks if you want the criterion to handle them safely
                    'pad_mask_B_train': pad_mask_B[:batch_size, :sep]
                })

            self.manifold_alignment_criterion(aux_data)

        # =====================================================================
        # STEP 3: INVARIANT CROSS-ATTENTION & PFN BLOCK (UPDATING 'C')
        # =====================================================================


        # 1. Build the Memory Bank (Keys and Values)
        K_A = self.W_K2(attn_A[:sep, :batch_size, :])
        K_B = self.W_K2(attn_B[:sep, :batch_size, :])
        K_final = torch.cat([K_A, K_B], dim=0)

        V_final = torch.cat([A[:sep, :batch_size, :], B[:sep, :batch_size, :]], dim=0)
        pad_mask_mem = torch.cat([pad_mask_A[:batch_size, :sep], pad_mask_B[:batch_size, :sep]], dim=1)

        # 2. Refine Stream C Queries (using the structural mask from earlier)
        C_refined, _ = self.stacked_attn(C, C, C, attn_mask=struct_mask, key_padding_mask=pad_mask_A)

        # 3. Dynamic Query Selection (Test Only vs. All)
        if self.update_C_train:
            Q_final = self.W_Q2(C_refined[:, :batch_size, :])
            src = C[:, :batch_size, :]
        else:
            Q_final = self.W_Q2(C_refined[sep:, :batch_size, :])
            src = C[sep:, :batch_size, :]


        # 4. Execute Cross-Attention
        cross_out, cross_attn_weights = self.cross_attn(
            query=Q_final,
            key=K_final,
            value=V_final,
            key_padding_mask=pad_mask_mem
        )

        # Assuming attn_weights shape is [Batch, Seq_Q, Seq_K]
        # =====================================================================
        # 3. Safe Attention Mass Distribution (Ignoring Padding NaNs)
        # =====================================================================
        with torch.no_grad():
            # attn_weights shape: [Batch, Seq_Q, Seq_K]

            # 1. Sum across the Key dimension to get the total mass going to each Bank
            mass_A_per_query = cross_attn_weights[:, :, :sep].sum(dim=-1)  # [Batch, Seq_Q]
            mass_B_per_query = cross_attn_weights[:, :, sep:].sum(dim=-1)  # [Batch, Seq_Q]

            # 2. Get the valid queries matching the actual Seq_Q length
            if self.update_C_train:
                valid_query_mask = ~pad_mask_A[:batch_size, :]  # [Batch, Seq_Train + Seq_Test]
            else:
                valid_query_mask = ~pad_mask_A[:batch_size, sep:]  # [Batch, Seq_Test]

            # 3. Calculate the mean ONLY over the valid, unpadded queries
            safe_mass_A = mass_A_per_query[valid_query_mask].mean().item()
            safe_mass_B = mass_B_per_query[valid_query_mask].mean().item()

            ForwardMetaContext.set('Telemetry/cross_attn_mass_to_A', safe_mass_A)
            ForwardMetaContext.set('Telemetry/cross_attn_mass_to_B', safe_mass_B)

            # 4. Train-Only ratio (ONLY applicable if we actually processed train tokens)
            if self.update_C_train:
                valid_train_mask = valid_query_mask[:, :sep]
                train_mass_A = mass_A_per_query[:, :sep][valid_train_mask].mean().item()
                train_mass_B = mass_B_per_query[:, :sep][valid_train_mask].mean().item()

                transfer_ratio = train_mass_B / (train_mass_A + 1e-9)

                ForwardMetaContext.set('Telemetry/cross_attn_mass_A_train', train_mass_A)
                ForwardMetaContext.set('Telemetry/cross_attn_mass_B_train', train_mass_B)
                ForwardMetaContext.set('Telemetry/transfer_ratio_train', transfer_ratio)        # =====================================================================
        # 4. EXACT PFN RESIDUAL & FFN BLOCK
        # =====================================================================
        # Apply the zero-initialized adapter gate
        src2 = self.gamma * cross_out

        # First Residual
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        # FFN Block
        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))

        # Second Residual
        src = src + self.dropout2(src2)
        if not self.pre_norm:
            src = self.norm2(src)

        # =====================================================================
        # 5. SEQUENCE SPLICING
        # =====================================================================
        if self.update_C_train:
            C_updated = src
        else:
            # Prepend the mathematically untouched original C_train
            C_train_original = C[:sep, :batch_size, :]
            C_updated = torch.cat([C_train_original, src], dim=0)

        # =====================================================================
        # TELEMETRY: ACTIVE REJECTION DETECTION
        # =====================================================================
        if self.training:
            with torch.no_grad():
                # 1. Payload Magnitude Ratio
                # We calculate the L2 norm of the injected tensor (src2) vs the original (C)
                injected_signal = self.dropout2(src2)

                # Match the slice based on what we are actually updating
                if self.update_C_train:
                    base_slice = C[:, :batch_size, :]
                else:
                    base_slice = C[sep:, :batch_size, :]

                base_magnitude = torch.norm(base_slice, dim=-1).mean()
                injection_magnitude = torch.norm(injected_signal, dim=-1).mean()

                payload_ratio = (injection_magnitude / (base_magnitude + 1e-9)).item()
                ForwardMetaContext.set('Telemetry/adapter_payload_ratio', payload_ratio)

                # 2. Cosine Drift
                c_original = base_slice
                c_new = src
                cos_sim = F.cosine_similarity(c_original, c_new, dim=-1).mean().item()
                """
                These two metrics are the ultimate lie-detectors for that "handshake." 
                They measure the Magnitude (Volume) and the Angle (Direction) of your adapter's intervention.
                Here is the deep intuition on what they actually measure.
                
                * adapter_payload_ratio (The Volume Knob)
                The Math: $\frac{\| \text{Injected Signal} \|_2}{\| \text{Base Signal} \|_2}$
                The Intuition: How "loud" is the adapter compared to the original latent state?
                
                * adapter_cosine_drift (The Steering Wheel)
                The Math: The cosine of the angle between the original vector and the updated vector.
                The Intuition: Did the adapter fundamentally change the meaning of the latent state, or just its intensity?
                
                In high-dimensional latent spaces, Direction = Semantic Meaning, while Magnitude = Intensity/Confidence.
                Cosine $\approx 1.0$ (No Drift): The vectors are perfectly parallel.
                Cosine $\approx 0.0$ (Orthogonal Drift): The adapter is aggressively twisting the latent vector into an entirely new dimension. 
                """
                ForwardMetaContext.set('Telemetry/adapter_cosine_drift', cos_sim)
        # ridden ourselves of the "shadow-batch"
        return A[:, :batch_size, :], B[:, :batch_size, :], C_updated


class ManifoldAlignmentCriterion(nn.Module):
    """
    Computes the geometric alignment losses for the Manifold Adapter.

    This module unpacks the auxiliary data dictionary generated by the
    ManifoldCrossAttnLayer during prior training. It calculates a Hard-CrossEntropy
    loss for train tokens (discrete identity matching) and a Soft-CrossEntropy
    loss for test tokens (continuous Top-k geometric interpolation) based on the
    ground-truth coordinate warps.

    Args:
        top_k (int): The number of nearest neighbors to target in the continuous
            geometric loss for test tokens. Defaults to 3.
    """

    def __init__(self, top_k=3):
        super().__init__()
        self.k = top_k

    def _get_topk_soft_targets(self, query_coords, key_coords):
        """
        Calculates exact geometric soft-targets using Euclidean distance.

        Args:
            query_coords (Tensor): The physical coordinates of the queries.
                Shape: `[Seq_Q, Batch, d_coord]`.
            key_coords (Tensor): The physical coordinates of the keys.
                Shape: `[Seq_K, Batch, d_coord]`.

        Returns:
            Tensor: A target probability distribution of shape `[Batch, Seq_Q, Seq_K]`
            where the `k` closest keys are assigned a probability of `1/k`, and all
            others are 0.0.
        """
        # Transpose for cdist: [Batch, Seq, d_coord]
        q_c = query_coords.transpose(0, 1)
        k_c = key_coords.transpose(0, 1)

        # Pairwise L2 Distance
        dist_matrix = torch.cdist(q_c, k_c, p=2)  # [Batch, Seq_Q, Seq_K]

        # Extract Top-k Indices
        _, top_k_indices = torch.topk(-dist_matrix, k=self.k, dim=-1)  # [Batch, Seq_Q, k]

        # Create Soft Target Probabilities
        targets = torch.zeros_like(dist_matrix)
        targets.scatter_(-1, top_k_indices, 1.0 / self.k)

        return targets

    def forward(self, aux_data):
        """
        Calculates the auxiliary losses and logs entropy and accuracy metrics to
        the ForwardMetaContext.

        Args:
            aux_data (dict): The dictionary yielded by `ManifoldCrossAttnLayer.forward`.
                Must contain permuted query matrices, scores, and raw coordinate
                states for both train and test splits.

        Returns:
            Tensor: A scalar loss tensor combining the train and test auxiliary losses.
            Returns 0.0 if `aux_data` is empty (e.g., during inference).
        """
        if not aux_data:
            return 0.0


        total_loss = 0.0

        batch_size = aux_data['batch_size']

        # =========================================================
        # 1. TRAIN TOKENS: Hard Identity Loss
        # =========================================================
        if 'scores_train' in aux_data:
            scores_train = aux_data['scores_train']
            perm_Q_train = aux_data['perm_Q_train']
            seq_train = aux_data['seq_train']

            # Target is the identity matrix, realigned to match permuted queries
            targets_train = torch.arange(seq_train, device=scores_train.device).unsqueeze(0).expand(batch_size, -1)
            targets_train = targets_train[:, perm_Q_train]

            flat_scores_train = scores_train.reshape(-1, seq_train)
            flat_targets_train = targets_train.reshape(-1)

            loss_train = F.cross_entropy(flat_scores_train, flat_targets_train)
            total_loss = total_loss + loss_train

            # Telemetry
            ForwardMetaContext.set('Telemetry/ce_aux_loss_train', loss_train.item())

            probs_train = F.softmax(scores_train, dim=-1)
            entropy_train = -(probs_train * torch.log(probs_train + 1e-9)).sum(dim=-1).mean()
            ForwardMetaContext.set('Telemetry/adapter_align_entropy_train', entropy_train.item())

            # Telemetry: Accuracy
            with torch.no_grad():
                preds_train = torch.argmax(flat_scores_train, dim=-1)
                train_acc = (preds_train == flat_targets_train).float().mean()
                ForwardMetaContext.set('Telemetry/adapter_align_acc_train', train_acc.item())

                _, top3_preds_train = torch.topk(flat_scores_train, k=3, dim=-1)
                train_top3_acc = (top3_preds_train == flat_targets_train.unsqueeze(-1)).any(dim=-1).float().mean()
                ForwardMetaContext.set('Telemetry/adapter_align_top3_acc_train', train_top3_acc.item())

        # =========================================================
        # 2. TEST TOKENS: Top-k Continuous Geometric Loss
        # =========================================================
        if 'scores_test' in aux_data:
            scores_test = aux_data['scores_test']

            # Retrieve exact geometric soft targets
            targets_test = self._get_topk_soft_targets(
                aux_data['X_test_warped_p'],
                aux_data['X_B_train']
            )

            # Masking out padded training tokens in the target distribution
            pad_mask_B_train = aux_data.get('pad_mask_B_train', None)
            if pad_mask_B_train is not None:
                # pad_mask_B_train is [Batch, Seq_K]. Expand to [Batch, 1, Seq_K]
                mask_K = pad_mask_B_train.unsqueeze(1)
                scores_test = scores_test.masked_fill(mask_K, float('-inf'))
                targets_test = targets_test.masked_fill(mask_K, 0.0)

                # Re-normalize targets so they sum to 1.0 after masking invalid keys
                targets_test = targets_test / (targets_test.sum(dim=-1, keepdim=True) + 1e-9)

            flat_scores_test = scores_test.reshape(-1, scores_test.size(-1))
            flat_targets_test = targets_test.reshape(-1, targets_test.size(-1))

            loss_test = F.cross_entropy(flat_scores_test, flat_targets_test)
            total_loss = total_loss + loss_test

            # Telemetry
            ForwardMetaContext.set('Telemetry/ce_aux_loss_test', loss_test.item())

            probs_test = F.softmax(scores_test, dim=-1)
            entropy_test = -(probs_test * torch.log(probs_test + 1e-9)).sum(dim=-1).mean()
            ForwardMetaContext.set('Telemetry/adapter_align_entropy_test', entropy_test.item())

            # Telemetry: Accuracy against Soft Targets
            with torch.no_grad():
                # Extract the indices of the actual k-nearest neighbors
                _, true_target_indices = torch.topk(flat_targets_test, k=self.k, dim=-1)

                # Top-1: Does the model's top guess match ANY of the valid nearest neighbors?
                preds_test = torch.argmax(flat_scores_test, dim=-1, keepdim=True)
                test_acc = (preds_test == true_target_indices).any(dim=-1).float().mean()
                ForwardMetaContext.set('Telemetry/adapter_align_acc_test', test_acc.item())

                # Top-3: Does ANY of the model's top 3 guesses match ANY of the valid nearest neighbors?
                _, top3_preds_test = torch.topk(flat_scores_test, k=3, dim=-1)
                test_top3_acc = (top3_preds_test.unsqueeze(2) == true_target_indices.unsqueeze(1)).any(dim=2).any(
                    dim=1).float().mean()
                ForwardMetaContext.set('Telemetry/adapter_align_top3_acc_test', test_top3_acc.item())

        ForwardMetaContext.set('ce_aux_loss', total_loss)

        return total_loss
