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
                mask_B_in_A = pad_mask_A[batch_size:, :sep]  # A's pad mask applies to B_in_A
                mask_A_in_B = pad_mask_B[batch_size:, :sep]  # B's pad mask applies to A_in_B
                mask_B_real = pad_mask_B[:batch_size, :sep]

                mask_Q = torch.cat([mask_A_real, mask_B_in_A], dim=1)[:, perm_Q]
                mask_K = torch.cat([mask_A_in_B, mask_B_real], dim=1)[:, perm_K]

                # Block padded keys
                scores = scores.masked_fill(mask_K.unsqueeze(1), float('-inf'))
                # Block padded queries from contributing to loss
                target_indices = target_indices.masked_fill(mask_Q, -100)

            # Flatten for CE
            flat_scores = scores.view(-1, scores.size(-1))
            flat_targets = target_indices.view(-1)

            ce_aux_loss = F.cross_entropy(flat_scores, flat_targets, ignore_index=-100, reduction='mean')

            ForwardMetaContext.set('ce_aux_loss', ce_aux_loss)

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
                pad_mask_mem = torch.cat([pad_mask_A[:batch_size, :sep], pad_mask_B[:batch_size, :sep]], dim=1)
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
            cross_out, _ = self.cross_attn(
                query=Q_final,
                key=K_final,
                value=V_final,
                key_padding_mask=pad_mask_mem
            )

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
