import torch
import torch.nn as nn


class AlternatingTriStreamBlock(nn.Module):
    """
    Implements the Chronos-2 style zigzag routing:
    Temporal Attention (seq) -> Feature Attention (cross-stream) -> FFN
    """

    def __init__(self, d_model, nhead, dropout=0.1, cross_type='temporal_query'):
        super().__init__()
        # 1. Temporal (Sequence) Attention - Independent per stream
        self.temporal_attn_AB = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.temporal_attn_B = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_attn_C = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm_temp_AB = nn.LayerNorm(d_model)
        # self.norm_temp_B = nn.LayerNorm(d_model)
        self.norm_temp_C = nn.LayerNorm(d_model)

        # 2. Feature (Stream) Attention - A/B paired
        # We use self-attention for the A/B pair (seq_len=2)
        # ==========================================
        # 2. Temporal Cross-Attention (C queries B over Sequence T)
        # ==========================================
        self.temporal_cross_attn_C = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_cross_C = nn.LayerNorm(d_model)
        self.norm_cross_B = nn.LayerNorm(d_model)

        # ==========================================
        # 3. Feature Attention (Cross-Stream at time t)
        # ==========================================
        self.feature_attn_C = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm_feat_A = nn.LayerNorm(d_model)
        self.norm_feat_B = nn.LayerNorm(d_model)
        self.norm_feat_C = nn.LayerNorm(d_model)

        # 3. Feed Forward Networks
        self.ffn_AB = self._build_ffn(d_model, dropout)
        # self.ffn_B = self._build_ffn(d_model, dropout)
        self.ffn_C = self._build_ffn(d_model, dropout)

        self.norm_ffn_AB = nn.LayerNorm(d_model)
        # self.norm_ffn_B = nn.LayerNorm(d_model)
        self.norm_ffn_C = nn.LayerNorm(d_model)

        self.query_prep_C = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        self.cross_type = cross_type

    def _build_ffn(self, d_model, dropout):
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, A, B, C, pad_mask_A=None, pad_mask_B=None, causal_mask=None):
        T, Batch, D = A.shape

        # ==========================================
        # STEP 1: TEMPORAL ATTENTION (Along Sequence Dim T)
        # ==========================================
        # Using Pre-Norm architecture
        A_norm = self.norm_temp_AB(A)
        A_temp, _ = self.temporal_attn_AB(
            A_norm, A_norm, A_norm,
            key_padding_mask=pad_mask_A, attn_mask=causal_mask
        )
        A = A + A_temp

        B_norm = self.norm_temp_AB(B)
        B_temp, _ = self.temporal_attn_AB(
            B_norm, B_norm, B_norm,
            key_padding_mask=pad_mask_B, attn_mask=causal_mask
        )
        B = B + B_temp

        C_norm = self.norm_temp_C(C)
        # C follows A's padding mask as it's initialized with A
        C_temp, _ = self.temporal_attn_C(
            C_norm, C_norm, C_norm,
            key_padding_mask=pad_mask_A, attn_mask=causal_mask
        )
        C = C + C_temp

        # ==========================================
        # STEP 2: ASYMMETRIC FEATURE ATTENTION (Across Streams at time t)
        # ==========================================
        # A and B are strictly Keys/Values here. They do NOT act as queries.
        A_kv = self.norm_feat_A(A).view(T * Batch, 1, D).transpose(0, 1)  # (1, T*Batch, D)
        B_kv = self.norm_feat_B(B).view(T * Batch, 1, D).transpose(0, 1)  # (1, T*Batch, D)

        # Stack A and B to form a Key/Value sequence of length 2
        AB_kv = torch.cat([A_kv, B_kv], dim=0).detach()  # (2, T*Batch, D)

        # # C is the Query
        if self.cross_type == 'direct':
            C_q = self.norm_feat_C(C).view(T * Batch, 1, D).transpose(0, 1)  # (1, T*Batch, D)

            # Workbench C cross-attends to A and B simultaneously
            C_feat_out, _ = self.feature_attn_C(query=C_q, key=AB_kv, value=AB_kv)

        elif self.cross_type == 'prep':
            # C prepares a specialized query state before asking A and B
            C_temp_state = self.norm_feat_C(C)
            C_query_state = self.query_prep_C(C_temp_state)  # <--- The decoupling step

            C_q = C_query_state.view(T * Batch, 1, D).transpose(0, 1)

            C_feat_out, _ = self.feature_attn_C(query=C_q, key=AB_kv, value=AB_kv)

        elif self.cross_type == 'temporal_query':
            # ==========================================
            # STEP 1.5: TEMPORAL CROSS-ATTENTION (The Heavyweight Addition)
            # C uses the entire sequence of B to find shifted features
            # ==========================================
            C_cross_norm = self.norm_cross_C(C)
            B_cross_norm = self.norm_cross_B(B)

            # causal_mask safely applies here too, preventing Test tokens in C
            # from peeking at Test tokens in B, maintaining strict PFN rules.
            C_cross, _ = self.temporal_cross_attn_C(
                query=C_cross_norm,
                key=B_cross_norm.detach(),
                value=B_cross_norm.detach(),
                key_padding_mask=pad_mask_B,
                attn_mask=causal_mask
            )
            C = C + C_cross

            # ==========================================
            # STEP 2: ASYMMETRIC FEATURE ATTENTION (At current time step t)
            # ==========================================
            A_kv = self.norm_feat_A(A).view(T * Batch, 1, D).transpose(0, 1)
            B_kv = self.norm_feat_B(B).view(T * Batch, 1, D).transpose(0, 1)
            AB_kv = torch.cat([A_kv, B_kv], dim=0).detach()  # Shape: (2, T*Batch, D)

            C_q = self.norm_feat_C(C).view(T * Batch, 1, D).transpose(0, 1)

            # C resolves any final local alignments at time `t`
            C_feat_out, _ = self.feature_attn_C(query=C_q, key=AB_kv, value=AB_kv)

        # Residual update is ONLY applied to C.
        # A and B remain untouched by this step.
        C = C + C_feat_out.view(T, Batch, D)

        # ==========================================
        # STEP 3: FEED FORWARD
        # ==========================================
        A = A + self.ffn_AB(self.norm_ffn_AB(A))
        B = B + self.ffn_AB(self.norm_ffn_AB(B))
        C = C + self.ffn_C(self.norm_ffn_C(C))

        return A, B, C


class MultiStageTriHarmonicModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1, num_bars=250, use_freq_enc_x=False,
                 use_cross_attn=True):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = nn.Linear(1, d_model)  # Fourier?
        self.y_encoder = nn.Linear(1, d_model)
        self.use_cross_attn = use_cross_attn

        # Multi-Stage Alternating Pipeline
        self.layers = nn.ModuleList([
            AlternatingTriStreamBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, num_bars)

    # ... [Keep your append_cross_domain_features method exactly as is] ...

    def forward(self, batch):
        X_train_A, Y_train_A = batch['train']['X_A'], batch['train']['Y_A']
        X_train_B, Y_train_B = batch['train']['X_B'], batch['train']['Y_B']
        X_test_A = batch['test']['X_A']
        X_test_B = batch['test']['X_B']
        pad_mask_A = batch['train']['padding_mask_A']
        pad_mask_B = batch['train']['padding_mask_B']
        single_eval_pos = batch['train']['X_B'].shape[0]

        X_A = torch.cat([X_train_A, X_test_A], dim=0)
        X_B = torch.cat([X_train_B, X_test_B], dim=0)

        emb_X_A = self.x_encoder(X_A)
        emb_X_B = self.x_encoder(X_B)
        emb_Y_A = self.y_encoder(Y_train_A)
        emb_Y_B = self.y_encoder(Y_train_B)

        A = emb_X_A.clone()
        B = emb_X_B.clone()
        batch_size = A.shape[1]

        A[:single_eval_pos, :, :] += emb_Y_A
        B[:single_eval_pos, :, :] += emb_Y_B

        # if self.use_cross_attn and 'X_A_in_B' in batch['train'].keys():
        #     A, pad_mask_A, X_A = self.append_cross_domain_features(
        #         batch, 'A_in_B', A, pad_mask_A, X_A, single_eval_pos)
        #     B, pad_mask_B, X_B = self.append_cross_domain_features(
        #         batch, 'B_in_A', B, pad_mask_B, X_B, single_eval_pos)

        X_C = X_A.clone()
        C = A.clone()

        # Optional: Implement PFN evaluation masking here if you don't want
        # train items looking at test items, or test items looking at test items.
        seq_len = A.shape[0]
        structural_mask = self._get_structural_mask(seq_len, single_eval_pos, A.device)

        # Forward pass through alternating blocks
        for layer in self.layers:
            A, B, C = layer(A, B, C, pad_mask_A, pad_mask_B, causal_mask=structural_mask)

        # Truncate Shadow Batches before decoding
        # if self.use_cross_attn and A.shape[0] > (X_train_A.shape[0] + X_test_A.shape[0]):
        #      true_seq_len = X_train_A.shape[0] + X_test_A.shape[0]
        #      A = A[:true_seq_len, :, :]
        #      B = B[:true_seq_len, :, :]
        #      C = C[:true_seq_len, :, :]

        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)

        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        return logits_A, logits_B, logits_C

    def _get_structural_mask(self, seq_len, sep, device):
        """
        Creates a block-causal mask with independent test evaluations.
        Rows = Queries, Cols = Keys. True means 'Do not attend'.
        """
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)

        # 1. Block Train from looking at Test (Top-Right quadrant)
        mask[:sep, sep:] = True

        # 2. Block Test from looking at other Test tokens (Bottom-Right quadrant)
        if seq_len > sep:
            test_len = seq_len - sep
            test_block = torch.ones((test_len, test_len), device=device, dtype=torch.bool)
            test_block.fill_diagonal_(False)
            mask[sep:, sep:] = test_block

        return mask
