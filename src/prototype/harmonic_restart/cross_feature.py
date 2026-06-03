import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class AlternatingTriStreamBlock(nn.Module):
    """
    Implements the Chronos-2 style zigzag routing:
    Temporal Attention (seq) -> Feature Attention (cross-stream) -> FFN
    """

    def __init__(self, d_model, nhead, dropout=0.1, cross_type='temporal_query', use_gate=True):
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
        # self.temporal_cross_attn_C = CosineMultiheadAttention(d_model, nhead, dropout=dropout)
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

        self.use_gate = use_gate
        if self.use_gate:
            self.c_update_gate = nn.Sequential(
                nn.LayerNorm(d_model * 2), # idea was that the gate might be cause for logit expl.
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

    def _build_ffn(self, d_model, dropout):
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    @staticmethod
    def sliced_attention(query_seq, sep_idx, attn_module, pad_mask):
        sliced_pad_mask = pad_mask[:, :sep_idx] if pad_mask is not None else None
        # Train -> Train
        out_left, _ = attn_module(
            query_seq[:sep_idx], query_seq[:sep_idx], query_seq[:sep_idx],
            key_padding_mask=sliced_pad_mask
        )
        # Test -> Train (Test tokens strictly query Train tokens)
        out_right, _ = attn_module(
            query_seq[sep_idx:], query_seq[:sep_idx], query_seq[:sep_idx],
            key_padding_mask=sliced_pad_mask
        )
        return torch.cat([out_left, out_right], dim=0)

    def forward(self, A, B, C, sep, pad_mask_A=None, pad_mask_B=None, causal_mask=None):
        # FIXME: having moved the pfn marginal into this cross_layer, the trainer in the warm_up phase
        #  will be unable to learn anything, other than using the encoder and decoder linears!

        T, Batch, D = A.shape

        A_norm = self.norm_temp_AB(A)
        B_norm = self.norm_temp_AB(B)
        C_norm = self.norm_temp_C(C)

        # Using the following code avoids passing the causal mask, which in turn has internal changes for MHA
        # as consequence, causing more efficient and precise attn.
        # Process A, B, and C using the FlashAttention-safe slicing method
        A_temp = self.sliced_attention(A_norm, sep, self.temporal_attn_AB, pad_mask_A)
        A = A + A_temp

        B_temp = self.sliced_attention(B_norm, sep, self.temporal_attn_AB, pad_mask_B)
        B = B + B_temp

        C_temp = self.sliced_attention(C_norm, sep, self.temporal_attn_C, pad_mask_A)
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
            C = C + C_feat_out.view(T, Batch, D)

        elif self.cross_type == 'prep':
            # C prepares a specialized query state before asking A and B
            C_temp_state = self.norm_feat_C(C)
            C_query_state = self.query_prep_C(C_temp_state)  # <--- The decoupling step

            C_q = C_query_state.view(T * Batch, 1, D).transpose(0, 1)

            C_feat_out, _ = self.feature_attn_C(query=C_q, key=AB_kv, value=AB_kv)
            C = C + C_feat_out.view(T, Batch, D)

        elif self.cross_type == 'temporal_query':
            # ==========================================
            # STEP 1.5: TEMPORAL CROSS-ATTENTION (The Heavyweight Addition)
            # C uses the entire sequence of B to find shifted features
            # ==========================================
            C_cross_norm = self.norm_cross_C(C)
            B_cross_norm = self.norm_cross_B(B)

            # causal_mask safely applies here too, preventing Test tokens in C
            # from peeking at Test tokens in B, maintaining strict PFN rules.
            # if False:
            # FIXME: what if we also here didn't use the mask, but two fwds with train test split
            # TODO: when deactivating fp16, we need to cast q, k, v and the mask to float()!
            with torch.cuda.amp.autocast(enabled=False):
                C_cross, _ = self.temporal_cross_attn_C(
                    query=C_cross_norm.float(),
                    key=B_cross_norm.detach().float(),
                    value=B_cross_norm.detach().float(),
                    key_padding_mask=pad_mask_B,
                    attn_mask=causal_mask.float() if causal_mask is not None else None
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
            with torch.cuda.amp.autocast(enabled=False):
                C_feat_out, _ = self.feature_attn_C(query=C_q.float(), key=AB_kv.float(), value=AB_kv.float())

        # Residual update is ONLY applied to C.
        # A and B remain untouched by this step.
        C_feat_out = C_feat_out.view(T, Batch, D)

        if self.use_gate:
            # Calculate the gate using the current state (A) and the target features (B)
            gate = self.c_update_gate(torch.cat([C, C_feat_out], dim=-1))

            if self.training:
                ForwardMetaContext.set(**{"Telemetry/last_layer_gate_mean": gate.mean().item(), })

            # THE FIX: Smooth Interpolation instead of Residual Addition
            # If gate = 0, keep C's history. If gate = 1, fully overwrite with B's features.
            C = (1.0 - gate) * C + gate * C_feat_out
        else:
            C = C + C_feat_out

        # ==========================================
        # STEP 3: FEED FORWARD
        # ==========================================
        A = A + self.ffn_AB(self.norm_ffn_AB(A))
        B = B + self.ffn_AB(self.norm_ffn_AB(B))
        C = C + self.ffn_C(self.norm_ffn_C(C))

        return A, B, C


class MultiStageTriHarmonicModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1, num_bars=250, use_freq_enc_x=False,
                 use_cross_attn=True, C_decoder_type='pass_through'):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = nn.Linear(1, d_model)  # Fourier?
        self.y_encoder = nn.Linear(1, d_model)
        self.use_cross_attn = use_cross_attn

        # Multi-Stage Alternating Pipeline
        self.cross_layers = nn.ModuleList([
            AlternatingTriStreamBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, num_bars, bias=False)  # false to avoid softmax drift and divergence.
        # self.decoder_C = nn.Linear(d_model, num_bars, bias=False)  # false to avoid softmax drift and divergence.

        if C_decoder_type not in {'with_grad', 'pass_through'}:
            raise ValueError('C_decoder_type is invalid.')

        self.C_decoder_type = C_decoder_type  # 'with_grad' or 'pass_through'

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
        C = A.detach().clone()
        # C = emb_X_A.detach().clone()

        # Optional: Implement PFN evaluation masking here if you don't want
        # train items looking at test items, or test items looking at test items.
        seq_len = A.shape[0]
        # structural_mask = self._get_structural_mask(seq_len, single_eval_pos, A.device)

        # Forward pass through alternating blocks
        for layer in self.cross_layers:
            A, B, C = layer(A, B, C, single_eval_pos, pad_mask_A, pad_mask_B, causal_mask=None)
                            # causal_mask=structural_mask)

        # Truncate Shadow Batches before decoding
        # if self.use_cross_attn and A.shape[0] > (X_train_A.shape[0] + X_test_A.shape[0]):
        #      true_seq_len = X_train_A.shape[0] + X_test_A.shape[0]
        #      A = A[:true_seq_len, :, :]
        #      B = B[:true_seq_len, :, :]
        #      C = C[:true_seq_len, :, :]

        out_A = self.final_norm(A)
        out_B = self.final_norm(B)

        logits_A = self.decoder(out_A[single_eval_pos:, :, :].float())
        logits_B = self.decoder(out_B[single_eval_pos:, :, :].float())

        if self.C_decoder_type == 'with_grad':
            out_C = self.final_norm(C)
            logits_C = self.decoder(out_C[single_eval_pos:, :, :].float())
        elif self.C_decoder_type == 'pass_through':
            out_C = F.layer_norm(
                C,
                self.final_norm.normalized_shape,
                weight=self.final_norm.weight.detach(),
                bias=self.final_norm.bias.detach() if self.final_norm.bias is not None else None,
                eps=self.final_norm.eps
            )

            c_features = out_C[single_eval_pos:, :, :].float()

            # Pass through F.linear using detached versions of the decoder weights
            # This allows out_C to receive full gradients, but completely immunizes self.decoder.weight
            logits_C = F.linear(
                input=c_features,
                weight=self.decoder.weight.detach().float(),
                bias=self.decoder.bias.detach() if self.decoder.bias is not None else None
            )

        if self.training:
            ForwardMetaContext.set(
                **{
                    "Telemetry/logits_A_max": logits_A.max().item(),
                    "Telemetry/logits_A_median": logits_A.median().item(),
                    "Telemetry/logits_B_max": logits_B.max().item(),
                    "Telemetry/logits_C_max": logits_C.max().item(),

                }
            )

        return logits_A.float(), logits_B.float(), logits_C.float()

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
