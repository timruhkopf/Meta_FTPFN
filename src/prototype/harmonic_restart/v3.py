import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class StableCopyMultiheadAttention(nn.Module):
    """L2-Normalized Cosine Attention for flawless, crash-free hard copying in pure fp16/bf16."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, logit_cap=30.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.logit_cap = logit_cap

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        S_q, B_q, _ = query.size()
        S_k, B_k, _ = key.size()

        q = self.q_proj(query).view(S_q, B_q, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(S_k, B_k, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(S_k, B_k, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Strict L2 Normalization to prevent median drift
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

        # Cosine Similarity bounded between [-1, 1]
        cos_sim = torch.matmul(q, k.transpose(-2, -1))

        # Soft-cap scalar guarantees logits never exceed +/- 30.0
        capped_logits = cos_sim * self.logit_cap

        if attn_mask is not None:
            capped_logits = capped_logits + attn_mask.to(capped_logits.dtype)

        if key_padding_mask is not None:
            padding_mask_expanded = key_padding_mask.view(B_k, 1, 1, S_k)
            capped_logits = capped_logits.masked_fill(padding_mask_expanded, float('-inf'))

        attn_weights = F.softmax(capped_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(2, 0, 1, 3).contiguous().view(S_q, B_q, -1)

        return self.out_proj(out), attn_weights


class DeltaPFNBlock(nn.Module):
    """
    Implements the asymmetric Delta routing with an omniscient Teacher prior.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        # 1. SHARED PFN Temporal Layer (For A, B, and the Teacher Target)
        self.temporal_attn_AB = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_temp_AB = nn.LayerNorm(d_model)

        # 2. Deform-Copy Layer (Stable Cosine Attention for C_neutral -> B)
        self.cross_attn_neutral = StableCopyMultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm_cross_C = nn.LayerNorm(d_model)
        self.norm_cross_B = nn.LayerNorm(d_model)

        # 3. SHARED Feed Forward Network
        self.ffn_AB = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn_AB = nn.LayerNorm(d_model)

    # --- Read-Only Functional Wrappers ---
    def _detached_layer_norm(self, x, norm_module):
        return F.layer_norm(
            x, norm_module.normalized_shape,
            weight=norm_module.weight.detach(),
            bias=norm_module.bias.detach() if norm_module.bias is not None else None,
            eps=norm_module.eps
        )

    def _detached_ffn(self, x, ffn_module):
        x = F.linear(x, ffn_module[0].weight.detach(),
                     ffn_module[0].bias.detach() if ffn_module[0].bias is not None else None)
        x = F.gelu(x)
        x = F.dropout(x, p=ffn_module[2].p, training=self.training)
        x = F.linear(x, ffn_module[3].weight.detach(),
                     ffn_module[3].bias.detach() if ffn_module[3].bias is not None else None)
        x = F.dropout(x, p=ffn_module[4].p, training=self.training)
        return x

    def _detached_sliced_attention(self, query_seq, sep_idx, norm_module, attn_module, pad_mask=None):
        q_norm = self._detached_layer_norm(query_seq, norm_module)
        sliced_pad_mask = pad_mask[:, :sep_idx] if pad_mask is not None else None

        kwargs = {
            'embed_dim_to_check': attn_module.embed_dim,
            'num_heads': attn_module.num_heads,
            'in_proj_weight': attn_module.in_proj_weight.detach(),
            'in_proj_bias': attn_module.in_proj_bias.detach() if attn_module.in_proj_bias is not None else None,
            'bias_k': attn_module.bias_k.detach() if attn_module.bias_k is not None else None,
            'bias_v': attn_module.bias_v.detach() if attn_module.bias_v is not None else None,
            'add_zero_attn': attn_module.add_zero_attn,
            'dropout_p': attn_module.dropout,
            'out_proj_weight': attn_module.out_proj.weight.detach(),
            'out_proj_bias': attn_module.out_proj.bias.detach() if attn_module.out_proj.bias is not None else None,
            'training': self.training,
            'need_weights': False,
            'key_padding_mask': sliced_pad_mask
        }

        out_left, _ = F.multi_head_attention_forward(query=q_norm[:sep_idx], key=q_norm[:sep_idx],
                                                     value=q_norm[:sep_idx], **kwargs)
        out_right, _ = F.multi_head_attention_forward(query=q_norm[sep_idx:], key=q_norm[:sep_idx],
                                                      value=q_norm[:sep_idx], **kwargs)
        return torch.cat([out_left, out_right], dim=0)

    @staticmethod
    def sliced_attention(query_seq, sep_idx, attn_module, pad_mask=None):
        sliced_pad_mask = pad_mask[:, :sep_idx] if pad_mask is not None else None
        out_left, _ = attn_module(query_seq[:sep_idx], query_seq[:sep_idx], query_seq[:sep_idx],
                                  key_padding_mask=sliced_pad_mask)
        out_right, _ = attn_module(query_seq[sep_idx:], query_seq[:sep_idx], query_seq[:sep_idx],
                                   key_padding_mask=sliced_pad_mask)
        return torch.cat([out_left, out_right], dim=0)

    def forward(self, A, B, C, sep, true_B_in_A=None, pad_mask_A=None, pad_mask_B=None, pad_mask_C=None):
        # ==========================================
        # STEP 1: DEFORM-COPY (Workspace queries B_train)
        # ==========================================
        C_neutral_train = C[sep: 2 * sep, :, :]
        B_train = B[:sep, :, :]

        C_neutral_norm = self.norm_cross_C(C_neutral_train)
        B_train_norm = self.norm_cross_B(B_train)

        sliced_pad_B = pad_mask_B[:, :sep] if pad_mask_B is not None else None

        C_cross_out, _ = self.cross_attn_neutral(
            query=C_neutral_norm,
            key=B_train_norm.detach(),
            value=B_train.detach(),
            key_padding_mask=sliced_pad_B
        )

        # Update Workspace and stitch C: [A_train || Workspace_updated || A_test]
        C = torch.cat([C[:sep], C_neutral_train + C_cross_out, C[2 * sep:]], dim=0)

        # ==========================================
        # STEP 2: SHARED PFN TEMPORAL ATTENTION
        # ==========================================
        A = A + self.sliced_attention(self.norm_temp_AB(A), sep, self.temporal_attn_AB, pad_mask_A)
        B = B + self.sliced_attention(self.norm_temp_AB(B), sep, self.temporal_attn_AB, pad_mask_B)

        # C uses the detached pass to preserve the shared weights. Train context is 2*sep.
        C = C + self._detached_sliced_attention(C, 2 * sep, self.norm_temp_AB, self.temporal_attn_AB, pad_mask_C)

        # ==========================================
        # STEP 3: LATENT MANIFOLD CONSTRAINT (Deep Supervision)
        # ==========================================
        if self.training and true_B_in_A is not None:
            # Construct the perfect latent trajectory using the actual coordinates
            target_seq = torch.cat([A[:sep].detach(), true_B_in_A], dim=0)

            target_norm = self.norm_temp_AB(target_seq)
            target_out = target_seq + self.temporal_attn_AB(target_norm, target_norm, target_norm)[0]

            latent_target_workspace = target_out[sep:, :, :]
            latent_pred_workspace = C[sep: 2 * sep, :, :]

            # Compute penalty to be collected by outer model telemetry
            latent_penalty = F.mse_loss(latent_pred_workspace, latent_target_workspace.detach())

            # Use your custom ForwardMetaContext
            try:
                ForwardMetaContext.set(**{
                    "Telemetry/latent_manifold_mse": latent_penalty.item(),
                    "Loss/latent_penalty": latent_penalty
                })
            except NameError:
                pass  # Safe fallback if context manager is missing during tests

        # ==========================================
        # STEP 4: SHARED FFN
        # ==========================================
        A = A + self.ffn_AB(self.norm_ffn_AB(A))
        B = B + self.ffn_AB(self.norm_ffn_AB(B))
        C = C + self._detached_ffn(self._detached_layer_norm(C, self.norm_ffn_AB), self.ffn_AB)

        return A, B, C


class MultiStageTriHarmonicModel(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=4, dropout=0.1, num_bars=250):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        # --- THE INFERENCE-SAFE WORKSPACE PROMPT ---
        # A learnable structural flag broadcasted to B's coordinates
        # to tell the network to shift them into A's domain.
        self.workspace_prompt = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.cross_layers = nn.ModuleList([
            DeltaPFNBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Decoder without bias to prevent softmax drift
        self.decoder = nn.Linear(d_model, num_bars, bias=False)

    def forward(self, batch):
        sep = batch['train']['X_B'].shape[0]

        X_A = torch.cat([batch['train']['X_A'], batch['test']['X_A']], dim=0)
        X_B = torch.cat([batch['train']['X_B'], batch['test']['X_B']], dim=0)

        emb_X_A = self.x_encoder(X_A)
        emb_X_B = self.x_encoder(X_B)

        A = emb_X_A.clone()
        B = emb_X_B.clone()

        A[:sep, :, :] += self.y_encoder(batch['train']['Y_A'])
        B[:sep, :, :] += self.y_encoder(batch['train']['Y_B'])

        # ==========================================
        # STREAM C SETUP: [A_train || Workspace || A_test]
        # ==========================================
        A_train = A[:sep].detach().clone()
        A_test = A[sep:].detach().clone()

        # Initialize neutral workspace with B's known coordinates + the learned flag
        B_train_coords = self.x_encoder(batch['train']['X_B'])
        C_workspace = B_train_coords + self.workspace_prompt

        # Stack C. Length = sep (A_train) + sep (workspace) + T_test (A_test)
        C = torch.cat([A_train, C_workspace, A_test], dim=0)

        # ==========================================
        # LATENT TEACHER SETUP (Strictly for Training)
        # ==========================================
        true_B_in_A = None
        if self.training:
            # Construct the omniscient target: Embedded true coords + true values
            true_coords = self.x_encoder(batch['train']['X_B_in_A'])
            true_values = self.y_encoder(batch['train']['Y_B_in_A'])
            true_B_in_A = true_coords + true_values

        # Construct C's combined padding mask
        pad_mask_A = batch['train']['padding_mask_A']
        pad_mask_B = batch['train']['padding_mask_B']

        pad_mask_C = None
        if pad_mask_A is not None and pad_mask_B is not None:
            # C's train segment is [A_train || Workspace], length is 2*sep
            pad_mask_C = torch.cat([pad_mask_A[:, :sep], pad_mask_B[:, :sep]], dim=1)

        # ==========================================
        # FORWARD PASS
        # ==========================================
        for layer in self.cross_layers:
            A, B, C = layer(A, B, C, sep, true_B_in_A=true_B_in_A,
                            pad_mask_A=pad_mask_A, pad_mask_B=pad_mask_B, pad_mask_C=pad_mask_C)

        # ==========================================
        # DECODING & EXTRACTION
        # ==========================================
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)

        # Detached pass for C so it doesn't skew the decoder's global alignment
        out_C = F.layer_norm(
            C, self.final_norm.normalized_shape,
            weight=self.final_norm.weight.detach(),
            bias=self.final_norm.bias.detach() if self.final_norm.bias is not None else None,
            eps=self.final_norm.eps
        )

        logits_A = self.decoder(out_A[sep:, :, :].float())
        logits_B = self.decoder(out_B[sep:, :, :].float())

        # C's test tokens start at index 2*sep
        logits_C_test = F.linear(
            out_C[2 * sep:, :, :].float(),
            self.decoder.weight.detach().float()
        )

        # Extract Workspace Train Logits for your explicit tracking/aux NLL
        logits_C_workspace = F.linear(
            out_C[sep: 2 * sep, :, :].float(),
            self.decoder.weight.detach().float()
        )

        if self.training:
            try:
                ForwardMetaContext.set(**{
                    "Telemetry/logits_A_max": logits_A.max().item(),
                    "Telemetry/logits_A_median": logits_A.median().item(),
                    "Telemetry/logits_C_workspace_max": logits_C_workspace.max().item(),
                })
            except NameError:
                pass

        return logits_A.float(), logits_B.float(), logits_C_test.float()