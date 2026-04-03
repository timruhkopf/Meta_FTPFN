import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor



class PreNormTriStreamTransformerLayer(nn.Module):
    # FIXME: make the architecture batch_first=False
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=True ) -> None:
        super().__init__()
        batch_first = False

        # Shared self-attention and conditional cross-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Shared Feedforward
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # Pre-Norms
        self.norm1 = LayerNorm(d_model)  # Pre Self-Attention
        self.norm_cross = LayerNorm(d_model)  # Pre Cross-Attention
        self.norm2 = LayerNorm(d_model)  # Pre Feedforward

        self.dropout1 = Dropout(dropout)
        self.dropout_cross = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.relu

        if use_B_attn_sink:
            self.B_attn_sink = nn.Parameter(torch.randn(1, 1, d_model))  # Learned token for C to attend to if it wants to ignore B
        self.use_B_attn_sink = use_B_attn_sink

    def _apply_self_attention(self, src: Tensor, eval_pos: int, pad_mask: Optional[Tensor]) -> Tensor:
        train_part = src[:eval_pos, :, :]
        test_part = src[eval_pos:, :, :]
        train_pad_mask = pad_mask[:, :eval_pos] if pad_mask is not None else None

        train_out = self.self_attn(train_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]
        test_out = self.self_attn(test_part, train_part, train_part, key_padding_mask=train_pad_mask)[0]

        return torch.cat([train_out, test_out], dim=0)

    def forward(
            self,
            A: Tensor,
            B: Tensor,
            C: Tensor,
            single_eval_pos: int,
            pad_mask_A: Optional[Tensor] = None,
            pad_mask_B: Optional[Tensor] = None
    ):
        # ==========================================
        # 1. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        normed_A = self.norm1(A)
        normed_B = self.norm1(B)
        normed_C = self.norm1(C)

        # C has the exact same structure/padding as A
        # FIXME: we could also just stack them and do one big attention call!
        src2_A = self._apply_self_attention(normed_A, single_eval_pos, pad_mask_A)
        src2_B = self._apply_self_attention(normed_B, single_eval_pos, pad_mask_B)
        src2_C = self._apply_self_attention(normed_C, single_eval_pos, pad_mask_A)

        A = A + self.dropout1(src2_A)
        B = B + self.dropout1(src2_B)
        C = C + self.dropout1(src2_C)

        # ==========================================
        # 2. CONDITIONAL CROSS-ATTENTION (Pre-Norm)
        # ==========================================
        # A and B do NOT cross attend. They bypass this block entirely.
        normed_cross_C = self.norm_cross(C)
        normed_cross_B = self.norm_cross(B)

        # B's train part serves as the memory
        B_train = normed_cross_B[:single_eval_pos, :, :]
        pad_mask_B_train = pad_mask_B[:, :single_eval_pos] if pad_mask_B is not None else None

        if self.use_B_attn_sink:
            batch_size = B_train.shape[1]
            # we append one learned token as an escape valve for C to not attend to B
            B_train = torch.cat([B_train, self.B_attn_sink.expand(1, batch_size, -1)], dim=0)

            if pad_mask_B is not None:
                pad_mask_B_train = torch.cat([pad_mask_B_train, torch.zeros(B_train.size(0), 1, dtype=torch.bool, device=B_train.device)], dim=1)

        cross_C = self.cross_attn(
            query=normed_cross_C,
            key=B_train,
            value=B_train,
            key_padding_mask=pad_mask_B_train
        )[0]

        C = C + self.dropout_cross(cross_C)

        # ==========================================
        # 3. SHARED FEEDFORWARD (Pre-Norm)
        # ==========================================
        normed_ff_A = self.norm2(A)
        normed_ff_B = self.norm2(B)
        normed_ff_C = self.norm2(C)

        # Helper for the MLP
        def ff_block(x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        A = A + self.dropout2(ff_block(normed_ff_A))
        B = B + self.dropout2(ff_block(normed_ff_B))
        C = C + self.dropout2(ff_block(normed_ff_C))

        return A, B, C


class TriHarmonicModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_bars=100, use_B_attn_sink=True):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)

        self.layer = PreNormTriStreamTransformerLayer(d_model, nhead=nhead, use_B_attn_sink=use_B_attn_sink)

        # Final norm for Pre-Norm architecture
        self.final_norm = LayerNorm(d_model)

        # Shared decoder
        self.decoder = nn.Linear(d_model, num_bars)

    def forward(self, batch):
        X_train_A, Y_train_A = batch['train']['X_A'], batch['train']['Y_A']
        X_train_B, Y_train_B = batch['train']['X_B'], batch['train']['Y_B']
        X_test_A = batch['test']['X_A']
        X_test_B = batch['test']['X_B']

        single_eval_pos = batch['train']['X_B'].shape[0]

        # Concat train and test X
        X_A = torch.cat([X_train_A, X_test_A], dim=0)
        X_B = torch.cat([X_train_B, X_test_B], dim=0)

        pad_mask_A = torch.isnan(X_A).transpose(0, 1)

        # Clean NaNs
        X_A_clean = torch.nan_to_num(X_A, nan=0.0).unsqueeze(-1)
        X_B_clean = torch.nan_to_num(X_B, nan=0.0).unsqueeze(-1)
        Y_A_train_clean = torch.nan_to_num(Y_train_A, nan=0.0).unsqueeze(-1)
        Y_B_train_clean = torch.nan_to_num(Y_train_B, nan=0.0).unsqueeze(-1)

        # Encode
        emb_X_A = self.x_encoder(X_A_clean)
        emb_X_B = self.x_encoder(X_B_clean)
        emb_Y_A = self.y_encoder(Y_A_train_clean)
        emb_Y_B = self.y_encoder(Y_B_train_clean)

        # Inject Y into train positions
        emb_X_A[:single_eval_pos, :, :] += emb_Y_A
        emb_X_B[:single_eval_pos, :, :] += emb_Y_B

        # INITIALIZE C AS A COPY OF A
        emb_X_C = emb_X_A.clone()

        # Pass through the Tri-Stream Layer
        out_A, out_B, out_C = self.layer(emb_X_A, emb_X_B, emb_X_C, single_eval_pos, pad_mask_A, pad_mask_B=None)

        # Apply final norm
        out_A = self.final_norm(out_A)
        out_B = self.final_norm(out_B)
        out_C = self.final_norm(out_C)


        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        return logits_A, logits_B, logits_C
