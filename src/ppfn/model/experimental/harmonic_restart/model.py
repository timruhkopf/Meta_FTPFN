import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor
from torch.nn.functional import dropout

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class PFNLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1,):
        super(PFNLayer, self).__init__()
        batch_first = False
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

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
        # Store original batch size to split them back apart later
        # Assumes A, B, and C all have the exact same shape: (Time, Batch, D)
        T, B_size, D = A.shape

        # ==========================================
        # 1. BATCH CONCATENATION
        # ==========================================
        # Stack along the Batch dimension (dim=1) -> Shape: (Time, 3 * Batch, D)
        combined = torch.cat([A, B, C], dim=1)

        # Handle padding masks (Shape: (Batch, Time))
        if pad_mask_A is not None or pad_mask_B is not None:
            device = combined.device
            # If one mask is provided but the other isn't, default the missing one to False
            pad_mask_A = torch.zeros((B_size, T), dtype=torch.bool, device=device) if pad_mask_A is None else pad_mask_A
            pad_mask_B = torch.zeros((B_size, T), dtype=torch.bool, device=device) if pad_mask_B is None else pad_mask_B

            # C uses the exact same padding mask as A
            combined_mask = torch.cat([pad_mask_A, pad_mask_B, pad_mask_A], dim=0)
        else:
            combined_mask = None

        # ==========================================
        # 2. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        normed_combined = self.norm1(combined)

        # One massive attention call
        src2_combined = self._apply_self_attention(
            normed_combined,
            single_eval_pos,
            combined_mask
        )

        combined = combined + self.dropout1(src2_combined)

        # ==========================================
        # 3. SHARED FEED-FORWARD
        # ==========================================
        normed_ff_combined = self.norm2(combined)

        def ff_block(x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        combined = combined + self.dropout2(ff_block(normed_ff_combined))

        # ==========================================
        # 4. UNPACK BACK TO A, B, C
        # ==========================================
        # Split the combined tensor back into 3 distinct tensors along the batch dimension
        A_out, B_out, C_out = torch.split(combined, split_size_or_sections=B_size, dim=1)

        return A_out, B_out, C_out


# TODO what about gating against negative transfer?
class PreNormTriStreamTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=False ) -> None:
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

        self.gate_linear = Linear(2 * d_model, 1)  # For the coherence check gating mechanism
        nn.init.constant_(self.gate_linear.bias, 1.0)  # Initialize bias to 1.0 to encourage initially trusting the cross-attention

        self.activation = F.relu

        self.use_B_attn_sink = use_B_attn_sink
        if use_B_attn_sink:
            self.B_attn_sink = nn.Parameter(torch.randn(1, 1, d_model))  # Learned token for C to attend to if it wants to ignore B

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

        cross_C, cross_attn_weights = self.cross_attn(
            query=normed_cross_C,
            key=B_train,
            value=B_train,
            key_padding_mask=pad_mask_B_train
        )

        # identify the sink relative weight
        if self.use_B_attn_sink:
            ForwardMetaContext.set('cross_attn_weights', cross_attn_weights)  # Store for later analysis


        # --- THE COHERENCE CHECK ---
        # We concatenate the original state and the proposed update
        # gate_input = torch.cat([normed_cross_C, cross_C], dim=-1)
        # gate = torch.sigmoid(self.gate_linear(gate_input))  # (Seq, Batch, 1)


        # ForwardMetaContext.set('gate', gate)  # Store for later analysis

        # If the cross_C (from B) contradicts normed_cross_C (from A),
        # the network can learn to drive 'gate' to 0.
        # C = C + self.dropout_cross(gate * cross_C)
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

class FourierEncoder(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super().__init__()
        # Initialize random frequencies
        self.frequencies = nn.Parameter(torch.randn(1, d_model // 2) * sigma, requires_grad=False)
        self.linear = nn.Linear(d_model, d_model) # Optional mixing layer

    def forward(self, x):
        # x is (Seq, Batch, 1)
        scaled_x = 2 * torch.pi * x @ self.frequencies
        fourier_features = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return self.linear(fourier_features)


class TriHarmonicModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_B_attn_sink=True, use_freq_enc_x=True):
        super().__init__()
        self.num_bars = num_bars
        self.x_encoder = FourierEncoder(d_model) if use_freq_enc_x else nn.Linear(1, d_model)
        self.y_encoder = nn.Linear(1, d_model)


        self.pfn_layer = PFNLayer(d_model, nhead=nhead, dropout=dropout)
        self.layer = PreNormTriStreamTransformerLayer(d_model, nhead=nhead, use_B_attn_sink=use_B_attn_sink)

        # Final norm for Pre-Norm architecture
        self.final_norm = LayerNorm(d_model)

        # Shared decoder
        self.decoder = nn.Linear(d_model, num_bars)

    def forward(self, batch):
        # FIXME: change the signature to accept this directly
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

        emb_X_C = emb_X_A.clone()

        A, B, C = self.pfn_layer(emb_X_A, emb_X_B, emb_X_C, single_eval_pos, pad_mask_A)


        # Pass through the Tri-Stream Layer
        # FIXME: pad_mask_B is only here none, because it is the reference size
        _, _, C = self.layer(A.detach(), B.detach(), C.detach(), single_eval_pos, pad_mask_A, pad_mask_B=None)

        # Apply final norm
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)


        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        # TODO: make sure that the decoder is frozen in the "fine-tuning" stage
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])
        return logits_A, logits_B, logits_C
