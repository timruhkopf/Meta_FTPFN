import torch
import torch.nn as nn

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class MHA_StreamAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        A baseline 3-stream adapter where the target stream (C) simply attends
        to the training points of the related task stream (B_train) via standard
        cross-attention, without any prior domain unwarping.
        """
        super().__init__()
        self.d_model = d_model

        # Cross-Attention Components
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # Feed-Forward Network (FFN) Modulator
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self,A, B, C, sep, **kwargs):
        """
        x: Latent representations packed as (T, 3*Batch, d_model)
        single_eval_pos: The sequence index separating train from test (sep)
        """

        # Isolate B_train for Stream C to query
        B_train = B[:sep]

        # CROSS-ATTENTION: C queries B_train
        attn_out, attn_weights = self.cross_attn(
            self.norm_q(C),
            self.norm_k(B_train),
            self.norm_v(B_train)
        )

        # First Residual Connection
        C = C + attn_out

        # Second Residual Connection
        # C = C + self.ffn(self.norm_ffn(C))

        ForwardMetaContext.set( "attn_scores", attn_weights)

        return A, B, C


