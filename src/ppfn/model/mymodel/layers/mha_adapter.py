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

        # Apply Identity Initialization
        self.init_as_identity()

    def init_as_identity(self):
        """
        Forces the adapter to output zeros initially,
        making the forward pass: C = C + 0
        """
        # 1. Zero out MHA output projection
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)

        # 2. Zero out the final linear layer in FFN
        # Accessing the last module in the Sequential block
        final_ffn_layer = self.ffn[-1]
        if isinstance(final_ffn_layer, nn.Linear):
            nn.init.zeros_(final_ffn_layer.weight)
            if final_ffn_layer.bias is not None:
                nn.init.zeros_(final_ffn_layer.bias)

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


