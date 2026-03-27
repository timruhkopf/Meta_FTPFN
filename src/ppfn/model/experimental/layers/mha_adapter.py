import torch
import torch.nn as nn


class MHA_StreamAdapter(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_task_pe=True):
        """
        A 3-stream adapter where the target stream (C) attends to a concatenated
        context of (A_train, B_train). Learned task embeddings are added to
        distinguish the source domain of the keys/values.
        """
        super().__init__()
        self.d_model = d_model

        # Learned Task Embeddings: Index 0 for Task A, Index 1 for Task B
        self.use_task_pe = use_task_pe
        if use_task_pe:
            self.task_embedding = nn.Embedding(2, d_model)

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
        final_ffn_layer = self.ffn[-1]
        if isinstance(final_ffn_layer, nn.Linear):
            nn.init.zeros_(final_ffn_layer.weight)
            if final_ffn_layer.bias is not None:
                nn.init.zeros_(final_ffn_layer.bias)

        # 3. Initialize task embeddings with small values so they don't
        # heavily disrupt the initial identity flow
        nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def forward(self, A, B, C, sep, **kwargs):
        """
        A, B, C: Latent representations of shape (T, Batch, d_model)
        sep: The sequence index separating train from test
        """
        device = A.device

        # 1. Isolate the training contexts
        A_train = A[:sep]
        B_train = B[:sep]

        if self.use_task_pe:
            # 2. Fetch and apply task embeddings (Broadcasting handles the Batch and T dimensions)
            emb_A = self.task_embedding(torch.tensor(0, device=device))
            emb_B = self.task_embedding(torch.tensor(1, device=device))

            A_train_pe = A_train + emb_A
            B_train_pe = B_train + emb_B
            C_query = C + emb_A

            # 3. Concatenate along the sequence dimension (dim=0)
            # Resulting shape: (2 * sep, Batch, d_model)
            context = torch.cat([A_train_pe, B_train_pe], dim=0)
        else:
            C_query = C
            context = torch.cat([A_train, B_train], dim=0)

        # 4. CROSS-ATTENTION: C queries the joint (A + B) context
        attn_out, attn_weights = self.cross_attn(
            self.norm_q(C_query),
            self.norm_k(context),
            self.norm_v(context)
        )

        # First Residual Connection
        C = C + attn_out

        # Second Residual Connection (Uncommented for completeness)
        # C = C + self.ffn(self.norm_ffn(C))

        # Note: attn_weights will now be of shape (Batch, T_C, 2 * sep)
        # The first half corresponds to attention on A, the second half on B
        # ForwardMetaContext.set("attn_scores", attn_weights)

        return A, B, C