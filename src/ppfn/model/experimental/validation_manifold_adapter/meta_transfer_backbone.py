import torch
from torch import nn

from ppfn.model.experimental.layers.glt_adapter import MLP

class MetaTransferModel(nn.Module):
    def __init__(self, transfer_layer, input_dim=1,  num_bins=100, pre_train=False, pre_norm=True, dropout=0.1, detach_to_transfer=True):  # <-- Added num_bins
        super().__init__()
        self.pre_train = pre_train
        self.pre_norm = pre_norm
        self.detach_to_transfer = detach_to_transfer

        self.transfer_layer = transfer_layer
        self.dmodel = self.transfer_layer.dmodel
        self.y_proj = MLP(input_dim, self.dmodel, self.dmodel)
        self.x_proj = MLP(input_dim, self.dmodel, self.dmodel)

        self.backbone_attention = nn.MultiheadAttention(self.dmodel, num_heads=4, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.dmodel)
        self.norm2 = nn.LayerNorm(self.dmodel)
        self.linear1 = nn.Linear(self.dmodel, self.dmodel * 4)
        self.linear2 = nn.Linear(self.dmodel * 4, self.dmodel)

        self.final_norm = nn.LayerNorm(self.dmodel)

        self.activation = nn.GELU()

        # CHANGED: Project out to the number of bins, not a single continuous value
        self.out_proj = MLP(self.dmodel, self.dmodel, num_bins)

    def pfn_fwd(self, x):
        """Standard Transformer Feed-Forward Network (FFN) block."""
        identity = x

        # 1. Pre-Norm
        if self.pre_norm:
            x = self.norm2(x)

        # 2. MLP Expansion
        x2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(x)
                )
            )
        )

        # 3. Residual
        x = identity + self.dropout2(x2)

        # 4. Post-Norm
        if not self.pre_norm:
            x = self.norm2(x)

        return x

    def forward(self, batch):
        sep = batch["sep"]
        device = batch["y_cA"].device

        # --- 1. Projections & Embeddings ---
        A = self.y_proj(batch["y_cA"])
        B = self.y_proj(batch["y_cB"])
        C = self.y_proj(batch["y_cA"])

        hp_A = self.x_proj(batch["x_cA"])
        hp_B = self.x_proj(batch["x_cB"])
        hp_C = self.x_proj(batch["x_cA"])

        A += hp_A
        B += hp_B
        C += hp_C

        # --- 2. Shared Backbone Attention ---
        # Concat along BATCH dimension (dim=1).
        # Shape goes from [T_total, Batch, dmodel] -> [T_total, 3 * Batch, dmodel]
        ABC = torch.cat([A, B, C], dim=1)
        # Concat padding masks along sequence dimension (dim=0) to match 3 * Batch
        padding = torch.cat([batch["mask_A"], batch["mask_B"], batch["mask_A"]], dim=0)

        # Train/Test Split
        identity_train = ABC[:sep]
        identity_test = ABC[sep:]

        # Pre-Norm Logic
        if self.pre_norm:
            normed_train = self.norm1(identity_train)
            normed_test = self.norm1(identity_test)
            kv_train = normed_train  # Keys/Values are always context (train)
        else:
            normed_train = identity_train
            normed_test = identity_test
            kv_train = identity_train

        # Attention Operations
        attn_out_train, _ = self.backbone_attention(
            normed_train, kv_train, kv_train,
            key_padding_mask=padding[:, :sep]
        )
        attn_out_test, _ = self.backbone_attention(
            normed_test, kv_train, kv_train,
            key_padding_mask=padding[:, :sep]
        )

        # Residual Addition
        ABC_train = identity_train + self.dropout1(attn_out_train)
        ABC_test = identity_test + self.dropout1(attn_out_test)

        # Post-Norm Logic
        if not self.pre_norm:
            ABC_train = self.norm1(ABC_train)
            ABC_test = self.norm1(ABC_test)

        # Recombine Sequence
        ABC = torch.cat([ABC_train, ABC_test], dim=0)

        if self.pre_train:
            return self.pfn_fwd(ABC)

        # --- 3. Transfer Layer ---
        # Chunking is 100% safe here because dim=1 is the batch dimension (3 * Batch)
        A, B, C = ABC.chunk(3, dim=1)

        if self.detach_to_transfer:
            A_ = A.detach()
            B_ = B.detach()
            C_ = C.detach()
        else:
            A_ = A
            B_ = B
            C_ = C

        _A, _B, C_out = self.transfer_layer(
            A_, B_, C_, sep=sep, hp=(hp_A, hp_B, hp_C),
            mask_A=batch["mask_A"], mask_B=batch["mask_B"]
        )

        # --- 4. Shared FFN (pfn_fwd) ---
        # Batch them together for a single efficient pass through the FFN
        ABC_out = torch.cat([A, B, C_out], dim=1)
        ABC_out = self.pfn_fwd(ABC_out)

        if self.pre_norm:
            ABC_out = self.final_norm(ABC_out)

        # Split back
        A_final, B_final, C_final = ABC_out.chunk(3, dim=1)

        # --- 5. Output Projection ---
        return (
            self.out_proj(A_final[sep:, :, :]),
            self.out_proj(B_final[sep:, :, :]),
            self.out_proj(C_final[sep:, :, :])
        )