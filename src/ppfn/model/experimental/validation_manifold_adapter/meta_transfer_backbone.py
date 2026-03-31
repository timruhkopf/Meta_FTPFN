import torch
from torch import nn

from ppfn.model.experimental.layers.glt_adapter import MLP

class MetaTransferModel(nn.Module):
    def __init__(self, transfer_layer, input_dim=1,  num_bins=100, pre_train=False, pre_norm=False, dropout=0.1):  # <-- Added num_bins
        super().__init__()
        self.pre_train = pre_train
        self.pre_norm = pre_norm

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

        self.activation = nn.GELU()

        # CHANGED: Project out to the number of bins, not a single continuous value
        self.out_proj = MLP(self.dmodel, self.dmodel, num_bins)

    def pfn_fwd(self, ABC):

        # flow taken from ifbo.layer.TransformerEncoderLayer
        ABC = ABC + self.dropout1(ABC)

        if not self.pre_norm:
            ABC = self.norm1(ABC)

        if self.pre_norm:
            ABC_ = self.norm2(ABC)
        else:
            ABC_ = ABC
        ABC2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(ABC_)
                )
            )
        )
        ABC = ABC + self.dropout2(ABC2)

        if not self.pre_norm:
            ABC = self.norm2(ABC)

        return ABC

    def forward(self, batch):
        sep = batch["sep"]

        A = self.y_proj(batch["y_cA"])
        B = self.y_proj(batch["y_cB"])
        C = self.y_proj(batch["y_cA"])

        device = A.device

        hp_A = self.x_proj(batch["x_cA"])
        hp_B = self.x_proj(batch["x_cB"])
        hp_C = self.x_proj(batch["x_cA"])

        A += hp_A
        B += hp_B
        C += hp_C

        # self attention on the concat of A, B, C with the appropriate masks and separation index
        # This is basically taken from the ifbo.layer.TransformerEncoderLayer forward function
        ABC = torch.cat([A, B, C], dim=1).to(device)  # Shape: [T_total*3, Batch, dmodel]
        padding = torch.cat([batch["mask_A"], batch["mask_B"], batch["mask_A"]], dim=0).to(device)
        ABC_train, _ = self.backbone_attention(
            ABC[:sep], ABC[:sep], ABC[:sep],
            key_padding_mask=padding[:, : sep]
        )

        ABC_test, _ = self.backbone_attention(
            ABC[sep:], ABC[:sep], ABC[:sep],
            key_padding_mask=padding[:, :sep]  # Test tokens can attend to all context tokens, but not to other test tokens
        )

        ABC = torch.cat([ABC_train, ABC_test], dim=0)

        if self.pre_train:
            ABC = self.pfn_fwd(ABC)
            return ABC

        # fixme: A may be of variable size during inference, then chunking fails
        A, B, C = ABC.chunk(3, dim=1)  # Split back into A, B, C based on the original concatenation

        A, B, C_out = self.transfer_layer(
            A, B, C, sep=sep, hp=(hp_A, hp_B, hp_C),
            mask_A=batch["mask_A"], mask_B=batch["mask_B"]
        )


        # ABC = torch.cat([A, B, C_out], dim=1)  # Shape: [T_total*3, Batch, dmodel]
        # ABC = self.pfn_fwd(ABC)

        # FIXME: will fail if A is variable length during inference
        # C_out = ABC.chunk(3, dim=1)[3]
        C_out = self.pfn_fwd(C_out)  # Apply the PFN block to C_out
        query_out = C_out[sep:, :, :]

        return self.out_proj(query_out)  # Shape will now be [T_query, Batch, num_bins]
