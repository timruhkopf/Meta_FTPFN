import torch
from torch import nn


class StandardCrossAttnLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        # 1. Standard Multihead Attention (It handles its own Q, K, V projections)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False
        )
        
        # 2. Standard Post-Norm and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, A, B, C, sep, pad_mask_A, pad_mask_B):
        # 1. Create the Memory Bank (Concatenate A and B)
        # Using .detach() to protect the marginal backbones
        memory = torch.cat([A.detach(), B.detach()], dim=0) 
        pad_mask_mem = torch.cat([pad_mask_A, pad_mask_B], dim=1)
        
        # 2. Vanilla Cross Attention
        # C is the Query. The concatenated [A, B] is the Key and Value.
        attn_out, _ = self.cross_attn(
            query=C, 
            key=memory, 
            value=memory, 
            key_padding_mask=pad_mask_mem
        )
        
        # 3. Standard Residual + FFN Block (Post-Norm)
        C = self.norm1(C + self.dropout1(attn_out))
        C = self.norm2(C + self.ffn(C))
        
        return A, B, C