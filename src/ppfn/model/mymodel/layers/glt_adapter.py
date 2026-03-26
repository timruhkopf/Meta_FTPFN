import torch
import torch.nn as nn



class MLP(nn.Module):
    """A standard 2-layer MLP to add non-linear depth."""

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class GatedLatentTransferLayer(nn.Module):
    """

    source for the a separated infomration flow, where we have k and v different from q in order to compute what to extract:
    https://arxiv.org/pdf/2107.14795 Perciever IO
    """

    def __init__(self, dmodel=128, use_gate=True):
        super().__init__()
        self.linear_AB1 = MLP(dmodel * 2, dmodel, dmodel * 2)
        self.self_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4, )
        self.linear_AB2 = nn.Linear(dmodel * 2, dmodel)
        self.linear_C = nn.Linear(dmodel * 2, dmodel)
        self.cross_attention = nn.MultiheadAttention(embed_dim=2 * dmodel, vdim=dmodel, num_heads=4, )

        self.C_test_attention = nn.MultiheadAttention(embed_dim=dmodel, num_heads=4, vdim=dmodel)

        self.out_proj = MLP(dmodel * 2, dmodel, dmodel * 2)

        self.use_gate = use_gate
        if use_gate:
            self.gate_proj = nn.Sequential(
                nn.Linear(4 * dmodel, dmodel),
                nn.Sigmoid()
            )

        # Learnable dummy key and value (Shape: [1, 1, feature_dim])
        self.dummy_key = nn.Parameter(torch.randn(1, 1, 2 * dmodel))
        # Note: Adjust dummy_value dimension based on whether you concatenated hp into value earlier
        self.dummy_value = nn.Parameter(torch.randn(1, 1, dmodel))

        self.init_weights()

    def init_weights(self):
        """Neutral Initialization to fade-in the adapters layers """

    def forward(self, A, B, C, sep, hp, mask_A=None, mask_B=None, *args, **kwargs):
        device = A.device
        hp_A, hp_B, hp_C = hp

        # 1. Parse Context Components ----------------------------------------
        A_train, B_train = A[:sep], B[:sep]
        C_train, C_test = C[:sep], C[sep:]
        hp_A_train, hp_B_train = hp_A[:sep], hp_B[:sep]
        hp_C_train, hp_C_test = hp_C[:sep], hp_C[sep:]

        # 2.0 self attend for A and B to extract relative features ------------------------------
        #  e.g. "i am a point in a local maximum", "we all are trending linearly"
        ABC = torch.cat([
            torch.cat([hp_A_train, A_train], dim=-1),
            torch.cat([hp_B_train, B_train], dim=-1),
            torch.cat([hp_C_train, C_train], dim=-1)
            # if we do this prior to any cross attention this is redundant (C=A)

        ], dim=1)

        # down project to dmodel
        ABC = self.linear_AB1(ABC)

        # 2.1. Deal with padding in self attention
        # since A and B have different sequence lengths, we need to build an attention mask
        # to prevent attending to the padded points.
        mask_A_train = mask_A[:, :sep] if mask_A is not None else None
        mask_B_train = mask_B[:, :sep] if mask_B is not None else None

        if mask_A_train is not None and mask_B_train is not None:
            # C doesn't have a mask (it uses all its points), so it's all False
            mask_C_train = mask_A_train.clone()

            # Concat along batch dimension (dim=0) because ABC has 3*Batch size
            self_attn_mask = torch.cat([mask_A_train, mask_B_train, mask_C_train], dim=0)
        else:
            self_attn_mask = None

        # 2.2  A,B,C Shared self attention to extract relational features within A, B, C respectively.
        ABC, _ = self.self_attention(ABC, ABC, ABC, key_padding_mask=self_attn_mask)

        # Extract the feature descriptors for each task after self-attention
        a_dim, b_dim = A_train.shape[1], B_train.shape[1]
        A_feat, B_feat, C_feat = ABC[:, :a_dim], ABC[:, a_dim:a_dim + b_dim], ABC[:, a_dim + b_dim:]

        # 3. Expand C_feat to C_test_feat by hp attention across coordinates. ---------------------------------------
        C_test_feat, _ = self.C_test_attention(hp_C_test, hp_C_train, C_feat, key_padding_mask=mask_C_train)

        # 4. Cross Attention from C to A's and B's features ---------------------------------------
        # throw in A_feat and B_feat into one context for C to cross attend to.
        batch_size = A.shape[1]

        # Here we don't want the softmax constraint, because it will have a sum-to-one constraint across A and B,
        # but we want the model to be able to choose to attend to B
        # Expand dummy tokens to match batch size: [1, Batch, D]
        d_key = self.dummy_key.expand(1, batch_size, -1)
        d_val = self.dummy_value.expand(1, batch_size, -1)

        query = torch.cat([
            torch.cat([hp_C_train, C_feat], dim=-1),
            torch.cat([hp_C_test, C_test_feat], dim=-1)
        ], dim=0)
        key = torch.cat([
            torch.cat([hp_A_train, A_feat], dim=-1),
            torch.cat([hp_B_train, B_feat], dim=-1),
            d_key  # <--- The Escape Valve
        ], dim=0)
        value = torch.cat([  # Value payload is without the hp, because only then it is in the
            A_feat, B_feat,
            d_val  # <--- The Escape Valve
        ], dim=0)  # we only want the features to be the value
        # we cannot attend to the B features, because they are not grounded in the same domain as C
        # TODO ideally, we'd know how B looks like in C's domain (B_train'), then we could make the value the raw B_train' payload (without hp).

        # Build the cross-attention mask
        # C is looking at A and B, so the keys sequence length is 2*T_max.
        # Mask shape needs to be [Batch, 2*T_max]
        # Build the cross-attention mask (Shape: [Batch, 2*sep])
        if mask_A_train is not None and mask_B_train is not None:
            # Concat along the sequence dimension (dim=1) for the mask
            cross_attn_mask = torch.cat([mask_A_train, mask_B_train], dim=1)
        else:
            cross_attn_mask = None

        # We must also append a 'False' to the cross_attn_mask so the dummy token is never masked out
        if cross_attn_mask is not None:
            dummy_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            cross_attn_mask = torch.cat([cross_attn_mask, dummy_mask], dim=1)

        # Pass the mask to cross-attention
        C_cross, _ = self.cross_attention(query, key, value, key_padding_mask=cross_attn_mask)

        # 4. Residual update to C --------------------------------------------
        # C is originally (T, B, D). We project C_cross down to match.
        # Evaluate the query AGAINST the retrieved context
        # This is heavily inspired by Highway Networks and GRUs.
        # It is local, point-by-point, and fully aware of the relationship between $C$ and $B$.
        # This allows the network to say: "I asked for a local maximum (Query),
        # but the feature vector I got back from B looks like a steep drop (C_cross).
        # This is useless to me. Close the gate."
        if self.use_gate:
            gate_input = torch.cat([query, C_cross], dim=-1)
            gate = self.gate_proj(gate_input)
        else:
            gate = 1.0

        # Gated Residual update
        C = C + gate * self.out_proj(C_cross)

        return A, B, C
