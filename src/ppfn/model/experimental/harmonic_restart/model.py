import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional
from torch import Tensor

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class PFNLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, ):
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
            if pad_mask_A is None:
                pad_mask_A = torch.zeros((B_size, T), dtype=torch.bool, device=device)
            if pad_mask_B is None:
                pad_mask_B = torch.zeros((B_size, T), dtype=torch.bool, device=device)

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

        return A_out.contiguous(), B_out.contiguous(), C_out.contiguous()


class GumbelGate(nn.Module):
    def __init__(self, input_dim, hard=True):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, 1)
        self.hard = hard
        # Initialize bias so it starts "undecided" or slightly open
        nn.init.constant_(self.gate_linear.bias, 0.5)

    def forward(self, x, tau=1.0, training=True):
        logits = self.gate_linear(x)  # (Batch, 1)

        if training:
            # 1. Sample Gumbel noise
            unif = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(unif + 1e-20) + 1e-20)

            # 2. Apply Gumbel-Sigmoid trick
            # We treat the single logit as the difference between two gumbel samples
            y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

            if self.hard:
                # 3. Straight-Through Estimator
                # Forward pass: 0 or 1. Backward pass: gradient of y_soft
                y_hard = (y_soft > 0.5).float()
                gate = y_hard - y_soft.detach() + y_soft
            else:
                gate = y_soft
        else:
            # Inference: Just a deterministic threshold or sigmoid
            gate = (torch.sigmoid(logits) > 0.5).float() if self.hard else torch.sigmoid(logits)

        return gate, logits


class HardSigmoidGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d_model, 1)
        # Initialize bias to 0.5 to 1.0.
        # This ensures gate output starts around 0.6 - 0.7 (inside the linear gradient zone)
        nn.init.constant_(self.gate_linear.bias, 0.8)

    def forward(self, x):
        # F.hardsigmoid is built into modern PyTorch (1.7+)
        logits = self.gate_linear(x)
        ForwardMetaContext.set('gate_logits', logits)
        return F.hardsigmoid(logits)


# TODO what about gating against negative transfer?
class PreNormTriStreamTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=128, dropout=0.1, use_B_attn_sink=False, use_hp=True,
                 gate_type=None, use_gate=False, use_add_pfn=True, use_post_attn=False) -> None:
        super().__init__()
        batch_first = False
        self.use_hp = use_hp

        # Shared self-attention and conditional cross-attention
        self.use_post_attn = use_post_attn
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.use_add_pfn = use_add_pfn
        if self.use_add_pfn:
            self.pfn_layer = PFNLayer(d_model, nhead, dim_feedforward=dim_feedforward)

        self.use_gate = use_gate
        self.gate_type = gate_type
        if self.gate_type == 'hard_sigmoid':
            self.gate_module = HardSigmoidGate(d_model)

        elif self.gate_type == 'gumbel':
            # 2 * d_model because input is concatenated
            self.gate_module = GumbelGate(2 * d_model, hard=False)  # Start soft, maybe flip to hard later
            self.tau = 1.0
        elif self.gate_type == "sigmoid":
            # Standard Sigmoid fallback
            self.gate_linear = Linear(2 * d_model, 1)
            nn.init.constant_(self.gate_linear.bias, 1.0)

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
        nn.init.constant_(self.gate_linear.bias,
                          1.0)  # Initialize bias to 1.0 to encourage initially trusting the cross-attention

        self.activation = F.relu

        self.use_B_attn_sink = use_B_attn_sink
        if use_B_attn_sink:
            self.B_attn_sink = nn.Parameter(
                torch.randn(1, 1, d_model))  # Learned token for C to attend to if it wants to ignore B

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
            hp_A: Tensor,
            hp_B: Tensor,
            hp_C: Tensor,
            sep: int,
            raw_hp_A: Tensor=None,
            raw_hp_B: Tensor=None,
            raw_hp_C: Tensor=None,
            pad_mask_A: Optional[Tensor] = None,
            pad_mask_B: Optional[Tensor] = None
    ):

        if self.use_hp:
            A = A + hp_A
            B = B + hp_B
            C = C + hp_C

        if self.use_add_pfn:
            A, B, C = self.pfn_layer(A, B, C, sep, pad_mask_A, pad_mask_B)

        # ==========================================
        # 1. SHARED SELF-ATTENTION (Pre-Norm)
        # ==========================================
        if not self.use_post_attn: # Consider: this is the most successful under the detached
            # normed_A = self.norm1(A)
            normed_B = self.norm1(B)
            normed_C = self.norm1(C)

            # C has the exact same structure/padding as A
            # FIXME: we could also just stack them and do one big attention call!
            # src2_A = self._apply_self_attention(normed_A, single_eval_pos, pad_mask_A)
            src2_B = self._apply_self_attention(normed_B, sep, pad_mask_B)
            src2_C = self._apply_self_attention(normed_C, sep, pad_mask_A)

            # A = A + self.dropout1(src2_A)
            B = B + self.dropout1(src2_B)
            C = C + self.dropout1(src2_C)

        # ==========================================
        # 2. CONDITIONAL CROSS-ATTENTION (Pre-Norm)
        # ==========================================
        # A and B do NOT cross attend. They bypass this block entirely.
        normed_cross_C = self.norm_cross(C)
        normed_cross_B = self.norm_cross(B)

        # B's train part serves as the memory
        B_train = normed_cross_B[:sep, :, :]
        pad_mask_B_train = pad_mask_B[:, :sep] if pad_mask_B is not None else None

        if self.use_B_attn_sink:
            batch_size = B_train.shape[1]
            # we append one learned token as an escape valve for C to not attend to B
            B_train = torch.cat([B_train, self.B_attn_sink.expand(1, batch_size, -1)], dim=0)

            if pad_mask_B is not None:
                pad_mask_B_train = torch.cat([pad_mask_B_train, torch.zeros(B_train.size(0), 1, ...)], dim=1)

        # TODO can we take the pre-sigmoid weights and use them as signal for the gating? wouldn't that measure the data support?
        cross_C, cross_attn_weights = self.cross_attn(
            query=normed_cross_C,
            # Avoiding task based gradient interference, we detach B here and get better uncertainty bounds
            key=B_train,
            value=B_train,
            key_padding_mask=pad_mask_B_train
        )

        # --------------------------------------
        # Adjusted Attn Entropy (Thermodynamic Energy)
        # --------------------------------------
        # If the space is perfectly aligned (just shifted/scaled), C should look at B and instantly know exactly which
        # token it corresponds to. The attention distribution will be a sharp peak (a Dirac delta).If the spaces are
        # highly warped, the relationships are ambiguous. C will hedge its bets and attend softly to many tokens in B.
        # In thermodynamics, this spreading of state is higher entropy.How to measure it: Calculate the Shannon Entropy
        # of the cross-attention weights.
        # $$E_{entropy} = -\frac{1}{N} \sum_{i} \sum_{j} A_{i,j} \log(A_{i,j} + \epsilon)$$

        # cross_attn_weights: (B, Heads, Tc, Tb)
        epsilon = 1e-9

        # Entropy over the source/key dimension (Tb)
        # Resulting shape: (B, Heads, Tc)
        entropy = - (cross_attn_weights * torch.log(cross_attn_weights + epsilon)).sum(dim=-1)

        # Mean over all dimensions to get a single scalar for logging
        thermodynamic_energy = entropy.mean()

        ForwardMetaContext.set('thermodynamic_energy', thermodynamic_energy)

        # --------------------------------------
        # Measure Transport Energy
        # --------------------------------------
        #  The Transport Work (The "Earth Mover's" Energy)Since you know the physical/hyperparameter locations
        #  (hp_A / hp_C and hp_B), you can measure the physical "distance" the attention mechanism has to reach
        #  across to find a match.If B is just a shifted A, a perfectly unwarped mapping would simply be a strict
        #  diagonal attention matrix offset by the shift $\Delta = hp_B - hp_C$. If the attention spreads out or
        #  reaches to completely wrong hp locations, the space is heavily warped.How to measure it:Treat the
        #  cross-attention matrix as a transport plan (or transition probability matrix), and calculate the
        #  Wasserstein-like expected distance:
        #  $$E_{transport} = \sum_{i} \sum_{j} A_{i,j} \cdot \mathcal{D}(hp_{C_i}, hp_{B_j})^2$$Where $A_{i,j}$
        #  is the cross-attention weight from C's $i$-th token to B's $j$-th token, and $\mathcal{D}$ is
        #  the distance between their known hyperparameter coordinates.

        # 1. Permute HP tensors to (Batch, Seq, Dhp) for cdist
        hp_B_train = hp_B[:sep, :, :]

        # 1. Permute HP tensors to (Batch, Seq, Dhp) for cdist
        hp_C_perm = hp_C.permute(1, 0, 2)
        hp_B_train_perm = hp_B_train.permute(1, 0, 2)

        # 2. Compute the distance/cost matrix: (Batch, Tc, sep)
        dist_matrix = torch.cdist(hp_C_perm, hp_B_train_perm, p=2)
        cost_matrix = dist_matrix ** 2

        # 3. Check if cross_attn_weights is 3D or 4D
        if cross_attn_weights.dim() == 4:
            # (B, Heads, Tc, sep+1) -> (B, Tc, sep+1)
            mean_attn = cross_attn_weights.mean(dim=1)
        else:
            # (B, Tc, sep+1)
            mean_attn = cross_attn_weights

        # 4. Slice the last dimension to remove the sink
        tb_len = cost_matrix.size(-1)  # This is now exactly `sep`
        mean_attn_no_sink = mean_attn[:, :, :tb_len]

        # 5. Energy calculation
        # Both tensors are now cleanly (Batch, Tc, sep)
        transport_energy = (mean_attn_no_sink * cost_matrix).sum(dim=-1).mean()

        ForwardMetaContext.set('transport_energy', transport_energy)

        # ------------------------------------------

        # identify the sink relative weight
        # if self.use_B_attn_sink:
        ForwardMetaContext.set('cross_attn_weights', cross_attn_weights)  # Store for later analysis

        # ------------------------------------------
        # Update strength (Kinetic Energy)
        # ------------------------------------------
        # If you want to measure the energy strictly in the embedding space (rather than the sequence/hp space),
        # look at the residual update itself.The cross-attention outputs a vector cross_C which is then added to C.
        # This vector represents the literal mathematical "force" applied to C to pull it towards B's manifold.
        # How to measure it:Calculate the Relative L2 Norm of the update.
        # $$E_{kinetic} = \frac{1}{N} \sum \frac{\| \text{cross\_C} \|_2}{\| C \|_2}$$
        # C is the pre-residual state, cross_C is the attention output


        # Calculate L2 norm along the feature dimension (dim=-1)
        # These results will be (Tc, B)
        update_norm = torch.norm(cross_C, p=2, dim=-1)
        state_norm = torch.norm(C, p=2, dim=-1)

        # Relative magnitude of the warp
        # Average across Sequence (dim=0) and Batch (dim=1)
        kinetic_energy = (update_norm / (state_norm + 1e-8)).mean()

        ForwardMetaContext.set('kinetic_energy', kinetic_energy)


        # ==========================================
        # 3. THE TASK-LEVEL COHERENCE CHECK
        # ==========================================
        if self.use_gate:
            global_target = normed_cross_C.mean(dim=0)
            global_proposal = cross_C.mean(dim=0)
            gate_input = torch.cat([global_target, global_proposal], dim=-1)  # (Batch, 2D)

            # Evaluate the chosen gate type
            if self.gate_type == 'hard_sigmoid':
                # HardSigmoidGate handles setting the 'gate_logits' internally
                gate_scalar = self.gate_module(gate_input)

            elif self.gate_type == 'gumbel':
                gate_scalar, gate_logits = self.gate_module(gate_input, tau=self.tau, training=self.training)
                ForwardMetaContext.set('gate_logits', gate_logits)

            elif self.gate_type == 'sigmoid':
                gate_logits = self.gate_linear(gate_input)
                ForwardMetaContext.set('gate_logits', gate_logits)
                gate_scalar = torch.sigmoid(gate_logits)

            # Store the activated probability for analysis
            ForwardMetaContext.set('gate', gate_scalar)

            # Broadcast the scalar back across the sequence dimension -> (1, Batch, 1)
            gate_broadcast = gate_scalar.unsqueeze(0)

            # Apply the gated residual
            C = C + self.dropout_cross(gate_broadcast * cross_C)

        else:
            C = C + self.dropout_cross(cross_C)

        # ==========================================
        # 3. SHARED FEEDFORWARD (Pre-Norm)
        # ==========================================
        # normed_ff_A = self.norm2(A)
        normed_ff_B = self.norm2(B)
        normed_ff_C = self.norm2(C)

        # Helper for the MLP
        def ff_block(x):
            return self.linear2(self.dropout(self.activation(self.linear1(x))))

        # A = A + self.dropout2(ff_block(normed_ff_A))
        B = B + self.dropout2(ff_block(normed_ff_B))
        C = C + self.dropout2(ff_block(normed_ff_C))

        if self.use_post_attn: # fixme: move this layer into the main model and detach C for it!
            normed_C = self.norm1(C)
            src2_C = self._apply_self_attention(normed_C, sep, pad_mask_A)
            C = C + self.dropout1(src2_C)

        return A, B, C


class FourierEncoder(nn.Module):
    def __init__(self, d_model, sigma=1.0):
        super().__init__()
        # Initialize random frequencies
        self.frequencies = nn.Parameter(torch.randn(1, d_model // 2) * sigma, requires_grad=False)
        self.linear = nn.Linear(d_model, d_model)  # Optional mixing layer

    def forward(self, x):
        # x is (Seq, Batch, 1)
        scaled_x = 2 * torch.pi * x @ self.frequencies
        fourier_features = torch.cat([torch.sin(scaled_x), torch.cos(scaled_x)], dim=-1)
        return self.linear(fourier_features)


class TriHarmonicModel(nn.Module):
    def __init__(self, d_model=64, nhead=4, dropout=0.1, num_bars=100, use_B_attn_sink=False, use_freq_enc_x=True):
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

        # ==========================================
        # FIX: Inject Y into train positions without truncating the sequence
        # ==========================================
        A = emb_X_A.clone()
        B = emb_X_B.clone()

        A[:single_eval_pos, :, :] += emb_Y_A
        B[:single_eval_pos, :, :] += emb_Y_B

        C = A.clone()

        # Pass through the Marginal PFN Layer
        A, B, C = self.pfn_layer(A, B, C, single_eval_pos, pad_mask_A)

        # ==========================================
        # FIX: Cleaned up kwargs to match PreNormTriStreamTransformerLayer signature
        # ==========================================
        _, _, C = self.layer(
            A.detach(), B.detach(), C.detach(),
            hp_A=emb_X_A.detach(), hp_B=emb_X_B.detach(), hp_C=emb_X_A.detach(),
            sep=single_eval_pos,
            raw_hp_A = X_A_clean,
            raw_hp_B = X_B_clean,
            raw_hp_C = X_A_clean,
            pad_mask_A=pad_mask_A,
            pad_mask_B=None
        )

        # Apply final norm
        out_A = self.final_norm(A)
        out_B = self.final_norm(B)
        out_C = self.final_norm(C)

        # Decode test positions into logits
        logits_A = self.decoder(out_A[single_eval_pos:, :, :])
        logits_B = self.decoder(out_B[single_eval_pos:, :, :])
        logits_C = self.decoder(out_C[single_eval_pos:, :, :])

        return logits_A, logits_B, logits_C
