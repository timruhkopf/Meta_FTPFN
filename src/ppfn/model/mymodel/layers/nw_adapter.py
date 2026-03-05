import torch
import torch.nn as nn

from ppfn.model.mymodel.meta_context import ForwardMetaContext


class NadarayaWatsonAdapter(nn.Module):
    def __init__(
            self,
            d_model,
            n_heads,
            dropout=0.1,
            nw_dropout=0.0,
            reuse_attn=True,
            hp_only_attn=True,
            seq_len=1000,
            address=None
    ):
        """
        A 3-stream meta-learning adapter that performs Non-Parametric Local Smoothing
        to align and extract knowledge from related tasks within a frozen Prior-Data Fitted Network (PFN).

        This architecture addresses the risk of negative transfer by treating frozen latent
        representations as a topological space where local domain distortions can be measured
        and corrected before cross-task extraction occurs.

        The forward pass operates on three parallel streams packed into a single batch dimension:
        - Stream A (Target Task): Acts as the ground-truth contextual anchor (Frozen).
        - Stream B (Related Task): The source of external knowledge, which may be distorted (Frozen).
        - Stream C (Modulated Target): The active stream being updated for the next (frozen) PFN layer.

        The adapter operates in two main stages:
        1. The Corrector (Nadaraya-Watson): Calculates the local distortion error between
        Stream A's training points and Stream B's "belief" of those points. It applies this
        error to B's domain using an NW-style attention kernel to undistort B into A's manifold.
        2. The Extractor: Stream C queries the newly undistorted B representations to extract
        relevant cross-task features safely, updating itself via a residual connection.

        Args:
         d_model (int): The embedding dimension of the PFN latent space.
         n_heads (int): The number of heads for the Multi-Head Attention mechanisms.
         dropout (float, optional): Standard dropout applied to the Extractor's attention
             matrices and the FFN modulator. Defaults to 0.1.
         nw_dropout (float, optional): Dropout applied specifically to the Nadaraya-Watson
             Corrector attention. It is highly recommended to leave this at 0.0. The NW step
             relies on exact local structural anchors to compute the geometric translation between
             domains. Randomly dropping out attention weights here can randomly sever those anchors,
             resulting in chaotic domain shifts and poisoned representations. Defaults to 0.0.
         reuse_attn (bool, optional): If True, shares the exact same Multi-Head Attention
             weights between the `attn_train` (C_train querying B_prime) and `attn_test`
             (C_test querying B_prime) operations in the Extractor. This mirrors the architectural
             design of the original PFN, which reuses attention for train/test splits on a
             single item to maintain unified feature extraction logic and saves learnable parameters.
             Defaults to True.
        Note: This module expects the hyperparameters (`hp`) to already be encoded
        by the PFN into the same `d_model` latent dimension.
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.hp_only_attn = hp_only_attn
        self.address = address # backlink for logging and debugging purposes
        self.reuse_attn = reuse_attn

        # 1. The Corrector
        self.norm_nw_q = nn.LayerNorm(d_model)
        self.norm_nw_k = nn.LayerNorm(d_model)
        self.norm_nw_v = nn.LayerNorm(d_model)

        self.mlp_err = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.nw_attn = nn.MultiheadAttention(d_model, n_heads, dropout=nw_dropout)

        # 2. The Extractor
        self.norm_train_q = nn.LayerNorm(d_model)
        self.norm_test_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

        self.attn_train = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        if reuse_attn:
            self.attn_test = self.attn_train
        else:
            self.attn_test = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        # 3. FFN Modulator
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.initialize_as_identity()

    def initialize_as_identity(self):
        """
        Initializes the adapter to act as an identity function initially.
        Zeroes out the final projections of the Extractor, FFN, and Corrector.
        """
        # 1. The Extractor: Zero out the attention output projection
        # This ensures `c_train_update` and `c_test_update` are strictly 0.
        nn.init.zeros_(self.attn_train.out_proj.weight)
        nn.init.zeros_(self.attn_train.out_proj.bias)

        if not self.reuse_attn:
            nn.init.zeros_(self.attn_test.out_proj.weight)
            nn.init.zeros_(self.attn_test.out_proj.bias)

        # 2. The FFN Modulator: Zero out the final linear layer
        # In self.ffn, the second Linear layer is at index 3.
        # (0: Linear, 1: GELU, 2: Dropout, 3: Linear, 4: Dropout)
        nn.init.zeros_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[3].bias)

        # 3. The Corrector: Zero out the NW attention output projection
        # This ensures `corr_out` is 0, meaning B_prime exactly equals B_active initially.
        # This prevents the network from applying chaotic domain shifts before learning the manifold.
        nn.init.zeros_(self.nw_attn.out_proj.weight)
        nn.init.zeros_(self.nw_attn.out_proj.bias)

        # Note: You do not need to zero out self.mlp_err because zeroing the
        # nw_attn output projection already bottlenecks the correction to 0.

    # TODO try a "near-identity" initialization where we initialize the weights to small random values instead of exact zeros.
    # def initialize_as_identity(self, epsilon=1e-4):
    #     """
    #     Initializes the adapter to a near-identity state.
    #     Allows for immediate gradient signal while preserving frozen model behavior.
    #     """
    #     # 1. The Extractor & Corrector (MultiheadAttention)
    #     for mha in [self.nw_attn, self.attn_train, self.attn_test]:
    #         # Zero the bias to prevent constant shifts
    #         nn.init.zeros_(mha.out_proj.bias)
    #         # Use a tiny normal distribution for weights
    #         nn.init.normal_(mha.out_proj.weight, std=epsilon)
    #
    #     # 2. The FFN Modulator
    #     # self.ffn[3] is the second Linear layer
    #     nn.init.zeros_(self.ffn[3].bias)
    #     nn.init.normal_(self.ffn[3].weight, std=epsilon)
    #
    #     # 3. The MLP Error (Optional but helpful)
    #     # Priming the error transformation ensures the NW corrector
    #     # starts looking for meaningful distortions immediately.
    #     nn.init.zeros_(self.mlp_err[-1].bias)
    #     nn.init.normal_(self.mlp_err[-1].weight, std=epsilon)


    def forward(self, A, B, C, sep, hp, **kwargs):
        """
        x: Latent representations (T, 3*Batch, d_model)
        hp: PFN-encoded hyperparameter coordinates (T, 3*Batch, d_model)
        """

        device = A.device

        total_batch = hp.shape[1]

        # Since we concatenated A, B, and C in StreamParser:
        R = total_batch // 3

        # --- Extract Encoded Hyperparameter Streams ---
        hp_A = hp[:, :R, :]
        hp_B = hp[:, R: 2 * R, :]
        hp_C = hp[:, 2 * R:, :]

        # --- Split Streams ---
        A_train = A[:sep]
        B_train, B_test, B_belief_A = B[:sep], B[sep:self.seq_len], B[self.seq_len:]
        C_train, C_test, C_belief_A = C[:sep], C[sep:self.seq_len], C[self.seq_len:]

        hp_A_train = hp_A[:sep]
        # this is train and test - this way, we can have a look at B_test's belief on the query
        hp_B_active = hp_B[:self.seq_len]
        hp_C_train = hp_C[:sep]
        hp_C_test = hp_C[sep:self.seq_len]

        # ==========================================
        # STAGE 1: THE NW error propagation (Corrector)
        # ==========================================
        # Calculates local distortion error: A_train - B_test(A_train)
        error = self.mlp_err(self.norm_nw_v(A_train - B_belief_A))
        B_active = B[:self.seq_len]

        q_nw = self.norm_nw_q(hp_B_active)
        k_nw = self.norm_nw_k(hp_A_train)

        corr_out, corr_weights = self.nw_attn(q_nw, k_nw, error)
        B_prime = B_active + corr_out

        # ==========================================
        # STAGE 2: THE EXTRACTOR
        # ==========================================
        k_ext = self.norm_k(hp_B_active)
        v_ext = B_prime

        if self.hp_only_attn:
            q_train, q_test = self.norm_train_q(hp_C_train), self.norm_test_q(hp_C_test)
        else:
            q_train, q_test = self.norm_train_q(C_train), self.norm_test_q(C_test)

        c_train_update, train_weights = self.attn_train(q_train, k_ext, v_ext)
        c_test_update, test_weights = self.attn_test(q_test, k_ext, v_ext)

        c_belief_zero_update = torch.zeros_like(C_belief_A)
        C_update = torch.cat([c_train_update, c_test_update, c_belief_zero_update], dim=0)

        # 3. Apply residual and FFN modulation
        C = C + C_update
        C = C + self.ffn(self.norm_ffn(C))

        ForwardMetaContext.log_stats(
            layer_name=self.address,
            stats_dict=dict(
                corrector_att_scores=corr_weights,
                train_attn_scores=train_weights,
                test_attn_scores=test_weights
            )
        )

        return A, B, C

