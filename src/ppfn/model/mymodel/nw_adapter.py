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


    def forward(self, hp, A, B, C, sep):
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
        hp_B_active = hp_B[
            :self.seq_len]  # this is train and test - this way, we can have a look at B_test's belief on the query
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

        C_update = torch.cat([c_train_update, c_test_update, C_belief_A], dim=0)

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

if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from copy import deepcopy


    # Make sure this matches your actual import path in the real script
    # from src.ppfn.model.refactor.meta_context import ForwardMetaContext

    # ==========================================
    # STRICTLY LINEAR HIGH-DIM WRAPPER
    # ==========================================
    class LinearHighDimWrapper(nn.Module):
        def __init__(self, input_dim=2, d_model=64, n_heads=4, dropout=0.0, seq_len=60):
            super().__init__()
            self.seq_len = seq_len
            self.up_proj = nn.Linear(input_dim, d_model)
            self.hp_proj = nn.Linear(input_dim - 1, d_model)  # hp dim

            # Pass an address so we can find our stats on the bulletin board
            self.adapter = NadarayaWatsonAdapter(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                seq_len=seq_len,
                address="nw_adapter_layer_1"
            )

            self.down_proj = nn.Linear(d_model, input_dim)

        def forward(self, x, single_eval_pos):
            # 1. Project inputs
            h = self.up_proj(x)

            # Explicitly isolate the hyperparameters (the x-coordinates, index 0)
            # We keep the trailing dimension so shape is (T, 3*Batch, 1)
            hp = x[:, :, 0:1]
            hp = self.hp_proj(hp)

            # 2. Split streams for the new adapter signature
            total_batch = h.shape[1]
            R = total_batch // 3
            A = h[:, :R, :]
            B = h[:, R:2 * R, :]
            C = h[:, 2 * R:, :]

            # 3. Apply Adapter
            A_out, B_out, C_out = self.adapter(hp=hp, A=A, B=B, C=C, sep=single_eval_pos)

            # 4. Reassemble and project down
            h_out = torch.cat([A_out, B_out, C_out], dim=1)
            out = self.down_proj(h_out)

            # 5. Retrieve telemetry and clear the board for the next run
            stats = ForwardMetaContext.get_stats()
            attn_weights = stats.get(self.adapter.address, {})
            ForwardMetaContext.clear()

            return out, attn_weights


    # ==========================================
    # NON-LINEAR DYNAMIC DATA GENERATOR
    # ==========================================
    def generate_shared_complex_batch(batch_size, T, sep, D=2):
        # 1. Task A (The Shared Irregular Target)
        t_A = torch.linspace(0, 2 * np.pi, T).view(T, 1, 1).expand(T, batch_size, 1)

        # Complex Base: A sum of two random frequencies
        f1 = torch.rand(1, batch_size, 1) * 2.0 + 1.0
        f2 = torch.rand(1, batch_size, 1) * 3.0 + 2.0

        # The "Unknown": A Gaussian bump that randomly appears ONLY in the test region (x > 3.0)
        bump_center = torch.rand(1, batch_size, 1) * 2.0 + 3.5
        bump_height = torch.rand(1, batch_size, 1) * 2.0 - 1.0
        bump_width = 0.4

        def shared_base_shape(x):
            base = torch.sin(f1 * x) + 0.5 * torch.cos(f2 * x)
            bump = bump_height * torch.exp(-((x - bump_center) ** 2) / (2 * bump_width ** 2))
            return base + bump

        A_data = torch.cat([t_A, shared_base_shape(t_A)], dim=-1)

        # 2. Task B (The Distorted Domain of the Shared Shape)
        t_B = torch.linspace(-0.5, 2 * np.pi + 0.5, T).view(T, 1, 1).expand(T, batch_size, 1)

        # Domain Distortion Parameters (Warping the x and y axes)
        warp = torch.rand(1, batch_size, 1) * 0.8 + 0.2
        curve = torch.rand(1, batch_size, 1) * 0.2 - 0.1

        def task_B_func(x):
            # We apply the distortion to the x-axis BEFORE passing it to the shared shape
            distorted_x = x + warp * torch.sin(x)
            # We apply the y-axis bend AFTER the shared shape
            return shared_base_shape(distorted_x) + curve * (x ** 2)

        B_data = torch.cat([t_B, task_B_func(t_B)], dim=-1)

        # 3. Stream C (Queries)
        C_data_init = torch.zeros(T, batch_size, D)

        # The x-coordinates (hyperparameters) are always fully known
        C_data_init[:, :, 0] = t_A[:, :, 0]

        # Allow the network to see the A_train ground truth
        C_data_init[:sep, :, 1] = A_data[:sep, :, 1]
        # but here we need to prevent peeking, because the test tokens contain the true y in this study!
        C_data_init[sep:, :, 1] = torch.randn(T - sep, batch_size) * 0.1

        # 4. Belief Alignment (Sequence Append)
        t_A_train = t_A[:sep]
        B_belief_A_data = torch.cat([t_A_train, task_B_func(t_A_train)], dim=-1)

        # Fix: Use T instead of self.seq_len
        B_data_corrected = torch.cat([B_data[:T], B_belief_A_data], dim=0)

        # Fix: Extend A and C so torch.cat doesn't crash on dim=0
        A_train_append = A_data[:sep].clone()
        C_train_append = C_data_init[:sep].clone()

        A_data_ext = torch.cat([A_data, A_train_append], dim=0)
        C_data_init_ext = torch.cat([C_data_init, C_train_append], dim=0)

        with torch.no_grad():
            distorted_t_B = t_B + warp * torch.sin(t_B)
            base_B = shared_base_shape(distorted_t_B)

            debug_info = {
                't_B': t_B.squeeze().cpu().numpy(),
                'distorted_t_B': distorted_t_B.squeeze().cpu().numpy(),
                'base_B': base_B.squeeze().cpu().numpy(),
                'task_B': B_data[:, :, 1].squeeze().cpu().numpy()
            }

        return A_data_ext, B_data_corrected, C_data_init_ext, B_belief_A_data, debug_info


    # ==========================================
    # MAIN EXECUTION
    # ==========================================
    def run_dynamic_sanity_check():
        T = 60
        sep = 20
        D = 2
        epochs = 20000
        batch_size = 128

        # Initialize the WRAPPER instead of the raw adapter
        # We give it d_model=64 and 4 attention heads for rich feature extraction
        adapter = LinearHighDimWrapper(input_dim=D, d_model=64, n_heads=4, dropout=0.1, seq_len=T)
        optimizer = optim.Adam(adapter.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print(f"Starting fully dynamic meta-training with High-Dim Latent Space...")
        loss_history = []
        pbar = tqdm(range(epochs), desc="Training Adapter")
        for epoch in pbar:
            optimizer.zero_grad()
            ForwardMetaContext.clear()  # Ensure clean state

            A_data, B_data_corrected, C_data_init, B_belief_A_data, debug_info = generate_shared_complex_batch(
                batch_size, T, sep)
            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)

            out_batch, _ = adapter(x, single_eval_pos=sep)

            C_out = out_batch[:, 2 * batch_size:, :]

            # Calculate loss only on the valid sequence part (ignoring appended belief)
            loss = criterion(C_out[:T], A_data[:T])

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            loss_history.append(current_loss)

            if (epoch + 1) % 200 == 0:
                pbar.set_postfix({"Loss": f"{current_loss:.6f}"})

            if (epoch + 1) % 2000 == 0:
                plot_results(sep, T, adapter, loss_history)

        print("\nTesting on a novel Task A and novel Task B distortion...")


    def plot_results(sep, T, adapter, loss_history):
        with torch.no_grad():
            ForwardMetaContext.clear()
            A_data, B_data_corrected, C_data_init, B_belief_A_data, debug_info = generate_shared_complex_batch(1, T,
                                                                                                               sep)

            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)

            adapter.eval()
            final_batch, attn_weights = adapter(x, single_eval_pos=sep)

            # batch_size is 1 here, so Stream C starts at index 2
            C_final = final_batch[:, 2:, :]

            # Use the exact key from your updated adapter logging
            corrector_attn = attn_weights['corrector_att_scores'].squeeze().cpu().numpy()

            # Data extraction for plotting
            A_plot = A_data[:sep * 2, 0, :].numpy()
            B_plot = B_data_corrected[:sep * 2, 0, :].numpy()
            C_plot = C_final[:sep * 2, 0, :].numpy()
            A_train = A_data[:sep, 0, :].numpy()
            B_belief_plot = B_belief_A_data[:, 0, :].numpy()

            # --- NEW: Calculate the Domain Boundary ---
            # 1. Get the maximum X-coordinate of the A_train anchors
            max_a_train_x = A_train[-1, 0]

            # 2. Find the first index in B_active where the X-coordinate exceeds max_a_train_x
            # np.argmax on a boolean array returns the first True index
            b_cutoff_idx = np.argmax(B_plot[:, 0] > max_a_train_x)

            # Fallback just in case B's domain is completely contained within A's (rare with our generation)
            if b_cutoff_idx == 0 and B_plot[-1, 0] <= max_a_train_x:
                b_cutoff_idx = len(B_plot) - 1

            # ==========================================
            # VISUALIZATIONS (2x2 Grid)
            # ==========================================
            fig, axs = plt.subplots(2, 2, figsize=(20, 14))

            # 1. The Learning Curve (Top Left)
            axs[0, 0].plot(loss_history, color='purple', alpha=0.8)
            axs[0, 0].set_title('Learning Curve (MSE Loss)')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].set_yscale('log')
            axs[0, 0].grid(True)

            # 2. The Alignment Plot (Top Right)
            axs[0, 1].scatter(B_plot[:, 0], B_plot[:, 1], c='gray', alpha=0.5, label='Task B (Novel Distorted)')
            axs[0, 1].plot(A_plot[:, 0], A_plot[:, 1], 'g--', linewidth=2, label='Task A (Ground Truth)')
            axs[0, 1].scatter(A_train[:, 0], A_train[:, 1], c='green', s=100, marker='*', label='A Train (Anchors)')
            axs[0, 1].scatter(B_belief_plot[:, 0], B_belief_plot[:, 1], c='red', s=40, marker='x',
                              label='B Belief of A_train')
            axs[0, 1].scatter(C_plot[:, 0], C_plot[:, 1], c='blue', s=30, label='Stream C (Adapter Output)')
            for i in range(sep):
                axs[0, 1].plot([A_train[i, 0], B_belief_plot[i, 0]], [A_train[i, 1], B_belief_plot[i, 1]], 'r:',
                               alpha=0.4)
            axs[0, 1].set_title('Zero-Shot Alignment on Novel Functions')
            axs[0, 1].legend()
            axs[0, 1].grid(True)

            # 3. GROUND TRUTH DEFORMATION (Bottom Left)
            t_B = debug_info['t_B']
            distorted_t_B = debug_info['distorted_t_B']
            base_B = debug_info['base_B']
            task_B = debug_info['task_B']

            axs[1, 0].plot(distorted_t_B, base_B, 'k--', alpha=0.4, linewidth=2, label='Pristine Manifold (Hidden)')
            axs[1, 0].scatter(t_B, task_B, c='gray', s=20, label='Task B (Observed)')

            # Draw displacement arrows
            for i in range(len(t_B)):
                axs[1, 0].annotate('', xy=(t_B[i], task_B[i]), xytext=(distorted_t_B[i], base_B[i]),
                                   arrowprops=dict(arrowstyle="->", color="orange", alpha=0.6))

            axs[1, 0].set_xlim(axs[0, 1].get_xlim())
            axs[1, 0].set_title('Ground Truth Displacement (Orange = The hidden warp)')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            # 4. The Corrector Attention Heatmap (Bottom Right)
            # Transpose the matrix AND set origin to 'lower' for Cartesian alignment
            im = axs[1, 1].imshow(corrector_attn.T, cmap='viridis', aspect='auto', origin='lower')

            axs[1, 1].set_title('Corrector Attention Distribution')
            axs[1, 1].set_xlabel('Queries: B_active Points (Index 0 to 39)')
            axs[1, 1].set_ylabel('Keys: A_train Anchors (Index 0 to 19)')

            # Draw the Boundary Line
            axs[1, 1].axvline(x=b_cutoff_idx, color='red', linestyle='--', linewidth=2,
                              label='End of A_train Domain (Extrapolation Boundary)')
            axs[1, 1].legend(loc='upper right', framealpha=0.9)
            fig.colorbar(im, ax=axs[1, 1], label='Attention Weight')

            plt.tight_layout()
            plt.show()

            adapter.train()


    run_dynamic_sanity_check()