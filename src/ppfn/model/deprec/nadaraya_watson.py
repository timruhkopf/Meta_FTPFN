import torch
import torch.nn as nn

from typing import Tuple

# class ContinuousPositionalEncoding(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.d_model = d_model
#         # Create a spectrum of frequency bands
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         self.register_buffer('div_term', div_term)
#
#     def forward(self, x):
#         # x shape: (T, Batch, 1)
#         pe = torch.zeros(x.shape[0], x.shape[1], self.d_model, device=x.device)
#         # Apply sine to even indices, cosine to odd indices
#         pe[:, :, 0::2] = torch.sin(x * self.div_term)
#         pe[:, :, 1::2] = torch.cos(x * self.div_term)
#         return pe

class NadarayaWatsonAdapter(nn.Module):
    # FIXME: d_hp needs to be removed, because we should take the pfn's encoded hyperparameters as input to account for variable sized hp spaces.
    def __init__(
            self,
            d_model,
            n_heads,
            dropout=0.1,
            nw_dropout=0.0,
            reuse_attn=True,
            hp_only_attn=True,
            seq_len=1000
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

        self.single_eval_pos = None
        self.hp = None
        self._attn_statistics = None


    def validate_forward_args(self, x, *args, **kwargs) -> Tuple[int, int]:
        single_eval_pos = kwargs.get("single_eval_pos", None)
        if single_eval_pos is None:
            single_eval_pos = self.single_eval_pos
        assert single_eval_pos is not None, "single_eval_pos must be provided"

        hp = kwargs.get("hp", None)
        if hp is None:
            hp = self.hp
        assert hp is not None, "hp (PFN-encoded hyperparameters) must be provided"

        B = x.shape[1]
        assert B % 3 == 0, "Batch size must be multiple of 3"
        return B, single_eval_pos, hp

    def forward(self, x, hp=None, *args, **kwargs):
        """
        x: Latent representations (T, 3*Batch, d_model)
        hp: PFN-encoded hyperparameter coordinates (T, 3*Batch, d_model)
        """
        self._attn_statistics = None # reset attn statistics on each forward pass
        B_dim, sep, hp = self.validate_forward_args(x, *args, **kwargs)
        R = B_dim // 3

        # --- Extract Latent Streams ---
        A = x[:, :R, :].detach()
        B = x[:, R: 2 * R, :].detach()
        C = x[:, 2 * R:, :]
        device = A.device

        # --- Extract Encoded Hyperparameter Streams ---
        hp_A = hp[:, :R, :]
        hp_B = hp[:, R: 2 * R, :]
        hp_C = hp[:, 2 * R:, :]

        # --- Split Streams ---
        A_train = A[:sep]
        B_train, B_test, B_belief_A = B[:sep], B[sep:self.seq_len], B[self.seq_len:]
        C_train, C_test, C_belief_A = C[:sep], C[sep:self.seq_len], C[self.seq_len:]

        hp_A_train = hp_A[:sep]
        hp_B_active = hp_B[:self.seq_len]
        hp_C_train = hp_C[:sep]
        hp_C_test = hp_C[sep:self.seq_len]

        # ==========================================
        # STAGE 1: THE NW error propagation
        # ==========================================
        error = self.mlp_err(self.norm_nw_v(A_train - B_belief_A))
        B_active = B[:self.seq_len]

        # Q and K use the PFN's encoded hyperparameters directly
        q_nw = self.norm_nw_q(hp_B_active)
        k_nw = self.norm_nw_k(hp_A_train)

        corr_out, corr_weights = self.nw_attn(
            q_nw,  # Query: Encoded spatial position of active B points
            k_nw,  # Key: Encoded spatial position of A anchors
            error  # Value: Latent feature distortion
        )

        B_prime = B_active + corr_out

        # ==========================================
        # STAGE 2: THE EXTRACTOR
        # ==========================================
        if self.hp_only_attn:
            # Keys for the extractor are the encoded HPs of B_active
            k_ext = self.norm_k(hp_B_active)
            v_ext = B_prime

            # Queries are the hyperparameters of C
            c_train_update, train_weights = self.attn_train(
                self.norm_train_q(hp_C_train), k_ext, v_ext
            )

            c_test_update, test_weights = self.attn_test(
                self.norm_test_q(hp_C_test), k_ext, v_ext
            )

        else:
            # Keys for the extractor are the encoded HPs of B_active
            k_ext = self.norm_k(hp_B_active)
            v_ext = B_prime

            # Queries use the ACTUAL latent features of C
            c_train_update, train_weights = self.attn_train(
                self.norm_train_q(C_train), k_ext, v_ext
            )

            c_test_update, test_weights = self.attn_test(
                self.norm_test_q(C_test), k_ext, v_ext
            )

        c_dummy_update = torch.zeros_like(C_belief_A).to(device)
        C_update = torch.cat([c_train_update, c_test_update, c_dummy_update], dim=0)

        C = C + C_update
        C = C + self.ffn(self.norm_ffn(C))

        batch = torch.cat([A, B, C], dim=1)

        self._attn_statistics =  {
            "corrector": corr_weights,
            "train_attn_scores": train_weights,
            "test_attn_scores": test_weights
        }
        return batch



if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from copy import deepcopy


    # ==========================================
    # STRICTLY LINEAR HIGH-DIM WRAPPER
    # ==========================================
    class LinearHighDimWrapper(nn.Module):
        def __init__(self, input_dim=2, d_model=64, n_heads=4,  dropout=0.0, seq_len=60):
            super().__init__()
            self.seq_len = seq_len
            self.up_proj = nn.Linear(input_dim, d_model)
            self.hp_proj = nn.Linear(input_dim -1, d_model) # hp dim
            self.adapter = NadarayaWatsonAdapter(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )

            self.down_proj = nn.Linear(d_model, input_dim)

        def forward(self, x, single_eval_pos):
            h = self.up_proj(x)

            # Explicitly isolate the hyperparameters (the x-coordinates, index 0)
            # We keep the trailing dimension so shape is (T, 3*Batch, 1)
            hp = x[:, :, 0:1]
            hp = self.hp_proj(hp)

            h_out, attn_weights = self.adapter(h, hp=hp, single_eval_pos=single_eval_pos)
            out = self.down_proj(h_out)
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
        bump_height = torch.rand(1, batch_size, 1) * 2.0 - 1.0  # Can go up or down
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

        # Force the network to figure out the test region from Task B
        C_data_init[sep:, :, 1] = torch.randn(T - sep, batch_size) * 0.1 #

        # 4. Belief Alignment
        # 4. Belief Alignment
        t_A_train = t_A[:sep]
        B_belief_A_data = torch.cat([t_A_train, task_B_func(t_A_train)], dim=-1)

        B_data_corrected = torch.cat([B_data[:self.seq_len], B_belief_A_data], dim=0)

        # --- NEW: Extract the exact ground truth distortion mapping for visualization ---
        with torch.no_grad():
            distorted_t_B = t_B + warp * torch.sin(t_B)  # The true X accordion
            base_B = shared_base_shape(distorted_t_B)  # The pure feature without the Y-bend

            debug_info = {
                't_B': t_B.squeeze().cpu().numpy(),
                'distorted_t_B': distorted_t_B.squeeze().cpu().numpy(),
                'base_B': base_B.squeeze().cpu().numpy(),
                'task_B': B_data[:, :, 1].squeeze().cpu().numpy()
            }

        return A_data, B_data_corrected, C_data_init, B_belief_A_data, debug_info

    # ==========================================
    # MAIN EXECUTION
    # ==========================================
    def run_dynamic_sanity_check():
        T = 60
        sep = 20
        D = 2
        epochs = 20000
        batch_size = 128  # We can handle larger batches now

        # Initialize the WRAPPER instead of the raw adapter
        # We give it d_model=64 and 4 attention heads for rich feature extraction
        adapter = LinearHighDimWrapper(input_dim=D, d_model=64, n_heads=4, dropout=0.1)

        # Notice: NO bypass_layernorms() here!

        optimizer = optim.Adam(adapter.parameters(), lr=0.001)  # Slightly lower LR for deep network
        criterion = nn.MSELoss()

        print(f"Starting fully dynamic meta-training with High-Dim Latent Space...")
        loss_history = []
        pbar = tqdm(range(epochs), desc="Training Adapter")
        for epoch in pbar:
            optimizer.zero_grad()

            A_data, B_data_corrected, C_data_init,  B_belief_A_data, debug_info  = generate_shared_complex_batch(batch_size, T, sep)

            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)

            # Forward pass through the wrapper
            out_batch = adapter(x, single_eval_pos=sep)

            # The indexing remains exactly the same
            C_out = out_batch[:, 2 * batch_size:, :]

            loss = criterion(C_out[:sep * 2], A_data[:sep * 2])

            loss.backward()
            optimizer.step()

            # Record the loss
            current_loss = loss.item()
            loss_history.append(current_loss)

            if (epoch + 1) % 200 == 0:
                # update pbar with current loss
                pbar.set_postfix({"Loss": f"{current_loss:.6f}"})


            if ( epoch+1 )% 2000 == 0:
                plot_results(sep, T, adapter, loss_history)

        # ==========================================
        # INFERENCE ON A NOVEL FUNCTION & DISTORTION
        # ==========================================
        print("\nTesting on a novel Task A and novel Task B distortion...")


    def plot_results(sep, T, adapter, loss_history):
        with torch.no_grad():
            A_data, B_data_corrected, C_data_init, B_belief_A_data, debug_info = generate_shared_complex_batch(1, T,
                                                                                                               sep)

            x = torch.cat([A_data, B_data_corrected, C_data_init], dim=1)
            # Using the continuous positional encodings for hp
            hp = x[:, :, 0:1] # adatper handles can extract the hyperparameters internally
            adapter.eval()
            final_batch = adapter(x, single_eval_pos=sep)
            attn_weights = adapter._attn_statistics

            C_final = final_batch[:, 2:, :]
            corrector_attn = attn_weights['corrector'].squeeze().cpu().numpy()

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