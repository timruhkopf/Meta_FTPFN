import torch
from torch.utils.data import IterableDataset
import numpy as np
import matplotlib.pyplot as plt




class InfiniteHarmonicsStream(IterableDataset):
    """
    The "Hidden Harmonic Mixture" Prior with Negative Transfer Injection.
    Refactored to match the Tri-Stream Pipeline expectations.
    """

    def __init__(self, batch_size=32, n_A=10, n_B=50, n_test=200, x_range=(-5, 5),
                 num_components=4, noise_std=0.05, share_unrelated=0.2, scale=True, shift=True, warp=True):
        super().__init__()
        self.batch_size = batch_size
        self.n_A = n_A
        self.n_B = n_B
        self.n_test = n_test
        self.min_x, self.max_x = x_range
        self.num_components = num_components
        self.noise_std = noise_std
        self.share_unrelated = share_unrelated

        self.scale = scale
        self.shift = shift
        self.warp = warp

    @staticmethod
    def _apply_spatial_warp(X, w_amp, w_freq, w_phase):
        """Applies mild non-linear spatial warping to coordinates."""
        w_amp_ext = w_amp.unsqueeze(0)
        w_freq_ext = w_freq.unsqueeze(0)
        w_phase_ext = w_phase.unsqueeze(0)
        return w_amp_ext * torch.sin(2 * torch.pi * w_freq_ext * X + w_phase_ext)

    @staticmethod
    def _eval_function(X, amps, freqs, phases):
        """Evaluates the sum of sinusoids based on explicit parameters."""
        X_ext = X.unsqueeze(0)
        a_ext = amps.unsqueeze(1)
        f_ext = freqs.unsqueeze(1)
        p_ext = phases.unsqueeze(1)
        terms = a_ext * torch.sin(2 * torch.pi * f_ext * X_ext + p_ext)
        return terms.sum(dim=0)

    def _generate_blueprints(self):
        """Generates the base sinusoidal parameters for Tasks A and B."""
        B = self.batch_size
        K = self.num_components

        # Task A Blueprint
        amps_A = torch.empty(K, B).uniform_(0.5, 2.0)
        freqs_A = torch.empty(K, B).uniform_(0.1, 0.5)
        phases_A = torch.empty(K, B).uniform_(0, 2 * torch.pi)

        # Task B Blueprint (Clone A, then modify unrelated traps)
        amps_B, freqs_B, phases_B = amps_A.clone(), freqs_A.clone(), phases_A.clone()

        num_unrelated = int(B * self.share_unrelated)
        is_unrelated = torch.zeros(B, dtype=torch.bool)

        if num_unrelated > 0:
            is_unrelated[-num_unrelated:] = True
            amps_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.5, 2.0)
            freqs_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.1, 0.5)
            phases_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0, 2 * torch.pi)

        params_A = (amps_A, freqs_A, phases_A)
        params_B = (amps_B, freqs_B, phases_B)

        return params_A, params_B, is_unrelated

    def _generate_transformations(self):
        """Generates affine and spatial warping transformations."""
        B = self.batch_size

        # Affine Shifts
        if self.shift:
            v_shift_A = torch.empty(B).uniform_(-2.0, 2.0)
            h_shift_A = torch.empty(B).uniform_(-1.5, 1.5)
        else:
            v_shift_A, h_shift_A = torch.zeros(B), torch.zeros(B)

        # Scale
        scale_A = torch.empty(B).uniform_(0.3, 1.3) if self.scale else torch.ones(B)

        # Spatial Warp
        if self.warp:
            warp_amp_A = torch.empty(B).uniform_(0.0, 0.7)
            warp_freq_A = torch.empty(B).uniform_(0.0, 0.2)
            warp_phase_A = torch.empty(B).uniform_(0, 2 * torch.pi)
        else:
            warp_amp_A, warp_freq_A, warp_phase_A = torch.zeros(B), torch.zeros(B), torch.zeros(B)

        shifts = (v_shift_A, h_shift_A)
        warps = (warp_amp_A, warp_freq_A, warp_phase_A)

        return shifts, scale_A, warps

    def _sample_x_coordinates(self):
        """Samples X coordinates for Train and Test sets."""
        B = self.batch_size
        # FIXME: B's domain is elongated here to ensure, we will always have a valid target for the
        #  test point projection. In reality, this will not always be the case -- we may want to learn a tokenwise valve
        X_train_B, _ = torch.sort(torch.empty(self.n_B, B).uniform_(self.min_x -2, self.max_x+2), dim=0)
        X_train_A, _ = torch.sort(torch.empty(self.n_A, B).uniform_(self.min_x, self.max_x), dim=0)

        # FIX: Removed .unsqueeze(1) here to match the shape of the train set (Seq, Batch)
        X_test_B = torch.empty(self.n_test, B).uniform_(self.min_x-2, self.max_x+2)
        X_test_A = torch.empty(self.n_test, B).uniform_(self.min_x, self.max_x)

        return X_train_A, X_test_A, X_train_B, X_test_B

    def _pad_task_a(self, X_train_A, Y_train_A):
        """Pads Task A to match Task B's sequence length."""
        B = self.batch_size
        pad_size = self.n_B - self.n_A

        if pad_size > 0:
            # Note: Changed from 0. to float('nan') so the isnan check below actually works
            pad = torch.full((pad_size, B), float('nan'))
            X_train_A_padded = torch.cat([X_train_A, pad], dim=0)
            Y_train_A_padded = torch.cat([Y_train_A, pad], dim=0)
        else:
            X_train_A_padded = X_train_A
            Y_train_A_padded = Y_train_A

        padding_mask_A = torch.isnan(X_train_A_padded).transpose(1, 0)

        # Replace NaNs with 0.0 after mask creation to prevent gradient issues
        X_train_A_padded = torch.nan_to_num(X_train_A_padded, nan=0.0)
        Y_train_A_padded = torch.nan_to_num(Y_train_A_padded, nan=0.0)

        return X_train_A_padded, Y_train_A_padded, padding_mask_A

    def warp_and_evaluate(self, X, params, shifts, scale, warps):
        v_shift, h_shift = shifts
        X_warped = X - h_shift + self._apply_spatial_warp(X, *warps)
        Y = scale * self._eval_function(X_warped, *params) + v_shift

        Y +=  torch.randn_like(X_warped) * self.noise_std
        return X_warped, Y

    def _sample_batch(self):
        """
        Refactored orchestrator: A and B_in_A share the same functional curve.
        B is the distorted observation of that curve.
        """
        # 1. Generate underlying truth parameters for the "Canonical" task A
        params_A, params_B, is_unrelated = self._generate_blueprints()
        shifts, scale_A, warps = self._generate_transformations()

        # 2. Sample coordinates
        # X_train_A: Points in clean domain
        # X_train_B: Points in distorted domain
        X_train_A, X_test_A, X_train_B_in_A, X_test_B_in_A = self._sample_x_coordinates()

        # --- THE TRUTH (Canonical Domain A) ---
        # A is the clean function evaluated on clean coordinates
        Y_train_A = self._eval_function(X_train_A, *params_A) + torch.randn_like(X_train_A) * self.noise_std
        Y_test_A = self._eval_function(X_test_A, *params_A)

        # B in A
        Y_train_B_in_A = self._eval_function(X_train_B_in_A, *params_B) + torch.randn_like(X_train_B_in_A) * self.noise_std
        Y_test_B_in_A = self._eval_function(X_test_B_in_A, *params_B)


        # --- THE TARGET (Domain B mapped into A's curve) ---
        # B_in_A: We take the source coordinates (X_train_B) and apply the
        # forward warp defined by this batch's params. This creates the
        # "Target" state that the flow matching model must reach.
        X_train_B, Y_train_B = self.warp_and_evaluate(X_train_B_in_A, params_B, shifts, scale_A, warps)
        X_test_B, Y_test_B = self.warp_and_evaluate(X_test_B_in_A, params_B, shifts, scale_A, warps)

        # --- THE DISTORTED OBSERVATION (Domain B) ---
        # B is the raw observation. We evaluate the harmonic function at the
        # warped locations to create the input for the model.
        # We use the same warp as B_in_A to ensure the model learns the inverse path.

        # 3. Pad Task A to match Task B shapes
        X_train_A_pad, Y_train_A_pad, padding_mask_A = self._pad_task_a(X_train_A, Y_train_A)

        # 4. Compile and return dictionary
        return {
            # FIXME: is_unrelated as flag. add padding mask
            'params': {
                'params_A': params_A, 'shifts': shifts, 'scale_A': scale_A, 'warps': warps
            },
            'train': {
                'X_B': X_train_B.unsqueeze(-1), 'Y_B': Y_train_B.unsqueeze(-1),
                'X_A': X_train_A_pad.unsqueeze(-1), 'Y_A': Y_train_A_pad.unsqueeze(-1),
                # This is the target for Flow Matching:
                'X_B_in_A': X_train_B_in_A.unsqueeze(-1), 'Y_B_in_A': Y_train_B_in_A.unsqueeze(-1),
                'padding_mask_A': padding_mask_A,
            },
            'test': {
                'X_B': X_test_B.unsqueeze(-1), 'Y_B': Y_test_B.unsqueeze(-1),
                'X_A': X_test_A.unsqueeze(-1), 'Y_A': Y_test_A.unsqueeze(-1),
                'X_B_in_A': X_test_B_in_A.unsqueeze(-1), 'Y_B_in_A': Y_test_B_in_A.unsqueeze(-1),
            }
        }
    def __iter__(self):
        while True:
            yield self._sample_batch()


import torch
import numpy as np
import matplotlib.pyplot as plt


class HeatmapVisualizer:

    @staticmethod
    def _to_np(val):
        """Safely converts a PyTorch tensor to a Numpy array."""
        return val.detach().cpu().numpy() if torch.is_tensor(val) else val

    @staticmethod
    def _get_param(p_dict, key, batch_idx, default=0.0):
        """Safely extracts a parameter from the batch dictionary, defaulting if missing."""
        if key in p_dict:
            val = p_dict[key]
            if val is None:
                return default
            if len(val.shape) > 1:  # Handle 2D arrays like amps/freqs [K, B]
                return HeatmapVisualizer._to_np(val[:, batch_idx])
            return HeatmapVisualizer._to_np(val[batch_idx])  # Handle 1D arrays [B]
        return default

    @staticmethod
    def _compute_logits(model, batch_data, logits_A, logits_B, logits_C):
        """Runs model inference if a model is provided and logits are missing."""
        if model is not None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                # Adapt this unpacking if your model returns a dictionary now
                logits_A, logits_B, logits_C = model(batch_data)
            if was_training:
                model.train()
        return logits_A, logits_B, logits_C

    @staticmethod
    def _get_binned_percentiles(probs_np, borders_np, percentiles=(0.025, 0.5, 0.975)):
        """Calculates continuous percentiles via CDF interpolation between bin edges."""
        cdf = np.cumsum(probs_np, axis=-1)
        cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=-1)
        results = {pct: np.zeros(probs_np.shape[0]) for pct in percentiles}

        for i in range(probs_np.shape[0]):
            for pct in percentiles:
                idx = np.searchsorted(cdf[i], pct)
                if idx == 0:
                    results[pct][i] = borders_np[0]
                elif idx >= len(borders_np):
                    results[pct][i] = borders_np[-1]
                else:
                    p_low, p_high = cdf[i, idx - 1], cdf[i, idx]
                    b_low, b_high = borders_np[idx - 1], borders_np[idx]
                    if p_high > p_low:
                        fraction = (pct - p_low) / (p_high - p_low)
                        results[pct][i] = b_low + fraction * (b_high - b_low)
                    else:
                        results[pct][i] = b_low
        return results

    @staticmethod
    def _build_ground_truth_curve(X, p_dict, batch_idx, stream_type='A'):
        """Reconstructs true harmonic lines. Safely handles missing/renamed parameters."""
        params_A = p_dict.get('params_A', {})

        # --- 1. Robust Extraction ---
        # Look for 'amps', fallback to 'amps_A', and default to an array (not a float)
        amps = HeatmapVisualizer._get_param(params_A, 'amps', batch_idx, default=np.array([0.0]))
        if np.array_equal(amps, np.array([0.0])):
            amps = HeatmapVisualizer._get_param(params_A, 'amps_A', batch_idx, default=np.array([0.0]))

        freqs = HeatmapVisualizer._get_param(params_A, 'freqs', batch_idx, default=np.array([0.0]))
        if np.array_equal(freqs, np.array([0.0])):
            freqs = HeatmapVisualizer._get_param(params_A, 'freqs_A', batch_idx, default=np.array([0.0]))

        phases = HeatmapVisualizer._get_param(params_A, 'phases', batch_idx, default=np.array([0.0]))
        if np.array_equal(phases, np.array([0.0])):
            phases = HeatmapVisualizer._get_param(params_A, 'phases_A', batch_idx, default=np.array([0.0]))

        # Force everything to be at least a 1D array to guarantee [:, None] won't crash
        amps = np.atleast_1d(amps)
        freqs = np.atleast_1d(freqs)
        phases = np.atleast_1d(phases)

        # --- 2. Transform Extraction ---
        # Base transformations (Stream A gets standard, B gets the shifts/scales applied)
        scale = HeatmapVisualizer._get_param(p_dict, 'scale_A', batch_idx, default=1.0) if stream_type == 'B' else 1.0

        shifts = p_dict.get('shifts', {})
        h_shift = HeatmapVisualizer._get_param(shifts, 'h_shift', batch_idx, default=0.0) if stream_type == 'B' else 0.0
        v_shift = HeatmapVisualizer._get_param(shifts, 'v_shift', batch_idx, default=0.0) if stream_type == 'B' else 0.0

        # Apply Global Shift
        X_warped = X - h_shift

        # --- 3. Compute Terms ---
        terms = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * X_warped[None, :] + phases[:, None])
        return scale * np.sum(terms, axis=0) + v_shift
    @staticmethod
    def _plot_single_axis(ax, stream_name, batch_idx, batch_data, logits, y_dense_A, y_dense_B,
                          x_dense, borders_np, centers):
        """Handles plotting the heatmap, lines, and scatter points for a single subplot."""

        # Stream A and C both pull from dataset 'A'
        data_key = 'A' if stream_name in ['A', 'C'] else 'B'
        is_ac_stream = stream_name in ['A', 'C']

        # 1. Plot Heatmap & Percentiles (if logits exist)
        if logits is not None:
            X_test_np = HeatmapVisualizer._to_np(batch_data['test'][f'X_{data_key}'][:, batch_idx]).flatten()
            probs = torch.softmax(logits[:, batch_idx, :], dim=-1).detach().cpu().numpy()

            sort_idx = np.argsort(X_test_np)
            X_test_np = X_test_np[sort_idx]
            probs = probs[sort_idx]

            percentiles = HeatmapVisualizer._get_binned_percentiles(probs, borders_np)

            ax.pcolormesh(X_test_np, centers, probs.T, cmap='viridis', shading='nearest', alpha=0.9, rasterized=True)
            ax.plot(X_test_np, percentiles[0.5], color='orange', linestyle='-', linewidth=1.0, zorder=10)
            ax.plot(X_test_np, percentiles[0.025], color='orange', linestyle=':', linewidth=1.0, zorder=10)
            ax.plot(X_test_np, percentiles[0.975], color='orange', linestyle=':', linewidth=1.0, zorder=10)

        # 2. Plot Ground Truth Lines
        true_y = y_dense_A if is_ac_stream else y_dense_B
        line_color, line_style, line_alpha = ('white', '-', 0.9) if is_ac_stream else ('red', '--', 0.5)
        ax.plot(x_dense, true_y, color=line_color, linestyle=line_style, linewidth=1.0, alpha=line_alpha, zorder=5)

        # 3. Scatter Training Points
        X_train = HeatmapVisualizer._to_np(batch_data['train'][f'X_{data_key}'][:, batch_idx]).flatten()
        Y_train = HeatmapVisualizer._to_np(batch_data['train'][f'Y_{data_key}'][:, batch_idx]).flatten()

        # Handle asymmetric padding masks cleanly
        mask_key = f'padding_mask_{data_key}'
        if mask_key in batch_data['train']:
            pad_mask_np = HeatmapVisualizer._to_np(batch_data['train'][mask_key])

            # Align mask dynamically based on sequence length
            seq_len = len(X_train)
            if pad_mask_np.shape[0] == seq_len:
                pad_mask = pad_mask_np[:, batch_idx]  # [Seq, Batch]
            else:
                pad_mask = pad_mask_np[batch_idx, :seq_len]  # [Batch, Seq]
            valid = ~pad_mask.astype(bool)
        else:
            # If no mask exists (e.g., Stream B), assume all points are valid
            valid = np.ones(len(X_train), dtype=bool)

        if is_ac_stream:
            ax.scatter(X_train[valid], Y_train[valid], c='white', s=50, edgecolors='black', linewidth=.5, zorder=20)
        else:
            ax.scatter(X_train[valid], Y_train[valid], c='red', s=40, marker='x', alpha=0.8, zorder=20)

        ax.set_ylim(borders_np[0] - 1.5, borders_np[-1] + 1.5)
        ax.grid(False)

    @classmethod
    def save_heatmaps(cls, fig, batch_data, borders, save_path, model=None, logits_A=None, logits_B=None, logits_C=None,
                      x_range=(-5, 5), plot=False):
        """Main orchestrator for building and saving the heatmap grid."""

        logits_A, logits_B, logits_C = cls._compute_logits(model, batch_data, logits_A, logits_B, logits_C)
        min_x, max_x = x_range
        p = batch_data['params']

        # We simply visualize the first two items in the batch now
        batch_size = batch_data['train']['X_A'].shape[1]
        idx_list = [0, 1] if batch_size > 1 else [0]

        # Initialize Plot (3 rows, N columns)
        axes = fig.subplots(3, len(idx_list), sharex=True, sharey=True, gridspec_kw={'hspace': 0.25, 'wspace': 0.1})

        # Handle 1D axes array if batch size is 1
        if len(idx_list) == 1:
            axes = np.expand_dims(axes, axis=1)

        borders_np = cls._to_np(borders)
        centers = (borders_np[:-1] + borders_np[1:]) / 2.0
        x_dense = np.linspace(min_x, max_x, 1000)

        for col_idx, batch_idx in enumerate(idx_list):
            y_dense_A = cls._build_ground_truth_curve(x_dense, p, batch_idx, stream_type='A')
            y_dense_B = cls._build_ground_truth_curve(x_dense, p, batch_idx, stream_type='B')

            for row_idx, stream_name in enumerate(['A', 'B', 'C']):
                ax = axes[row_idx, col_idx]
                curr_logits = {'A': logits_A, 'B': logits_B, 'C': logits_C}[stream_name]

                cls._plot_single_axis(
                    ax, stream_name, batch_idx, batch_data, curr_logits,
                    y_dense_A, y_dense_B, x_dense, borders_np, centers
                )

                if row_idx == 0:
                    ax.set_title(f"BATCH ITEM {batch_idx}", fontweight='bold', fontsize=11, pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Stream {stream_name}", fontsize=10, fontweight='bold')

        # Set x-labels on the bottom row
        for col_idx in range(len(idx_list)):
            axes[2, col_idx].set_xlabel("x-coordinate")

        plt.tight_layout()
        if plot:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)



class HeatmapVisualizerOLD:
    # (Assuming this sits inside a class based on your @staticmethod decorator)

    @staticmethod
    def _to_np(val):
        """Safely converts a PyTorch tensor to a Numpy array."""
        return val.detach().cpu().numpy() if torch.is_tensor(val) else val

    @staticmethod
    def _get_param(p_dict, key, batch_idx, default=0.0):
        """Safely extracts a parameter from the batch dictionary, defaulting if missing."""
        if key in p_dict:
            val = p_dict[key]
            if len(val.shape) > 1:  # Handle 2D arrays like amps/freqs [K, B]
                return HeatmapVisualizer._to_np(val[:, batch_idx])
            return HeatmapVisualizer._to_np(val[batch_idx])  # Handle 1D arrays [B]
        return default

    @staticmethod
    def _compute_logits(model, batch_data, logits_A, logits_B, logits_C):
        """Runs model inference if a model is provided and logits are missing."""
        if model is not None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                logits_A, logits_B, logits_C = model(batch_data)
            if was_training:
                model.train()
        return logits_A, logits_B, logits_C

    @staticmethod
    def _get_binned_percentiles(probs_np, borders_np, percentiles=(0.025, 0.5, 0.975)):
        """Calculates continuous percentiles via CDF interpolation between bin edges."""
        cdf = np.cumsum(probs_np, axis=-1)
        # Pad with 0.0 to align exactly with the N+1 borders array
        cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=-1)
        results = {pct: np.zeros(probs_np.shape[0]) for pct in percentiles}

        for i in range(probs_np.shape[0]):
            for pct in percentiles:
                idx = np.searchsorted(cdf[i], pct)
                if idx == 0:
                    results[pct][i] = borders_np[0]
                elif idx >= len(borders_np):
                    results[pct][i] = borders_np[-1]
                else:
                    p_low, p_high = cdf[i, idx - 1], cdf[i, idx]
                    b_low, b_high = borders_np[idx - 1], borders_np[idx]
                    if p_high > p_low:
                        fraction = (pct - p_low) / (p_high - p_low)
                        results[pct][i] = b_low + fraction * (b_high - b_low)
                    else:
                        results[pct][i] = b_low
        return results

    @staticmethod
    def _build_ground_truth_curve(X, p_dict, batch_idx, stream_type='A'):
        """Reconstructs true harmonic lines with spatial/phase warps based on stream type."""
        amps = HeatmapVisualizer._get_param(p_dict, f'amps_{stream_type}', batch_idx)
        freqs = HeatmapVisualizer._get_param(p_dict, f'freqs_{stream_type}', batch_idx)
        phases = HeatmapVisualizer._get_param(p_dict, f'phases_{stream_type}', batch_idx)

        # Default transformations to 0/1 for Stream B, pull actuals for Stream A
        scale = HeatmapVisualizer._get_param(p_dict, f'scale_{stream_type}', batch_idx,
                                             default=1.0) if stream_type == 'A' else 1.0
        h_shift = HeatmapVisualizer._get_param(p_dict, f'h_shift_{stream_type}', batch_idx,
                                               default=0.0) if stream_type == 'A' else 0.0
        v_shift = HeatmapVisualizer._get_param(p_dict, f'v_shift_{stream_type}', batch_idx,
                                               default=0.0) if stream_type == 'A' else 0.0

        w_amp = HeatmapVisualizer._get_param(p_dict, 'warp_amp_A', batch_idx) if stream_type == 'A' else 0.0
        w_freq = HeatmapVisualizer._get_param(p_dict, 'warp_freq_A', batch_idx) if stream_type == 'A' else 0.0
        w_phase = HeatmapVisualizer._get_param(p_dict, 'warp_phase_A', batch_idx) if stream_type == 'A' else 0.0

        p_w_amp = HeatmapVisualizer._get_param(p_dict, 'p_warp_amp', batch_idx) if stream_type == 'A' else 0.0
        p_w_freq = HeatmapVisualizer._get_param(p_dict, 'p_warp_freq', batch_idx) if stream_type == 'A' else 0.0
        p_w_phase = HeatmapVisualizer._get_param(p_dict, 'p_warp_phase', batch_idx) if stream_type == 'A' else 0.0

        # Apply Global Shift
        X_warped = X - h_shift

        # 1. Spatial Sinusoidal Wobble
        if w_amp > 0: X_warped += w_amp * np.sin(2 * np.pi * w_freq * X + w_phase)
        # 2. Phase Warp
        if p_w_amp > 0: X_warped += p_w_amp * np.sin(2 * np.pi * p_w_freq * X + p_w_phase)

        # Evaluate harmonic function
        terms = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * X_warped[None, :] + phases[:, None])
        return scale * np.sum(terms, axis=0) + v_shift

    @staticmethod
    def _plot_single_axis(ax, stream_name, batch_idx, batch_data, logits, y_dense_A, y_dense_B,
                          x_dense, borders_np, centers, is_trap):
        """Handles plotting the heatmap, lines, and scatter points for a single subplot."""
        if is_trap:
            ax.set_facecolor('#fffafa')

        # EXACT MATCH GUARANTEE: Stream A and C both pull from dataset 'A'
        data_key = 'A' if stream_name in ['A', 'C'] else 'B'
        is_ac_stream = stream_name in ['A', 'C']

        # 1. Plot Heatmap & Percentiles (if logits exist)
        # 1. Plot Heatmap & Percentiles (if logits exist)
        if logits is not None:
            # Get X and flatten the trailing (1,) dimension to make it strictly 1D
            X_test_np = HeatmapVisualizer._to_np(batch_data['test'][f'X_{data_key}'][:, batch_idx]).flatten()
            probs = torch.softmax(logits[:, batch_idx, :], dim=-1).detach().cpu().numpy()

            # CRITICAL: pcolormesh requires sorted X coordinates.
            # Since test points are uniformly sampled, we must sort them for visualization.
            sort_idx = np.argsort(X_test_np)
            X_test_np = X_test_np[sort_idx]
            probs = probs[sort_idx]

            # Get smooth interpolated percentiles
            percentiles = HeatmapVisualizer._get_binned_percentiles(probs, borders_np)

            # Plot Smooth Heatmap
            ax.pcolormesh(X_test_np, centers, probs.T, cmap='viridis', shading='nearest', alpha=0.9, rasterized=True)
            ax.plot(X_test_np, percentiles[0.5], color='orange', linestyle='-', linewidth=1.0, zorder=10)
            ax.plot(X_test_np, percentiles[0.025], color='orange', linestyle=':', linewidth=1.0, zorder=10)
            ax.plot(X_test_np, percentiles[0.975], color='orange', linestyle=':', linewidth=1.0, zorder=10)

        # 2. Plot Ground Truth Lines
        true_y = y_dense_A if is_ac_stream else y_dense_B
        line_color, line_style, line_alpha = ('white', '-', 0.9) if is_ac_stream else ('red', '--', 0.5)
        ax.plot(x_dense, true_y, color=line_color, linestyle=line_style, linewidth=1.0, alpha=line_alpha, zorder=5)

        # 3. Scatter Training Points
        X_train = HeatmapVisualizer._to_np(batch_data['train'][f'X_{data_key}'][:, batch_idx]).flatten()
        Y_train = HeatmapVisualizer._to_np(batch_data['train'][f'Y_{data_key}'][:, batch_idx]).flatten()

        pad_mask_np = HeatmapVisualizer._to_np(batch_data['train'][f'padding_mask_{data_key}'])
        sep = batch_data['train']['X_B'].shape[0]
        # Bulletproof alignment: Force the mask to yield a vector exactly the length of X_train
        if pad_mask_np.shape[0] == len(X_train) and pad_mask_np.shape[1] != len(X_train):
            pad_mask = pad_mask_np[:sep, batch_idx]  # Mask is [Seq, Batch]
        else:
            pad_mask = pad_mask_np[batch_idx, :sep]  # Mask is [Batch, Seq] (Standard dataloader behavior)

        valid = ~pad_mask.astype(bool)

        if is_ac_stream:
            ax.scatter(X_train[valid], Y_train[valid], c='white', s=50, edgecolors='black', linewidth=.5, zorder=20)
        else:
            ax.scatter(X_train[valid], Y_train[valid], c='red', s=40, marker='x', alpha=0.8, zorder=20)

        ax.set_ylim(borders_np[0] - 1.5, borders_np[-1] + 1.5)
        ax.grid(False)

    @classmethod
    def save_heatmaps(cls, fig, batch_data, borders, save_path, model=None, logits_A=None, logits_B=None, logits_C=None,
                      x_range=(-5, 5), plot=False):
        """Main orchestrator for building and saving the heatmap grid."""

        # 1. Setup Data & Logits
        logits_A, logits_B, logits_C = cls._compute_logits(model, batch_data, logits_A, logits_B, logits_C)
        min_x, max_x = x_range
        p = batch_data['params']

        # 2. Resolve Columns (Related vs Unrelated)
        is_unrelated_flat = cls._to_np(p['is_unrelated']).astype(bool).flatten()
        related_indices = np.where(~is_unrelated_flat)[0]
        unrelated_indices = np.where(is_unrelated_flat)[0]

        idx_list = [
            related_indices[0] if len(related_indices) > 0 else 0,
            unrelated_indices[0] if len(unrelated_indices) > 0 else (
                related_indices[1] if len(related_indices) > 1 else 0)
        ]

        # 3. Initialize Plot
        axes = fig.subplots(3, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0.25, 'wspace': 0.1})
        borders_np = cls._to_np(borders)
        centers = (borders_np[:-1] + borders_np[1:]) / 2.0
        x_dense = np.linspace(min_x, max_x, 1000)

        # 4. Iterate over Columns and Rows
        for col_idx, batch_idx in enumerate(idx_list):
            is_trap = is_unrelated_flat[batch_idx]

            # Pre-calculate ground truth curves for this column
            y_dense_A = cls._build_ground_truth_curve(x_dense, p, batch_idx, stream_type='A')
            y_dense_B = cls._build_ground_truth_curve(x_dense, p, batch_idx, stream_type='B')

            for row_idx, stream_name in enumerate(['A', 'B', 'C']):
                ax = axes[row_idx, col_idx]
                curr_logits = {'A': logits_A, 'B': logits_B, 'C': logits_C}[stream_name]

                # Delegate plotting logic to helper
                cls._plot_single_axis(
                    ax, stream_name, batch_idx, batch_data, curr_logits,
                    y_dense_A, y_dense_B, x_dense, borders_np, centers, is_trap
                )

                # Aesthetics
                if row_idx == 0:
                    ax.set_title("UNRELATED TRAP" if is_trap else "RELATED CONTEXT", fontweight='bold', fontsize=11,
                                 pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Stream {stream_name}", fontsize=10, fontweight='bold')

        axes[2, 0].set_xlabel("x-coordinate")
        axes[2, 1].set_xlabel("x-coordinate")

        # 5. Finalize
        plt.tight_layout()
        if plot:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)


if __name__ == '__main__':
    # 1. Setup Dataset with enough batch size to guarantee a Trap (10 * 0.2 = 2 traps)
    dataset = InfiniteHarmonicsStream(batch_size=10, n_A=10, n_B=50, n_test=200,
                                      num_components=1, noise_std=0.05, share_unrelated=0.0)
    batch_data = next(iter(dataset))

    import matplotlib.pyplot as plt


    def visualize_batch(batch_data, b_idx=0):
        # Extract data from the batch
        # Note: Using .squeeze(-1) to flatten the [Seq, Batch, 1] structure
        X_A = batch_data['train']['X_A'][:, b_idx].squeeze().numpy()
        Y_A = batch_data['train']['Y_A'][:, b_idx].squeeze().numpy()

        X_B = batch_data['train']['X_B'][:, b_idx].squeeze().numpy()
        Y_B = batch_data['train']['Y_B'][:, b_idx].squeeze().numpy()

        X_B_in_A = batch_data['train']['X_B_in_A'][:, b_idx].squeeze().numpy()
        Y_B_in_A = batch_data['train']['Y_B_in_A'][:, b_idx].squeeze().numpy()

        # Create a dense X range for the Ground Truth line
        x_dense = np.linspace(-5, 5, 500)

        # Use the helper to get the GT curve for Task B (the 'Truth')
        # We use params_B and the specific warps/shifts for this batch
        params_B = batch_data['params']['params_A']  # This is a tuple (amps, freqs, phases)
        # Extracting these require indexing into the batch dimension
        amps = params_B[0][:, b_idx]
        freqs = params_B[1][:, b_idx]
        phases = params_B[2][:, b_idx]

        # We reconstruct the 'Truth' line (the curve B_in_A lives on)
        # We call your _eval_function directly
        y_gt = InfiniteHarmonicsStream._eval_function(torch.tensor(x_dense), amps, freqs, phases).numpy()

        plt.figure(figsize=(10, 6))

        # 1. The GT Line (The underlying manifold)
        plt.plot(x_dense, y_gt, color='gray', linestyle='--', alpha=0.5, label='Truth Manifold (Params A)')

        # 2. A (Clean/Reference)
        # We filter out padded NaNs/zeros if present
        mask_A = batch_data['train']['padding_mask_A'][b_idx]
        plt.scatter(X_A[~mask_A], Y_A[~mask_A], c='blue', s=40, label='A (Canonical)')

        # 3. B_in_A (The target for Flow Matching)
        plt.scatter(X_B_in_A, Y_B_in_A, c='green', s=40, marker='o', label='B_in_A (Target)')

        # 4. B (The distorted observation)
        plt.scatter(X_B, Y_B, c='red', s=40, marker='x', label='B (Distorted Observation)')

        plt.legend()
        plt.title(f"Batch Alignment - Manifold View (Batch {b_idx})")
        plt.grid(True, alpha=0.3)
        plt.show()


    # Run this after generating your batch
    # visualize_batch(batch_data)

    # 2. Setup Bins
    num_bars = 50
    borders = torch.linspace(-5, 5, num_bars + 1)
    bin_centers = (borders[:-1] + borders[1:]) / 2.0


    def generate_smooth_logits(Y_true, centers, confidence=1.0):
        """
        Generates Gaussian-shaped logits centered on the true Y values.
        Y_true: (Seq, Batch)
        centers: (Num_Bars)
        """
        # Expand for broadcasting: (Seq, Batch, Num_Bars)
        y_target = Y_true.unsqueeze(-1)
        c = centers.view(1, 1, -1)

        # Calculate squared distance from bin centers (Gaussian Kernel)
        # Higher confidence = narrower peaks
        logits = -((y_target - c) ** 2) / (2 * (0.5 / confidence) ** 2)
        return logits


    # 3. Create synthetic predictions for each stream
    # Stream A: Very low confidence (wide blur)
    logits_A = generate_smooth_logits(batch_data['test']['Y_A'], bin_centers, confidence=0.8)

    # Stream B: High confidence (sharp line)
    logits_B = generate_smooth_logits(batch_data['test']['Y_B'], bin_centers, confidence=5.0)

    # Stream C: High confidence on related, but falls back to blur on traps
    # We simulate this by checking the is_unrelated mask
    logits_C = logits_A.clone()
    # is_unrelated = batch_data['params']['is_unrelated']
    # for b in range(logits_C.shape[1]):
    #     if not is_unrelated[b]:
    #         # If related, make it sharp like B
    #         logits_C[:, b, :] = generate_smooth_logits(batch_data['test']['Y_A'], bin_centers, confidence=4.0)[
    #             :, b, :]

    # 4. Plot
    fig = plt.figure(figsize=(12, 14))
    HeatmapVisualizer.save_heatmaps(
        fig=fig,
        batch_data=batch_data,
        borders=borders,
        save_path="test_heatmap.png",
        logits_A=logits_A,
        logits_B=logits_B,
        logits_C=logits_C,
        plot=True
    )
