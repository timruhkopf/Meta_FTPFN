import torch
from torch.utils.data import IterableDataset
import numpy as np
import matplotlib.pyplot as plt


class BaseHarmonicsStream(IterableDataset):
    """
    Abstract Base Class for Hidden Harmonic Mixture.
    Handles blueprints, transforms, and evaluations. Subclasses define X-coordinate sampling.
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
    def apply_spatial_warp(X, w_amp, w_freq, w_phase):
        w_amp_ext = w_amp.unsqueeze(0)
        w_freq_ext = w_freq.unsqueeze(0)
        w_phase_ext = w_phase.unsqueeze(0)
        return w_amp_ext * torch.sin(2 * torch.pi * w_freq_ext * X + w_phase_ext)

    @staticmethod
    def eval_function(X, amps, freqs, phases):
        X_ext = X.unsqueeze(0)
        a_ext = amps.unsqueeze(1)
        f_ext = freqs.unsqueeze(1)
        p_ext = phases.unsqueeze(1)
        terms = a_ext * torch.sin(2 * torch.pi * f_ext * X_ext + p_ext)
        return terms.sum(dim=0)

    def _sample_blueprints(self):
        B, K = self.batch_size, self.num_components
        amps_A = torch.empty(K, B).uniform_(0.5, 2.0)
        freqs_A = torch.empty(K, B).uniform_(0.1, 0.5)
        phases_A = torch.empty(K, B).uniform_(0, 2 * torch.pi)

        amps_B, freqs_B, phases_B = amps_A.clone(), freqs_A.clone(), phases_A.clone()
        is_unrelated = torch.zeros(B, dtype=torch.bool)

        num_unrelated = int(B * self.share_unrelated)
        if num_unrelated > 0:
            amps_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.5, 2.0)
            freqs_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.1, 0.5)
            phases_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0, 2 * torch.pi)
            is_unrelated[-num_unrelated:] = True

        return (amps_A, freqs_A, phases_A), (amps_B, freqs_B, phases_B), is_unrelated

    def _sample_transforms(self):
        B = self.batch_size
        v_shift_A = torch.empty(B).uniform_(-2.0, 2.0) if self.shift else torch.zeros(B)
        h_shift_A = torch.empty(B).uniform_(-1.5, 1.5) if self.shift else torch.zeros(B)
        scale_A = torch.empty(B).uniform_(0.3, 1.3) if self.scale else torch.ones(B)

        warp_amp_A = torch.empty(B).uniform_(0.0, 0.7) if self.warp else torch.zeros(B)
        warp_freq_A = torch.empty(B).uniform_(0.0, 0.2) if self.warp else torch.zeros(B)
        warp_phase_A = torch.empty(B).uniform_(0, 2 * torch.pi) if self.warp else torch.zeros(B)

        return scale_A, v_shift_A, h_shift_A, warp_amp_A, warp_freq_A, warp_phase_A

    def _sample_x_coordinates(self):
        """MUST BE IMPLEMENTED BY SUBCLASS."""
        raise NotImplementedError

    def _sample_batch(self):
        # 1. Get Base Parameters
        (amps_A, freqs_A, phases_A), (amps_B, freqs_B, phases_B), is_unrelated = self._sample_blueprints()
        scale_A, v_shift_A, h_shift_A, warp_amp_A, warp_freq_A, warp_phase_A = self._sample_transforms()

        # 2. Get Spatial Coordinates (Handled by Subclass)
        X_train_A, X_train_B, X_test_A, X_test_B = self._sample_x_coordinates()

        # 3. Evaluate B
        Y_train_B = self.eval_function(X_train_B, amps_B, freqs_B, phases_B) + torch.randn_like(
            X_train_B) * self.noise_std
        Y_test_B = self.eval_function(X_test_B, amps_B, freqs_B, phases_B)

        # 4. Evaluate B in A's Domain (Warped)
        X_train_B_in_A_domain = X_train_B - h_shift_A + self.apply_spatial_warp(
            X_train_B, warp_amp_A, warp_freq_A, warp_phase_A)
        X_test_B_in_A_domain = X_test_B - h_shift_A + self.apply_spatial_warp(
            X_test_B, warp_amp_A, warp_freq_A, warp_phase_A)

        # [CORRECTION]: Added scale_A to match A's true domain scaling
        Y_train_B_in_A_domain = scale_A * self.eval_function(X_train_B_in_A_domain, amps_A, freqs_A, phases_A) + v_shift_A
        Y_test_B_in_A_domain = scale_A * self.eval_function(X_test_B_in_A_domain, amps_A, freqs_A, phases_A) + v_shift_A

        # 5. Evaluate A
        X_train_A_warped = X_train_A - h_shift_A + self.apply_spatial_warp(
            X_train_A, warp_amp_A, warp_freq_A, warp_phase_A)
        X_test_A_warped = X_test_A - h_shift_A + self.apply_spatial_warp(
            X_test_A, warp_amp_A, warp_freq_A, warp_phase_A)

        Y_train_A_clean = scale_A * self.eval_function(X_train_A_warped, amps_A, freqs_A, phases_A) + v_shift_A
        Y_train_A = Y_train_A_clean + torch.randn_like(X_train_A) * self.noise_std
        Y_test_A = scale_A * self.eval_function(X_test_A_warped, amps_A, freqs_A, phases_A) + v_shift_A

        # --- NEW: Evaluate A in B's Domain (Canonical) ---
        # A's observed coordinates map to their warped counterparts in B's base space.
        X_train_A_in_B_domain = X_train_A_warped
        X_test_A_in_B_domain = X_test_A_warped

        # We evaluate these canonical coordinates using B's clean blueprint
        Y_train_A_in_B_domain = self.eval_function(X_train_A_in_B_domain, amps_B, freqs_B, phases_B)
        Y_test_A_in_B_domain = self.eval_function(X_test_A_in_B_domain, amps_B, freqs_B, phases_B)

        # 6. Pad A (and A_in_B) to match n_B
        pad_size = self.n_B - self.n_A
        if pad_size > 0:
            nan_pad = torch.full((pad_size, self.batch_size), float('nan'), device=X_train_A.device)

            X_train_A_padded = torch.cat([X_train_A, nan_pad], dim=0)
            Y_train_A_padded = torch.cat([Y_train_A, nan_pad], dim=0)

            # Pad the new A_in_B variables
            X_train_A_in_B_padded = torch.cat([X_train_A_in_B_domain, nan_pad], dim=0)
            Y_train_A_in_B_padded = torch.cat([Y_train_A_in_B_domain, nan_pad], dim=0)
        else:
            X_train_A_padded, Y_train_A_padded = X_train_A, Y_train_A
            X_train_A_in_B_padded, Y_train_A_in_B_padded = X_train_A_in_B_domain, Y_train_A_in_B_domain

        return {
            'params': {
                'amps_A': amps_A, 'freqs_A': freqs_A, 'phases_A': phases_A,
                'amps_B': amps_B, 'freqs_B': freqs_B, 'phases_B': phases_B,
                'scale_A': scale_A, 'v_shift_A': v_shift_A, 'h_shift_A': h_shift_A,
                'warp_amp_A': warp_amp_A, 'warp_freq_A': warp_freq_A, 'warp_phase_A': warp_phase_A,
                'is_unrelated': is_unrelated
            },
            'train': {
                'X_B': X_train_B, 'Y_B': Y_train_B,
                'X_A': X_train_A_padded, 'Y_A': Y_train_A_padded,
                'X_B_in_A': X_train_B_in_A_domain, 'Y_B_in_A': Y_train_B_in_A_domain,
                'X_A_in_B': X_train_A_in_B_padded, 'Y_A_in_B': Y_train_A_in_B_padded, # Added
            },
            'test': {
                'X_B': X_test_B, 'Y_B': Y_test_B,
                'X_A': X_test_A, 'Y_A': Y_test_A,
                'X_B_in_A': X_test_B_in_A_domain, 'Y_B_in_A': Y_test_B_in_A_domain,
                'X_A_in_B': X_test_A_in_B_domain, 'Y_A_in_B': Y_test_A_in_B_domain,   # Added
            }
        }

    def __iter__(self):
        while True:
            yield self._sample_batch()


class GlobalSparseHarmonicsStream(BaseHarmonicsStream):
    """Case 1: A is observed over the FULL domain, but is highly sparse."""

    def _sample_x_coordinates(self):
        B = self.batch_size

        # Both A and B are sampled uniformly across the entire [min_x, max_x]
        X_train_B, _ = torch.sort(torch.empty(self.n_B, B).uniform_(self.min_x, self.max_x), dim=0)
        X_train_A, _ = torch.sort(torch.empty(self.n_A, B).uniform_(self.min_x, self.max_x), dim=0)

        # Test sets always span the full domain
        X_test_B = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)
        X_test_A = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)

        return X_train_A, X_train_B, X_test_A, X_test_B


class SubdomainHarmonicsStream(BaseHarmonicsStream):
    """Case 2: A is only observed in a specific subdomain (e.g., the first 50%)."""

    def __init__(self, subdomain_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.subdomain_ratio = subdomain_ratio

    def _sample_x_coordinates(self):
        B = self.batch_size
        split_x = self.min_x + (self.max_x - self.min_x) * self.subdomain_ratio

        # B is sampled across the FULL domain
        X_train_B, _ = torch.sort(torch.empty(self.n_B, B).uniform_(self.min_x, self.max_x), dim=0)

        # A is constrained ONLY to the subdomain [min_x, split_x]
        X_train_A, _ = torch.sort(torch.empty(self.n_A, B).uniform_(self.min_x, split_x), dim=0)

        # Test sets always span the full domain (so we can measure extrapolation)
        X_test_B = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)
        X_test_A = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)

        return X_train_A, X_train_B, X_test_A, X_test_B


class HarmonicsVisualizer:
    """Handles all plotting and visualization for the Harmonics Streams."""

    @staticmethod
    def save_heatmaps(fig, batch_data, borders, save_path, model=None, logits_A=None, logits_B=None, logits_C=None,
                      x_range=(-5, 5), plot=False):

        # 1. Dynamic Logit Computation (if model is passed)
        if model is not None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                logits_A, logits_B, logits_C = model(batch_data)
            if was_training:
                model.train()

        min_x, max_x = x_range
        p = batch_data['params']

        def to_np(val):
            return val.detach().cpu().numpy() if torch.is_tensor(val) else val

        # --- Helper for Smooth Percentiles ---
        def get_binned_percentiles(probs_np, borders_np, percentiles=[0.025, 0.5, 0.975]):
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

        # 2. Safely flatten to numpy boolean array
        is_unrelated_flat = to_np(p['is_unrelated']).astype(bool).flatten()
        related_indices = np.where(~is_unrelated_flat)[0]
        unrelated_indices = np.where(is_unrelated_flat)[0]

        # 3. Enforce Column 0 = Related, Column 1 = Unrelated
        idx_list = [0, 0]
        if len(related_indices) > 0:
            idx_list[0] = related_indices[0]
        if len(unrelated_indices) > 0:
            idx_list[1] = unrelated_indices[0]
        else:
            if len(related_indices) > 1:
                idx_list[1] = related_indices[1]  # Fallback

        axes = fig.subplots(3, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0.25, 'wspace': 0.1})
        borders_np = to_np(borders)
        centers = (borders_np[:-1] + borders_np[1:]) / 2.0
        x_dense = np.linspace(min_x, max_x, 1000)

        for col_idx, batch_idx in enumerate(idx_list):
            is_trap = is_unrelated_flat[batch_idx]
            col_title = "RELATED CONTEXT" if not is_trap else "UNRELATED TRAP"

            # Reconstruct true lines with spatial/phase warps
            def build_func(X, amps, freqs, phases, scale=1.0, h_shift=0.0, v_shift=0.0,
                           w_amp=0.0, w_freq=0.0, w_phase=0.0,
                           p_w_amp=0.0, p_w_freq=0.0, p_w_phase=0.0):

                # Apply both Global Shift and Local Non-Linear Warps
                X_warped = X - h_shift

                # 1. Spatial Sinusoidal Wobble (if active)
                if w_amp > 0:
                    X_warped += w_amp * np.sin(2 * np.pi * w_freq * X + w_phase)

                # 2. Phase Warp (if active)
                if p_w_amp > 0:
                    X_warped += p_w_amp * np.sin(2 * np.pi * p_w_freq * X + p_w_phase)

                # Evaluate harmonic function at the warped coordinates
                terms = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * X_warped[None, :] + phases[:, None])
                return scale * np.sum(terms, axis=0) + v_shift

            # Safely extract warp parameters (default to 0.0 if not present in older batches)
            def get_param(key, default=0.0):
                return to_np(p[key][batch_idx]) if key in p else default

            w_amp = get_param('warp_amp_A')
            w_freq = get_param('warp_freq_A')
            w_phase = get_param('warp_phase_A')

            p_w_amp = get_param('p_warp_amp')
            p_w_freq = get_param('p_warp_freq')
            p_w_phase = get_param('p_warp_phase')

            y_dense_A = build_func(
                x_dense,
                to_np(p['amps_A'][:, batch_idx]),
                to_np(p['freqs_A'][:, batch_idx]),
                to_np(p['phases_A'][:, batch_idx]),
                scale=to_np(p['scale_A'][batch_idx]),
                h_shift=to_np(p['h_shift_A'][batch_idx]),
                v_shift=to_np(p['v_shift_A'][batch_idx]),
                w_amp=w_amp, w_freq=w_freq, w_phase=w_phase,
                p_w_amp=p_w_amp, p_w_freq=p_w_freq, p_w_phase=p_w_phase
            )

            # Stream B evaluates its own blueprint unwarped
            y_dense_B = build_func(
                x_dense,
                to_np(p['amps_B'][:, batch_idx]),
                to_np(p['freqs_B'][:, batch_idx]),
                to_np(p['phases_B'][:, batch_idx])
            )

            # 4. Iterate through Rows (Streams)
            for row_idx, stream_name in enumerate(['A', 'B', 'C']):
                ax = axes[row_idx, col_idx]

                # EXACT MATCH GUARANTEE: Stream A and C both pull from dataset 'A'
                data_key = 'A' if stream_name in ['A', 'C'] else 'B'

                if is_trap:
                    ax.set_facecolor('#fffafa')

                curr_logits = {'A': logits_A, 'B': logits_B, 'C': logits_C}[stream_name]
                X_test_np = to_np(batch_data['test'][f'X_{data_key}'][:, batch_idx])

                if curr_logits is not None:
                    probs = torch.softmax(curr_logits[:, batch_idx, :], dim=-1).detach().cpu().numpy()

                    # Get smooth interpolated percentiles
                    percentiles = get_binned_percentiles(probs, borders_np, percentiles=[0.025, 0.5, 0.975])

                    # Plot Smooth Heatmap
                    ax.pcolormesh(X_test_np, centers, probs.T, cmap='viridis', shading='nearest', alpha=0.9,
                                  rasterized=True)

                    # Plot Median and Smooth Confidence Intervals
                    ax.plot(X_test_np, percentiles[0.5], color='orange', linestyle='-', linewidth=1., zorder=10)
                    ax.plot(X_test_np, percentiles[0.025], color='orange', linestyle=':', linewidth=1, zorder=10)
                    ax.plot(X_test_np, percentiles[0.975], color='orange', linestyle=':', linewidth=1, zorder=10)

                # Plot Ground Truth Lines (A & C share the same true function)
                true_y = y_dense_A if stream_name in ['A', 'C'] else y_dense_B
                line_color = 'white' if stream_name in ['A', 'C'] else 'red'
                line_style = '-' if stream_name in ['A', 'C'] else '--'
                line_alpha = 0.9 if stream_name in ['A', 'C'] else 0.5
                ax.plot(x_dense, true_y, color=line_color, linestyle=line_style, linewidth=1.0, alpha=line_alpha,
                        zorder=5)

                # Scatter Training Points
                X_train = to_np(batch_data['train'][f'X_{data_key}'][:, batch_idx])
                Y_train = to_np(batch_data['train'][f'Y_{data_key}'][:, batch_idx])
                valid = ~np.isnan(X_train)

                if stream_name in ['A', 'C']:
                    ax.scatter(X_train[valid], Y_train[valid], c='white', s=50, edgecolors='black', linewidth=.5,
                               zorder=20)
                else:
                    ax.scatter(X_train[valid], Y_train[valid], c='red', s=40, marker='x', alpha=0.8, zorder=20)

                # Labels and Aesthetics
                if row_idx == 0:
                    ax.set_title(col_title, fontweight='bold', fontsize=11, pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"Stream {stream_name}", fontsize=10, fontweight='bold')

                ax.set_ylim(borders_np[0] - 1.5, borders_np[-1] + 1.5)
                ax.grid(False)

        axes[2, 0].set_xlabel("x-coordinate")
        axes[2, 1].set_xlabel("x-coordinate")

        plt.tight_layout()
        if plot:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close(fig)



if __name__ == '__main__':

    # 1. Setup Dataset with enough batch size to guarantee a Trap (10 * 0.2 = 2 traps)
    # Using the new Extrapolation subclass where A is only observed on the left half
    dataset = GlobalSparseHarmonicsStream(
        batch_size=10, n_A=10, n_B=50, n_test=200,
        num_components=4, noise_std=0.05, share_unrelated=0.2,
    )
    batch_data = next(iter(dataset))

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
    is_unrelated = batch_data['params']['is_unrelated']
    for b in range(logits_C.shape[1]):
        if not is_unrelated[b]:
            # If related, make it sharp like B
            logits_C[:, b, :] = generate_smooth_logits(batch_data['test']['Y_A'], bin_centers, confidence=4.0)[
                :, b, :]

    # 4. Plot
    fig = plt.figure(figsize=(12, 14))

    # Updated to use the separated Visualizer class
    HarmonicsVisualizer.save_heatmaps(
        fig=fig,
        batch_data=batch_data,
        borders=borders,
        save_path="test_heatmap.png",
        logits_A=logits_A,
        logits_B=logits_B,
        logits_C=logits_C,
        plot=True
    )
