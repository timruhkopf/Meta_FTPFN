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
                 num_components=4, noise_std=0.05, share_unrelated=0.2, scale=True, shift=True):
        super().__init__()
        self.batch_size = batch_size
        self.n_A = n_A
        self.n_B = n_B
        self.n_test = n_test
        self.min_x ,  self.max_x = x_range
        self.num_components = num_components
        self.noise_std = noise_std
        self.share_unrelated = share_unrelated

        self.scale = scale
        self.shift = shift

    def _sample_batch(self):
        B = self.batch_size
        K = self.num_components

        # 1. Randomize "Hidden Blueprint" with LOWER frequencies
        # Old: uniform_(0.2, 2.0) -> New: uniform_(0.1, 0.5)
        # This ensures about 1-2 full cycles across the x_range of 10 units.
        amps_A = torch.empty(K, B).uniform_(0.5, 2.0)
        freqs_A = torch.empty(K, B).uniform_(0.1, 0.5)
        phases_A = torch.empty(K, B).uniform_(0, 2 * torch.pi)

        # 2. Prepare Blueprint for B
        amps_B = amps_A.clone()
        freqs_B = freqs_A.clone()
        phases_B = phases_A.clone()

        # Unrelated Traps: also use lower frequencies for consistency
        num_unrelated = int(B * self.share_unrelated)
        if num_unrelated > 0:
            amps_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.5, 2.0)
            freqs_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0.1, 0.5)
            phases_B[:, -num_unrelated:] = torch.empty(K, num_unrelated).uniform_(0, 2 * torch.pi)

        # 3. Constrain the Transformations
        # Reducing h_shift makes the relatedness more obvious for the cross-attn
        if self.shift:
            v_shift_A = torch.empty(B).uniform_(-2.0, 2.0)
            h_shift_A = torch.empty(B).uniform_(-1.5, 1.5)
        else:
            v_shift_A = torch.zeros(B)
            h_shift_A = torch.zeros(B)

        if self.scale:
            scale_A = torch.empty(B).uniform_(0.3, 1.3)
        else:
            scale_A = torch.ones(B)

        # 4. Sample X coordinates (Seq, Batch)
        # Train points are sorted uniformly random, Test points are linspace for smooth heatmaps
        X_train_B, _ = torch.sort(torch.empty(self.n_B, B).uniform_(self.min_x, self.max_x), dim=0)
        X_train_A, _ = torch.sort(torch.empty(self.n_A, B).uniform_(self.min_x, self.max_x), dim=0)

        X_test_B = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)
        X_test_A = torch.linspace(self.min_x, self.max_x, self.n_test).unsqueeze(1).expand(self.n_test, B)

        # Helper to evaluate based on explicit parameters
        def eval_function(X, amps, freqs, phases):
            # X: (Seq, B), amps/freqs/phases: (K, B)
            X_ext = X.unsqueeze(0)  # (1, Seq, B)
            a_ext = amps.unsqueeze(1)  # (K, 1, B)
            f_ext = freqs.unsqueeze(1)  # (K, 1, B)
            p_ext = phases.unsqueeze(1)  # (K, 1, B)
            # Standard harmonic formula with 2*pi
            terms = a_ext * torch.sin(2 * torch.pi * f_ext * X_ext + p_ext)
            return terms.sum(dim=0)  # (Seq, B)

        # 5. Evaluate Y coordinates
        # Task B evaluates its own blueprint (which might be unrelated)
        Y_train_B = eval_function(X_train_B, amps_B, freqs_B, phases_B) + torch.randn_like(X_train_B) * self.noise_std
        Y_test_B = eval_function(X_test_B, amps_B, freqs_B, phases_B)

        # for training purposes: we know the true underlying warp, so we can look at B's actual location before transforming it into B's domain
        X_train_B_in_A_domain = X_train_B-h_shift_A
        X_test_B_in_A_domain = X_test_B -h_shift_A
        Y_train_B_in_A_domain = eval_function(X_train_B_in_A_domain, amps_A, freqs_A, phases_A) + v_shift_A
        Y_test_B_in_A_domain = eval_function(X_test_B_in_A_domain, amps_A, freqs_A, phases_A) + v_shift_A

        # Task A evaluates Blueprint A with spatial and amplitude transformations
        Y_train_A_clean = scale_A * eval_function(X_train_A - h_shift_A, amps_A, freqs_A, phases_A) + v_shift_A
        Y_train_A = Y_train_A_clean + torch.randn_like(X_train_A) * self.noise_std
        Y_test_A = scale_A * eval_function(X_test_A - h_shift_A, amps_A, freqs_A, phases_A) + v_shift_A

        # 6. Pad A to match B's sequence length (Pipeline Requirement)
        pad_size = self.n_B - self.n_A
        if pad_size > 0:
            nan_pad = torch.full((pad_size, B), float('nan'))
            X_train_A_padded = torch.cat([X_train_A, nan_pad], dim=0)
            Y_train_A_padded = torch.cat([Y_train_A, nan_pad], dim=0)
        else:
            X_train_A_padded = X_train_A
            Y_train_A_padded = Y_train_A

        # 7. Add boolean mask for tracking which tasks are unrelated traps
        is_unrelated = torch.zeros(B, dtype=torch.bool)
        if num_unrelated > 0:
            is_unrelated[-num_unrelated:] = True

        return {
            'params': {
                'amps_A': amps_A, 'freqs_A': freqs_A, 'phases_A': phases_A,
                'amps_B': amps_B, 'freqs_B': freqs_B, 'phases_B': phases_B,
                'scale_A': scale_A, 'v_shift_A': v_shift_A, 'h_shift_A': h_shift_A,
                'is_unrelated': is_unrelated
            },
            'train': {
                'X_B': X_train_B, 'Y_B': Y_train_B,
                'X_A': X_train_A_padded, 'Y_A': Y_train_A_padded,
                'X_B_in_A': X_train_B_in_A_domain, 'Y_B_in_A': Y_train_B_in_A_domain,
            },
            'test': {
                'X_B': X_test_B, 'Y_B': Y_test_B,
                'X_A': X_test_A, 'Y_A': Y_test_A,
                'X_B_in_A': X_test_B_in_A_domain, 'Y_B_in_A': Y_test_B_in_A_domain,
            }
        }

    def __iter__(self):
        while True:
            yield self._sample_batch()

    # FIXME: have two plots: both have the exact same target task, but they have different related.
    #  one is with a related, one is an unrelated

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

            # Reconstruct true lines
            def build_func(X, amps, freqs, phases, scale=1.0, h_shift=0.0, v_shift=0.0):
                X_shifted = X - h_shift
                terms = amps[:, None] * np.sin(2 * np.pi * freqs[:, None] * X_shifted[None, :] + phases[:, None])
                return scale * np.sum(terms, axis=0) + v_shift

            y_dense_A = build_func(
                x_dense, to_np(p['amps_A'][:, batch_idx]), to_np(p['freqs_A'][:, batch_idx]),
                to_np(p['phases_A'][:, batch_idx]),
                scale=to_np(p['scale_A'][batch_idx]), h_shift=to_np(p['h_shift_A'][batch_idx]),
                v_shift=to_np(p['v_shift_A'][batch_idx])
            )
            y_dense_B = build_func(
                x_dense, to_np(p['amps_B'][:, batch_idx]), to_np(p['freqs_B'][:, batch_idx]),
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


# if __name__ == '__main__':
    # Quick test to visualize a batch
    # dataset = InfiniteHarmonicsStream(batch_size=10, n_A=10, n_B=50, n_test=200,
    #                                  num_components=4, noise_std=0.05, share_unrelated=0.2)
    # batch_data = next(iter(dataset))
    #
    # borders = np.linspace(-5, 5, 50)
    # fig = plt.figure(figsize=(8, 12))
    # InfiniteHarmonicsStream.save_heatmaps(fig, batch_data, borders, "test_heatmap.png",  plot=True)

if __name__ == '__main__':
    # 1. Setup Dataset with enough batch size to guarantee a Trap (10 * 0.2 = 2 traps)
    dataset = InfiniteHarmonicsStream(batch_size=10, n_A=10, n_B=50, n_test=200,
                                      num_components=4, noise_std=0.05, share_unrelated=0.2)
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
    InfiniteHarmonicsStream.save_heatmaps(
        fig=fig,
        batch_data=batch_data,
        borders=borders,
        save_path="test_heatmap.png",
        logits_A=logits_A,
        logits_B=logits_B,
        logits_C=logits_C,
        plot=True
    )
