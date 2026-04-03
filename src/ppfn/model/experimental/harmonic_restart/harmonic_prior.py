import torch
from torch.utils.data import IterableDataset
import numpy as np
import matplotlib.pyplot as plt

# FIXME: make X (T, B, D)
class InfiniteHarmonicsStream(IterableDataset):
    def __init__(self, batch_size=32, n_A=20, n_B=100, n_test=50, max_x=10.0, num_components=3):
        super().__init__()
        self.batch_size = batch_size
        self.n_A = n_A
        self.n_B = n_B
        self.n_test = n_test
        self.max_x = max_x
        self.num_components = num_components

    def _sample_batch(self):
        B, K = self.batch_size, self.num_components
        freq_B = np.random.uniform(0.2, 2.0, (B, K))
        phase_B = np.random.uniform(0, 2 * np.pi, (B, K))
        amp_B = np.random.uniform(0.5, 1.5, (B, K))

        d_freq = np.random.uniform(-0.1, 0.1, (B, K))
        d_phase = np.random.uniform(-0.5, 0.5, (B, K))
        d_amp = np.random.uniform(-0.2, 0.2, (B, K))
        offset_A = np.random.uniform(-1.0, 1.0, (B, 1))

        freq_A = freq_B + d_freq
        phase_A = phase_B + d_phase
        amp_A = amp_B + d_amp

        def eval_harmonic(X, f, p, a, offset=0):
            X_exp = np.expand_dims(X, axis=1)
            f_exp = np.expand_dims(f, axis=2)
            p_exp = np.expand_dims(p, axis=2)
            a_exp = np.expand_dims(a, axis=2)
            return np.sum(a_exp * np.sin(2 * np.pi * f_exp * X_exp + p_exp), axis=1) + offset

        X_train_B = np.sort(np.random.uniform(0, self.max_x, (B, self.n_B)), axis=1)
        X_train_A = np.sort(np.random.uniform(0, self.max_x, (B, self.n_A)), axis=1)
        X_test_B = np.sort(np.random.uniform(0, self.max_x, (B, self.n_test)), axis=1)
        X_test_A = np.sort(np.random.uniform(0, self.max_x, (B, self.n_test)), axis=1)

        Y_train_B = eval_harmonic(X_train_B, freq_B, phase_B, amp_B)
        Y_train_A = eval_harmonic(X_train_A, freq_A, phase_A, amp_A, offset_A)
        Y_test_B = eval_harmonic(X_test_B, freq_B, phase_B, amp_B)
        Y_test_A = eval_harmonic(X_test_A, freq_A, phase_A, amp_A, offset_A)

        pad_size = self.n_B - self.n_A
        if pad_size > 0:
            nan_pad = np.full((B, pad_size), np.nan)
            X_train_A_padded = np.concatenate([X_train_A, nan_pad], axis=1)
            Y_train_A_padded = np.concatenate([Y_train_A, nan_pad], axis=1)
        else:
            X_train_A_padded = X_train_A
            Y_train_A_padded = Y_train_A

        # batch_first = False (T, B, D)

        X_train_B = X_train_B.T  # (n_B, B)
        Y_train_B = Y_train_B.T  # (n_B, B)
        X_test_B = X_test_B.T  # (n_test, B)
        Y_test_B = Y_test_B.T  # (n_test, B)

        # Ensure A's padded train and test are also (Seq, Batch)
        X_train_A_padded = X_train_A_padded.T
        Y_train_A_padded = Y_train_A_padded.T
        X_test_A = X_test_A.T
        Y_test_A = Y_test_A.T


        # Convert everything to float32 for PyTorch out of the gate
        return {
            'params': {k: v.astype(np.float32) for k, v in locals().items() if
                       k in ['freq_B', 'phase_B', 'amp_B', 'freq_A', 'phase_A', 'amp_A', 'offset_A']},
            'train': {
                'X_B': X_train_B.astype(np.float32), 'Y_B': Y_train_B.astype(np.float32),
                'X_A': X_train_A_padded.astype(np.float32), 'Y_A': Y_train_A_padded.astype(np.float32),
            },
            'test': {
                'X_B': X_test_B.astype(np.float32), 'Y_B': Y_test_B.astype(np.float32),
                'X_A': X_test_A.astype(np.float32), 'Y_A': Y_test_A.astype(np.float32),
            }
        }

    def __iter__(self):
        while True:
            yield self._sample_batch()

    @staticmethod
    def save_heatmaps(fig, batch_data, borders, save_path, logits_A=None, logits_B=None, logits_C=None, batch_idx=0,
                      max_x=10.0):
        axs = fig.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.15})
        ax_A, ax_B, ax_C = axs[0], axs[1], axs[2]

        def to_np(val):
            return val.detach().cpu().numpy() if torch.is_tensor(val) else val

        borders_np = to_np(borders)
        centers = (borders_np[:-1] + borders_np[1:]) / 2.0
        p = batch_data['params']

        # Helper to plot a single stream (A, B, or C) with the 'Turbo' colormap
        def plot_stream_data(ax, name, logits, X_test, true_y_dense):
            X_test_np = to_np(X_test)

            # --- Heatmap & Distribution Logic ---
            if logits is not None:
                logits_np = to_np(logits)
                probs = np.exp(logits_np) / np.sum(np.exp(logits_np), axis=-1, keepdims=True)

                mean_pred = np.sum(probs * centers, axis=-1)
                cumsum = np.cumsum(probs, axis=-1)
                lower_bound = centers[np.argmax(cumsum >= 0.025, axis=-1)]
                upper_bound = centers[np.argmax(cumsum >= 0.975, axis=-1)]

                dx = np.diff(X_test_np)
                if len(dx) > 0:
                    midpoints = X_test_np[:-1] + dx / 2
                    edges_x = np.concatenate([[X_test_np[0] - dx[0] / 2], midpoints, [X_test_np[-1] + dx[-1] / 2]])
                else:
                    edges_x = np.array([X_test_np[0] - 0.1, X_test_np[0] + 0.1])

                # --- NEW COLOURMAP (High probability = Bright Yellow, Low = Dark Blue) ---
                pcm = ax.pcolormesh(edges_x, borders_np, probs.T, cmap='turbo', alpha=0.9, shading='flat',
                                    rasterized=True)

                # Plot statistics with new colors for visibility on dark turbo background
                # We use cyan/white for the lines so they are legible against the dark blue and bright yellow
                ax.plot(X_test_np, mean_pred, color='white', linewidth=1.8, label='Mean Prediction', zorder=10)
                ax.plot(X_test_np, lower_bound, color='cyan', linestyle='--', linewidth=1.2, label='95% CI', zorder=10)
                ax.plot(X_test_np, upper_bound, color='cyan', linestyle='--', linewidth=1.2, zorder=10)

            # --- Ground Truth & Samples Logic ---
            data_key = 'A' if name.startswith('C') else name[:1]

            if name.startswith('C'):
                main_marker_color, marker = 'lime', 's'  # bright green for C contrast
            elif name.startswith('A'):
                main_marker_color, marker = 'tomato', 's'  # bright red for A contrast
            else:
                main_marker_color, marker = 'dodgerblue', 'o'  # bright blue for B contrast

            X_train = to_np(batch_data['train'][f'X_{data_key}'][batch_idx])
            Y_train = to_np(batch_data['train'][f'Y_{data_key}'][batch_idx])
            X_test_targets = to_np(batch_data['test'][f'X_{data_key}'][batch_idx])
            Y_test_targets = to_np(batch_data['test'][f'Y_{data_key}'][batch_idx])

            ax.plot(np.linspace(0, max_x, 1000), true_y_dense, color='grey', alpha=0.5, label=f'True {data_key}',
                    zorder=1)

            # Plot Training Samples (Sparse)
            valid_idx = ~np.isnan(X_train)
            ax.scatter(X_train[valid_idx], Y_train[valid_idx], c=main_marker_color, s=35, marker=marker,
                       edgecolors='black', linewidth=0.5, zorder=20, label=f'Train {data_key}')

            # Subsample Test Targets (e.g., every 10th point) so we don't hide the heatmap
            ax.scatter(X_test_targets[::10], Y_test_targets[::10], facecolors='none', edgecolors=main_marker_color,
                       s=30, marker=marker, alpha=0.8, linewidth=0.5, zorder=20, label=f'Test {data_key} (subsampled)')

            # Formatting with darker grid for contrast
            ax.set_title(f"Stream {name}", fontsize=10, pad=5)
            ax.set_ylabel("y")
            ax.set_ylim(borders_np[0], borders_np[-1])
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), fontsize='small')
            ax.grid(True, color='grey', alpha=0.3, zorder=0)

        # Reconstruct true lines
        def build_func(X, f, p_phase, a, offset=0):
            return np.sum(a[:, None] * np.sin(2 * np.pi * f[:, None] * X[None, :] + p_phase[:, None]), axis=0) + offset

        x_dense = np.linspace(0, max_x, 1000)
        y_dense_A = build_func(x_dense, to_np(p['freq_A'][batch_idx]), to_np(p['phase_A'][batch_idx]),
                               to_np(p['amp_A'][batch_idx]), to_np(p['offset_A'][batch_idx]))
        y_dense_B = build_func(x_dense, to_np(p['freq_B'][batch_idx]), to_np(p['phase_B'][batch_idx]),
                               to_np(p['amp_B'][batch_idx]))

        plot_stream_data(
            ax_A, 'A',
            logits_A[:, batch_idx, :] if logits_A is not None else None,
            batch_data['test']['X_A'][:, batch_idx],
            y_dense_A
        )

        plot_stream_data(
            ax_B, 'B',
            logits_B[:, batch_idx, :] if logits_B is not None else None,
            batch_data['test']['X_B'][:, batch_idx],
            y_dense_B
        )

        plot_stream_data(
            ax_C, 'C (A|B)',
            logits_C[:, batch_idx, :] if logits_C is not None else None,
            batch_data['test']['X_A'][:, batch_idx],
            y_dense_A
        )

        ax_C.set_xlabel("x coordinate")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"Saved heatmaps to: {save_path}")
