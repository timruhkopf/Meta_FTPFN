import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mlflow

from ppfn.trainer import AbstractCallback


class HeatmapCallback(AbstractCallback):
    def __init__(self, plot_every: int, plot_dir: str, start_plotting_epoch=2000, **kwargs):
        super().__init__(**kwargs)
        self.plot_every = plot_every
        self.plot_dir = plot_dir
        self.start_plotting = start_plotting_epoch
        os.makedirs(self.plot_dir, exist_ok=True)

    def on_epoch_end(self, epoch, **kwargs):
        if epoch < self.start_plotting or epoch % self.plot_every != 0:
            return

        batch, _ = self.trainer._get_next_batch()

        # Isolate inference state
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        with torch.no_grad():
            logits_A, logits_B, logits_C = self.trainer.model(batch)
        if was_training:
            self.trainer.model.train()

        # Instantiate the stateful visualizer
        visualizer = HeatmapVisualizer(
            borders=self.trainer.criterion.criterion_backend.borders,
            x_range=(-5, 5)
        )

        fig = visualizer.generate_figure(
            batch_data=batch,
            logits=(logits_A, logits_B, logits_C)
        )

        plot_path = os.path.join(self.plot_dir, f"heatmaps_step_{epoch:05d}.png")
        fig.savefig(plot_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

        mlflow.log_artifact(plot_path, "heatmap_plots")


class HeatmapVisualizer:
    def __init__(self, borders, x_range=(-5, 5)):
        self.min_x, self.max_x = x_range
        self.borders_np = self._to_np(borders)
        self.centers = (self.borders_np[:-1] + self.borders_np[1:]) / 2.0
        # Cache the dense X tensor for ground truth reconstruction
        self.x_dense = torch.linspace(self.min_x, self.max_x, 1000)
        self.x_dense_np = self.x_dense.numpy()

    @staticmethod
    def _to_np(val):
        return val.detach().cpu().numpy() if torch.is_tensor(val) else val

    def _reconstruct_ground_truth(self, params, batch_idx, stream_type='A'):
        """Reconstructs the curve perfectly using PyTorch based on the Prior's output format."""
        # Unpack params_A: (amps, freqs, phases) shape [K, B] -> Select batch idx -> shape [K, 1]
        amps = params['params_A'][0][:, batch_idx].unsqueeze(1)
        freqs = params['params_A'][1][:, batch_idx].unsqueeze(1)
        phases = params['params_A'][2][:, batch_idx].unsqueeze(1)

        x_eval = self.x_dense.unsqueeze(0)  # [1, 1000]

        if stream_type == 'A':
            terms = amps * torch.sin(2 * torch.pi * freqs * x_eval + phases)
            y_eval = terms.sum(dim=0)
            return y_eval.squeeze(0).numpy()

        # Stream B applies the full spatial warping and affine transformations
        v_shift = params['shifts'][0][batch_idx]
        h_shift = params['shifts'][1][batch_idx]
        scale = params['scale_A'][batch_idx]

        w_amp = params['warps'][0][batch_idx]
        w_freq = params['warps'][1][batch_idx]
        w_phase = params['warps'][2][batch_idx]

        x_warped = x_eval - h_shift + w_amp * torch.sin(2 * torch.pi * w_freq * x_eval + w_phase)
        terms = amps * torch.sin(2 * torch.pi * freqs * x_warped + phases)
        y_eval = scale * terms.sum(dim=0) + v_shift

        return y_eval.squeeze(0).numpy()

    def _get_binned_percentiles(self, probs_np, percentiles=(0.025, 0.5, 0.975)):
        """Calculates continuous percentiles via CDF interpolation between bin edges."""
        cdf = np.cumsum(probs_np, axis=-1)
        cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=-1)
        results = {pct: np.zeros(probs_np.shape[0]) for pct in percentiles}

        for i in range(probs_np.shape[0]):
            for pct in percentiles:
                idx = np.searchsorted(cdf[i], pct)
                if idx == 0:
                    results[pct][i] = self.borders_np[0]
                elif idx >= len(self.borders_np):
                    results[pct][i] = self.borders_np[-1]
                else:
                    p_low, p_high = cdf[i, idx - 1], cdf[i, idx]
                    b_low, b_high = self.borders_np[idx - 1], self.borders_np[idx]

                    if p_high > p_low:
                        fraction = (pct - p_low) / (p_high - p_low)
                        results[pct][i] = b_low + fraction * (b_high - b_low)
                    else:
                        results[pct][i] = b_low
        return results

    def _draw_heatmap_layer(self, ax, X_coords, logits):
        """Renders the colored probability density and percentile lines."""
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # Sort spatially so pcolormesh and plot lines don't zig-zag
        sort_idx = np.argsort(X_coords)
        X_sorted = X_coords[sort_idx]
        probs_sorted = probs[sort_idx]

        percentiles = self._get_binned_percentiles(probs_sorted)

        ax.pcolormesh(X_sorted, self.centers, probs_sorted.T, cmap='viridis', shading='nearest', alpha=0.9,
                      rasterized=True)
        ax.plot(X_sorted, percentiles[0.5], color='orange', linestyle='-', linewidth=1.0, zorder=10)
        ax.plot(X_sorted, percentiles[0.025], color='orange', linestyle=':', linewidth=1.0, zorder=10)
        ax.plot(X_sorted, percentiles[0.975], color='orange', linestyle=':', linewidth=1.0, zorder=10)

    def _draw_scatter_layer(self, ax, batch_data, data_key, batch_idx, is_ac_stream):
        """Renders the underlying training coordinates, respecting padding masks."""
        X_train = self._to_np(batch_data['train'][f'X_{data_key}'][:, batch_idx]).flatten()
        Y_train = self._to_np(batch_data['train'][f'Y_{data_key}'][:, batch_idx]).flatten()

        mask_key = f'padding_mask_{data_key}'
        if mask_key in batch_data['train']:
            pad_mask = self._to_np(batch_data['train'][mask_key][:, batch_idx])
            valid = ~pad_mask.astype(bool)
        else:
            valid = np.ones(len(X_train), dtype=bool)

        if is_ac_stream:
            ax.scatter(X_train[valid], Y_train[valid], c='white', s=50, edgecolors='black', linewidth=0.5, zorder=20)
        else:
            ax.scatter(X_train[valid], Y_train[valid], c='red', s=40, marker='x', alpha=0.8, zorder=20)

    def generate_figure(self, batch_data, logits):
        """Main orchestrator for building the full Matplotlib figure."""
        logits_A, logits_B, logits_C = logits
        params = batch_data['params']

        batch_size = batch_data['train']['X_A'].shape[1]
        idx_list = [0, 1] if batch_size > 1 else [0]

        fig, axes = plt.subplots(3, len(idx_list), figsize=(10, 8), sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0.25, 'wspace': 0.1})

        # Standardize axes to 2D array even if batch size is 1
        if len(idx_list) == 1:
            axes = np.expand_dims(axes, axis=1)

        for col, batch_idx in enumerate(idx_list):
            y_dense_A = self._reconstruct_ground_truth(params, batch_idx, stream_type='A')
            y_dense_B = self._reconstruct_ground_truth(params, batch_idx, stream_type='B')

            for row, stream_name in enumerate(['A', 'B', 'C']):
                ax = axes[row, col]
                is_ac_stream = stream_name in ['A', 'C']
                data_key = 'A' if is_ac_stream else 'B'

                curr_logits = {'A': logits_A, 'B': logits_B, 'C': logits_C}[stream_name]

                # 1. Heatmap Base
                if curr_logits is not None:
                    X_test = self._to_np(batch_data['test'][f'X_{data_key}'][:, batch_idx]).flatten()
                    self._draw_heatmap_layer(ax, X_test, curr_logits[:, batch_idx, :])

                # 2. Ground Truth Lines
                true_y = y_dense_A if is_ac_stream else y_dense_B
                line_color, line_style, line_alpha = ('white', '-', 0.9) if is_ac_stream else ('red', '--', 0.5)
                ax.plot(self.x_dense_np, true_y, color=line_color, linestyle=line_style, linewidth=1.0,
                        alpha=line_alpha, zorder=5)

                # 3. Scatter Points
                self._draw_scatter_layer(ax, batch_data, data_key, batch_idx, is_ac_stream)

                ax.set_ylim(self.borders_np[0] - 1.5, self.borders_np[-1] + 1.5)

                if row == 0:
                    ax.set_title(f"BATCH ITEM {batch_idx}", fontweight='bold', fontsize=11, pad=10)
                if col == 0:
                    ax.set_ylabel(f"Stream {stream_name}", fontsize=10, fontweight='bold')

        for col in range(len(idx_list)):
            axes[2, col].set_xlabel("x-coordinate")

        return fig