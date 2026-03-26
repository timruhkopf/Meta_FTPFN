import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_binned_percentiles(probs_np, borders_np, percentiles=[0.025, 0.5, 0.975]):
    """Calculates continuous percentiles from binned probabilities via CDF interpolation."""
    # 1. Calculate Cumulative Distribution Function (CDF)
    cdf = np.cumsum(probs_np, axis=-1)

    # Pad with 0.0 at the beginning to match borders shape for interpolation
    cdf = np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], axis=-1)

    results = {pct: np.zeros(probs_np.shape[0]) for pct in percentiles}

    for i in range(probs_np.shape[0]):
        for pct in percentiles:
            # Find the bin index where the CDF crosses our target percentile
            idx = np.searchsorted(cdf[i], pct)

            if idx == 0:
                results[pct][i] = borders_np[0]
            elif idx >= len(borders_np):
                results[pct][i] = borders_np[-1]
            else:
                # Linear interpolation within the specific bin for a smooth line
                p_low, p_high = cdf[i, idx - 1], cdf[i, idx]
                b_low, b_high = borders_np[idx - 1], borders_np[idx]

                if p_high > p_low:
                    fraction = (pct - p_low) / (p_high - p_low)
                    results[pct][i] = b_low + fraction * (b_high - b_low)
                else:
                    results[pct][i] = b_low

    return results


def plot_single_task(ax, batch, probs, batch_idx, n_A, n_B, title, borders):
    """Helper to plot a specific batch index on a given axis with a predictive heatmap and CIs."""
    sep = batch["sep"]

    mask_A_np = batch["mask_A"][batch_idx, :sep].cpu().numpy()
    mask_B_np = batch["mask_B"][batch_idx, :sep].cpu().numpy()

    x_cA = batch["x_cA"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_A_np]
    y_cA = batch["y_cA"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_A_np]
    x_cB = batch["x_cB"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_B_np]
    y_cB = batch["y_cB"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_B_np]

    x_q = batch["x_qA"][:, batch_idx, 0].float().cpu().numpy()
    y_qA_true = batch["y_qA_true"][:, batch_idx, 0].float().cpu().numpy()
    y_qB_true = batch["y_qB_true"][:, batch_idx, 0].float().cpu().numpy()

    p = probs[:, batch_idx, :].detach().float().cpu().numpy()

    sort_idx = np.argsort(x_q)
    x_q_sorted = x_q[sort_idx]
    p_sorted = p[sort_idx, :]
    y_qA_sorted = y_qA_true[sort_idx]
    y_qB_sorted = y_qB_true[sort_idx]

    borders_np = borders.cpu().numpy()
    bin_centers = (borders_np[:-1] + borders_np[1:]) / 2.0

    # --- Plot the Predictive Heatmap ---
    ax.pcolormesh(
        x_q_sorted,
        bin_centers,  # <-- Changed from borders_np to bin_centers
        p_sorted.T,
        cmap='viridis',
        shading='nearest',  # <-- Explicitly tell it X and Y are centers
        alpha=0.9
    )

    # --- NEW: Calculate and Plot Median & 95% CI ---
    percentiles = get_binned_percentiles(p_sorted, borders_np, percentiles=[0.025, 0.5, 0.975])

    # Plot Confidence Intervals
    ax.plot(x_q_sorted, percentiles[0.025], color='orange', linestyle=':', linewidth=2, label='95% CI')
    ax.plot(x_q_sorted, percentiles[0.975], color='orange', linestyle=':', linewidth=2)

    # Plot Median
    ax.plot(x_q_sorted, percentiles[0.5], color='orange', linestyle='-', linewidth=2.5, label='Median Pred')

    # --- Plot True Data and Context Points ---
    ax.plot(x_q_sorted, y_qB_sorted, color='red', linestyle='--', alpha=0.5, label='Source Task B (Full)')
    ax.scatter(x_cB, y_cB, c='red', marker='x', s=20, alpha=0.8, label=f'Source Pts ({n_B})')

    ax.plot(x_q_sorted, y_qA_sorted, color='white', linewidth=2.0, alpha=0.9, label='Target Task A (GT)')
    ax.scatter(x_cA, y_cA, c='white', edgecolors='black', s=80, zorder=5, label=f'Target Pts ({n_A})')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(borders_np[0], borders_np[-1])
    ax.legend(loc='upper right', fontsize='small', frameon=True)
    ax.grid(False)

def plot_training_step(step, batch, probs, loss_hist_rel, loss_hist_unrel, n_A, n_B, borders, save_path):
    # CHANGED: Added 'borders' to arguments and passed 'probs' instead of 'y_pred'
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    is_unrelated = batch["is_unrelated"].cpu().numpy()

    idx_rel = np.where(~is_unrelated)[0]
    idx_unrel = np.where(is_unrelated)[0]

    # Plot Related Task
    if len(idx_rel) > 0:
        plot_single_task(ax1, batch, probs, idx_rel[0], n_A, n_B, f"Step {step}: Related (Helpful B)", borders)
    else:
        ax1.set_title("No Related Task in this Eval Batch")

    # Plot Unrelated Task
    if len(idx_unrel) > 0:
        plot_single_task(ax2, batch, probs, idx_unrel[0], n_A, n_B, f"Step {step}: Unrelated (Same A, Random B)", borders)
    else:
        ax2.set_title("No Unrelated Task in this Eval Batch")

    # Plot Loss Curves
    if len(loss_hist_rel) > 0:
        ax3.plot(loss_hist_rel, color='blue', alpha=0.2, label='Related (Raw)')
        ax3.plot(loss_hist_unrel, color='red', alpha=0.2, label='Unrelated (Raw)')

        if len(loss_hist_rel) > 50:
            avg_rel = np.convolve(loss_hist_rel, np.ones(50) / 50, mode='valid')
            avg_unrel = np.convolve(loss_hist_unrel, np.ones(50) / 50, mode='valid')
            ax3.plot(np.arange(49, len(loss_hist_rel)), avg_rel, color='blue', linewidth=2, label='Related (Avg)')
            ax3.plot(np.arange(49, len(loss_hist_unrel)), avg_unrel, color='red', linewidth=2, label='Unrelated (Avg)')

    ax3.set_title("Training Loss Separated (NLL)") # CHANGED title
    ax3.set_yscale('log')
    ax3.set_xlabel("Steps")
    ax3.legend(loc='upper right')
    ax3.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"step_{step:05d}.png", dpi=150)
    plt.close(fig)