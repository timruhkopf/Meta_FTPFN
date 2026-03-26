import numpy as np
import matplotlib.pyplot as plt


def plot_single_task(ax, batch, y_pred, batch_idx, n_A, n_B, title):
    """Helper to plot a specific batch index on a given axis."""
    sep = batch["sep"]

    # Slice for the specific batch_idx
    mask_A_np = batch["mask_A"][batch_idx, :sep].cpu().numpy()
    mask_B_np = batch["mask_B"][batch_idx, :sep].cpu().numpy()

    x_cA = batch["x_cA"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_A_np]
    y_cA = batch["y_cA"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_A_np]
    x_cB = batch["x_cB"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_B_np]
    y_cB = batch["y_cB"][:sep, batch_idx, 0].float().cpu().numpy()[~mask_B_np]

    x_q = batch["x_qA"][:, batch_idx, 0].float().cpu().numpy()
    y_qA_true = batch["y_qA_true"][:, batch_idx, 0].float().cpu().numpy()
    y_qB_true = batch["y_qB_true"][:, batch_idx, 0].float().cpu().numpy()

    y_p = y_pred[:, batch_idx, 0].detach().float().cpu().numpy()

    ax.plot(x_q, y_qB_true, color='red', linestyle='--', alpha=0.2, label='Source Task B (Full)')
    ax.scatter(x_cB, y_cB, c='red', marker='x', s=20, alpha=0.4, label=f'Source Pts ({n_B})')

    ax.plot(x_q, y_qA_true, color='blue', linewidth=1.5, alpha=0.6, label='Target Task A (GT)')
    ax.scatter(x_cA, y_cA, c='blue', edgecolors='black', s=80, zorder=5, label=f'Target Pts ({n_A})')

    ax.plot(x_q, y_p, color='green', linewidth=2.5, label='Model Prediction (A|B)')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='upper right', fontsize='small', frameon=True)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)


def plot_training_step(step, batch, y_pred, loss_hist_rel, loss_hist_unrel, n_A, n_B, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    is_unrelated = batch["is_unrelated"].cpu().numpy()

    # Find the first available indices for related and unrelated tasks
    idx_rel = np.where(~is_unrelated)[0]
    idx_unrel = np.where(is_unrelated)[0]

    # Plot Related Task (Counterfactual A)
    if len(idx_rel) > 0:
        plot_single_task(ax1, batch, y_pred, idx_rel[0], n_A, n_B, f"Step {step}: Related (Helpful B)")
    else:
        ax1.set_title("No Related Task in this Eval Batch")

    # Plot Unrelated Task (Counterfactual B)
    if len(idx_unrel) > 0:
        plot_single_task(ax2, batch, y_pred, idx_unrel[0], n_A, n_B, f"Step {step}: Unrelated (Same A, Random B)")
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

    ax3.set_title("Training Loss Separated (MSE)")
    ax3.set_yscale('log')
    ax3.set_xlabel("Steps")
    ax3.legend(loc='upper right')
    ax3.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"step_{step:05d}.png", dpi=150)
    plt.close(fig)
