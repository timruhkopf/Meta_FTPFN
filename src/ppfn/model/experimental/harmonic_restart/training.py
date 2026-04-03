import os

import mlflow
import torch
from matplotlib import pyplot as plt
from torch import optim, GradScaler, amp
from torch.utils.data import DataLoader
from tqdm import tqdm

from pfns4hpo.bar_distribution import FullSupportBarDistribution
from ppfn.model.experimental.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream
from ppfn.model.experimental.harmonic_restart.model import TriHarmonicModel


def calculate_gradient_sparsity(model, threshold=1e-8):
    """Calculates the percentage of gradients that are near zero."""
    zero_grads, total_grads = 0, 0
    for p in model.parameters():
        if p.grad is not None:
            grads = p.grad.abs()
            zero_grads += (grads < threshold).sum().item()
            total_grads += grads.numel()
    return (zero_grads / total_grads) * 100 if total_grads > 0 else 0.0

def train(
        experiment_name: str = "TriHarmonic_Transfer",
        run_name: str = "PreNorm_AdamW",
        output_dir: str = "./run_outputs",  # user specified folder
        batch_size: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_steps: int = 100000,
        warmup_pct: float = 0.05,
        grad_clip_norm: float = 1.0,
        log_every: int = 100,
        plot_every: int = 2000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Trains the Tri-Stream model using NLL loss, AMP, MLflow, and saves heatmap figures."""
    print(f"Starting Tri-Stream training on {device}. Outputs to: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Setup MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(locals())  # log all function arguments cleanly

        # 2. Setup Data, Borders, and Criterion
        # (Assuming you have imported the actual stream class from dataset.py)
        # stream = InfiniteHarmonicsStream(batch_size=batch_size, n_A=20, n_B=100, n_test=50)
        # dataloader = DataLoader(stream, batch_size=None)

        # --- STUB DATASET FOR UNIFIED SCRIPT RUNNABILITY ---
        import sys

        stream = InfiniteHarmonicsStream(batch_size=batch_size, n_A=20, n_B=100, n_test=50)
        dataloader = DataLoader(stream, batch_size=None)
        # --- END STUB ---

        borders = torch.linspace(-4.0, 4.0, steps=101).to(device)
        criterion = FullSupportBarDistribution(borders, smoothing=0.05).to(device)

        # 3. Setup Model
        # (Assuming you imported the actual model from model.py)
        model = TriHarmonicModel(d_model=d_model, nhead=nhead, num_bars=criterion.num_bars).to(device)

        # 4. Setup Optimizer, Scheduler, and AMP Scaler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=max_steps,
            pct_start=warmup_pct
        )

        scaler = GradScaler(enabled=(device == "cuda"))
        model.train()

        # 5. Training Loop
        pbar = tqdm(enumerate(dataloader), total=max_steps)
        for step, batch in pbar:
            if step >= max_steps:
                break

            optimizer.zero_grad(set_to_none=True)

            # Extract and move targets to device
            Y_test_A = batch['test']['Y_A'].to(device)
            Y_test_B = batch['test']['Y_B'].to(device)

            # Pre-move dictionary inputs (or update model forward)
            for k1 in ['train', 'test']:
                for k2 in batch[k1]:
                    batch[k1][k2] = batch[k1][k2].to(device)

            # --- Forward Pass (AMP) ---
            with amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', enabled=(device == "cuda")):
                logits_A, logits_B, logits_C = model(batch)

            # NLL Losses
            loss_A = criterion(logits_A, Y_test_A).mean()
            loss_B = criterion(logits_B, Y_test_B).mean()
            # C targets A!
            loss_C = criterion(logits_C, Y_test_A).mean()

            total_loss = loss_A + loss_B + loss_C

            # --- Backward Pass (AMP) ---
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)  # Required before sparsity calc and clipping
            grad_sparsity = calculate_gradient_sparsity(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # --- Logging (Metrics) ---
            if step % log_every == 0:
                metrics = {
                    "Loss/Total": total_loss.item(),
                    "NLL/A_Unconditional": loss_A.item(),
                    "NLL/B_Dense": loss_B.item(),
                    "NLL/C_Conditional": loss_C.item(),
                    "Metrics/NLL_Diff_C_minus_A": (loss_C - loss_A).item(),
                    "Metrics/Gradient_Sparsity_Pct": grad_sparsity,
                    "Optim/Learning_Rate": scheduler.get_last_lr()[0]
                }
                mlflow.log_metrics(metrics, step=step)

                # print(f"Step {step:05d} | NLL A: {loss_A.item():.3f} | NLL C: {loss_C.item():.3f} | "
                #       f"Diff (C-A): {(loss_C - loss_A).item():.3f} | Sparse: {grad_sparsity:.1f}%")
                pbar.set_description(f"Step {step:05d} - Total Loss: {total_loss.item():.4f} - NLL A: {loss_A.item():.3f} - "
                                        f"NLL C: {loss_C.item():.3f} - Sparse: {grad_sparsity:.1f}% -- Diff (C-A): {(loss_C - loss_A).item():.3f}")
            # --- Logging (Plots to File) ---
            if step % plot_every == 0 or step == max_steps - 1:
                model.eval()
                with torch.no_grad():
                    l_A, l_B, l_C = model(batch)
                model.train()

                fig = plt.figure(figsize=(10, 8))
                plot_name = f"heatmaps_step_{step:05d}.png"
                save_path = os.path.join(plot_dir, plot_name)

                InfiniteHarmonicsStream.save_heatmaps(
                    fig=fig, batch_data=batch, borders=borders, save_path=save_path,
                    logits_A=l_A, logits_B=l_B, logits_C=l_C, batch_idx=0
                )

                # Log plot to MLflow as artifact
                mlflow.log_artifact(save_path, "heatmap_plots")

        print("Training complete.")


if __name__ == "__main__":
    import fire
    fire.Fire(train)