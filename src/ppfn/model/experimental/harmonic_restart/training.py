import os
from pathlib import Path

import mlflow
import torch
from matplotlib import pyplot as plt
from torch import optim, GradScaler, amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from pfns4hpo.bar_distribution import FullSupportBarDistribution
from ppfn.model.experimental.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream
from ppfn.model.experimental.harmonic_restart.model import TriHarmonicModel
from ppfn.model.mymodel.meta_context import ForwardMetaContext

import logging

logger = logging.getLogger(__name__)


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
        batch_size: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        max_steps: int = 100000,
        warmup_steps: int = 10000,
        save_chkpt: bool = True,
        load_chkpt: str = "",  # path to checkpoint file to load (if any)
        warmup_pct: float = 0.05,
        grad_clip_norm: float = 1.0,
        log_every: int = 100,
        plot_every: int = 2000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path="."
):
    """Trains the Tri-Stream model using NLL loss, AMP, MLflow, and saves heatmap figures."""
    logger.info(f"Starting Tri-Stream training on {device}. Outputs to: {save_path}")

    # xor arguments
    assert not (save_chkpt and bool(
        load_chkpt)), "Cannot both save and load checkpoints in the same run. Please choose one."
    assert not (bool(load_chkpt) and warmup_steps > 0), "Either load a checkpoint or do warmup training"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, f"{experiment_name}_{run_name}_{timestamp}")

    # Ensure output directory exists

    Path(save_path).mkdir(exist_ok=True, parents=True)
    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    logger.info("Output directories set up at: {}".format(save_path))

    # 1. Setup MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(locals())  # log all function arguments cleanly

        stream = InfiniteHarmonicsStream(batch_size=batch_size, n_A=10, n_B=50, n_test=300)
        dataloader = DataLoader(stream, batch_size=None)

        borders = torch.linspace(-7.0, 7.0, steps=150).to(device)
        criterion = FullSupportBarDistribution(borders, smoothing=0.05).to(device)

        # 3. Setup Model
        # (Assuming you imported the actual model from model.py)
        model = TriHarmonicModel(d_model=d_model, nhead=nhead, num_bars=criterion.num_bars).to(device)

        if bool(load_chkpt):
            model.load_state_dict(torch.load(load_chkpt, map_location=device, weights_only=True))

        # 4. Setup Optimizer, Scheduler, and AMP Scaler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=max_steps,
            pct_start=warmup_pct
        )

        scaler = GradScaler(enabled=(device == "cuda"))
        model.train()

        # 5. Training Loop
        pbar = tqdm(enumerate(dataloader), total=max_steps, )
        for step, batch in pbar:
            ForwardMetaContext.clear()
            if step >= max_steps:
                break

            if step == warmup_steps:
                # freeze the backend model
                logger.info("Warmup complete. Freezing marginal backend; unlocking Adapter C.")
                for param in set(model.parameters()) - set(model.layer.parameters()):
                    param.requires_grad = False  # Unfreeze all parameters after warmup

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

            if step < warmup_steps:
                total_loss = loss_A + loss_B

            else:
                total_loss = loss_C

            # --- Backward Pass (AMP) ---
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)  # Required before sparsity calc and clipping
            grad_sparsity = calculate_gradient_sparsity(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step == warmup_steps) and warmup_steps > 0 and save_chkpt:
                torch.save(model.state_dict(), os.path.join(save_path, f"model_checkpoint_step_{step:05d}.pt"))

            # --- Logging (Metrics) ---
            if step % log_every == 0:

                # 1. Get the boolean mask from the batch
                is_unrelated = batch['params']['is_unrelated'].to(device)

                # 2. Get unreduced losses (Shape: [Seq, Batch])
                loss_C_unreduced = criterion(logits_C, Y_test_A)
                loss_A_unreduced = criterion(logits_A, Y_test_A)
                diff_unreduced = loss_C_unreduced - loss_A_unreduced

                # 3. Safely calculate means based on the mask
                # We index along dimension 1 because your tensors are (Seq, Batch)
                diff_all = diff_unreduced.mean().item()

                if (~is_unrelated).any():
                    diff_related = diff_unreduced[:, ~is_unrelated].mean().item()
                else:
                    diff_related = 0.0  # Fallback if no related samples in this batch

                if is_unrelated.any():
                    diff_unrelated = diff_unreduced[:, is_unrelated].mean().item()
                else:
                    diff_unrelated = 0.0  # Fallback if no unrelated samples in this batch

                metrics = {
                    "nll/Total": total_loss.item(),
                    "nll/A": loss_A.item(),
                    "nll/B": loss_B.item(),
                    "nll/C": loss_C.item(),
                    "nll/C-A_all": diff_all,
                    "nll/C-A_related": diff_related,  # TARGET: Strongly Negative (C is much better)
                    "nll/C-A_unrelated": diff_unrelated,  # TARGET: Around 0.0 (C falls back to A)
                    "Metrics/Gradient_Sparsity_Pct": grad_sparsity,
                    "Metrics/Learning_Rate": scheduler.get_last_lr()[0]
                }
                mlflow.log_metrics(metrics, step=step)

                # log attn sink metrics:
                cross_attn_weights = ForwardMetaContext.get('cross_attn_weights')

                if cross_attn_weights is not None:
                    # cross_attn_weights shape: (B, Seq_C, Seq_B + 1)
                    # 1. Isolate attention paid specifically to the sink (last token)
                    # 2. Average it across all queries in Stream C
                    sink_attn = cross_attn_weights[:, :, -1].mean(dim=1)  # Shape: (B,)

                    # Split by your prior's relationship mask
                    is_unrelated = batch['params']['is_unrelated'].to(device)

                    sink_attn_unrelated = sink_attn[is_unrelated].mean().item() if is_unrelated.any() else 0.0
                    sink_attn_related = sink_attn[~is_unrelated].mean().item() if (~is_unrelated).any() else 0.0

                    mlflow.log_metrics({
                        "Gate/AttnSink: Unrelated_Trap": sink_attn_unrelated,
                        "Gate/AttnSink: Related_Task": sink_attn_related,
                    }, step=step)

                gate_vals = ForwardMetaContext.get('gate')
                is_unrelated = batch['params']['is_unrelated'].to(device)

                if gate_vals is not None:
                    # Average across the sequence dimension to get one value per batch item
                    avg_gate_per_batch = gate_vals.mean(dim=0).squeeze(-1)  # (Batch,)

                    gate_unrelated = avg_gate_per_batch[is_unrelated].mean().item() if is_unrelated.any() else 0.0
                    gate_related = avg_gate_per_batch[~is_unrelated].mean().item() if (~is_unrelated).any() else 0.0

                    mlflow.log_metrics({
                        "Gate/Related": gate_related,  # Target: High (~1.0)
                        "Gate/Unrelated": gate_unrelated,  # Target: Low (~0.0)
                    }, step=step)

                # print(f"Step {step:05d} | NLL A: {loss_A.item():.3f} | NLL C: {loss_C.item():.3f} | "
                #       f"Diff (C-A): {(loss_C - loss_A).item():.3f} | Sparse: {grad_sparsity:.1f}%")
                pbar.set_description(
                    f"Step {step:05d} - "
                    f"Total Loss: {total_loss.item():.4f} - "
                    f"NLL A: {loss_A.item():.3f} - "
                    f"NLL C: {loss_C.item():.3f} - "
                    f"Sparse: {grad_sparsity:.1f}% -- "
                    f"Diff (C-A): {(loss_C - loss_A).item():.3f}"
                )

            # --- Logging (Plots to File) ---
            if step % plot_every == 0 or step == max_steps - 1:
                fig = plt.figure(figsize=(10, 8))
                plot_name = f"heatmaps_step_{step:05d}.png"
                plot = os.path.join(plot_dir, plot_name)
                os.makedirs(os.path.dirname(plot), exist_ok=True)

                # Pass the model; it will handle the eval()/no_grad() context automatically
                InfiniteHarmonicsStream.save_heatmaps(
                    fig=fig,
                    batch_data=batch,
                    borders=borders,
                    save_path=plot,
                    model=model  # <-- Dynamic Logits
                )

                # Log plot to MLflow as artifact
                mlflow.log_artifact(plot, "heatmap_plots")

        print("Training complete.")


if __name__ == "__main__":
    import fire

    fire.Fire(train)
