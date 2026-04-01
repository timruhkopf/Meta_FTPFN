import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

import mlflow

from pfns4hpo.bar_distribution import FullSupportBarDistribution
from ppfn.model.experimental.layers.glt_adapter import GatedLatentTransferLayer
from ppfn.model.experimental.validation_manifold_adapter.meta_transfer_backbone import MetaTransferModel
from ppfn.model.experimental.validation_manifold_adapter.plotting import plot_training_step
from ppfn.model.experimental.validation_manifold_adapter.prior import create_padded_batch, \
    VectorizedComplexTaskGenerator

from ppfn.model.mymodel.meta_context import ForwardMetaContext


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_meta_model(
        device,
        model,
        criterion,
        generator,
        optimizer,
        scheduler,
        steps=2000,
        batch_size=32,
        n_A=5,
        n_B=30,
        n_query=50,
        save_path=Path('.'),
        plot_every=200,
        compile_model=True,
        clip_norm=1.0
):
    logger.info('Starting training loop...')
    model = model.to(device)

    # Add PyTorch compilation for massive speedups on Ampere GPUs
    if compile_model and torch.__version__.startswith('2.') and device.type == 'cuda':
        print("Compiling model for faster execution...")
        model = torch.compile(model)

    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    # scaler = torch.amp.GradScaler('cuda', enabled=True)

    # CHANGED: Track related and unrelated losses separately
    loss_history_rel = []
    loss_history_unrel = []

    iterator = tqdm(range(steps + 1), disable=None, mininterval=10.0)

    for step in iterator:
        ForwardMetaContext.clear()
        model.train()
        optimizer.zero_grad()

        batch = create_padded_batch(generator, batch_size, n_A, n_B, n_query, device, share_unrelated=0.2)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # It is critical, that the BarDistribution criterion is outside the autocast!
            y_pred = model(batch)  # Shape: [T_query, Batch, num_bins]

        # Calculate unreduced loss using BarDistribution
        # BarDistribution returns shape [T_query, Batch]
        # loss_tensor = criterion(logits=y_pred, y=batch["y_qA_true"]) # notice, that here the loss was based on bfp16
        logits_fp32 = y_pred.float()
        y_true_fp32 = batch["y_qA_true"].float()

        # Now compute the loss with full precision
        loss_tensor = criterion(logits=logits_fp32, y=y_true_fp32)

        # Average over sequence dimension (dim=0)
        # Note: Removed dim=2 because BarDistribution drops the trailing '1' dimension
        loss_per_item = loss_tensor.mean(dim=0)

        # The total loss to actually backpropagate
        total_loss = loss_per_item.mean()

        # Use the scaler to backward (prevents underflow in FP16/BF16)
        # scaler.scale(total_loss).backward()
        total_loss.backward()

        # --- GRADIENT CLIPPING BLOCK ---
        # 1. Unscale the gradients so the norm is calculated correctly
        # scaler.unscale_(optimizer)

        # 2. Clip the gradients (max_norm=1.0 is standard for Transformers/MLPs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        mlflow.log_metric("grad_norm/unclipped", unclipped_norm, step=step)
        # scaler.step(optimizer)

        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=step)
        optimizer.step()
        scheduler.step()
        # scaler.update()

        # --- LOSS SPLITTING & LOGGING ---
        # Split the loss based on the boolean mask (we cast to float32 before calling item
        # to ensure compatibility with BF16)
        is_unrel = batch["is_unrelated"]

        if (~is_unrel).any():
            l_rel = loss_per_item[~is_unrel].mean().float().item()
            loss_history_rel.append(l_rel)
            mlflow.log_metric("nll/loss_rel", loss_history_rel[-1], step=step)
        else:
            loss_history_rel.append(loss_history_rel[-1] if loss_history_rel else 0.0)
            mlflow.log_metric("nll/loss_rel", loss_history_rel[-1], step=step)

        if is_unrel.any():
            l_unrel = loss_per_item[is_unrel].mean().float().item()
            loss_history_unrel.append(l_unrel)
            mlflow.log_metric("nll/loss_unrel", loss_history_unrel[-1], step=step)
        else:
            loss_history_unrel.append(loss_history_unrel[-1] if loss_history_unrel else 0.0)
            mlflow.log_metric("nll/loss_unrel", loss_history_unrel[-1], step=step)

        if step == 0:
            mem_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            print(f"\n[GPU Monitor] Step 0: Max Allocated: {mem_allocated:.2f}GB | Max Reserved: {mem_reserved:.2f}GB")
            torch.cuda.reset_peak_memory_stats(device)

        metrics = ForwardMetaContext._state.__dict__
        metrics = {k: float(v.cpu().item()) for k, v in metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
        mlflow.log_metrics(metrics, step=step)

        # TODO track nll/C-A

        if step % 10 == 0:
            iterator.set_description(
                f"Step {step}/{steps} | Rel: {loss_history_rel[-1]:.4f} | Unrel: {loss_history_unrel[-1]:.4f}")

        if step % plot_every == 0:
            model.eval()
            with torch.no_grad():
                # 1. Generate a batch of 2 purely RELATED tasks
                eval_batch = create_padded_batch(generator, 2, n_A, n_B, n_query, device, share_unrelated=0.0)

                # 2. The Counterfactual Injection:
                # Force Batch Item 1 to have the EXACT same Target (A) as Batch Item 0,
                # but keep its own randomly generated Source (B).
                keys_to_copy = ["x_cA", "y_cA", "x_qA", "y_qA_true", "x_cC", "y_cC"]
                for k in keys_to_copy:
                    eval_batch[k][:, 1, :] = eval_batch[k][:, 0, :]

                # 3. Explicitly label Batch Item 1 as the unrelated trap
                eval_batch["is_unrelated"] = torch.tensor([False, True], device=device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    eval_pred_logits = model(eval_batch)

                    # Convert logits to probabilities for the heatmap
                    eval_probs = torch.softmax(eval_pred_logits, dim=-1)

                plot_training_step(
                    step,
                    eval_batch,
                    eval_probs,  # Passed probabilities instead of logits/means
                    loss_history_rel,
                    loss_history_unrel,
                    n_A, n_B,
                    borders,  # Passed borders down to the plotting func
                    save_path
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Meta-Transfer Model with Gated Latent Transfer Layer")
    parser.add_argument('--mlflow_experiment', type=str, default="MetaTransfer_Validation",
                        help='MLflow experiment name')
    parser.add_argument('--mlflow_run_name', type=str, default="GatedLatentTransfer_Run", help='MLflow run name')
    parser.add_argument('--mlflow_tracking_uri', type=str, default="sqlite:////home/ruhkopf/PycharmProjects/Meta_FTPFN/mlflow.db", help='MLflow tracking URI')
    parser.add_argument('--pool_mode', type=str, default="mean",
                        choices=["mean", "softmax", "softmin", "distributional"], help='Pooling mode for the gate')
    parser.add_argument('--hp_mode', type=str, default="concat",
                        choices=["concat", "add", "ignore_hp"],
                        help='How to incorporate hyperparameters into the transfer layer')
    parser.add_argument('--pointwise', action='store_true', help='Whether to use pointwise gating')
    parser.add_argument('--global_gate', action='store_true', help='Whether to use a global gate in addition to the pointwise gate')


    parser.add_argument('--steps', type=int, default=100000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size for training')
    parser.add_argument('--n_A', type=int, default=5, help='Number of context points for target task A')
    parser.add_argument('--n_B', type=int, default=30, help='Number of context points for source task B')
    parser.add_argument('--n_query', type=int, default=50, help='Number of query points')
    parser.add_argument('--plot_every', type=int, default=1000, help='Plot every N steps')
    parser.add_argument('--save_path', type=str, default="./outputs/debug/", help='Path to save training plots')
    parser.add_argument('--max_lr', type=float, default=4e-4, help='Maximum learning rate for OneCycleLR')
    parser.add_argument('--pct_start', type=float, default=0.20,
                        help='Percentage of steps to spend on the warmup phase of OneCycleLR')

    args = parser.parse_args()

    logger.info('Parsed command-line arguments:')
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    logger.info('Setting up training environment...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    num_bins = 128  # e.g., 100 bins means 101 borders
    borders = torch.linspace(-8.0, 8.0, num_bins + 1, device=device)
    criterion = FullSupportBarDistribution(borders=borders, smoothing=0.05)  # Using your custom class

    generator = VectorizedComplexTaskGenerator()

    model = MetaTransferModel(
        transfer_layer=GatedLatentTransferLayer(
            dmodel=128, use_struct_gate=True,
            gate_params={
                "use_pointwise": args.pointwise,
                "use_global": args.global_gate,
                "pool_mode": args.pool_mode
            },
            hp_mode=args.hp_mode
        ),
        input_dim=1,
        num_bins=num_bins,
        pre_train=False
    )
    model = model.to(device)

    STEPS = args.steps
    max_lr = args.max_lr
    optimizer = optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=STEPS + 1,
        pct_start=args.pct_start,  # Spends the first 10% of steps warming up
        anneal_strategy='cos'
    )

    import os
    from datetime import datetime

    logger.info('Initializing MLflow tracking...')
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_dir = f"{args.mlflow_run_name}_{timestamp}"
    # FIXME: for local debugging, this will end up in the validation_manifold_adapter
    # fixme: log save_path as a param & logger.info
    args.save_path = os.path.join("outputs", args.mlflow_experiment, run_dir)

    # Ensure the directory exists
    os.makedirs(args.save_path, exist_ok=True)

    with mlflow.start_run(run_name=args.mlflow_run_name ):
        mlflow.log_params(vars(args))

        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_metric("model/num_params", n_params)


        train_meta_model(
            device=device,
            model=model,
            criterion=criterion,
            generator=generator,
            optimizer=optimizer,
            scheduler=scheduler,

            steps=STEPS,
            batch_size=args.batch_size,
            n_A=args.n_A,  # sparse target
            n_B=args.n_B,  # dense related
            plot_every=args.plot_every,
            save_path=Path(args.save_path),
            compile_model=False
        )

    logger.info("done")
