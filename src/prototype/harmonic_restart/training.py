import os
from pathlib import Path

import mlflow
import torch
from matplotlib import pyplot as plt
from torch import optim, GradScaler, amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

import torch.nn.functional as F

from pfns4hpo.bar_distribution import FullSupportBarDistribution
from prototype.harmonic_restart import HarmonicsVisualizer
from prototype.harmonic_restart.harmonic_prior import GlobalSparseHarmonicsStream
from prototype.harmonic_restart.model import TriHarmonicModel
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
        save_path=".",
        share_unrelated=0.2,
        train_jointly=False,
        use_attn_bonus=False,
        n_A=10,
        n_B=50,
        scale=True,
        shift=True,
        warp=True,
):
    assert train_jointly == False
    import subprocess
    import os
    from pathlib import Path

    # --- Git Extraction Logic ---
    try:
        # Get the current commit hash (short version)
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

        # Capture uncommitted changes (the diff)
        git_diff = subprocess.check_output(['git', 'diff']).decode('utf-8')

        # Path for a temporary diff file
        diff_file_path = os.path.join(save_path, "uncommitted_changes.diff")
        with open(diff_file_path, "w") as f:
            f.write(git_diff)
    except Exception as e:
        git_hash = "unknown"
        diff_file_path = None
        logger.warning(f"Could not capture Git state: {e}")

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
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        logger.info(f'mlflow db location: {mlflow.get_tracking_uri()}')
        # Log hyperparameters
        mlflow.log_params(locals())  # log all function arguments cleanly

        # 2. Specifically log the diff file as an artifact if it exists
        if diff_file_path and os.path.getsize(diff_file_path) > 0:
            mlflow.log_artifact(diff_file_path, artifact_path="git_metadata")
            logger.info("Git diff logged to MLflow.")
        elif diff_file_path:
            logger.info("No uncommitted changes to log.")

        # class imbalance for the gate loss:
        pos_weight = share_unrelated / (1 - share_unrelated)
        global_pos_weight = torch.tensor([pos_weight], device=device)

        stream = GlobalSparseHarmonicsStream(
            batch_size=batch_size, n_A=n_A, n_B=n_B, n_test=300,
            share_unrelated=share_unrelated, scale=scale, shift=shift, warp=warp)
        dataloader = DataLoader(stream, batch_size=None)

        borders = torch.linspace(-7.0, 7.0, steps=250).to(device)
        criterion = FullSupportBarDistribution(borders, smoothing=0.05).to(device)

        # 3. Setup Model
        # (Assuming you imported the actual model from model.py)
        from prototype.harmonic_restart.layer import TriStreamLayer
        model = TriHarmonicModel(
            cross_attn_layer=TriStreamLayer(
                d_model=d_model, dim_feedforward=d_model, nhead=nhead, use_B_attn_sink=False,
                use_hp=False, use_add_pfn=True, use_post_attn=False, cross_attn_type='deform',
                num_align_steps=2, use_spectral_norm=False
            ),
            d_model=d_model, nhead=nhead, num_bars=criterion.num_bars).to(device)

        if bool(load_chkpt):
            model.load_state_dict(torch.load(load_chkpt, map_location=device, weights_only=True), strict=False)

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

            if train_jointly:
                pass
            else:
                if step == 0:
                    # freeze the adapter
                    logger.info("Freezing Adapter A and B for warmup phase.")
                    for param in model.layer.parameters():
                        param.requires_grad = False  # Freeze adapter parameters during warmup

                if step == warmup_steps:
                    # freeze the backend model
                    logger.info("Warmup complete. Freezing marginal backend; unlocking Adapter C.")
                    for param in set(model.parameters()) - set(model.layer.parameters()):
                        param.requires_grad = False  # Unfreeze all parameters after warmup

                    for param in model.layer.parameters():
                        param.requires_grad = True

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

            # ==========================================================
            # SUPERVISED GUIDED ATTENTION (Proposal B)
            # ==========================================================
            # 1. Retrieve the cross attention weights from the layer
            # Shape: (Batch * nhead, Seq_C, Seq_B + 1)
            loss_guided_attn = torch.tensor(0.0, device=device)
            raw_attn_weights = ForwardMetaContext.get('cross_attn_weights')
            if use_attn_bonus and raw_attn_weights is not None:
                # Fixme: this should not be a hill bonus, but an attractor valley with flat valley.
                #  But the idea is flawed, because deeper layers might be "lobotomized" if we force it to attend based on
                #  spatial attention only. Discovering the optimal manifold alignment on its own is a better way fwd.
                if raw_attn_weights.dim() == 4:
                    attn_weights = raw_attn_weights.mean(dim=1)  # (Batch, Seq_C, Seq_B or Seq_B+1)
                else:
                    attn_weights = raw_attn_weights

                # 1. Reconstruct Coordinates Safely
                X_C = torch.cat([batch['train']['X_A'], batch['test']['X_A']], dim=0)
                X_C_clean = torch.nan_to_num(X_C, nan=0.0).transpose(0, 1)  # (Batch, Seq_C)

                X_B_train = batch['train']['X_B']
                X_B_clean = torch.nan_to_num(X_B_train, nan=0.0).transpose(0, 1)  # (Batch, Seq_B)

                # 2. Target coordinates in B's space
                # FIXME: warping is not accounted for
                h_shift = batch['params']['h_shift_A'].to(device).view(-1, 1)
                X_target_b = X_C_clean - h_shift  # (Batch, Seq_C)

                # 3. Pairwise Squared Distances
                dist_sq = (X_target_b.unsqueeze(2) - X_B_clean.unsqueeze(1)) ** 2

                # 4. Create the "Bonus Hill" (Values from 0.0 to 1.0)
                # sigma defines the width of your "neighborhood"
                sigma = 0.5
                bonus_landscape = torch.exp(-dist_sq / (2 * sigma ** 2))

                # 5. Calculate Expected Bonus (Reward)
                Seq_B = X_B_clean.shape[1]
                attn_to_B = attn_weights[:, :, :Seq_B]

                # Expected Reward = Sum of (Probability * Bonus_Value)
                expected_bonus = (attn_to_B * bonus_landscape).sum(dim=-1)  # (Batch, Seq_C)

                # 6. Convert Reward to Loss (Negative Log)
                # If expected_bonus is 1.0 (perfect), loss is 0.
                # If expected_bonus is 0.001 (terrible), loss is ~6.9.
                bonus_loss = -torch.log(expected_bonus + 1e-8)

                # 7. Handle Traps & Sinks
                is_unrelated = batch['params']['is_unrelated'].to(device)  # (Batch,)

                if model.layer.use_B_attn_sink:
                    # For traps, the "neighborhood" is simply the sink token
                    sink_attn = attn_weights[:, :, -1]
                    trap_loss = -torch.log(sink_attn + 1e-8)

                    loss_per_batch = torch.where(is_unrelated.unsqueeze(1), trap_loss, bonus_loss)
                else:
                    loss_per_batch = torch.where(is_unrelated.unsqueeze(1), torch.zeros_like(bonus_loss), bonus_loss)

                # 8. Final Mean
                loss_guided_attn = loss_per_batch.mean()

                if step % log_every:
                    mlflow.log_metrics({
                        "metrics/guided_Attn_Loss": loss_guided_attn.item()
                    }, step=step)

            # NLL Losses
            loss_A = criterion(logits_A, Y_test_A).mean()
            loss_B = criterion(logits_B, Y_test_B).mean()

            # C targets A!
            loss_C = criterion(logits_C, Y_test_A).mean()

            if train_jointly:
                total_loss = loss_A + loss_B + loss_C
            else:
                if step < warmup_steps:
                    total_loss = loss_A + loss_B

                else:
                    total_loss = loss_C + loss_guided_attn

            aux = ForwardMetaContext.get('B_in_A_domain')
            if aux is not None and 'kl_loss' in aux:
                total_loss += 1e-5 * aux['kl_loss']

            if aux is not None and 'cycle_loss' in aux:
                total_loss += 1e-5 * aux['cycle_loss']

            total_nll_loss = total_loss.clone().item()

            # 2. Fetch the gate values and the mask
            gate_logits_val = ForwardMetaContext.get('gate_logits')
            is_unrelated = batch['params']['is_unrelated'].to(device)

            if gate_logits_val is not None:
                # 2. Squeeze and cast to float32 for stable loss calculation
                gate_logits = gate_logits_val.squeeze(-1).float()

                # 3. Create the Ideal Target (1.0 for Related, 0.0 for Trap)
                ideal_gate = (~is_unrelated).float()

                # 4. Compute Weighted Loss using the built-in PyTorch parameter
                loss_gate = F.binary_cross_entropy_with_logits(
                    gate_logits,
                    ideal_gate,
                    pos_weight=global_pos_weight  # <--- The Magic Bullet for Imbalance
                )

                # 5. Add to main loss
                gate_loss_weight = 1
                total_loss += (gate_loss_weight * loss_gate)

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
                    "nll/Total": total_nll_loss,
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

                # therm = ForwardMetaContext.get('thermodynamic_energy')
                # transport = ForwardMetaContext.get('transport_energy')
                # kinetic_energy = ForwardMetaContext.get('kinetic_energy')
                #
                # metrics = {
                #     "energy/thermodynamic_energy": therm.mean().item() if therm is not None else 0.0,
                #     "energy/transport": transport.mean().item() if transport is not None else 0.0,
                #     "energy/kinetic_energy": kinetic_energy.mean().item(),
                # }
                #
                # mlflow.log_metrics(metrics, step=step)

                # log attn sink metrics:
                # cross_attn_weights = ForwardMetaContext.get('cross_attn_weights')
                #
                # if cross_attn_weights is not None:
                #     # cross_attn_weights shape: (B, Seq_C, Seq_B + 1)
                #     # 1. Isolate attention paid specifically to the sink (last token)
                #     # 2. Average it across all queries in Stream C
                #     sink_attn = cross_attn_weights[:, :, -1].mean(dim=1)  # Shape: (B,)
                #
                #     # Split by your prior's relationship mask
                #     is_unrelated = batch['params']['is_unrelated'].to(device)
                #
                #     sink_attn_unrelated = sink_attn[is_unrelated].mean().item() if is_unrelated.any() else 0.0
                #     sink_attn_related = sink_attn[~is_unrelated].mean().item() if (~is_unrelated).any() else 0.0
                #
                #     mlflow.log_metrics({
                #         "Gate/AttnSink: Unrelated_Trap": sink_attn_unrelated,
                #         "Gate/AttnSink: Related_Task": sink_attn_related,
                #     }, step=step)

                # TODO: check auc_roc, precision-recall, Type I / Type II error etc. for the gate predictions (if they exist in the context)
                gate_vals = ForwardMetaContext.get('gate')
                is_unrelated = batch['params']['is_unrelated'].to(device)

                if gate_vals is not None:
                    # The gate is ALREADY task-level: shape is (Batch, 1)
                    # We just squeeze the last dimension to match the mask: shape -> (Batch,)
                    gate_per_batch = gate_vals.squeeze(-1)

                    # Apply masks
                    gate_unrelated = gate_per_batch[is_unrelated].mean().item() if is_unrelated.any() else 0.0
                    gate_related = gate_per_batch[~is_unrelated].mean().item() if (~is_unrelated).any() else 0.0

                    mlflow.log_metrics({
                        "Gate/Related": gate_related,  # Target: High (~1.0)
                        "Gate/Unrelated": gate_unrelated  # Target: Low (~0.0)
                    }, step=step)

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
                raise NotImplementedError('This section needs updating after refactor' )
                fig = plt.figure(figsize=(10, 8))
                plot_name = f"heatmaps_step_{step:05d}.png"
                plot = os.path.join(plot_dir, plot_name)
                os.makedirs(os.path.dirname(plot), exist_ok=True)

                # Pass the model; it will handle the eval()/no_grad() context automatically
                batch, _ = self.trainer._get_next_batch()

                logits_A, logits_B, logits_C = self.trainer.model(batch)

                fig = plt.figure(figsize=(10, 8))
                plot_name = f"heatmaps_step_{step:05d}.png"
                plot_path = os.path.join(self.plot_dir, plot_name)

                # Updated to use the separated Visualizer class
                HarmonicsVisualizer.save_heatmaps(
                    fig=fig,
                    batch_data=batch,
                    borders=self.trainer.criterion.criterion_backend.borders,
                    save_path=plot_path,
                    logits_A=logits_A,
                    logits_B=logits_B,
                    logits_C=logits_C,
                    plot=False
                )

                # Log plot to MLflow as artifact
                mlflow.log_artifact(plot, "heatmap_plots")

        print("Training complete.")


if __name__ == "__main__":
    import fire
    import os
    import sys

    # --- Setup paths for the printout ---
    # project_dir = os.getcwd()
    # python_exec = sys.executable
    # script_name = sys.argv[0]
    # args = ' '.join(sys.argv[1:])
    # db_uri = f"sqlite:///{project_dir}/mlflow.db"
    #
    # # --- The Cheat Sheet String ---
    # print("\n" + "═" * 60)
    # print(" 🚀 REMOTE TRAINING COCKPIT")
    # print("═" * 60)
    #
    # print(f"\n1. START/ATTACH TO TMUX WHEN LOGGED IN ON REMOTE:")
    # print(f"   tmux attach -t training || tmux new -s training")
    #
    # print(f"\n2. RUN YOUR TRAINING (Copy & Paste into tmux):")
    # print(f"   cd {project_dir} && {python_exec} {script_name} {args}")
    #
    # print(f"\n3. LAUNCH MLFLOW UI (In a second tmux window or separate terminal):")
    # print(f"   mlflow ui --backend-store-uri {db_uri} --port 5010 --host 0.0.0.0")
    #
    # print(f"\n4. TMUX EMERGENCY CHEAT SHEET:")
    # print(f"   • Detach (Keep running!):  Ctrl+B, then D")
    # print(f"   • Create new window:       Ctrl+B, then C")
    # print(f"   • Switch windows:          Ctrl+B, then [0-9]")
    # print(f"   • Scroll (Copy mode):      Ctrl+B, then [    (Use arrows, 'q' to exit)")
    #
    # print("═" * 60 + "\n")

    fire.Fire(train)
