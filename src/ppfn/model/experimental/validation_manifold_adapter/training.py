import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from pfns4hpo.bar_distribution import BarDistribution, FullSupportBarDistribution
from ppfn.model.experimental.layers.glt_adapter import MLP, GatedLatentTransferLayer
from ppfn.model.experimental.validation_manifold_adapter.plotting import plot_training_step
from ppfn.model.experimental.validation_manifold_adapter.prior import create_padded_batch, \
    VectorizedComplexTaskGenerator


class MetaTransferModel(nn.Module):
    def __init__(self, transfer_layer, input_dim=1, num_bins=100):  # <-- Added num_bins
        super().__init__()

        self.transfer_layer = transfer_layer
        self.dmodel = self.transfer_layer.dmodel
        self.y_proj = MLP(input_dim, self.dmodel, self.dmodel)
        self.x_proj = MLP(input_dim, self.dmodel, self.dmodel)

        # CHANGED: Project out to the number of bins, not a single continuous value
        self.out_proj = MLP(self.dmodel, self.dmodel, num_bins)

    def forward(self, batch):
        A = self.y_proj(batch["y_cA"])
        B = self.y_proj(batch["y_cB"])
        C = self.y_proj(batch["y_cC"])

        hp_A = self.x_proj(batch["x_cA"])
        hp_B = self.x_proj(batch["x_cB"])
        hp_C = self.x_proj(batch["x_cC"])

        A += hp_A
        B += hp_B
        C += hp_C

        _, _, C_out = self.transfer_layer(
            A, B, C, sep=batch["sep"], hp=(hp_A, hp_B, hp_C),
            mask_A=batch["mask_A"], mask_B=batch["mask_B"]
        )

        sep = batch["sep"]
        query_out = C_out[sep:, :, :]

        return self.out_proj(query_out)  # Shape will now be [T_query, Batch, num_bins]

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
        compile_model=True
):
    model = model.to(device)

    # Add PyTorch compilation for massive speedups on Ampere GPUs
    if compile_model and torch.__version__.startswith('2.') and device.type == 'cuda':
        print("Compiling model for faster execution...")
        model = torch.compile(model)

    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # CHANGED: Track related and unrelated losses separately
    loss_history_rel = []
    loss_history_unrel = []

    iterator = tqdm(range(steps + 1))

    for step in iterator:
        model.train()
        optimizer.zero_grad()

        batch = create_padded_batch(generator, batch_size, n_A, n_B, n_query, device, share_unrelated=0.2)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # It is critical, that the BarDistribution criterion is outside the autocast!
            y_pred = model(batch)  # Shape: [T_query, Batch, num_bins]

        # Calculate unreduced loss using BarDistribution
        # BarDistribution returns shape [T_query, Batch]
        loss_tensor = criterion(logits=y_pred, y=batch["y_qA_true"])

        # Average over sequence dimension (dim=0)
        # Note: Removed dim=2 because BarDistribution drops the trailing '1' dimension
        loss_per_item = loss_tensor.mean(dim=0)

        # The total loss to actually backpropagate
        total_loss = loss_per_item.mean()

        # Use the scaler to backward (prevents underflow in FP16/BF16)
        scaler.scale(total_loss).backward()

        # --- GRADIENT CLIPPING BLOCK ---
        # 1. Unscale the gradients so the norm is calculated correctly
        scaler.unscale_(optimizer)

        # 2. Clip the gradients (max_norm=1.0 is standard for Transformers/MLPs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # --- LOSS SPLITTING & LOGGING ---
        # Split the loss based on the boolean mask (we cast to float32 before calling item
        # to ensure compatibility with BF16)
        is_unrel = batch["is_unrelated"]

        if (~is_unrel).any():
            l_rel = loss_per_item[~is_unrel].mean().float().item()
            loss_history_rel.append(l_rel)
        else:
            loss_history_rel.append(loss_history_rel[-1] if loss_history_rel else 0.0)

        if is_unrel.any():
            l_unrel = loss_per_item[is_unrel].mean().float().item()
            loss_history_unrel.append(l_unrel)
        else:
            loss_history_unrel.append(loss_history_unrel[-1] if loss_history_unrel else 0.0)

        if step == 0:
            mem_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            mem_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            print(f"\n[GPU Monitor] Step 0: Max Allocated: {mem_allocated:.2f}GB | Max Reserved: {mem_reserved:.2f}GB")
            torch.cuda.reset_peak_memory_stats(device)

        scheduler.step()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    num_bins = 128  # e.g., 100 bins means 101 borders
    borders = torch.linspace(-8.0, 8.0, num_bins + 1, device=device)
    criterion = FullSupportBarDistribution(borders=borders, smoothing=0.05)  # Using your custom class

    generator = VectorizedComplexTaskGenerator()

    model = MetaTransferModel(
        transfer_layer=GatedLatentTransferLayer(dmodel=128),
        input_dim=1,
        num_bins=num_bins
    )

    STEPS = 100000
    max_lr = 3e-4
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=STEPS + 1,
        pct_start=0.10,  # Spends the first 10% of steps warming up
        anneal_strategy='cos'
    )

    train_meta_model(
        device=device,
        model=model,
        criterion=criterion,
        generator=generator,
        optimizer=optimizer,
        scheduler=scheduler,

        steps=STEPS,
        batch_size=8192,  # make use of VRAM
        n_A=5,  # sparse target
        n_B=30,  # dense related
        plot_every=500,
        save_path=Path("/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/struct_gate_long/"),
        compile_model=False
    )

    print("Training complete! Check the 'training_plots' folder.")
