import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from ppfn.model.mymodel.layers.glt_adapter import MLP, GatedLatentTransferLayer
from ppfn.model.mymodel.layers.validation_manifold_adapter.plotting import plot_training_step
from ppfn.model.mymodel.layers.validation_manifold_adapter.prior import create_padded_batch, \
    VectorizedComplexTaskGenerator


class MetaTransferModel(nn.Module):
    """
    A wrapper class to fit the size of a 1d dataset and match the three stream layer requirement.
    """

    def __init__(self, input_dim=1, dmodel=128):
        super().__init__()
        # Use MLPs for better representation as discussed
        self.y_proj = MLP(input_dim, dmodel, dmodel)
        self.x_proj = MLP(input_dim, dmodel, dmodel)
        self.transfer_layer = GatedLatentTransferLayer(dmodel=dmodel)
        self.out_proj = MLP(dmodel, input_dim, dmodel)

    def forward(self, batch):
        # Data is already padded and concatenated in create_padded_batch
        A = self.y_proj(batch["y_cA"])
        B = self.y_proj(batch["y_cB"])
        C = self.y_proj(batch["y_cC"])

        hp_A = self.x_proj(batch["x_cA"])
        hp_B = self.x_proj(batch["x_cB"])
        hp_C = self.x_proj(batch["x_cC"])

        # Call GatedLatentTransferLayer
        _, _, C_out = self.transfer_layer(
            A, B, C, sep=batch["sep"], hp=(hp_A, hp_B, hp_C),
            mask_A=batch["mask_A"], mask_B=batch["mask_B"]
        )

        # We only care about predicting the query portion of C_out
        sep = batch["sep"]
        query_out = C_out[sep:, :, :]

        return self.out_proj(query_out)


def train_meta_model(
        steps=2000,
        batch_size=32,
        n_A=5,
        n_B=30,
        n_query=50,
        save_path=Path('.'),
        plot_every=200,
        compile_model=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    generator = VectorizedComplexTaskGenerator()
    model = MetaTransferModel(input_dim=1, dmodel=128).to(device)

    # Add PyTorch compilation for massive speedups on Ampere GPUs
    if compile_model and torch.__version__.startswith('2.') and device.type == 'cuda':
        print("Compiling model for faster execution...")
        model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)

    # CHANGED: reduction='none' so we can split the loss before averaging
    criterion = nn.MSELoss(reduction='none')

    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    os.makedirs("training_plots", exist_ok=True)

    # CHANGED: Track related and unrelated losses separately
    loss_history_rel = []
    loss_history_unrel = []

    iterator = tqdm(range(steps + 1))

    for step in iterator:
        model.train()
        optimizer.zero_grad()

        # CHANGED: Inject 20% unrelated tasks to force the model to learn gating/escape token
        batch = create_padded_batch(generator, batch_size, n_A, n_B, n_query, device, share_unrelated=0.2)

        # Wrap the forward pass and loss calculation in autocast (bfloat16)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            y_pred = model(batch)

            # Calculate unreduced loss: Shape [T_query, Batch, 1]
            loss_tensor = criterion(y_pred, batch["y_qA_true"])

            # Average over sequence dimension (dim=0) and feature dimension (dim=2)
            # to get loss per batch item: Shape [Batch]
            loss_per_item = loss_tensor.mean(dim=(0, 2))

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

                # Ensure eval also runs in autocast
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    eval_pred = model(eval_batch)

                plot_training_step(
                    step,
                    eval_batch,
                    eval_pred,
                    loss_history_rel,
                    loss_history_unrel,
                    n_A, n_B,
                    save_path
                )


if __name__ == "__main__":
    train_meta_model(
        steps=25000,
        batch_size=8192,  # make use of VRAM
        n_A=5,  # sparse target
        n_B=30,  # dense related
        plot_every=500,
        save_path=Path("/home/ruhkopf/PycharmProjects/Meta_FTPFN/outputs/active_gate_new_plotting2/"))

    print("Training complete! Check the 'training_plots' folder.")
