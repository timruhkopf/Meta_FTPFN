import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.data import IterableDataset, DataLoader

from pfns4hpo.bar_distribution import FullSupportBarDistribution

import torch
import math
from typing import Tuple
from torch.utils.data import IterableDataset

from prototype.harmonic_restart import InfiniteHarmonicsStream


# Assuming BarDistribution is available
# from bar_distribution import BarDistribution

class UnifiedPFNLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        # Note: batch_first=False expects (Seq, Batch, D)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, eval_pos: int, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed_x = self.norm1(x)
        train_part = normed_x[:eval_pos]
        test_part = normed_x[eval_pos:]

        train_pad_mask = pad_mask[:, :eval_pos] if pad_mask is not None else None

        # Context self-attends
        train_out = self.self_attn(
            query=train_part, key=train_part, value=train_part,
            key_padding_mask=train_pad_mask, need_weights=False
        )[0]

        # Query cross-attends using identical projection weights
        if test_part.shape[0] > 0:
            test_out = self.self_attn(
                query=test_part, key=train_part, value=train_part,
                key_padding_mask=train_pad_mask, need_weights=False
            )[0]
            attn_out = torch.cat([train_out, test_out], dim=0)
        else:
            attn_out = train_out

        x = x + self.dropout1(attn_out)
        normed_ff = self.norm2(x)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(normed_ff))))
        return x + self.dropout2(ff_out)


class PFNStack(nn.Module):
    def __init__(self, d_model: int, num_layers: int, nhead: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([UnifiedPFNLayer(d_model, nhead) for _ in range(num_layers)])

    def forward(self, combined: torch.Tensor, eval_pos: int, pad_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            combined = layer(combined, eval_pos, pad_mask)
        return combined


class JEPAPFN(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, num_bins: int, d_model: int = 256,
                 num_layers: int = 6, nhead: int = 8, ema_decay: float = 0.996, borders=(-11, 11)):
        super().__init__()
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.num_bins = num_bins

        self.x_proj = nn.Linear(x_dim, d_model)
        self.y_proj = nn.Linear(y_dim, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # FIXME: use Tabpfn as backend model?
        self.student = PFNStack(d_model, num_layers, nhead)
        self.teacher = PFNStack(d_model, num_layers, nhead)

        for param in self.teacher.parameters():
            param.requires_grad = False

        # Detached Decoder Component
        self.decoder_norm = nn.LayerNorm(d_model)
        self.decoder_linear = nn.Linear(d_model, num_bins)

        borders = torch.linspace(*(*borders, num_bins +1))
        self.bar_dist = FullSupportBarDistribution( borders=borders)

    @torch.no_grad()
    def update_teacher_ema(self):
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def forward(self, x_train, y_train, x_test, y_test=None):
        # Transpose inputs to (Seq, Batch, Dim) for batch_first=False
        x_train, y_train = x_train.transpose(0, 1), y_train.transpose(0, 1)
        x_test = x_test.transpose(0, 1)
        if y_test is not None:
            y_test = y_test.transpose(0, 1)

        eval_pos = x_train.shape[0]
        seq_test = x_test.shape[0]
        B = x_train.shape[1]

        def student_fwd(x_train, y_train, x_test):
            # Shared Train Context
            train_emb = self.x_proj(x_train) + self.y_proj(y_train)

            # Student sequence
            student_test_emb = self.x_proj(x_test) + self.mask_token.expand(seq_test, B, -1)
            student_combined = torch.cat([train_emb, student_test_emb], dim=0)

            # 1. Get Backbone representations
            s_out = self.student(student_combined, eval_pos)
            student_test_reps = s_out[eval_pos:]  # Shape: (Seq, Batch, Dim)

            # 2. Pass through Predictor (Only for the Student!)
            # This is what we compare to the Teacher in the L2 loss
            predicted_teacher_reps = self.predictor(student_test_reps)

            dec_in = self.decoder_norm(student_test_reps_t.detach())
            logits = self.decoder_linear(dec_in)

            return y_test_hat, logits

        @torch.no_grad()
        def teacher_fwd(x_train, y_train, x_test, y_test):
            train_emb = self.x_proj(x_train) + self.y_proj(y_train)
            teacher_test_emb = self.x_proj(x_test) + self.y_proj(y_test)
            teacher_combined = torch.cat([train_emb, teacher_test_emb], dim=0)

            t_out = self.teacher(teacher_combined, eval_pos)

            # FIXME: layernorm for the teacher
            teacher_test_reps = t_out[eval_pos:]
            return

        # Fixme student input: B_train, predictor input A_train, A_test
        s_y_test_hat = student_fwd(x_train, y_train, x_test)
        if self.training:
            # FIXME: check that x_train is [A_train, B_in_A, A_test],
            #  and y_train is [A_train, B_in_A, A_test] as well!
            t_y_test_hat = teacher_fwd(x_train, y_train)

        # 4. Detached Decoder (Uses the Student's BACKBONE representation, not the prediction)
        student_test_reps_t = student_test_reps.transpose(0, 1)



        # We must return the predicted reps for the L2 loss!
        if teacher_test_reps is not None:
            teacher_test_reps = teacher_test_reps.transpose(0, 1)
            predicted_teacher_reps = predicted_teacher_reps.transpose(0, 1)

        return predicted_teacher_reps, teacher_test_reps, logits


import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow


@torch.no_grad()
def plot_posterior_heatmap(model, dataset, batch_dict, device, step, x_range=(-6, 6), resolution=200):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import mlflow

    @torch.no_grad()
    def plot_posterior_heatmap(model, dataset, batch_dict, device, step, x_range=(-6, 6), resolution=200):
        model.eval()

        # 1. Extract Batch 0 Context
        # Raw dataloader output is (Seq, Batch, Dim).
        # We transpose to (Batch, Seq, Dim) FIRST, then slice batch index 0.
        X_context = batch_dict['train']['X_B'].transpose(0, 1)[0:1, :, :].to(device)  # Shape: (1, Seq, 1)
        Y_context = batch_dict['train']['Y_B'].transpose(0, 1)[0:1, :, :].to(device)  # Shape: (1, Seq, 1)

        # 2. Create Dense Grid for X_test
        # Must be shaped (Batch, Seq, Dim) -> (1, 200, 1)
        X_grid = torch.linspace(x_range[0], x_range[1], resolution, device=device).unsqueeze(0).unsqueeze(-1)

        # 3. Standardize context
        # Now that shape is (1, Seq, 1), we compute stats over dim=1 (the Sequence dimension)
        y_mean = Y_context.mean(dim=1, keepdim=True)
        y_std = Y_context.std(dim=1, keepdim=True) + 1e-8
        Y_context_norm = (Y_context - y_mean) / y_std

        # 4. Model Forward Pass
        # y_test=None gracefully bypasses the Teacher logic in JEPAPFN
        _, _, logits = model(X_context, Y_context_norm, X_grid, y_test=None)

        # 5. Extract probabilities
        # logits shape is (Batch, Seq, num_bars) -> (1, 200, num_bars)
        probs = torch.softmax(logits.float(), dim=-1)
        probs_2d = probs[0].cpu().numpy().T  # Shape: (num_bars, 200)

        # 6. Un-normalize Y borders for plotting
        borders_norm = model.bar_dist.borders.cpu().numpy()
        borders_real = borders_norm * y_std[0, 0, 0].item() + y_mean[0, 0, 0].item()
        x_plot = X_grid[0, :, 0].cpu().numpy()  # Shape: (200,)

        # 7. Calculate True Continuous Function (Clean, without noise)
        params_A = tuple(p[:, 0:1] for p in batch_dict['params']['params_A'])
        shifts = tuple(s[0:1] for s in batch_dict['params']['shifts'])
        scale_A = batch_dict['params']['scale_A'][0:1]
        warps = tuple(w[0:1] for w in batch_dict['params']['warps'])

        X_grid_flat = X_grid[0, :, 0].cpu()  # Shape: (200,)

        # Replicate dataset warping logic
        v_shift, h_shift = shifts
        X_warped = X_grid_flat - h_shift + dataset._apply_spatial_warp(X_grid_flat, *warps).squeeze(0)
        Y_grid_true = scale_A * dataset._eval_function(X_warped, *params_A).squeeze(0) + v_shift

        # 8. Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        X_mesh, Y_mesh = np.meshgrid(x_plot, borders_real)
        c = ax.pcolormesh(X_mesh, Y_mesh, probs_2d, cmap='Blues', shading='auto', alpha=0.9)
        fig.colorbar(c, ax=ax, label='Predictive Probability Density')

        ax.plot(x_plot, Y_grid_true.numpy(), color='crimson', label='True Function (B)', linewidth=2.5, linestyle='--')

        # Scatter expects 1D arrays
        ax.scatter(X_context[0, :, 0].cpu().numpy(), Y_context[0, :, 0].cpu().numpy(),
                   color='black', label='Context Points (B)', zorder=5, edgecolor='white', s=60)

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(borders_real[0], borders_real[-1])
        ax.set_title(f"Posterior Predictive Heatmap (Step {step})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right")
        plt.tight_layout()

        return fig
import torch
import torch.nn.functional as F



def train_jepa_pfn_refactored(model, dataloader, optimizer,
                              epochs=100, device='cuda', plot_every_n_epochs=100):
    model.to(device)

    # Initialize the gradient scaler for AMP
    # Note: If running on CPU, GradScaler is a safe no-op.
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Start MLflow tracking
    with mlflow.start_run(run_name="JEPA_PFN_Harmonics"):
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "d_model": model.d_model,
            "num_bins": model.num_bins
        })
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_l2 = 0
        total_nll = 0

        for step, batch_dict in enumerate(dataloader):
            if step >= dataloader.dataset.batches_per_epoch:
                break

            optimizer.zero_grad()

            # 1. Unpack and prepare inputs (Keep data preparation in float32)
            X_B = batch_dict['train']['X_B'].transpose(0, 1).to(device)
            Y_B = batch_dict['train']['Y_B'].transpose(0, 1).to(device)
            X_B_test = batch_dict['test']['X_B'].transpose(0, 1).to(device)
            Y_B_test = batch_dict['test']['Y_B'].transpose(0, 1).to(device)
            sep = X_B.shape[1]

            # padding_mask_A = batch_dict['train']['padding_mask_A'].to(device)

            # --- Per-Task Standardization ---
            # valid_Y_B = Y_B.masked_fill(padding_mask_A.unsqueeze(-1), float('nan'))
            y_mean = torch.nanmean(Y_B, dim=1, keepdim=True)

            y_centered = Y_B - y_mean
            y_var = torch.nanmean(y_centered ** 2, dim=1, keepdim=True)
            y_std = torch.sqrt(y_var) + 1e-8
            #
            # Y_A_norm = (Y_A - y_mean) / y_std
            Y_B_norm = (Y_B - y_mean) / y_std
            # Y_A_norm = Y_A_norm.masked_fill(padding_mask_A.unsqueeze(-1), 0.0)

            # 2. Mixed Precision Forward Pass
            # On modern GPUs, 'bfloat16' is highly recommended over 'float16' for stability
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                student_reps, teacher_reps, logits = model(X_B, Y_B, X_B_test, Y_B_test)

                s_reps_norm = F.normalize(student_reps[-sep:], p=2, dim=-1)
                t_reps_norm = F.normalize(teacher_reps[-sep:], p=2, dim=-1)

                # Compute L2 loss inside autocast since MSE is stable under half-precision
                loss_l2 = F.mse_loss(s_reps_norm, t_reps_norm)

            # 3. Precision-Safe Loss Evaluation (Outside Autocast)
            # Explicitly cast logits back to float32 to ensure mathematical stability in the BarDistribution
            logits_f32 = logits.float()

            bar_logits = logits_f32.transpose(0, 1)
            bar_y_B = Y_B_norm.squeeze(-1).transpose(0, 1)

            # BarDistribution runs fully in standard float32
            loss_nll_map = model.bar_dist(bar_logits[-sep:], bar_y_B)
            loss_nll = loss_nll_map.mean()

            # Combine the losses (both are now float32)
            loss = loss_l2 + loss_nll

            # 4. Scaled Backward and Step Pass
            scaler.scale(loss).backward()

            # Unscale gradients prior to clipping to maintain true scale thresholds
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step and scaler update
            scaler.step(optimizer)
            scaler.update()

            # Update the Teacher EMA weights after gradients are settled
            model.update_teacher_ema()
            # MLflow Metrics Logging
            mlflow.log_metric("train/l2_loss", loss_l2.item(), step=global_step)
            mlflow.log_metric("train/nll_loss", loss_nll.item(), step=global_step)

            total_l2 += loss_l2.item()
            total_nll += loss_nll.item()
            global_step += 1

        print(f"Epoch {epoch + 1} | L2: {total_l2 / (step + 1):.4f} | NLL: {total_nll / (step + 1):.4f}")

        # MLflow Heatmap Generation
        if (epoch + 1) % plot_every_n_epochs == 0 :
            fig = plot_posterior_heatmap(model, dataloader.dataset, batch_dict, device, global_step)
            plt.show()
            plt.close(fig)  # Free memory

if __name__ == '__main__':

    # Example execution setup:
    dataset = InfiniteHarmonicsStream(batch_size=32, n_A=10, n_B=50, n_test=200)
    dataset.batches_per_epoch = 10
    # The crucial part is batch_size=None
    dataloader = DataLoader(dataset, batch_size=None)


    model = JEPAPFN(x_dim=1, y_dim=1, num_bins=200) # num_bins abstracted for standard MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train_jepa_pfn_refactored(model, dataloader, optimizer, epochs=20000, device='cuda')