from prototype.cross_feat_attn.model import DualTaskTabPFN
from prototype.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream


import torch


def prepare_harmonics_batch(batch_dict, device='cpu'):
    """
    Extracts Support (Train) and Query (Test) sets, injecting dummy zeros
    for the Query targets to prevent label leakage.
    """
    # --- 1. Extract Support (Train Context) ---
    X_A_tr = batch_dict['train']['X_A'].permute(1, 0, 2).squeeze(-1).to(device)
    Y_A_tr = batch_dict['train']['Y_A'].permute(1, 0, 2).squeeze(-1).to(device)

    X_B_tr = batch_dict['train']['X_B'].permute(1, 0, 2).squeeze(-1).to(device)
    Y_B_tr = batch_dict['train']['Y_B'].permute(1, 0, 2).squeeze(-1).to(device)

    mask_A_tr = batch_dict['train']['padding_mask_A'].to(device)

    # --- 2. Extract Query (Test Targets) ---
    X_A_te = batch_dict['test']['X_A'].permute(1, 0, 2).squeeze(-1).to(device)
    Y_A_te = batch_dict['test']['Y_A'].permute(1, 0, 2).squeeze(-1).to(device)

    X_B_te = batch_dict['test']['X_B'].permute(1, 0, 2).squeeze(-1).to(device)
    Y_B_te = batch_dict['test']['Y_B'].permute(1, 0, 2).squeeze(-1).to(device)

    # --- 3. Build Blocks ---
    # Support: Real inputs, Real targets
    Support_A = torch.stack([X_A_tr, Y_A_tr], dim=-1)
    Support_B = torch.stack([X_B_tr, Y_B_tr], dim=-1)

    # Query: Real inputs, Zeros for targets
    Query_A = torch.stack([X_A_te, torch.zeros_like(Y_A_te)], dim=-1)
    Query_B = torch.stack([X_B_te, torch.zeros_like(Y_B_te)], dim=-1)

    # Return the raw blocks so the loop can strip padding dynamically per-task
    return Support_A, Query_A, mask_A_tr, Support_B, Query_B, Y_B_te

# ==========================================
# 2. The Meta-Training Loop
# ==========================================
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn


def train_harmonics_tabpfn(model, dataset, num_steps=5000, lr=1e-4, device='cuda'):
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    data_iter = iter(dataset)

    pbar = tqdm(range(num_steps), desc="Meta-Training Dual-TabPFN")
    loss_history = []

    for step in pbar:
        optimizer.zero_grad()
        batch_dict = next(data_iter)

        # Fetch blocks
        Supp_A, Qry_A, mask_A, Supp_B, Qry_B, Y_B_te_true = prepare_harmonics_batch(batch_dict, device)
        batch_size = Supp_A.shape[0]
        batch_loss = 0.0

        for b in range(batch_size):
            # --- Assemble Task A (Handling Padding) ---
            valid_A_idx = ~mask_A[b]
            clean_supp_A = Supp_A[b][valid_A_idx]
            qry_A = Qry_A[b]

            task_A_table = torch.cat([clean_supp_A, qry_A], dim=0)
            sep_A = len(clean_supp_A)

            # --- Assemble Task B ---
            supp_B = Supp_B[b]
            qry_B = Qry_B[b]

            task_B_table = torch.cat([supp_B, qry_B], dim=0)
            sep_B = len(supp_B)

            # Ground truth for Task B queries
            task_B_targets = Y_B_te_true[b]

            # Forward Pass with sep indices
            preds_B = model(task_A_table, task_B_table, sep_A, sep_B)

            # Loss and Accumulation
            loss = criterion(preds_B, task_B_targets)
            (loss / batch_size).backward()
            batch_loss += loss.item() / batch_size

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(batch_loss)
        if step % 50 == 0:
            pbar.set_postfix({'MSE': f"{batch_loss:.4f}"})

    return loss_history

# ==========================================
# 3. Execution Script
# ==========================================
if __name__ == "__main__":
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the Prior (your exact code)
    # n_A=100 (clean points), n_B=50 (distorted points)
    prior = InfiniteHarmonicsStream(
        # fixme: this version has inverted notation regarding A and B
        batch_size=16,
        n_A=50,
        n_B=10,
        num_components=2,
        noise_std=0.05,
        scale=True,
        shift=True,
        warp=True
    )

    # We have 2 features in the final table: X (coordinate) and Y (value)
    NUM_FEATURES = 2

    # Instantiate the Architecture
    model = DualTaskTabPFN(
        num_features=NUM_FEATURES,
        emsize=128,
        nhead=4,
        num_layers=4
    )

    print(f"Starting training on {device}...")
    # Run the loop (start with a small number of steps to test)
    history = train_harmonics_tabpfn(model, prior, num_steps=2000, lr=3e-4, device=device)
    print("Training complete!")

    import matplotlib.pyplot as plt
    import torch


    def verify_harmonics_predictions(model, dataset, device='cuda'):
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            batch_dict = next(iter(dataset))
            Supp_A, Qry_A, mask_A, Supp_B, Qry_B, Y_B_te_true = prepare_harmonics_batch(batch_dict, device)

            b = 0  # Visualize first item

            # Assemble A
            valid_A_idx = ~mask_A[b]
            clean_supp_A = Supp_A[b][valid_A_idx]
            task_A_table = torch.cat([clean_supp_A, Qry_A[b]], dim=0)
            sep_A = len(clean_supp_A)

            # Assemble B
            task_B_table = torch.cat([Supp_B[b], Qry_B[b]], dim=0)
            sep_B = len(Supp_B[b])
            task_B_targets_true = Y_B_te_true[b]

            # Predict
            preds_B = model(task_A_table, task_B_table, sep_A, sep_B)

        # --- Extraction for Matplotlib ---
        # Task A Context
        X_A_tr = task_A_table[:sep_A, 0].cpu().numpy()
        Y_A_tr = task_A_table[:sep_A, 1].cpu().numpy()

        # Task B Context (Support) vs Test (Query)
        X_B_tr = task_B_table[:sep_B, 0].cpu().numpy()
        Y_B_tr = task_B_table[:sep_B, 1].cpu().numpy()

        X_B_te = task_B_table[sep_B:, 0].cpu().numpy()
        Y_B_te_true = task_B_targets_true.cpu().numpy()
        Y_B_pred = preds_B.cpu().numpy()

        # Hidden Canonical Targets (B mapped to A) for Test Set
        X_B_in_A_te = batch_dict['test']['X_B_in_A'][:, b, 0].numpy()
        Y_B_in_A_te = batch_dict['test']['Y_B_in_A'][:, b, 0].numpy()

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: The Domain Shift
        ax1.scatter(X_A_tr, Y_A_tr, alpha=0.3, color='blue', label='Task A Support (Clean)')
        ax1.scatter(X_B_in_A_te, Y_B_in_A_te, alpha=0.3, color='cyan', label='Task B Queries (Mapped to A)')
        ax1.scatter(X_B_tr, Y_B_tr, alpha=0.3, color='orange', marker='s', label='Task B Support (Distorted)')
        ax1.scatter(X_B_te, Y_B_te_true, alpha=0.8, color='red', marker='x', label='Task B Queries (Distorted)')

        for i in range(len(X_B_te)):
            ax1.plot([X_B_te[i], X_B_in_A_te[i]], [Y_B_te_true[i], Y_B_in_A_te[i]], color='gray', linestyle='--',
                     alpha=0.2)

        ax1.set_title("The Harmonic Shift\n(Context vs Targets)")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: The Predictions
        # Plot Support points lightly to show what the model based its prediction on
        ax2.scatter(X_B_tr, Y_B_tr, alpha=0.2, color='orange', marker='s', label='Task B Context')

        # Plot Test targets and predictions
        ax2.scatter(X_B_te, Y_B_te_true, alpha=0.6, color='red', s=60, marker='o', label='Task B Ground Truth')
        ax2.scatter(X_B_te, Y_B_pred, alpha=0.9, color='limegreen', s=80, marker='X', edgecolor='black',
                    label='TabPFN Predictions')

        for i in range(len(X_B_te)):
            ax2.plot([X_B_te[i], X_B_te[i]], [Y_B_te_true[i], Y_B_pred[i]], color='gray', linestyle=':', alpha=0.5)

        ax2.set_title("Query Predictions via Train/Test Split\n(Did it reverse the warp?)")
        ax2.set_xlabel("X coordinate")
        ax2.set_ylabel("Y value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    # --- Execution ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verify_harmonics_predictions(model, prior, device=device)