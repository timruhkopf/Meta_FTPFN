import torch
import torch.nn as nn




class TabPFNEmbedding(nn.Module):
    """
    Projects a 2D table [N, F] into a 3D latent tensor [N, F, d].
    """

    def __init__(self, num_features, emsize):
        super().__init__()
        self.emsize = emsize
        self.value_encoder = nn.Linear(1, emsize)
        self.feature_embeddings = nn.Embedding(num_features, emsize)

    def forward(self, x):
        N, F_dim = x.shape
        val_embs = self.value_encoder(x.unsqueeze(-1))
        feat_ids = torch.arange(F_dim, device=x.device).unsqueeze(0)
        feat_embs = self.feature_embeddings(feat_ids)
        return val_embs + feat_embs


class TabPFNBlock(nn.Module):
    """
    Executes Sample-wise and Feature-wise attention with strict
    train/test isolation using the separator index (sep).
    """

    def __init__(self, emsize, nhead, dim_feedforward=512, dropout=0.0):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(emsize, nhead, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(emsize, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(emsize, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, emsize)
        self.norm1 = nn.LayerNorm(emsize)
        self.norm2 = nn.LayerNorm(emsize)
        self.norm3 = nn.LayerNorm(emsize)
        self.activation = nn.GELU()

    def forward(self, h, sep):
        N, F_dim, d = h.shape

        # --- 1. Sample-Wise (Row) Attention ---
        h_row = h.transpose(0, 1)  # [F, N, d]

        # PFN Contextual Masking: True means "blocked from attending".
        # We allow all rows (Train + Test) to attend to Train rows (0 to sep).
        # We block all rows from attending to Test rows (sep to N).
        # This prevents label leakage and preserves test-set exchangeability.
        attn_mask = torch.ones((N, N), dtype=torch.bool, device=h.device)
        attn_mask[:, :sep] = False

        h_row_out, _ = self.row_attn(h_row, h_row, h_row, attn_mask=attn_mask)
        h = h + h_row_out.transpose(0, 1)
        h = self.norm1(h)

        # --- 2. Feature-Wise (Column) Attention ---
        # Features within the SAME row interact. No inter-row masking needed here.
        h_col_out, _ = self.col_attn(h, h, h)
        h = h + h_col_out
        h = self.norm2(h)

        # --- 3. Position-Wise FeedForward Layer ---
        h_ff = self.linear2(self.activation(self.linear1(h)))
        h = h + h_ff
        return self.norm3(h)


class CrossFeatureAttention(nn.Module):
    """
    Uni-directional bridge: Task B queries Task A's structure.
    """

    def __init__(self, emsize, nhead, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(emsize, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emsize)

    def forward(self, h_A, h_B, sep_A):
        N_B, F_dim, d = h_B.shape

        # Global Structural Pooling: Only pool the Train (Support) rows of Task A.
        # Test rows have dummy targets and shouldn't pollute the structural prototype.
        p_A = torch.mean(h_A[:sep_A], dim=0, keepdim=True)
        p_A = p_A.expand(N_B, -1, -1)

        # Cross-Attention: Task B features map themselves against Task A's blueprint
        h_B_out, _ = self.cross_attn(query=h_B, key=p_A, value=p_A)

        return self.norm(h_B + h_B_out)


class DualTaskTabPFN(nn.Module):
    """
    Uni-directional Dual-Task PFN predicting Task B based on Task A's prior.
    """

    def __init__(self, num_features, emsize=128, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = TabPFNEmbedding(num_features, emsize)

        self.layers = nn.ModuleList([
            TabPFNBlock(emsize, nhead) for _ in range(num_layers)
        ])

        self.cross_feature_layer = CrossFeatureAttention(emsize, nhead)

        self.regression_head = nn.Sequential(
            nn.Linear(emsize, emsize // 2),
            nn.GELU(),
            nn.Linear(emsize // 2, 1)
        )

    def forward(self, x_A, x_B, sep_A, sep_B):
        # Embeddings
        h_A = self.embedding(x_A)
        h_B = self.embedding(x_B)

        # Independent processing with strict Train/Test contextual splits
        for layer in self.layers:
            h_A = layer(h_A, sep_A)
            h_B = layer(h_B, sep_B)

        # Uni-directional Bridge: B updates using A's train structure
        h_B_bridged = self.cross_feature_layer(h_A, h_B, sep_A)

        # We only care about predicting the Test section of Task B.
        # We slice from sep_B onwards, and extract the target feature column (index -1).
        test_target_tokens = h_B_bridged[sep_B:, -1, :]

        return self.regression_head(test_target_tokens).squeeze(-1)

import numpy as np


class MutatingMetaSCMPrior:
    """
    Generates synthetic training episodes containing pairs of structurally
    related but functionally mutated datasets to train the Dual-Task TabPFN.
    """

    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale

    def sample_episode(self, n_A=1000, n_B=50):
        # 1. Randomize base structural weights for this specific episode
        w_shared = np.random.uniform(1.5, 4.0)

        # --- TASK A: Base Causal Mechanism ---
        # Sample covariate space
        x1_A = np.random.normal(0.0, 1.0, n_A)
        # Mechanism 1: Sine wave interaction
        x2_A = np.sin(x1_A * w_shared) + np.random.normal(0, self.noise_scale, n_A)
        # Target Function: Combination of linear and non-linear interactions
        y_A = (2.0 * x1_A) - (1.5 * x2_A) + np.random.normal(0, self.noise_scale, n_A)

        # --- TASK B: Mutated Function and Covariate Shift ---
        # Apply a heavy shift to the root cause feature distribution
        x1_B = np.random.normal(2.0, 0.4, n_B)

        # MUTATION: Change the underlying math function while preserving causal dependency.
        # Task B changes from a Sine wave interaction to an Exponential/Cosine blend.
        # This forces the network to find the abstract link via cross-feature attention.
        x2_B = np.cos(x1_B * (w_shared * 0.8)) * 1.2 + np.random.normal(0, self.noise_scale, n_B)
        y_B = (2.0 * x1_B) - (1.5 * x2_B) + np.random.normal(0, self.noise_scale, n_B)

        # Package tables cleanly: Columns are [X1, X2, Y]
        dataset_A = np.stack([x1_A, x2_A, y_A], axis=1)
        dataset_B_features = np.stack([x1_B, x2_B], axis=1)
        dataset_B_targets = y_B

        return (torch.tensor(dataset_A, dtype=torch.float32),
                torch.tensor(dataset_B_features, dtype=torch.float32),
                torch.tensor(dataset_B_targets, dtype=torch.float32))



# ==========================================
# 2. Episodic Meta-Training Loop
# ==========================================
def train_dual_tabpfn(model, prior, optimizer, criterion, num_episodes=10000):
    model.train()
    device = next(model.parameters()).device

    for episode in range(num_episodes):
        optimizer.zero_grad()

        # --- Step A: Generate Synthetic Episode ---
        # x_A:        [1000, 3] -> (X1, X2, y)
        # x_B_feat:   [50, 2]   -> (X1, X2)
        # y_B_true:   [50]      -> Ground truth targets for Task B
        x_A, x_B_feat, y_B_true = prior.sample_episode(n_A=1000, n_B=50)

        x_A = x_A.to(device)
        x_B_feat = x_B_feat.to(device)
        y_B_true = y_B_true.to(device)

        # --- Step B: Dummy Target Injection ---
        # Task B needs a 3rd column (target) to match Task A's shape.
        # We initialize it with zeros. The Cross-Feature Attention will
        # overwrite these zeros with the structural mapping from Task A.
        N_B = x_B_feat.shape[0]
        dummy_targets = torch.zeros((N_B, 1), dtype=torch.float32, device=device)
        x_B = torch.cat([x_B_feat, dummy_targets], dim=1)  # Shape becomes [50, 3]

        # --- Step C: Forward Pass ---
        # The model processes them independently, pools A, cross-attends B to A,
        # and outputs the final predictions from B's dummy column token.
        y_B_pred = model(x_A, x_B, sep_A=10, sep_B=50)

        # --- Step D: Loss and Optimization ---
        loss = criterion(y_B_pred, y_B_true)
        loss.backward()

        # Gradient clipping is highly recommended for TabPFN/Transformers
        # to prevent massive spikes during early meta-training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # --- Step E: Logging ---
        if episode % 500 == 0:
            print(f"Episode {episode:05d} | Task B MSE Loss: {loss.item():.4f}")


import matplotlib.pyplot as plt


def verify_and_plot(model, prior):
    """
    Evaluates the trained Dual-Task TabPFN on a brand new episode
    and plots the structural alignment and predictions.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Generate a completely unseen episode
    with torch.no_grad():
        x_A, x_B_feat, y_B_true = prior.sample_episode(n_A=1000, n_B=50)

        x_A_dev = x_A.to(device)
        x_B_feat_dev = x_B_feat.to(device)

        # Inject dummy target column for Task B
        dummy_targets = torch.zeros((x_B_feat_dev.shape[0], 1), device=device)
        x_B_dev = torch.cat([x_B_feat_dev, dummy_targets], dim=1)

        # Predict
        y_B_pred = model(x_A_dev, x_B_dev).cpu().numpy()

    # Convert tensors back to numpy for plotting
    x_A = x_A.numpy()
    x_B_feat = x_B_feat.numpy()
    y_B_true = y_B_true.numpy()

    # 2. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: The Mutated Feature Space ---
    ax1.scatter(x_A[:, 0], x_A[:, 1], alpha=0.2, color='blue', label='Task A (Base Sine)')
    ax1.scatter(x_B_feat[:, 0], x_B_feat[:, 1], alpha=0.8, color='red', edgecolor='black',
                label='Task B (Mutated Cosine)')
    ax1.set_title("Feature Space Shift\n(Notice Task B obeys a different curve)")
    ax1.set_xlabel("Feature X1")
    ax1.set_ylabel("Feature X2")
    ax1.legend()

    # --- Plot 2: Target Predictions vs Ground Truth ---
    # We plot the target y against X1 to see the regression fit
    ax2.scatter(x_B_feat[:, 0], y_B_true, alpha=0.8, color='red', s=60, edgecolor='black',
                label='Task B (Ground Truth)')
    ax2.scatter(x_B_feat[:, 0], y_B_pred, alpha=0.8, color='limegreen', s=60, marker='X', edgecolor='black',
                label='TabPFN Predictions')

    # Draw lines connecting predictions to their ground truth to visualize error
    for i in range(len(y_B_true)):
        ax2.plot([x_B_feat[i, 0], x_B_feat[i, 0]], [y_B_true[i], y_B_pred[i]], color='gray', linestyle='--', alpha=0.5)

    ax2.set_title("Zero-Shot Task B Predictions\n(Did it infer the mutation?)")
    ax2.set_xlabel("Feature X1")
    ax2.set_ylabel("Target y")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # ==========================================
    # 1. Initialization
    # ==========================================
    # We have 3 columns total: X1, X2, and y.
    NUM_FEATURES = 3

    # Instantiate the architecture built previously
    model = DualTaskTabPFN(num_features=NUM_FEATURES, emsize=128, nhead=4, num_layers=4)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the data generator (Prior)
    prior = MutatingMetaSCMPrior(noise_scale=0.05)

    # Standard Transformer hyperparameters
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Execute the training pipeline
    train_dual_tabpfn(model, prior, optimizer, criterion)
    verify_and_plot(model, prior)
