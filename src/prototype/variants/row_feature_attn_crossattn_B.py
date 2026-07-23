import torch
import torch.nn as nn
import torch.optim as optim

"""
row and featurewise attention: https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2_6.py
"""


class CrossTableICLModel(nn.Module):
    def __init__(self, num_x_features, d_model=64, num_heads=4):
        super().__init__()
        self.num_x = num_x_features
        self.total_features = num_x_features + 2  # F features + 1 (y) + 1 (mask)

        # Shared Tokenizer for both tables
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(self.total_features)
        ])

        # 1. Feature-Wise Attention (Shared weights for both tables)
        self.feature_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, batch_first=True
        )

        # 2. Row-Wise Self-Attention for Table B (The Prior)
        self.row_attn_B = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, batch_first=True
        )

        # 3. Row-Wise Cross-Attention (A queries B)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )

        # 4. Row-Wise Self-Attention for Table A (The Target Task)
        self.row_attn_A = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, batch_first=True
        )

        # Predictor (Only predicts for Table A)
        self.output_layer = nn.Sequential(
            nn.Linear(self.total_features * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def _tokenize(self, x, y, mask):
        inputs = [x[:, i].unsqueeze(1) for i in range(self.num_x)]
        inputs.append(y)
        inputs.append(mask)
        embedded = [self.feature_embeddings[i](feat).unsqueeze(1) for i, feat in enumerate(inputs)]
        return torch.cat(embedded, dim=1)  # [N, F+2, d_model]

    def forward(self, x_A, y_A, mask_A, x_B, y_B, mask_B):
        n_A = x_A.shape[0]

        # --- TOKENIZATION ---
        emb_A = self._tokenize(x_A, y_A, mask_A)  # [n_A, F+2, d]
        emb_B = self._tokenize(x_B, y_B, mask_B)  # [n_B, F+2, d]

        # --- 1. FEATURE-WISE ATTENTION ---
        feat_A = self.feature_attention(emb_A)
        feat_B = self.feature_attention(emb_B)

        # Transpose for Row-wise operations -> Batch becomes Features, Seq becomes Rows
        row_in_A = feat_A.transpose(0, 1)  # [F+2, n_A, d]
        row_in_B = feat_B.transpose(0, 1)  # [F+2, n_B, d]

        # --- 2. TABLE B BUILDS THE PRIOR ---
        row_B = self.row_attn_B(row_in_B)  # [F+2, n_B, d]

        # --- 3. CROSS-ATTENTION (A asks B) ---
        # Query = Table A, Key/Value = Table B
        cross_A, _ = self.cross_attention(query=row_in_A, key=row_B, value=row_B)

        # Add residual connection from A's original features
        A_combined = row_in_A + cross_A

        # --- 4. TABLE A SELF-ATTENTION (Adapting the prior) ---
        row_A = self.row_attn_A(A_combined)  # [F+2, n_A, d]

        # Transpose back: [n_A, F+2, d]
        final_repr_A = row_A.transpose(0, 1)

        # --- PREDICTION FOR TABLE A ---
        flat_features = final_repr_A.reshape(n_A, -1)
        predictions = self.output_layer(flat_features)

        return predictions


import torch
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_dynamic_complex_prior():
    torch.manual_seed(100)  # Seed ensures the random phase shift is identical every time you run this cell

    # 1. Unpack the 7 variables from our new neural-network-ready generator
    X_A, y_A_masked, mask_A, true_y_A, X_B, y_B, mask_B = generate_dynamic_complex_prior_task(
        n_A_context=6, n_A_query=100, n_B=300
    )

    # 2. Use the binary mask to separate Context (Support) from Query
    ctx_idx = (mask_A == 0).squeeze()
    qry_idx = (mask_A == 1).squeeze()

    X_A_ctx = X_A[ctx_idx]
    y_A_ctx = true_y_A[ctx_idx]

    X_A_qry = X_A[qry_idx]
    true_y_A_qry = true_y_A[qry_idx]

    # 3. Sort the query points! (Otherwise Matplotlib connects the random points in a zig-zag)
    sort_idx = torch.argsort(X_A_qry.squeeze())
    X_A_qry = X_A_qry[sort_idx]
    true_y_A_qry = true_y_A_qry[sort_idx]

    # --- PLOTTING ---
    plt.figure(figsize=(12, 7))

    # Plot Table B (The Prior)
    plt.scatter(X_B.numpy(), y_B.numpy(), color='lightgray', alpha=0.5, label='Table B (Prior Distribution)')

    # Plot Table A Context
    plt.scatter(X_A_ctx.numpy(), y_A_ctx.numpy(), color='blue', s=100, edgecolor='white', zorder=5,
                label='Table A Support (Only 6 points!)')

    # Plot the True Rule A
    plt.plot(X_A_qry.numpy(), true_y_A_qry.numpy(), 'g-', lw=3, label='True Rule A (Model must guess this)')

    # Simulate a Naive Fit (A polynomial degree 5 fit on only the 6 blue points)
    # This proves that standard regression fails here without Table B
    z = np.polyfit(X_A_ctx.squeeze().numpy(), y_A_ctx.squeeze().numpy(), 5)
    p = np.poly1d(z)
    plt.plot(X_A_qry.numpy(), p(X_A_qry.numpy()), 'r:', lw=2, label='Naive Fit (Overfitting 6 points)')

    plt.xlim(-5, 5)
    plt.ylim(-10, 15)
    plt.xlabel("Feature (X)")
    plt.ylabel("Target (Y)")
    plt.title("Dynamic Non-Linear Prior\n(Model must adapt to this random phase shift using B)")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. THE DYNAMIC COMPLEX DATA GENERATOR
# ==========================================
def generate_dynamic_complex_prior_task(n_A_context=6, n_A_query=50, n_B=300):
    """
    Generates a complex wave where the phase and bias shift of Table A
    are completely randomized for every single batch.
    """
    # Base Rule for Table B (The consistent "rules of the universe")
    freq_B = 2.0
    amp_B = 3.0
    slope_B = 0.5
    bias_B = 1.0

    # Randomly generate a NEW Delta Rule for Table A for this specific batch
    # Phase shift between -pi and pi
    phase_shift_A = (torch.rand(1).item() * 2 - 1) * 3.14159
    bias_delta = torch.randn(1).item() * 2.0

    # --- Create Table B (The Prior) ---
    X_B = (torch.rand(n_B, 1) * 10) - 5
    y_B = amp_B * torch.sin(freq_B * X_B) + slope_B * X_B + bias_B
    y_B += torch.randn(n_B, 1) * 0.8  # Add moderate noise
    mask_B = torch.zeros(n_B, 1)  # All of B is context

    # --- Create Table A (The Sparse Task) ---
    n_A = n_A_context + n_A_query

    # Context (Clustered, so it can't see the whole wave)
    X_A_ctx = (torch.rand(n_A_context, 1) * 4) - 2
    y_A_ctx = amp_B * torch.sin(freq_B * X_A_ctx + phase_shift_A) + slope_B * X_A_ctx + (bias_B + bias_delta)
    y_A_ctx += torch.randn(n_A_context, 1) * 0.3  # Add slight noise

    # Query (Spread out across the whole domain)
    X_A_qry = (torch.rand(n_A_query, 1) * 10) - 5
    true_y_A_qry = amp_B * torch.sin(freq_B * X_A_qry + phase_shift_A) + slope_B * X_A_qry + (bias_B + bias_delta)

    # Combine Context and Query into a single Table A tensor
    X_A = torch.cat([X_A_ctx, X_A_qry], dim=0)
    true_y_A = torch.cat([y_A_ctx, true_y_A_qry], dim=0)

    # Mask out the query answers so the model has to predict them
    y_A_masked = true_y_A.clone()
    y_A_masked[n_A_context:] = 0.0

    mask_A = torch.ones(n_A, 1)
    mask_A[:n_A_context] = 0  # 0 for context, 1 for query

    return X_A, y_A_masked, mask_A, true_y_A, X_B, y_B, mask_B


# ==========================================
# 2. THE META-TRAINING LOOP
# ==========================================

# Initialize for a 1D Feature space
torch.manual_seed(42)
num_features = 1
model = CrossTableICLModel(num_features, d_model=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Meta-Training: Learning to anchor sparse non-linear data to a massive prior...\n")

for step in range(100000):
    # Every step generates a wave with a completely new phase and vertical shift for A
    X_A, y_A, mask_A, true_y_A, X_B, y_B, mask_B = generate_dynamic_complex_prior_task(
        n_A_context=6, n_A_query=50, n_B=300
    )

    optimizer.zero_grad()

    # Forward pass uses both tables (A Cross-Attends to B)
    predictions = model(X_A, y_A, mask_A, X_B, y_B, mask_B)

    # ONLY calculate loss on Table A's query rows
    query_idx = (mask_A == 1).squeeze()
    loss = criterion(predictions[query_idx], true_y_A[query_idx])

    loss.backward()

    # Gradient clipping is highly recommended when dealing with dynamic non-linear data
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:04d} | Non-Linear Meta-Loss: {loss.item():.4f}")

# ==========================================
# 3. VERIFICATION
# ==========================================
print("\n" + "=" * 50)
print("VERIFICATION: Testing on a brand new phase shift")
print("=" * 50)

model.eval()
with torch.no_grad():
    X_A_test, y_A_test, mask_A_test, true_y_A_test, X_B_test, y_B_test, mask_B_test = generate_dynamic_complex_prior_task(
        n_A_context=6, n_A_query=10, n_B=300
    )
    preds = model(X_A_test, y_A_test, mask_A_test, X_B_test, y_B_test, mask_B_test)

query_idx = (mask_A_test == 1).squeeze()
true_queries = true_y_A_test[query_idx]
pred_queries = preds[query_idx]

for i in range(5):
    print(f"True Target A: {true_queries[i, 0].item():7.4f}  |  Predicted Target A: {pred_queries[i, 0].item():7.4f}")


plot_dynamic_complex_prior()
