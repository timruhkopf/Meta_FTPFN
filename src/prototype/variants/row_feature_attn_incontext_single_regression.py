import torch
import torch.nn as nn
import torch.optim as optim


class InContextTabularModel(nn.Module):
    def __init__(self, num_x_features, d_model=64, num_heads=4):
        super().__init__()
        self.num_x = num_x_features

        # We now have F features + 1 (the label y) + 1 (a binary mask indicating if it's a query)
        self.total_features = num_x_features + 2

        # Tokenizer for all inputs
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(self.total_features)
        ])

        # 1. Feature-Wise Attention Block
        self.feature_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, batch_first=True
        )

        # 2. Row-Wise Attention Block
        self.row_attention = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2, batch_first=True
        )

        # Predictor
        self.output_layer = nn.Sequential(
            nn.Linear(self.total_features * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, y_masked, query_mask):
        """
        x: [N, F] - The input features
        y_masked: [N, 1] - The labels (Context rows have actual values, Query rows are 0.0)
        query_mask: [N, 1] - Binary indicator (0 for context, 1 for query)
        """
        N = x.shape[0]

        # --- TOKENIZATION ---
        inputs = [x[:, i].unsqueeze(1) for i in range(self.num_x)]
        inputs.append(y_masked)
        inputs.append(query_mask)

        embedded = [self.feature_embeddings[i](feat).unsqueeze(1) for i, feat in enumerate(inputs)]

        # Shape: [Rows (N), Features (F+2), Embedding (d_model)]
        tensor_3d = torch.cat(embedded, dim=1)

        # --- 1. FEATURE-WISE ATTENTION ---
        # Sequence = Features, Batch = Rows
        # "How do the features within each row relate to each other?"
        feat_attended = self.feature_attention(tensor_3d)

        # --- THE AXIS FLIP ---
        # We transpose the tensor to swap Rows and Features
        # Shape becomes: [Features (F+2), Rows (N), Embedding (d_model)]
        row_input = feat_attended.transpose(0, 1)

        # --- 2. ROW-WISE ATTENTION ---
        # Sequence = Rows, Batch = Features
        # "How does Row A relate to Row B for this specific feature?"
        row_attended = self.row_attention(row_input)

        # Transpose back: [Rows (N), Features (F+2), Embedding (d_model)]
        final_repr = row_attended.transpose(0, 1)

        # --- PREDICTION ---
        flat_features = final_repr.reshape(N, -1)
        predictions = self.output_layer(flat_features)

        return predictions


def generate_icl_task(num_rows=100, num_features=4):
    """Generates a dataset with a completely random linear rule."""
    X = torch.randn(num_rows, num_features)

    # 1. Create a brand new rule for this specific task: y = X * W + b
    W = torch.randn(num_features, 1) * 2
    b = torch.randn(1) * 2
    true_y = X @ W + b

    # 2. Split dataset into Context (Support) and Query sets
    n_context = num_rows // 2

    # query_mask: 0 if context, 1 if query
    query_mask = torch.ones(num_rows, 1)
    query_mask[:n_context] = 0

    # y_masked: Hide the answers for the query rows so the model has to predict them
    y_masked = true_y.clone()
    y_masked[n_context:] = 0.0

    return X, y_masked, query_mask, true_y, W, b


# --- Initialization ---
torch.manual_seed(1337)
num_features = 4
model = InContextTabularModel(num_features, d_model=64)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Meta-Training: Model must learn to learn from context...\n")
for step in range(10000):
    # Every step is a brand new rule!
    X, y_masked, query_mask, true_y, W, b = generate_icl_task(100, num_features)

    optimizer.zero_grad()
    predictions = model(X, y_masked, query_mask)

    # ONLY calculate loss on the query rows (the ones it didn't see the answers for)
    query_idx = (query_mask == 1).squeeze()
    loss = criterion(predictions[query_idx], true_y[query_idx])

    loss.backward()
    optimizer.step()

    if step % 400 == 0:
        print(f"Step {step:04d} | Meta-Loss on unseen rules: {loss.item():.4f}")

# --- Verification on a Never-Before-Seen Rule ---
print("\n" + "=" * 50)
print("VERIFICATION: Zero-Shot Evaluation on a New Rule")
print("=" * 50)
X_test, y_test_masked, mask_test, true_y_test, W_test, b_test = generate_icl_task(20, num_features)

model.eval()
with torch.no_grad():
    preds = model(X_test, y_test_masked, mask_test)

print(f"Randomly generated rule for this test:")
print(f"Weights: {W_test.squeeze().numpy().round(2)}, Bias: {b_test.item():.2f}\n")

print("Predictions on Query rows (Model was not given these targets):")
query_idx = (mask_test == 1).squeeze()
true_queries = true_y_test[query_idx]
pred_queries = preds[query_idx]

for i in range(5):
    print(f"True Y: {true_queries[i, 0].item():7.4f}  |  Predicted Y: {pred_queries[i, 0].item():7.4f}")

# ---------------------------------------------------------------------
# Plot example data
# ---------------------------------------------------------------------
# here we have a true vs predicted plot, where the diagonal indicates perfect in-context
# inference.
import matplotlib.pyplot as plt

# Generate a slightly larger dataset for a denser plot
X_plot, y_masked_plot, mask_plot, true_y_plot, W_plot, b_plot = generate_icl_task(200, num_features)

model.eval()
with torch.no_grad():
    preds_plot = model(X_plot, y_masked_plot, mask_plot)

# Isolate ONLY the Query rows (the ones the model had to guess)
query_idx = (mask_plot == 1).squeeze()
y_true_query = true_y_plot[query_idx].numpy()
y_pred_query = preds_plot[query_idx].numpy()

# Create the 2D Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_true_query, y_pred_query, alpha=0.7, color='royalblue', edgecolors='k', label='Query Predictions')

# Plot the perfect prediction baseline (y = x)
min_val = min(y_true_query.min(), y_pred_query.min())
max_val = max(y_true_query.max(), y_pred_query.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Accuracy (y=x)')

plt.xlabel("True Values (Ground Truth)")
plt.ylabel("Predicted Values (Model Output)")
plt.title(f"In-Context Regression Validation\n(Rule: y = X*W + {b_plot.item():.2f})")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()