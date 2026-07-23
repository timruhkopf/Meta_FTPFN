import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from prototype.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream


# ==========================================
# 1. THE ARCHITECTURE
# ==========================================
class MaskedSetEncoder(nn.Module):
    """Encodes the scarce context A into a global latent vector, respecting padding masks."""

    def __init__(self, dim_in=2, dim_hidden=64, dim_context=32):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        self.global_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_context),
            nn.ReLU()
        )

    def forward(self, A, mask_A):
        # A shape: [Batch, Seq_A, 2]
        # mask_A shape: [Batch, Seq_A] (True for padded elements)

        features = self.point_net(A)  # [Batch, Seq_A, Hidden]

        # Mask out the padded points before max pooling by setting them to -infinity
        # (unsqueeze mask to match hidden dim: [Batch, Seq_A, 1])
        mask_expanded = mask_A.unsqueeze(-1).expand_as(features)
        features = features.masked_fill(mask_expanded, float('-inf'))

        # Max pooling over the sequence dimension
        global_feature = torch.max(features, dim=1)[0]  # [Batch, Hidden]

        # In case a batch was entirely masked (shouldn't happen here, but for safety)
        global_feature = torch.where(torch.isinf(global_feature), torch.zeros_like(global_feature), global_feature)

        context = self.global_net(global_feature)  # [Batch, Context]
        return context


class HarmonicFlowNetwork(nn.Module):
    """Predicts velocity v(S_t, t, c_A) for 2D points S = (X, Y)."""

    def __init__(self, dim_in=2, dim_context=64, dim_hidden=128):
        super().__init__()
        self.encoder = MaskedSetEncoder(dim_in, dim_hidden=128, dim_context=dim_context)

        # Input: S_t (2) + t (1) + context (64) = 67
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + 1 + dim_context, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_in)
        )

    def forward(self, S_t, t, A, mask_A):
        batch_size, seq_len, _ = S_t.shape

        # 1. Encode Context (Scarce Task A)
        c_A = self.encoder(A, mask_A)  # [Batch, Context]

        # 2. Expand context and time to match dense points sequence length
        c_A_expanded = c_A.unsqueeze(1).expand(-1, seq_len, -1)  # [Batch, Seq, Context]
        t_expanded = t.expand(-1, seq_len, -1)  # [Batch, Seq, 1]

        # 3. Concatenate and predict
        nn_input = torch.cat([S_t, t_expanded, c_A_expanded], dim=-1)
        velocity = self.mlp(nn_input)
        return velocity


class TransformerEncoder(nn.Module):
    def __init__(self, dim_in=2, dim_hidden=128, n_heads=4):
        super().__init__()
        # Internal processing of Task A (the "Prior" knowledge)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)
        # Mapping Task B to Task A
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim_hidden, num_heads=n_heads, batch_first=True)

        self.input_proj = nn.Linear(dim_in, dim_hidden)
        self.query_proj = nn.Linear(3, dim_hidden)  # S_t(2) + t(1)

    def forward(self, S_t, t, A, mask_A):
        # 1. Project A to hidden space
        A_emb = self.input_proj(A)

        # 2. Self-Attention: Learn the shape of Task A
        # (Pass mask_A where True indicates padding)
        A_self, _ = self.self_attn(A_emb, A_emb, A_emb, key_padding_mask=mask_A)

        # 3. Cross-Attention: S_t (B) queries the features of A

        seq_len = S_t.size(1)
        t_expanded = t.expand(-1, seq_len, -1)  # Now [Batch, Seq_B, 1]

        # Now cat works: [Batch, Seq_B, 2] + [Batch, Seq_B, 1] -> [Batch, Seq_B, 3]
        query_input = torch.cat([S_t, t_expanded], dim=-1)

        queries = self.query_proj(query_input)
        S_ctx, _ = self.cross_attn(queries, A_self, A_self)

        return S_ctx


class HarmonicFlowNetwork(nn.Module):
    def __init__(self, dim_in=2, dim_hidden=128):
        super().__init__()
        self.encoder = TransformerEncoder(dim_in, dim_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden + dim_in, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_in)
        )

    def forward(self, S_t, t, A, mask_A):
        # Features capture the mapping from B to A
        ctx = self.encoder(S_t, t, A, mask_A)
        # Predict velocity
        return self.mlp(torch.cat([S_t, ctx], dim=-1))

# ==========================================
# 2. THE TRAINING LOOP
# ==========================================
print("Initializing Model & Prior...")
# Assume InfiniteHarmonicsStream is defined above exactly as you provided
stream = InfiniteHarmonicsStream(batch_size=64, n_A=10, n_B=50, num_components=2)
data_iter = iter(stream)

model = HarmonicFlowNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 10000

print("Starting Training...")
model.train()
for epoch in range(epochs):
    batch = next(data_iter)
    train_data = batch['train']

    # 1. Extract and format Source B (Dense, unwarped) -> S_0
    # Prior outputs [Seq, Batch, 1]. We permute to [Batch, Seq, 1] and concatenate X, Y
    X_B = train_data['X_B'].permute(1, 0, 2)
    Y_B = train_data['Y_B'].permute(1, 0, 2)
    S_0 = torch.cat([X_B, Y_B], dim=-1)  # Shape: [Batch, n_B, 2]

    # 2. Extract and format Target B_in_A (Dense, warped) -> S_1
    X_B_in_A = train_data['X_B_in_A'].permute(1, 0, 2)
    Y_B_in_A = train_data['Y_B_in_A'].permute(1, 0, 2)
    S_1 = torch.cat([X_B_in_A, Y_B_in_A], dim=-1)  # Shape: [Batch, n_B, 2]

    # 3. Extract and format Context A (Scarce, padded) -> c_A
    X_A = train_data['X_A'].permute(1, 0, 2)
    Y_A = train_data['Y_A'].permute(1, 0, 2)
    A = torch.cat([X_A, Y_A], dim=-1)  # Shape: [Batch, n_B, 2]

    # Extract mask for Context A (slice to n_B since prior pads it to n_B + n_test)
    n_B = S_0.shape[1]
    mask_A = train_data['padding_mask_A'][:, :n_B]

    # 4. Sample Time t ~ U(0,1)
    t = torch.rand(S_0.size(0), 1, 1)

    # 5. Exact Paired Interpolation (Straight lines in 2D space)
    S_t = (1 - t) * S_0 + t * S_1

    # 6. Target Velocity
    v_target = S_1 - S_0

    # 7. Predict & Optimize
    optimizer.zero_grad()
    v_pred = model(S_t, t, A, mask_A)
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print(f"Step {epoch:4d} | Loss: {loss.item():.4f}")

print("Training Complete!\n")


# ==========================================
# 3. INFERENCE & VISUALIZATION (Zero-Shot)
# ==========================================
@torch.no_grad()
def generate_inference(model, S_0, A, mask_A, steps=50):
    model.eval()
    S_t = S_0.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.ones(S_0.size(0), 1, 1) * (i * dt)
        v = model(S_t, t, A, mask_A)
        S_t = S_t + v * dt
    return S_t

# Get a batch directly from the generator
eval_batch = next(data_iter)
train_data = eval_batch['train']

# 1. Extract Source (S_0) from TRAIN split
X_B_train = train_data['X_B'].permute(1, 0, 2)
Y_B_train = train_data['Y_B'].permute(1, 0, 2)
S_0_train = torch.cat([X_B_train, Y_B_train], dim=-1)

# 2. Extract Ground Truth Target (S_1) from TRAIN split
X_B_in_A_true = train_data['X_B_in_A'].permute(1, 0, 2)
Y_B_in_A_true = train_data['Y_B_in_A'].permute(1, 0, 2)
S_1_true = torch.cat([X_B_in_A_true, Y_B_in_A_true], dim=-1)

# 3. Extract Context A (with padding) from TRAIN split
X_A_train = train_data['X_A'].permute(1, 0, 2)
Y_A_train = train_data['Y_A'].permute(1, 0, 2)
A_train = torch.cat([X_A_train, Y_A_train], dim=-1)

# Extract the mask for Context A (slice to n_B size)
n_B = S_0_train.shape[1]
mask_A_train = train_data['padding_mask_A'][:, :n_B]

# 4. Run Inference
S_1_pred = generate_inference(model, S_0_train, A_train, mask_A_train)

# ==========================================
# Plotting the first item in the batch
# ==========================================
b_idx = 0
plt.figure(figsize=(10, 6))

# Plot True Target B_in_A (The hidden warp)
plt.scatter(S_1_true[b_idx, :, 0], S_1_true[b_idx, :, 1],
            c='lightgray', s=20, label='True Warped Domain (Target)')

# Plot Predicted B_in_A (Model's output)
plt.scatter(S_1_pred[b_idx, :, 0], S_1_pred[b_idx, :, 1],
            c='blue', s=30, label='Predicted Flow Output')

# 5. Filter and Plot Context A
# mask_A is True for padded values. We only want valid points (~mask)
valid_A_idx = ~mask_A_train[b_idx]
valid_A_points = A_train[b_idx][valid_A_idx]

plt.scatter(valid_A_points[:, 0], valid_A_points[:, 1],
            c='red', s=80, marker='x', linewidth=2, label='Context A (Condition)')

plt.title("Amortized Flow Matching (Train Split Visualized)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#
# import torch
# import torch.nn as nn
# import math
# import matplotlib.pyplot as plt
#
#
# # ==========================================
# # 1. THE PRIOR: Data Generating Process
# # ==========================================
# import torch
# import math
#
#
# def generate_task_data(batch_size, n_dense=500, n_sparse=25):
#     # 1. Source B: Dense standard unit circle
#     theta_B = torch.linspace(0, 2 * math.pi, n_dense).unsqueeze(0).repeat(batch_size, 1)
#     B = torch.stack([torch.cos(theta_B), torch.sin(theta_B)], dim=-1)
#
#     # 2. Harmonic Parameters
#     k = torch.randint(2, 6, (batch_size, 1)).float()
#     a = torch.rand(batch_size, 1) * 0.5 + 0.2
#     phi = torch.rand(batch_size, 1) * 2 * math.pi
#
#     # 3. Target B_in_A (Un-transformed)
#     R_B = 1.0 + a * torch.sin(k * theta_B + phi)
#     B_in_A = torch.stack([R_B * torch.cos(theta_B), R_B * torch.sin(theta_B)], dim=-1)
#
#     # 4. Context A (Un-transformed)
#     theta_A = torch.rand(batch_size, n_sparse) * 2 * math.pi
#     R_A = 1.0 + a * torch.sin(k * theta_A + phi)
#     A = torch.stack([R_A * torch.cos(theta_A), R_A * torch.sin(theta_A)], dim=-1)
#
#     # --- NEW: Random Transformation ---
#     # Random Shift: move in range [-0.5, 0.5] for both x and y
#     shift = (torch.rand(batch_size, 1, 2) - 0.5)
#
#     # Random Scale: factor between 0.5 and 1.5
#     scale = (torch.rand(batch_size, 1, 1) + 0.5)
#
#     # Apply to B_in_A and A
#     # Formula: transformed = (original * scale) + shift
#     B_in_A = (B_in_A * scale) + shift
#     A = (A * scale) + shift
#
#     return B, A, B_in_A
# # ==========================================
# # 2. THE ARCHITECTURE: Amortized Inference
# # ==========================================
# class SetEncoder(nn.Module):
#     """Encodes the scarce context A into a global latent vector."""
#
#     def __init__(self, dim_in=2, dim_hidden=64, dim_context=32):
#         super().__init__()
#         self.point_net = nn.Sequential(
#             nn.Linear(dim_in, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden)
#         )
#         self.global_net = nn.Sequential(
#             nn.Linear(dim_hidden, dim_context),
#             nn.ReLU()
#         )
#
#     def forward(self, A):
#         # A shape: [Batch, K, 2]
#         features = self.point_net(A)  # [Batch, K, Hidden]
#         global_feature = torch.max(features, dim=1)[0]  # Max pooling over set -> [Batch, Hidden]
#         context = self.global_net(global_feature)  # [Batch, Context]
#         return context
#
#
# class ConditionalFlowNetwork(nn.Module):
#     """Predicts velocity v(x_t, t, A)."""
#
#     def __init__(self, dim_in=2, dim_context=32, dim_hidden=128):
#         super().__init__()
#         self.encoder = SetEncoder(dim_in, dim_hidden=64, dim_context=dim_context)
#
#         # Input to MLP: x_t (2) + t (1) + context (32) = 35
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_in + 1 + dim_context, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_in)
#         )
#
#     def forward(self, x, t, A):
#         batch_size, n_dense, _ = x.shape
#
#         # 1. Encode Context
#         c_A = self.encoder(A)  # [Batch, Context]
#
#         # 2. Expand context and time to match dense points
#         c_A_expanded = c_A.unsqueeze(1).expand(-1, n_dense, -1)  # [Batch, N, Context]
#         t_expanded = t.expand(-1, n_dense, -1)  # [Batch, N, 1]
#
#         # 3. Concatenate and predict
#         nn_input = torch.cat([x, t_expanded, c_A_expanded], dim=-1)
#         velocity = self.mlp(nn_input)
#         return velocity
#
#
# # ==========================================
# # 3. THE TRAINING PROCESS: Exact Paired Flow
# # ==========================================
# print("Initializing Model...")
# model = ConditionalFlowNetwork()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# epochs = 1500
# batch_size = 64
#
# print("Starting Training...")
# model.train()
# for epoch in range(epochs):
#     # 1. Generate data on the fly (Implicitly learning f)
#     B, A, B_in_A = generate_task_data(batch_size)
#
#     # 2. Sample random time t ~ U(0,1)
#     t = torch.rand(batch_size, 1, 1)
#
#     # 3. Calculate interpolant (exact straight line path)
#     x_t = (1 - t) * B + t * B_in_A
#
#     # 4. Calculate target velocity
#     u_target = B_in_A - B
#
#     # 5. Predict and Optimize
#     optimizer.zero_grad()
#     v_pred = model(x_t, t, A)
#     loss = nn.MSELoss()(v_pred, u_target)
#
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 300 == 0:
#         print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")
#
# print("Training Complete!\n")
#
#
# # ==========================================
# # 4. INFERENCE & VISUALIZATION (Zero-Shot)
# # ==========================================
# @torch.no_grad()
# def euler_ode_solver(model, B, A, steps=50):
#     """Integrates the learned vector field from t=0 to t=1."""
#     model.eval()
#     x = B.clone()
#     dt = 1.0 / steps
#
#     for i in range(steps):
#         t = torch.ones(B.size(0), 1, 1) * (i * dt)
#         v = model(x, t, A)
#         x = x + v * dt
#
#     return x
#
#
# # Generate a single unseen test task
# B_test, A_test, B_in_A_test_true = generate_task_data(batch_size=1)
#
# # Run Inference (Notice B_in_A_test_true is NOT passed to the model)
# B_in_A_pred = euler_ode_solver(model, B_test, A_test)
#
# # Plotting
# B_np = B_test[0].numpy()
# A_np = A_test[0].numpy()
# Truth_np = B_in_A_test_true[0].numpy()
# Pred_np = B_in_A_pred[0].numpy()
#
# plt.figure(figsize=(15, 5))
#
# plt.subplot(1, 3, 1)
# plt.title("Domain B (Source)")
# plt.scatter(B_np[:, 0], B_np[:, 1], c='blue', s=10, alpha=0.5, label='Dense B')
# plt.xlim(-5, 5);
# plt.ylim(-5, 5);
# plt.legend()
#
# plt.subplot(1, 3, 2)
# plt.title("Domain A (Scarce Context)")
# plt.scatter(A_np[:, 0], A_np[:, 1], c='red', s=50, marker='x', label='Scarce A')
# plt.xlim(-5, 5);
# plt.ylim(-5, 5);
# plt.legend()
#
# plt.subplot(1, 3, 3)
# plt.title("Inference vs Ground Truth")
# plt.scatter(Truth_np[:, 0], Truth_np[:, 1], c='gray', s=10, alpha=0.3, label='True Warped B')
# plt.scatter(Pred_np[:, 0], Pred_np[:, 1], c='green', s=10, alpha=0.7, label='Predicted B_in_A')
# plt.scatter(A_np[:, 0], A_np[:, 1], c='red', s=50, marker='x', label='Scarce Context A')
# plt.xlim(-5, 5);
# plt.ylim(-5, 5);
# plt.legend()
#
# plt.tight_layout()
# plt.show()