import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
from prototype.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream



class TimeEmbedding(nn.Module):
    """Projects scalar time t into a high-dimensional sinusoidal embedding."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, t):
        # t shape: (Batch, 1)
        half_dim = self.d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return self.mlp(emb)


class FlowMatchingTransformer(nn.Module):
    """
    Time-conditioned Domain Alignment Transformer.
    Instead of predicting coordinates, it predicts the velocity vector dx/dt.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.proj_A = nn.Linear(2, d_model)
        self.proj_B = nn.Linear(2, d_model)
        self.time_embed = TimeEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output is the 2D velocity vector (v_x, v_y)
        self.velocity_head = nn.Linear(d_model, 2)

    def forward(self, X_A, Y_A, X_B_t, Y_B_t, t, padding_mask_A=None):
        # Dimensions:
        # X_A, Y_A: (n_A, Batch, 1)
        # X_B_t, Y_B_t: (n_B, Batch, 1) -> The intermediate locations at time t
        # t: (Batch, 1)

        seq_B = X_B_t.size(0)

        # 1. Project Context A
        feat_A = torch.cat([X_A, Y_A], dim=-1)
        emb_A = self.proj_A(feat_A)

        # 2. Project Intermediate State B_t
        feat_B_t = torch.cat([X_B_t, Y_B_t], dim=-1)
        emb_B_t = self.proj_B(feat_B_t)

        # 3. Inject Time Embedding
        t_emb = self.time_embed(t)  # (Batch, d_model)
        t_emb = t_emb.unsqueeze(0).expand(seq_B, -1, -1)  # Broadcast to (seq_B, Batch, d_model)
        emb_B_t = emb_B_t + t_emb

        # 4. Attention Pass
        encoded_A = self.encoder(emb_A, src_key_padding_mask=padding_mask_A)
        decoded_B = self.decoder(tgt=emb_B_t, memory=encoded_A, memory_key_padding_mask=padding_mask_A)

        # 5. Predict Velocity
        velocity = self.velocity_head(decoded_B)  # (seq_B, Batch, 2)

        v_X = velocity[..., 0:1]
        v_Y = velocity[..., 1:2]

        return v_X, v_Y


# ==========================================
# Training Pipeline
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Initialize Data and Model
data_stream = InfiniteHarmonicsStream(batch_size=64, n_A=10, n_B=50, share_unrelated=0.0)
data_iter = iter(data_stream)

model = FlowMatchingTransformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

model.train()
num_steps = 10000

for step in range(1, num_steps + 1):
    batch = next(data_iter)
    train_data = batch['train']

    # Static Context
    X_A = train_data['X_A'].to(device)
    Y_A = train_data['Y_A'].to(device)
    mask_A = train_data['padding_mask_A'].to(device)

    # Source (x_0) and Target (x_1)
    X_0 = train_data['X_B'].to(device)
    Y_0 = train_data['Y_B'].to(device)

    X_1 = train_data['X_B_in_A'].to(device)
    Y_1 = train_data['Y_B_in_A'].to(device)

    B_sz = X_0.size(1)

    # Sample random times t ~ U(0, 1) for each batch element
    t = torch.rand((B_sz, 1), device=device)
    t_expanded = t.unsqueeze(0)  # (1, Batch, 1) for broadcasting over sequence

    # 1. Compute Intermediate Path x_t = (1-t)*x_0 + t*x_1
    X_t = (1 - t_expanded) * X_0 + t_expanded * X_1
    Y_t = (1 - t_expanded) * Y_0 + t_expanded * Y_1

    # 2. Compute Target Velocity (Optimal Transport straight-line velocity)
    target_v_X = X_1 - X_0
    target_v_Y = Y_1 - Y_0

    optimizer.zero_grad()

    # 3. Predict Velocity
    pred_v_X, pred_v_Y = model(X_A, Y_A, X_t, Y_t, t, padding_mask_A=mask_A)

    # 4. CFM Vector Field Loss (MSE on velocities)
    loss_x = criterion(pred_v_X, target_v_X)
    loss_y = criterion(pred_v_Y, target_v_Y)
    loss = loss_x + loss_y

    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == 1:
        print(f"Step {step:04d}/{num_steps} | Vector Field Loss: {loss.item():.5f}")

# ==========================================
# Inference: Euler Integration and Plotting
# ==========================================

model.eval()
integration_steps = 20

with torch.no_grad():
    eval_batch = next(data_iter)
    test_data = eval_batch['test']

    # Get a single batch for visualization
    X_A_val = eval_batch['train']['X_A'].to(device)
    Y_A_val = eval_batch['train']['Y_A'].to(device)
    mask_A_val = eval_batch['train']['padding_mask_A'].to(device)

    # x_0 state (Starting positions in Domain B)
    X_curr = test_data['X_B'].to(device)
    Y_curr = test_data['Y_B'].to(device)

    # Store trajectories
    trajectories_X = [X_curr.cpu().numpy()]
    trajectories_Y = [Y_curr.cpu().numpy()]

    dt = 1.0 / integration_steps

    # Unroll the vector field from t=0 to t=1
    for i in range(integration_steps):
        t_val = torch.full((64, 1), i * dt, device=device)

        # Predict velocity vector at current position and time
        v_X, v_Y = model(X_A_val, Y_A_val, X_curr, Y_curr, t_val, padding_mask_A=mask_A_val)

        # Euler Step
        X_curr = X_curr + v_X * dt
        Y_curr = Y_curr + v_Y * dt

        trajectories_X.append(X_curr.cpu().numpy())
        trajectories_Y.append(Y_curr.cpu().numpy())

# Stack trajectories into shape (Steps, Seq, Batch, 1)
traj_X = np.stack(trajectories_X)
traj_Y = np.stack(trajectories_Y)

# ==========================================
# Plotting the Unrolled Trajectories
# ==========================================

idx = 0  # Batch index to plot
valid_A_len = (~eval_batch['train']['padding_mask_A'][idx]).sum().item()

x_a_plot = X_A_val[:valid_A_len, idx, 0].cpu().numpy()
y_a_plot = Y_A_val[:valid_A_len, idx, 0].cpu().numpy()

x_target = test_data['X_B_in_A'][:, idx, 0].cpu().numpy()
y_target = test_data['Y_B_in_A'][:, idx, 0].cpu().numpy()

plt.figure(figsize=(12, 7))

# 1. Plot Context Anchors (A)
plt.scatter(x_a_plot, y_a_plot, color='black', s=100, label='Context Anchors (A)', zorder=10)

# 2. Plot True Target Domain (B in A)
plt.scatter(x_target, y_target, color='green', marker='*', s=80, alpha=0.6, label='True Canonical (B in A)')

# 3. Plot Unrolled Trajectories
num_points = traj_X.shape[1]
colors = plt.cm.jet(np.linspace(0, 1, integration_steps + 1))

for p in range(num_points):
    x_path = traj_X[:, p, idx, 0]
    y_path = traj_Y[:, p, idx, 0]

    # Plot the path as a line
    plt.plot(x_path, y_path, color='gray', alpha=0.2, linewidth=1.5, zorder=1)

    # Scatter intermediate steps with color gradient representing time
    plt.scatter(x_path, y_path, c=colors, s=15, zorder=2)

# Mark the start and end of predictions clearly
plt.scatter(traj_X[0, :, idx, 0], traj_Y[0, :, idx, 0], color='red', marker='x', label='Source Domain ($t=0$)',
            zorder=5)
plt.scatter(traj_X[-1, :, idx, 0], traj_Y[-1, :, idx, 0], color='blue', marker='o', alpha=0.7,
            label='Final Prediction ($t=1$)', zorder=5)

plt.title(f"Conditional Flow Matching Integration (Steps: {integration_steps})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from prototype.harmonic_restart.harmonic_prior import InfiniteHarmonicsStream
#
#
# class DomainAlignmentTransformer(nn.Module):
#     def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
#         super().__init__()
#
#         # Projections from 2D point cloud space (X, Y) to latent space
#         self.proj_A = nn.Linear(2, d_model)
#         self.proj_B = nn.Linear(2, d_model)
#
#         # Encoder to build a representation of the canonical domain using Task A
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
#             dropout=dropout, batch_first=False
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         # Decoder where B queries information from the encoded A context
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
#             dropout=dropout, batch_first=False
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#
#         # Final head to directly regress the aligned coordinates in Domain A
#         self.regressor = nn.Linear(d_model, 2)
#
#     def forward(self, X_A, Y_A, X_B, Y_B, padding_mask_A=None):
#         # Inputs shapes: (Seq, Batch, 1)
#         # padding_mask_A shape: (Batch, Seq_A)
#
#         # 1. Prepare and project features
#         feat_A = torch.cat([X_A, Y_A], dim=-1)  # (n_B, B, 2)
#         feat_B = torch.cat([X_B, Y_B], dim=-1)  # (n_B, B, 2)
#
#         emb_A = self.proj_A(feat_A)  # (n_B, B, d_model)
#         emb_B = self.proj_B(feat_B)  # (n_B, B, d_model)
#
#         # 2. Encode Context Domain A
#         # src_key_padding_mask expects (Batch, Seq)
#         encoded_A = self.encoder(emb_A, src_key_padding_mask=padding_mask_A)
#
#         # 3. Cross-attend from Domain B to Domain A
#         # Target (Q) = emb_B, Memory (K, V) = encoded_A
#         decoded_B = self.decoder(
#             tgt=emb_B,
#             memory=encoded_A,
#             memory_key_padding_mask=padding_mask_A
#         )
#
#         # 4. Predict aligned coordinates
#         predictions = self.regressor(decoded_B)  # (n_B, B, 2)
#
#         pred_X_B_in_A = predictions[..., 0:1]
#         pred_Y_B_in_A = predictions[..., 1:2]
#
#         return pred_X_B_in_A, pred_Y_B_in_A
#
#
# # ==========================================
# # Training Pipeline
# # ==========================================
#
# # Initialize Prior Stream and Model
# data_stream = InfiniteHarmonicsStream(batch_size=64, n_A=10, n_B=50, share_unrelated=0.0)
# data_iter = iter(data_stream)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DomainAlignmentTransformer().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# criterion = nn.MSELoss()
#
# print(f"Training on device: {device}")
# model.train()
#
# # Short training loop for demonstration
# num_steps = 10000
# for step in range(1, num_steps + 1):
#     batch = next(data_iter)
#     train_data = batch['train']
#
#     # Move batch to device
#     X_A = train_data['X_A'].to(device)
#     Y_A = train_data['Y_A'].to(device)
#     X_B = train_data['X_B'].to(device)
#     Y_B = train_data['Y_B'].to(device)
#     mask_A = train_data['padding_mask_A'].to(device)
#
#     # Ground truth targets
#     target_X = train_data['X_B_in_A'].to(device)
#     target_Y = train_data['Y_B_in_A'].to(device)
#
#     optimizer.zero_grad()
#
#     # Forward Pass
#     pred_X, pred_Y = model(X_A, Y_A, X_B, Y_B, padding_mask_A=mask_A)
#
#     # Compute reconstruction loss
#     loss_x = criterion(pred_X, target_X)
#     loss_y = criterion(pred_Y, target_Y)
#     loss = loss_x + loss_y
#
#     loss.backward()
#     optimizer.step()
#
#     if step % 200 == 0 or step == 1:
#         print(
#             f"Step {step:04d}/{num_steps} | Total Loss: {loss.item():.4f} (Loss X: {loss_x.item():.4f}, Loss Y: {loss_y.item():.4f})")
#
# # ==========================================
# # Evaluation and Plotting
# # ==========================================
# model.eval()
# with torch.no_grad():
#     eval_batch = next(data_iter)
#     test_data = eval_batch['test']  # Using validation/test coordinates
#
#     # Prepare single batch elements for plotting (Batch index 0)
#     X_A_val = eval_batch['train']['X_A'][:, 0:1, :]  # Fetch unmasked elements via mask if needed, or just plot A
#     Y_A_val = eval_batch['train']['Y_A'][:, 0:1, :]
#     mask_A_val = eval_batch['train']['padding_mask_A'][0:1, :]
#
#     X_B_val = test_data['X_B'].to(device)
#     Y_B_val = test_data['Y_B'].to(device)
#
#     # Run model on validation instance
#     # We pass the train A context to resolve the alignment of test B inputs
#     pred_X_val, pred_Y_val = model(
#         eval_batch['train']['X_A'].to(device),
#         eval_batch['train']['Y_A'].to(device),
#         X_B_val, Y_B_val,
#         padding_mask_A=eval_batch['train']['padding_mask_A'].to(device)
#     )
#
# # Bring everything to CPU for plotting
# idx = 0  # Batch index to visualize
# valid_A_len = (~eval_batch['train']['padding_mask_A'][idx]).sum().item()
#
# x_a_plot = eval_batch['train']['X_A'][:valid_A_len, idx, 0].cpu().numpy()
# y_a_plot = eval_batch['train']['Y_A'][:valid_A_len, idx, 0].cpu().numpy()
#
# x_b_plot = test_data['X_B'][:, idx, 0].cpu().numpy()
# y_b_plot = test_data['Y_B'][:, idx, 0].cpu().numpy()
#
# x_target = test_data['X_B_in_A'][:, idx, 0].cpu().numpy()
# y_target = test_data['Y_B_in_A'][:, idx, 0].cpu().numpy()
#
# x_pred = pred_X_val[:, idx, 0].cpu().numpy()
# y_pred = pred_Y_val[:, idx, 0].cpu().numpy()
#
# # Generate Visualization
# plt.figure(figsize=(10, 6))
# plt.scatter(x_b_plot, y_b_plot, color='red', alpha=0.4, label='Raw Distorted Input (B)', marker='x')
# plt.scatter(x_a_plot, y_a_plot, color='black', s=80, label='Sparse Context (A)', zorder=5)
# plt.scatter(x_target, y_target, color='green', alpha=0.5, label='Ground Truth Target (B in A)')
# plt.scatter(x_pred, y_pred, color='blue', alpha=0.7, label='Transformer Prediction ($\hat{B}$ in A)', marker='o')
#
# # Draw displacement vectors to show trajectory matching
# for i in range(len(x_target)):
#     plt.plot([x_target[i], x_pred[i]], [y_target[i], y_pred[i]], color='gray', linestyle='--', alpha=0.3)
#
# plt.title("Domain Alignment: Mapping Warped Space $B$ Back to Canonical Space $A$")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.show()