import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# 1. THE PRIOR & DATA GENERATOR
# =====================================================================
def distortion_f(B_in_A):
    """
    The distortion function 'f' that maps B_in_A (Target Domain) -> B (Source Domain).
    We add a non-linear twist/sinusoidal wave and scale it up (more detailed/spread out).
    """
    B = B_in_A.clone()
    # Apply a non-linear spatial warping
    B[:, 0] = B_in_A[:, 0] * 1.5 + torch.sin(B_in_A[:, 1] * 3.0) * 0.3
    B[:, 1] = B_in_A[:, 1] * 1.5 + torch.cos(B_in_A[:, 0] * 3.0) * 0.3
    return B


# def generate_task_data():
#     """
#     Simulates dynamic data creation for a single task.
#     Returns:
#         A: Anchor cloud in target domain (Sparse, e.g., 80 points)
#         B_in_A: Ground-truth target for B (Dense, e.g., 200 points)
#         B: Detailed source cloud (Dense, e.g., 200 points) via distortion f
#     """
#     # Let's say domain A's underlying geometry is a semi-circle arch
#     theta_A = torch.rand(80) * np.pi
#     A = torch.stack([torch.cos(theta_A), torch.sin(theta_A)], dim=1)
#
#     # B_in_A shares the same underlying geometry but is sampled denser
#     theta_B = torch.rand(200) * np.pi
#     B_in_A = torch.stack([torch.cos(theta_B), torch.sin(theta_B)], dim=1)
#
#     # B is constructed by pulling B_in_A through the distorting prior 'f'
#     B = distortion_f(B_in_A)
#
#     return A, B_in_A, B

# def generate_task_data():
#     """
#     Simulates dynamic function sampling.
#     Every task has a completely different underlying geometry.
#     """
#     # 1. Sample the underlying function parameters for THIS specific task
#     # E.g., a quadratic curve where curvature and slope change wildly
#     alpha = (torch.rand(1) - 0.5) * 4.0  # Curvature: [-2.0, 2.0]
#     beta = (torch.rand(1) - 0.5) * 2.0  # Slope: [-1.0, 1.0]
#
#     # 2. Sample points for Anchor A (Sparse)
#     x_A = (torch.rand(80) - 0.5) * 4.0  # x in [-2, 2]
#     y_A = alpha * (x_A ** 2) + beta * x_A
#     A = torch.stack([x_A, y_A], dim=1)
#
#     # 3. Sample points for Target B_in_A (Dense, but strictly on the same function)
#     x_B = (torch.rand(200) - 0.5) * 4.0
#     y_B = alpha * (x_B ** 2) + beta * x_B
#     B_in_A = torch.stack([x_B, y_B], dim=1)
#
#     # 4. Apply the known class of distortions (e.g., spatial spreading / warping)
#     # This represents the "relationship" the network must learn to invert
#     B = torch.zeros_like(B_in_A)
#     B[:, 0] = B_in_A[:, 0] * 1.5 + torch.sin(B_in_A[:, 1] * 2.0) * 0.5
#     B[:, 1] = B_in_A[:, 1] * 1.5 + torch.cos(B_in_A[:, 0] * 2.0) * 0.5
#
#     return A, B_in_A, B

def generate_task_data():
    # 1. Sample BASE function parameters (Bounds: [-2, 2] and [-1, 1])
    alpha = (torch.rand(1) - 0.5) * 4.0
    beta = (torch.rand(1) - 0.5) * 2.0

    # Generate A and B_in_A using the base function
    x_A = (torch.rand(80) - 0.5) * 4.0
    y_A = alpha * (x_A ** 2) + beta * x_A
    A = torch.stack([x_A, y_A], dim=1)

    x_B = (torch.rand(200) - 0.5) * 4.0
    y_B = alpha * (x_B ** 2) + beta * x_B
    B_in_A = torch.stack([x_B, y_B], dim=1)

    # 2. Sample WARP function parameters (Bounds defined by domain knowledge)
    gamma = torch.rand(1) * 1.5 + 0.5  # Stretch factor: [0.5, 2.0]
    delta = torch.rand(1) * 3.0  # Wave frequency: [0.0, 3.0]

    # Apply the dynamic distortion
    B = torch.zeros_like(B_in_A)
    B[:, 0] = B_in_A[:, 0] * gamma + torch.sin(B_in_A[:, 1] * delta) * 0.5
    B[:, 1] = B_in_A[:, 1] * gamma + torch.cos(B_in_A[:, 0] * delta) * 0.5

    return A, B_in_A, B


# =====================================================================
# 2. MODEL ARCHITECTURE (Mini Point Transformer + Flow Net)
# =====================================================================
class MiniPointTransformerEncoder(nn.Module):
    """Encodes Anchor Cloud A using localized Vector Attention via k-NN."""

    def __init__(self, in_channels=2, out_channels=32, k=12):
        super().__init__()
        self.k = k
        self.linear_q = nn.Linear(in_channels, out_channels)
        self.linear_k = nn.Linear(in_channels, out_channels)
        self.linear_v = nn.Linear(in_channels, out_channels)

        # Position encoding network
        self.pos_net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, A):
        # A shape: [N_A, 2]
        # Compute pairwise distance matrix to find k-NN
        dist = torch.cdist(A, A)
        idx = dist.topk(self.k, largest=False, dim=-1).indices  # [N_A, k]

        # Gather neighbors
        N_A = A.shape[0]
        A_neighbors = A[idx]  # [N_A, k, 2]
        A_center = A.unsqueeze(1).expand(-1, self.k, -1)  # [N_A, k, 2]

        # Positional encoding: delta = pos_net(p_i - p_j)
        pos_enc = self.pos_net(A_center - A_neighbors)  # [N_A, k, out_channels]

        # Linear projections
        q = self.linear_q(A)  # [N_A, out_channels]
        k_val = self.linear_k(A_neighbors)  # [N_A, k, out_channels]
        v = self.linear_v(A_neighbors)  # [N_A, k, out_channels]

        # Vector Attention: Attn = Softmax(Q - K + Pos)
        attn = F.softmax(q.unsqueeze(1) - k_val + pos_enc, dim=1)  # [N_A, k, out_channels]

        # Aggregate local neighborhoods
        out = torch.sum(attn * (v + pos_enc), dim=1)  # [N_A, out_channels]

        # Pool globally to a single vector to handle dynamic target domain shapes
        global_condition = torch.mean(out, dim=0)  # [out_channels]
        return global_condition


class ConditionalVelocityNet(nn.Module):
    """Predicts the straight-line velocity vector field given (x_t, t, condition)."""

    def __init__(self, point_dim=2, cond_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(point_dim + 1 + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, point_dim)
        )

    def forward(self, x_t, t, condition):
        # x_t: [N_B, 2]
        # t: scalar or [N_B, 1]
        # condition: [cond_dim]
        N_B = x_t.shape[0]

        t_emb = torch.ones(N_B, 1, device=x_t.device) * t
        cond_emb = condition.unsqueeze(0).expand(N_B, -1)

        feat = torch.cat([x_t, t_emb, cond_emb], dim=-1)  # [N_B, 2 + 1 + 32]
        return self.net(feat)


# =====================================================================
# 3. CONDITIONAL FLOW MATCHING (TRAINING)
# =====================================================================
# Initialize networks
encoder = MiniPointTransformerEncoder()
flow_net = ConditionalVelocityNet(cond_dim=64)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(flow_net.parameters()), lr=1e-3)

print("Training Conditional Flow Matcher...")
for epoch in range(40000):
    optimizer.zero_grad()

    # Dynamically generate points for this task (simulating changing counts/resampling)
    A, B_in_A, B = generate_task_data()

    # 1. Encode Anchor Cloud A
    cond_A = encoder(A)
    cond_B = encoder(B)
    cond = torch.cat([cond_A, cond_B], dim=-1)

    # 2. Sample random time t ~ U(0, 1)
    t = torch.rand(1).item()

    # 3. Create probability path (Straight line interpolation)
    x_t = (1.0 - t) * B + t * B_in_A

    # 4. Define ground truth target velocity (u_t = x_1 - x_0)
    target_velocity = B_in_A - B

    # 5. Predict velocity
    pred_velocity = flow_net(x_t, t, cond)

    # 6. CFM Loss
    loss = F.mse_loss(pred_velocity, target_velocity)

    loss.backward()
    optimizer.step()

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.5f}")

# =====================================================================
# 4. INFERENCE via ODE INTEGRATION (Euler Method)
# =====================================================================
print("\nRunning Inference / ODE Integration...")
encoder.eval()
flow_net.eval()

with torch.no_grad():
    # Generate an unseen test task
    A_test, B_in_A_test, B_test = generate_task_data()

    # Get condition code from anchor A
    cond_test_A = encoder(A_test)
    cond_test_B = encoder(B_test)

    cond_test = torch.cat([cond_test_A, cond_test_B], dim=-1)

    # Trajectory tracking for visualization
    trajectory = [B_test.clone().numpy()]

    # Euler ODE Steps
    steps = 20
    dt = 1.0 / steps
    x_curr = B_test.clone()  # Start at t = 0 (Source Domain)

    for step in range(steps):
        t_curr = step * dt
        v_pred = flow_net(x_curr, t_curr, cond_test)
        x_curr = x_curr + v_pred * dt  # Take vector field step
        trajectory.append(x_curr.numpy())

# =====================================================================
# 5. PLOTTING RESULTS
# =====================================================================
trajectory = np.array(trajectory)  # [Steps, N_B, 2]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Input Setup
axes[0].scatter(B_test[:, 0], B_test[:, 1], color='crimson', alpha=0.6, label='Detailed Source (B)')
axes[0].scatter(A_test[:, 0], A_test[:, 1], color='darkblue', marker='x', s=60, label='Sparse Anchor (A)')
axes[0].set_title("Inputs given at Inference")
axes[0].legend()
axes[0].grid(True)

# Plot 2: Trajectories over Time
axes[1].scatter(trajectory[0, :, 0], trajectory[0, :, 1], color='crimson', alpha=0.3, label='t = 0')
axes[1].scatter(trajectory[steps // 2, :, 0], trajectory[steps // 2, :, 1], color='orange', alpha=0.5, label='t = 0.5')
axes[1].scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], color='teal', alpha=0.8, label='t = 1.0')
# Draw a few sample vector lines
for i in range(0, B_test.shape[0], 15):
    axes[1].plot(trajectory[:, i, 0], trajectory[:, i, 1], color='gray', linestyle='--', alpha=0.5)
axes[1].set_title("CFM Vector Field Trajectories ($B \\rightarrow A$)")
axes[1].legend()
axes[1].grid(True)

# Plot 3: Predicted Output vs Hidden Ground Truth Target
axes[2].scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], color='teal', alpha=0.7, label='Morphed Output')
axes[2].scatter(B_in_A_test[:, 0], B_in_A_test[:, 1], facecolors='none', edgecolors='black', s=50, alpha=0.5,
                label='True Target ($B_{in\\,A}$)')
axes[2].set_title("Final Registered Output vs. Target Domain")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()