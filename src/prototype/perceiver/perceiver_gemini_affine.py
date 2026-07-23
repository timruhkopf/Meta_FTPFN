import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)


# ---------------------------------------------------------
# DATA (unchanged from your code)
# ---------------------------------------------------------
def generate_few_shot_cube_data(batch_size):
    cube = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ], dtype=torch.float32)
    B_train_input = cube.unsqueeze(0).repeat(batch_size, 1, 1)
    R = torch.randn(batch_size, 3, 3) * 0.5
    T = torch.rand(batch_size, 1, 3) * 4 - 2
    B_gt_transformed = torch.bmm(B_train_input, R) + T
    basis_indices = [0, 1, 2, 3]
    A_train = B_gt_transformed[:, basis_indices, :]
    return A_train, B_train_input, B_gt_transformed


def closed_form_oracle(A_train, B_query):
    y0 = A_train[:, 0:1, :]
    R_rows = A_train[:, 1:4, :] - y0
    return torch.bmm(B_query, R_rows) + y0


# ---------------------------------------------------------
# THE FIX: Pure Linear Affine Attention
# ---------------------------------------------------------
class LinearAffineAttention(nn.Module):
    """
    Solves the geometric alignment perfectly by mirroring exact affine math.
    1. No Softmax: Allows interpolation AND extrapolation natively.
    2. Q & K in Source Domain: Compares invariant geometric relationships.
    3. V in Target Domain: Applies the weights directly to the transformed space.
    4. Homogeneous Coordinates: Naturally handles translation.
    """

    def __init__(self):
        super().__init__()
        # 4D mappings for homogeneous coordinates (x, y, z, 1)
        self.W_q = nn.Linear(4, 4, bias=False)
        self.W_k = nn.Linear(4, 4, bias=False)
        # 3D mapping for target values (x, y, z)
        self.W_v = nn.Linear(3, 3, bias=False)
        self.W_o = nn.Linear(3, 3, bias=False)

        # Initialize near identity to speed up learning of the linear system
        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)
        nn.init.eye_(self.W_v.weight)
        nn.init.eye_(self.W_o.weight)

    def to_homogeneous(self, x):
        # Append 1s to handle translations natively, replacing the need for an MLP bias
        ones = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)

    def forward(self, A_target, B_canonical):
        # A_target: (Batch, 4, 3)  - Target domain support points (Values)
        # B_canonical: (Batch, 8, 3) - Source domain points (Queries)

        # Extract Source domain support points (the first 4 points are the basis)
        # FIXME: directly assumes index ordering is also matching in points!
        A_canonical = B_canonical[:, :4, :]

        # Map Q and K strictly in the fixed canonical space (Homogeneous)
        Q = self.W_q(self.to_homogeneous(B_canonical))  # (Batch, 8, 4)
        K = self.W_k(self.to_homogeneous(A_canonical))  # (Batch, 4, 4)

        # V lives purely in the dynamically moving target domain
        V = self.W_v(A_target)  # (Batch, 4, 3)

        # LINEAR ATTENTION (No Softmax)
        # Step 1: Compute geometric relationship weights
        # K^T @ V -> (Batch, 4, 4) @ (Batch, 4, 3) = (Batch, 4, 3)
        context = torch.bmm(K.transpose(1, 2), V)

        # Step 2: Apply weights to the Queries to push them out into Target space
        # Q @ context -> (Batch, 8, 4) @ (Batch, 4, 3) = (Batch, 8, 3)
        out = torch.bmm(Q, context)

        return self.W_o(out)


import matplotlib.pyplot as plt


def plot_cube_transformation(B_in, B_gt, A_train, B_pred=None):
    """
    Plots the 3D cubes. Expects single instances (shape: [N, 3]),
    so strip the batch dimension before passing them in.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # The vertices connected by edges in our specific cube generation layout
    edges = [
        (0, 1), (0, 2), (0, 3),  # edges connected to origin (0,0,0)
        (1, 4), (1, 5),  # edges from (1,0,0)
        (2, 4), (2, 6),  # edges from (0,1,0)
        (3, 5), (3, 6),  # edges from (0,0,1)
        (4, 7), (5, 7), (6, 7)  # edges connecting to (1,1,1)
    ]

    def draw_cube(pts, color, label_prefix, linestyle='-', alpha=1.0, marker='o', size=40):
        # Draw vertices
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   color=color, s=size, label=f'{label_prefix} Vertices', alpha=alpha, marker=marker)
        # Draw edges
        for i, (p1, p2) in enumerate(edges):
            ax.plot([pts[p1, 0], pts[p2, 0]],
                    [pts[p1, 1], pts[p2, 1]],
                    [pts[p1, 2], pts[p2, 2]],
                    color=color, linestyle=linestyle, alpha=alpha,
                    label=f'{label_prefix} Edges' if i == 0 else "")

    # 1. Plot Canonical Source Cube (B_in)
    draw_cube(B_in, color='gray', label_prefix='Source (B_in)', linestyle=':', alpha=0.4)

    # 2. Plot Ground Truth Target Cube (B_gt)
    draw_cube(B_gt, color='green', label_prefix='Target GT (B_gt)', alpha=0.6)

    # 3. Plot Predicted Target Cube (B_pred)
    if B_pred is not None:
        draw_cube(B_pred, color='red', label_prefix='Predicted (B_pred)',
                  linestyle='--', alpha=0.8, marker='x', size=60)

    # 4. Highlight the 4 Support Points (A_train)
    # These are geometrically identical to the first 4 points of B_gt
    ax.scatter(A_train[:, 0], A_train[:, 1], A_train[:, 2],
               color='gold', s=200, edgecolors='black',
               label='Support Points (A_train)', zorder=5)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Few-Shot Affine Solver: Source vs Target vs Prediction')

    # Put legend outside the plot so it doesn't overlap the cubes
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    A, B_in, B_gt = generate_few_shot_cube_data(batch_size=4)
    oracle_pred = closed_form_oracle(A, B_in)
    print("Oracle MSE (should be ~0):", F.mse_loss(oracle_pred, B_gt).item())

    model = LinearAffineAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.5)

    # Note: Reduced steps from 10k to 2500 since this architecture converges rapidly
    pbar = tqdm(range(1000))
    losses = []
    for step in pbar:
        A_train, B_in_canonical, B_gt = generate_few_shot_cube_data(batch_size=512)

        # B_in_canonical is the static source cube (what you called B_transformed)
        B_pred = model(A_train, B_in_canonical)
        loss = F.mse_loss(B_pred, B_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())

        if step % 200 == 0:
            pbar.set_description(f"Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    print("\nFinal loss (avg last 100 steps):", sum(losses[-100:]) / 100)

    model.eval()
    with torch.no_grad():
        A, B_in, B_gt = generate_few_shot_cube_data(batch_size=8)
        B_pred = model(A, B_in)
        eval_loss = F.mse_loss(B_pred, B_gt)
    print("Held-out eval MSE:", eval_loss.item())
    print("\nPer-vertex error (first cube in batch):")
    err = (B_pred[0] - B_gt[0]).norm(dim=-1)
    for i in range(8):
        marker = " (was support point)" if i < 4 else " (had to be inferred)"
        print(
            f"  vertex {i}: target={B_gt[0, i].tolist()}, pred={B_pred[0, i].tolist()}, err={err[i].item():.4f}{marker}")

    # Detach from graph, move to CPU, and convert to numpy for matplotlib

    for _ in range(3):
        with torch.no_grad():
            A, B_in, B_gt = generate_few_shot_cube_data(batch_size=8)
            B_pred = model(A, B_in)
        b_in_np = B_in[0].cpu().numpy()
        b_gt_np = B_gt[0].cpu().numpy()
        b_pred_np = B_pred[0].cpu().numpy()
        a_train_np = A[0].cpu().numpy()

        print("\nRendering 3D Plot...")
        plot_cube_transformation(b_in_np, b_gt_np, a_train_np, b_pred_np)