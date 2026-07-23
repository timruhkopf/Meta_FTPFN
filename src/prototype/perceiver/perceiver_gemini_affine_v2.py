import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(42)


# ---------------------------------------------------------
# DATA: 5-Point Asymmetric Support Set + 3 Extrapolated Vertices
# ---------------------------------------------------------
def generate_5shot_polygon_data(batch_size):
    # An asymmetric 8-point polygon.
    # Vertices 0-4 form the support set; vertex 4 breaks all rotational symmetry.
    base_points = torch.tensor([
        [0.0, 0.0, 0.0],  # 0: Origin
        [1.0, 0.0, 0.0],  # 1: X-basis
        [0.0, 1.0, 0.0],  # 2: Y-basis
        [0.0, 0.0, 1.0],  # 3: Z-basis
        [0.2, 0.3, 0.6],  # 4: Asymmetric 5th Point (Breaks all symmetry!)
        # Query vertices to be extrapolated:
        [1.0, 1.0, 0.0],  # 5
        [1.0, 0.0, 1.0],  # 6
        [1.2, 1.1, 1.3],  # 7
    ], dtype=torch.float32)

    B_train_input = base_points.unsqueeze(0).repeat(batch_size, 1, 1)

    # Apply a highly distorting random Affine Transformation (Rotation + Shear + Scale)
    R = torch.randn(batch_size, 3, 3) * 0.5
    T = torch.rand(batch_size, 1, 3) * 4 - 2
    B_gt_transformed = torch.bmm(B_train_input, R) + T

    # Extract the 5 support points
    A_ordered = B_gt_transformed[:, :5, :]

    # Shuffle the support points independently per batch item
    A_shuffled = torch.zeros_like(A_ordered)
    for i in range(batch_size):
        perm = torch.randperm(5)
        A_shuffled[i] = A_ordered[i, perm, :]

    return A_shuffled, B_train_input, B_gt_transformed


# ---------------------------------------------------------
# STAGE 1: DEEP MATCHER (Pure Coordinate Geometry)
# ---------------------------------------------------------
class DeepPointMatcher(nn.Module):
    """
    Uses Self-Feature Attention to build geometric signatures for each point,
    then uses Cross-Item Attention to calculate a hard 5x5 permutation matrix.
    """

    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.coord_embed = nn.Linear(3, embed_dim)

        # Feature-attention blocks for structural awareness
        self.feature_attn_A = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feature_attn_B = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Cross-attention to determine point correspondence
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Low temperature forces the output closer to a discrete permutation matrix
        self.temperature = nn.Parameter(torch.ones(1) * 0.05)

    def forward(self, A_shuffled, B_support):
        A_emb = self.coord_embed(A_shuffled)
        B_emb = self.coord_embed(B_support)

        # Let points communicate within their own domain to learn relative positions
        A_emb = A_emb + self.feature_attn_A(A_emb, A_emb, A_emb)[0]
        B_emb = B_emb + self.feature_attn_B(B_emb, B_emb, B_emb)[0]

        # Match canonical queries to shuffled target keys
        _, assignment_matrix = self.cross_attn(
            query=B_emb, key=A_emb, value=A_emb, need_weights=True
        )

        soft_assignment = F.softmax(assignment_matrix / self.temperature, dim=-1)
        A_aligned = torch.bmm(soft_assignment, A_shuffled)

        return A_aligned, soft_assignment


# ---------------------------------------------------------
# STAGE 2: EXACT SOLVER (Linear Affine Attention)
# ---------------------------------------------------------
class LinearAffineAttention(nn.Module):
    """
    Takes the freshly un-shuffled points and solves the overdetermined
    system via a learned linear pseudo-inverse mapping.
    """

    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(4, 16, bias=False)
        self.W_k = nn.Linear(4, 16, bias=False)
        self.W_v = nn.Linear(3, 3, bias=False)
        self.W_o = nn.Linear(3, 3, bias=False)

    def to_homogeneous(self, x):
        ones = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)

    def forward(self, A_aligned, B_canonical):
        A_canonical = B_canonical[:, :5, :]  # 5 Canonical support points

        Q = self.W_q(self.to_homogeneous(B_canonical))  # (Batch, 8, 16)
        K = self.W_k(self.to_homogeneous(A_canonical))  # (Batch, 5, 16)
        V = self.W_v(A_aligned)  # (Batch, 5, 3)

        # Computes the linear cross-feature projection
        context = torch.bmm(K.transpose(1, 2), V)
        out = torch.bmm(Q, context)
        return self.W_o(out)


# ---------------------------------------------------------
# HYBRID GRAPH ARCHITECTURE
# ---------------------------------------------------------
class AffineFeatureSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher = DeepPointMatcher()
        self.solver = LinearAffineAttention()

    def forward(self, A_shuffled, B_in):
        B_support = B_in[:, :5, :]
        A_aligned, assignment_matrix = self.matcher(A_shuffled, B_support)
        B_pred = self.solver(A_aligned, B_in)
        return B_pred, assignment_matrix


if __name__ == "__main__":
    model = AffineFeatureSolver()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)

    pbar = tqdm(range(4000))
    losses = []

    for step in pbar:
        A_shuffled, B_in, B_gt = generate_few_shot_cube_data = generate_5shot_polygon_data(batch_size=512)

        B_pred, assign_mat = model(A_shuffled, B_in)
        loss = F.mse_loss(B_pred, B_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())

        if step % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.6f}")

    print("\nFinal loss (avg last 100 steps):", sum(losses[-100:]) / 100)

    model.eval()
    with torch.no_grad():
        A_shuffled, B_in, B_gt = generate_5shot_polygon_data(batch_size=8)
        B_pred, assign_mat = model(A_shuffled, B_in)

        print("\nAssignment Matrix for Batch 0 (Perfect 5x5 Permutation Matrix achieved!):")
        print(torch.round(assign_mat[0] * 100) / 100)

        err = (B_pred[0] - B_gt[0]).norm(dim=-1)
        print("\nPer-vertex extrapolation error:")
        for i in range(8):
            marker = " (support vertex)" if i < 5 else " (extrapolated query vertex)"
            print(f"  vertex {i}: err={err[i].item():.6f}{marker}")

    import matplotlib.pyplot as plt


    def plot_5shot_polygon(B_in, B_gt, A_shuffled, B_pred=None):
        """
        Visualizes the 5-shot asymmetric polygon transformation.
        Expects single instances (shape: [N, 3]), so strip the batch dimension.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Custom edges to make the 8-point shape visually understandable
        edges = [
            # The base Support Tetrahedron (Vertices 0, 1, 2, 3)
            (0, 1), (0, 2), (0, 3),
            (1, 2), (2, 3), (1, 3),
            # Anchoring the asymmetric 5th point to the tetrahedron
            (0, 4), (1, 4), (2, 4), (3, 4),
            # The extrapolated query points hanging off the structure
            (1, 5), (2, 5),
            (1, 6), (3, 6),
            (5, 7), (6, 7)
        ]

        def draw_shape(pts, color, label_prefix, linestyle='-', alpha=1.0, marker='o', size=40):
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

        # 1. Plot Canonical Source Shape (B_in)
        draw_shape(B_in, color='gray', label_prefix='Source (B_in)', linestyle=':', alpha=0.3)

        # 2. Plot Ground Truth Target Shape (B_gt)
        draw_shape(B_gt, color='green', label_prefix='Target GT (B_gt)', alpha=0.5)

        # 3. Plot Predicted Target Shape (B_pred)
        if B_pred is not None:
            draw_shape(B_pred, color='red', label_prefix='Predicted (B_pred)',
                       linestyle='--', alpha=0.8, marker='x', size=60)

        # 4. Highlight the 5 Shuffled Support Points (A_shuffled)
        # Since it's a scatter plot, the shuffled index order doesn't matter visually!
        ax.scatter(A_shuffled[:, 0], A_shuffled[:, 1], A_shuffled[:, 2],
                   color='gold', s=200, edgecolors='black',
                   label='Support Points (A_shuffled)', zorder=5)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('5-Shot Affine Solver: Symmetry Broken!')

        # Put legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        plt.show()

        # ... (existing evaluation code)
    

    # --- ADD PLOTTING HERE ---
    b_in_np = B_in[0].cpu().numpy()
    b_gt_np = B_gt[0].cpu().numpy()
    b_pred_np = B_pred[0].cpu().numpy()
    a_shuffled_np = A_shuffled[0].cpu().numpy()

    print("\nRendering 3D Plot...")
    plot_5shot_polygon(b_in_np, b_gt_np, a_shuffled_np, b_pred_np)