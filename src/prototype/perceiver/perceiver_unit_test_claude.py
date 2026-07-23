import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)

# ---------------------------------------------------------
# DATA (same as before)
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
    """Exact affine solve from the 4 correspondences. Used to sanity-check identifiability."""
    y0 = A_train[:, 0:1, :]               # (Batch,1,3)
    R_rows = A_train[:, 1:4, :] - y0       # (Batch,3,3)  rows = R^T essentially
    # y = x @ R + T  =>  R = R_rows (already arranged so x@R_rows recreates the right combo)
    return torch.bmm(B_query, R_rows) + y0


# ---------------------------------------------------------
# TRUE FEATURE-WISE DUAL ATTENTION (TabPFN-style)
# ---------------------------------------------------------
class ItemSelfAttnMLP(nn.Module):
    """Lets A's own 4 points attend to EACH OTHER, across R, shared weights across C.
    Crucial: a single B->A cross-attention layer can only ever produce a
    non-negative (softmax) combination of A's points, so it can't form a
    difference like A_i - A_0. By letting A self-attend first, the value
    projection (a free linear map, can encode a sign flip) plus the residual
    connection CAN form exactly that difference -- which cross-attention
    then just has to copy out."""
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.item_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio), nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, X):
        # X: (Batch, R, C, E) - self-attend across R, shared weights per C
        Bsz, R, C, E = X.shape
        Xr = X.permute(0, 2, 1, 3).reshape(Bsz * C, R, E)
        attn_out, _ = self.item_attn(Xr, Xr, Xr)
        Xr = self.ln1(Xr + attn_out)
        Xr = self.ln2(Xr + self.mlp(Xr))
        return Xr.reshape(Bsz, C, R, E).permute(0, 2, 1, 3)


class FeatureSelfAttnMLP(nn.Module):
    """Mixes info ACROSS the C=3 coordinate channels, independently per point.
    Needed because rotations require cross-channel mixing."""
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.feat_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio), nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, X):
        # X: (Batch, R, C, E)
        Bsz, R, C, E = X.shape
        Xc = X.reshape(Bsz * R, C, E)
        attn_out, _ = self.feat_attn(Xc, Xc, Xc)
        Xc = self.ln1(Xc + attn_out)
        Xc = self.ln2(Xc + self.mlp(Xc))
        return Xc.reshape(Bsz, R, C, E)


class FeatureWiseDualAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.pre_feat_mix = FeatureSelfAttnMLP(embed_dim, num_heads, mlp_ratio)  # NEW
        self.item_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.feat_mix = FeatureSelfAttnMLP(embed_dim, num_heads, mlp_ratio)  # kept, now a second mix after

    def forward(self, A, B):
        B = self.pre_feat_mix(B)          # <-- NEW: B now knows its own (x,y,z) jointly before querying A

        Bsz, RA, C, E = A.shape
        _, RB, _, _ = B.shape

        A_ic = A.permute(0, 2, 1, 3).reshape(Bsz * C, RA, E)
        B_ic = B.permute(0, 2, 1, 3).reshape(Bsz * C, RB, E)
        attn_out, _ = self.item_attn(query=B_ic, key=A_ic, value=A_ic)
        B_ic = self.ln1(B_ic + attn_out)
        B_new = B_ic.reshape(Bsz, C, RB, E).permute(0, 2, 1, 3)

        B_new = self.feat_mix(B_new)
        return B_new

class FeatureWisePerceiver(nn.Module):
    def __init__(self, embedding_size=32, num_blocks=2, num_support_points=4):
        super().__init__()
        E = embedding_size
        self.embedder = nn.Linear(1, E)
        self.decoder = nn.Linear(E, 1)
        # Positional tag per support point: without this, A's 4 points carry
        # no content signal saying "I am the image of canonical basis vector i"
        # (their raw values are randomly transformed every batch), and
        # nn.MultiheadAttention has no positional encoding of its own.
        self.support_pos_emb = nn.Parameter(torch.randn(num_support_points, 1, E) * 1.0)
        self.A_item_blocks = nn.ModuleList([ItemSelfAttnMLP(E) for _ in range(num_blocks)])
        self.A_feat_blocks = nn.ModuleList([FeatureSelfAttnMLP(E) for _ in range(num_blocks)])
        self.blocks = nn.ModuleList([FeatureWiseDualAttentionBlock(E) for _ in range(num_blocks)])

    def forward(self, A_BRC, B_BRC):
        A = self.embedder(A_BRC.unsqueeze(-1))  # (Batch, RA, C, E)
        B = self.embedder(B_BRC.unsqueeze(-1))  # (Batch, RB, C, E)
        # broadcast positional tag: (RA,1,E) -> (1,RA,1,E) added to (Batch,RA,C,E)
        A = A + self.support_pos_emb.permute(1, 0, 2).unsqueeze(2)
        for item_blk, feat_blk in zip(self.A_item_blocks, self.A_feat_blocks):
            A = item_blk(A)
            A = feat_blk(A)
        for block in self.blocks:
            B = block(A, B)
        return self.decoder(B).squeeze(-1)


if __name__ == "__main__":
    # Sanity check: closed-form oracle should solve this exactly
    A, B_in, B_gt = generate_few_shot_cube_data(batch_size=4)
    oracle_pred = closed_form_oracle(A, B_in)
    print("Oracle MSE (should be ~0):", F.mse_loss(oracle_pred, B_gt).item())

    model = FeatureWisePerceiver(embedding_size=32, num_blocks=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=600, factor=0.5)

    pbar = tqdm(range(10000))
    losses = []
    for step in pbar:
        A_train, B_transformed, B_gt = generate_few_shot_cube_data(batch_size=512)
        B_pred = model(A_train, B_transformed)
        loss = F.mse_loss(B_pred, B_gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())

        if step % 100 == 0:
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
        print(f"  vertex {i}: target={B_gt[0,i].tolist()}, pred={B_pred[0,i].tolist()}, err={err[i].item():.4f}{marker}")