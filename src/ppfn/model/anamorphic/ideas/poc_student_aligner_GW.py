import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple


# =====================================================================
# 1. ARCHITECTURE & LOSS DEFINITIONS
# =====================================================================

class CellTokenizer(nn.Module):
    """
    Prototyping cell tokenizer for continuous inputs and targets.
    Safely handles 3D inputs [T, B, D] by expanding them to [T, B, D, 1]
    before linear projection.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.target_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # Ensure x has trailing value dimension: [T, B, D] -> [T, B, D, 1]
        if x.dim() == 3:
            x = x.unsqueeze(-1)

        h = self.value_proj(x)

        if y is not None:
            # Ensure y is [T, B, 1]
            if y.dim() == 2:
                y = y.unsqueeze(-1)
            # Project to [T, B, d_model], then broadcast across D features -> [T, B, 1, d_model]
            y_emb = self.target_proj(y).unsqueeze(2)
            h = h + y_emb

        return self.norm(h)

class AlternatingBlockLayer(nn.Module):
    """
    Staged Alternating Block Layer:
    1. Intra-domain self-attention on diag blocks (A and B).
    2. Feature-wise cross-attention (Schema Translation).
    3. Row-wise cross-attention (Manifold Grounding).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Intra-Domain Self-Attention
        self.row_attn_A = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.col_attn_A = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.row_attn_B = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.col_attn_B = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # Off-Diagonal Feature-Wise Cross-Attention
        self.feat_cross_B_in_A = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.feat_cross_A_in_B = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # Off-Diagonal Row-Wise Cross-Attention
        self.row_ground_B_in_A = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.row_ground_A_in_B = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)

        # Norms and FFNs
        self.norm_A = nn.LayerNorm(d_model)
        self.norm_B = nn.LayerNorm(d_model)
        self.norm_BA = nn.LayerNorm(d_model)
        self.norm_AB = nn.LayerNorm(d_model)
        self.ffn_A = self._build_ffn(d_model, dropout)
        self.ffn_B = self._build_ffn(d_model, dropout)
        self.ffn_BA = self._build_ffn(d_model, dropout)
        self.ffn_AB = self._build_ffn(d_model, dropout)

    def _build_ffn(self, d_model: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def _apply_row_attn(self, x: torch.Tensor, attn_mod: nn.MultiheadAttention) -> torch.Tensor:
        T, B, D, M = x.shape
        x_flat = x.permute(0, 1, 2, 3).reshape(T, B * D, M)
        out, _ = attn_mod(x_flat, x_flat, x_flat)
        return out.reshape(T, B, D, M)

    def _apply_col_attn(self, x: torch.Tensor, attn_mod: nn.MultiheadAttention) -> torch.Tensor:
        T, B, D, M = x.shape
        x_flat = x.permute(2, 0, 1, 3).reshape(D, T * B, M)
        out, _ = attn_mod(x_flat, x_flat, x_flat)
        return out.reshape(D, T, B, M).permute(1, 2, 0, 3)

    def _apply_feat_cross(self, q: torch.Tensor, kv: torch.Tensor, attn_mod: nn.MultiheadAttention) -> torch.Tensor:
        T_q, B, D_q, M = q.shape
        T_kv, _, D_kv, _ = kv.shape
        q_flat = q.permute(2, 0, 1, 3).reshape(D_q, T_q * B, M)
        kv_flat = kv.permute(2, 0, 1, 3).reshape(D_kv, T_kv * B, M)
        out, _ = attn_mod(query=q_flat, key=kv_flat, value=kv_flat)
        return out.reshape(D_q, T_q, B, M).permute(1, 2, 0, 3)

    def _apply_row_cross(self, q: torch.Tensor, kv: torch.Tensor, attn_mod: nn.MultiheadAttention) -> torch.Tensor:
        T_q, B, D, M = q.shape
        T_kv, _, _, _ = kv.shape
        q_flat = q.permute(0, 1, 2, 3).reshape(T_q, B * D, M)
        kv_flat = kv.permute(0, 1, 2, 3).reshape(T_kv, B * D, M)
        out, _ = attn_mod(query=q_flat, key=kv_flat, value=kv_flat)
        return out.reshape(T_q, B, D, M)

    def forward(
            self,
            h_A: torch.Tensor,
            h_B: torch.Tensor,
            h_B_in_A: torch.Tensor,
            h_A_in_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Diagonal Blocks Intra-Task Updates
        h_A = h_A + self._apply_row_attn(h_A, self.row_attn_A)
        h_A = h_A + self._apply_col_attn(h_A, self.col_attn_A)
        h_A = self.norm_A(h_A + self.ffn_A(h_A))

        h_B = h_B + self._apply_row_attn(h_B, self.row_attn_B)
        h_B = h_B + self._apply_col_attn(h_B, self.col_attn_B)
        h_B = self.norm_B(h_B + self.ffn_B(h_B))

        # 2. Schema Translation: Cross-attend over features of the source domain
        h_B_in_A = h_B_in_A + self._apply_feat_cross(q=h_B_in_A, kv=h_B, attn_mod=self.feat_cross_B_in_A)
        h_A_in_B = h_A_in_B + self._apply_feat_cross(q=h_A_in_B, kv=h_A, attn_mod=self.feat_cross_A_in_B)

        # 3. Manifold Grounding: Cross-attend over rows of the target domain
        h_B_in_A = h_B_in_A + self._apply_row_cross(q=h_B_in_A, kv=h_A, attn_mod=self.row_ground_B_in_A)
        h_B_in_A = self.norm_BA(h_B_in_A + self.ffn_BA(h_B_in_A))

        h_A_in_B = h_A_in_B + self._apply_row_cross(q=h_A_in_B, kv=h_B, attn_mod=self.row_ground_A_in_B)
        h_A_in_B = self.norm_AB(h_A_in_B + self.ffn_AB(h_A_in_B))

        return h_A, h_B, h_B_in_A, h_A_in_B


class StudentManifoldAligner(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = CellTokenizer(d_model)
        self.layers = nn.ModuleList([
            AlternatingBlockLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.init_prompt_BA = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.init_prompt_AB = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)

    def forward(
        self,
        X_A: torch.Tensor,
        Y_A: torch.Tensor,
        X_B: torch.Tensor,
        Y_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Correctly unpack 3D tensor shape: [T, Batch, D_features]
        T_A, B, D_A = X_A.shape[:3]
        T_B, _, D_B = X_B.shape[:3]

        h_A = self.tokenizer(X_A, Y_A)  # -> [T_A, B, D_A, d_model]
        h_B = self.tokenizer(X_B, Y_B)  # -> [T_B, B, D_B, d_model]

        h_B_in_A = self.init_prompt_BA.expand(T_B, B, D_A, self.d_model).clone()
        h_A_in_B = self.init_prompt_AB.expand(T_A, B, D_B, self.d_model).clone()

        for layer in self.layers:
            h_A, h_B, h_B_in_A, h_A_in_B = layer(h_A, h_B, h_B_in_A, h_A_in_B)

        return h_B_in_A, h_A_in_B


def compute_relational_gw_loss(Z_pred: torch.Tensor, Z_gt: torch.Tensor) -> torch.Tensor:
    """
    Computes relational Gram-matrix Frobenius loss on true cosine similarities,
    masking out the constant diagonal (self-similarity = 1.0) to measure
    pure off-diagonal relational geometry.
    """
    T, B, D, M = Z_pred.shape

    # Pool across feature columns: [B, T, M]
    pred_sample = Z_pred.mean(dim=2).permute(1, 0, 2)
    gt_sample = Z_gt.mean(dim=2).permute(1, 0, 2)

    # Unit-sphere normalization -> dot products are true Cosine Similarities in [-1, 1]
    pred_norm = F.normalize(pred_sample, p=2, dim=-1)
    gt_norm = F.normalize(gt_sample, p=2, dim=-1)

    # Compute Cosine Similarity Gram matrices: [B, T, T] (NO / sqrt(M) division!)
    G_pred = torch.bmm(pred_norm, pred_norm.transpose(1, 2))
    G_gt = torch.bmm(gt_norm, gt_norm.transpose(1, 2))

    # Mask out the diagonal (which is trivially 1.0 for all unit vectors)
    eye_mask = torch.eye(T, device=Z_pred.device, dtype=torch.bool).unsqueeze(0)

    G_pred_off = G_pred.masked_fill(eye_mask, 0.0)
    G_gt_off = G_gt.masked_fill(eye_mask, 0.0)

    # Compute MSE strictly over the off-diagonal entries
    off_diag_elements = T * (T - 1)
    loss = F.mse_loss(G_pred_off, G_gt_off, reduction='sum') / (B * off_diag_elements)

    return loss

# =====================================================================
# 2. TRAINING STEP (AUTOCAST FORWARD -> FLOAT32 LOSS OUTSIDE)
# =====================================================================

def train_step_aligner(
        model: StudentManifoldAligner,
        optimizer: torch.optim.Optimizer,
        batch_dict: Dict[str, Dict[str, torch.Tensor]],
        device: torch.device
) -> Tuple[float, float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    train_data = batch_dict['train']
    X_B = train_data['X_B'].to(device)
    Y_B = train_data['Y_B'].to(device)
    X_A = train_data['X_A'].to(device)
    Y_A = train_data['Y_A'].to(device)

    X_B_in_A_gt = train_data['X_B_in_A'].to(device)
    Y_B_in_A_gt = train_data['Y_B_in_A'].to(device)
    X_A_in_B_gt = train_data['X_A_in_B'].to(device)
    Y_A_in_B_gt = train_data['Y_A_in_B'].to(device)

    # 1. Forward Pass INSIDE Autocast
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        Z_B_in_A_pred, Z_A_in_B_pred = model(X_A, Y_A, X_B, Y_B)

        with torch.no_grad():
            Z_B_in_A_gt = model.tokenizer(X_B_in_A_gt, Y_B_in_A_gt)
            Z_A_in_B_gt = model.tokenizer(X_A_in_B_gt, Y_A_in_B_gt)

    # 2. Loss Computation OUTSIDE Autocast in full float32 precision
    loss_BA = compute_relational_gw_loss(Z_B_in_A_pred.float(), Z_B_in_A_gt.float())
    loss_AB = compute_relational_gw_loss(Z_A_in_B_pred.float(), Z_A_in_B_gt.float())
    total_loss = loss_BA + loss_AB

    # 3. Backward Pass & Optimizer Step
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), loss_BA.item(), loss_AB.item()


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from typing import Tuple


class PointwiseReconstructionProbe(nn.Module):
    """
    Honest pointwise diagnostic probe.
    Decodes physical (X_hat, Y_hat) coordinates directly from each token Z[i]
    independently. Contains NO cross-attention and NO access to external anchors.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2)  # Predicts physical (X_hat, Y_hat)
        )

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Z shape: [T_seq, Batch, D_features, d_model] -> pool feature dim D
        if Z.dim() == 4:
            Z_pooled = Z.mean(dim=2)
        else:
            Z_pooled = Z

        # Strict stop-gradient so coordinate MSE never backprops into the aligner
        out = self.net(Z_pooled.detach())

        # Return X_hat and Y_hat matching the [T, Batch, 1] target shape
        return out[..., 0:1], out[..., 1:2]

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = StudentManifoldAligner(d_model=128, n_heads=4, n_layers=2, dropout=0.1).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #
    # # Dummy batch dictionary simulating prior outputs
    # batch_dict = {
    #     'train': {
    #         'X_B': torch.randn(10, 8, 5, 1), 'Y_B': torch.randn(10, 8, 1),
    #         'X_A': torch.randn(12, 8, 6, 1), 'Y_A': torch.randn(12, 8, 1),
    #         'X_B_in_A': torch.randn(10, 8, 5, 1), 'Y_B_in_A': torch.randn(10, 8, 1),
    #         'X_A_in_B': torch.randn(12, 8, 6, 1), 'Y_A_in_B': torch.randn(12, 8, 1),
    #     }
    # }
    #
    # loss = train_step_aligner(model, optimizer, batch_dict, device)
    # print(f"Training step completed successfully. Loss: {loss:.6f}")

    import os
    import math
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from typing import Dict, Tuple

    import numpy as np
    import matplotlib.pyplot as plt

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from typing import Dict, Tuple, Optional


    class UnifiedManifoldVisualizer:
        """
        3-row visualizer per batch item:
        Row 0: Canonical Domain A space (A_train support, B_in_A target, Z_{B->A} probed stars).
        Row 1: Distorted Domain B space (B_train support, A_in_B target, Z_{A->B} probed stars).
        Row 2: Spatially-ordered Gram matrix heatmaps (Sorted ascending by x-coordinate).
        """

        def __init__(self, x_range: Tuple[float, float] = (-5.0, 5.0)):
            self.min_x, self.max_x = x_range
            self.x_dense = torch.linspace(self.min_x, self.max_x, 1000)
            self.x_dense_np = self.x_dense.numpy()

        @staticmethod
        def _to_np(val):
            return val.detach().cpu().numpy() if torch.is_tensor(val) else val

        def _reconstruct_curve(self, params: Dict, batch_idx: int, stream_type: str = 'A') -> np.ndarray:
            amps = params['params_A'][0][:, batch_idx].unsqueeze(1)
            freqs = params['params_A'][1][:, batch_idx].unsqueeze(1)
            phases = params['params_A'][2][:, batch_idx].unsqueeze(1)

            x_eval = self.x_dense.unsqueeze(0)

            if stream_type == 'A':
                terms = amps * torch.sin(2 * torch.pi * freqs * x_eval + phases)
                return terms.sum(dim=0).squeeze(0).numpy()

            # Domain B: apply the horizontal warp BEFORE evaluating the harmonic,
            # then the affine scale/shift -- x_eval is the raw dense grid, it is
            # NOT pre-warped, so this step can't be skipped.
            v_shift = params['shifts'][0][batch_idx].item()
            h_shift = params['shifts'][1][batch_idx].item()
            scale = params['scale_A'][batch_idx].item()
            w_amp = params['warps'][0][batch_idx].item()
            w_freq = params['warps'][1][batch_idx].item()
            w_phase = params['warps'][2][batch_idx].item()

            x_warped = x_eval - h_shift + w_amp * torch.sin(2 * torch.pi * w_freq * x_eval + w_phase)
            terms = amps * torch.sin(2 * torch.pi * freqs * x_warped + phases)
            y_eval = scale * terms.sum(dim=0) + v_shift

            return y_eval.squeeze(0).numpy()

        def _compute_sorted_gram_matrix(
                self,
                Z: torch.Tensor,
                x_coords: np.ndarray,
                batch_idx: int
        ) -> np.ndarray:
            sample_emb = Z[:, batch_idx, :, :].mean(dim=1)  # [T, d_model]
            norm_emb = F.normalize(sample_emb, p=2, dim=-1)
            gram = torch.mm(norm_emb, norm_emb.t())
            gram_np = self._to_np(gram)

            sort_idx = np.argsort(x_coords)
            return gram_np[np.ix_(sort_idx, sort_idx)]

        def generate_figure(
                self,
                batch_data: Dict,
                Z_BA_pred: torch.Tensor,
                Z_BA_gt: torch.Tensor,
                probe_preds_BA: Tuple[torch.Tensor, torch.Tensor],
                probe_preds_AB = None,  # Optional default
                batch_indices: Tuple[int, int] = (0, 1)
        ) -> plt.Figure:
            params = batch_data['params']
            X_hat_BA, Y_hat_BA = probe_preds_BA
            X_hat_AB, Y_hat_AB = probe_preds_AB

            fig = plt.figure(figsize=(14, 12))
            gs = fig.add_gridspec(3, len(batch_indices), hspace=0.35, wspace=0.15)

            for col, batch_idx in enumerate(batch_indices):
                # =================================================================
                # ROW 0: CANONICAL DOMAIN A SPACE
                # =================================================================
                ax_A = fig.add_subplot(gs[0, col])
                ax_A.set_facecolor('#1e1e1e')

                y_dense_A = self._reconstruct_curve(params, batch_idx, stream_type='A')
                ax_A.plot(self.x_dense_np, y_dense_A, color='white', linestyle='-',
                          linewidth=1.8, alpha=0.9, label='True A Curve', zorder=5)

                X_A = self._to_np(batch_data['train']['X_A'][:, batch_idx]).flatten()
                Y_A = self._to_np(batch_data['train']['Y_A'][:, batch_idx]).flatten()
                ax_A.scatter(X_A, Y_A, c='white', s=35, edgecolors='black',
                             linewidth=0.6, zorder=10, label='A_train Support Pts')

                X_BA_gt = self._to_np(batch_data['train']['X_B_in_A'][:, batch_idx]).flatten()
                Y_BA_gt = self._to_np(batch_data['train']['Y_B_in_A'][:, batch_idx]).flatten()
                ax_A.scatter(X_BA_gt, Y_BA_gt, c='cyan', s=45, edgecolors='black',
                             linewidth=0.6, zorder=15, label='B_in_A Target (GT)')

                X_mod_BA = self._to_np(X_hat_BA[:, batch_idx]).flatten()
                Y_mod_BA = self._to_np(Y_hat_BA[:, batch_idx]).flatten()
                ax_A.scatter(X_mod_BA, Y_mod_BA, c='yellow', s=65, marker='*',
                             edgecolors='black', linewidth=0.5, zorder=20, label='Model Probed (B->A)')

                for xt, yt, xm, ym in zip(X_BA_gt, Y_BA_gt, X_mod_BA, Y_mod_BA):
                    ax_A.plot([xt, xm], [yt, ym], color='yellow', linestyle=':',
                              linewidth=0.8, alpha=0.5, zorder=12)

                ax_A.set_xlim(self.min_x, self.max_x)
                ax_A.grid(True, color='#333333', linestyle=':', alpha=0.7)
                ax_A.legend(loc='upper right', fontsize=8, facecolor='#2b2b2b', edgecolor='none', framealpha=0.85)
                ax_A.set_title(f"BATCH ITEM {batch_idx}: Domain A Alignment", fontweight='bold', fontsize=11, pad=10)
                if col == 0:
                    ax_A.set_ylabel("Domain A Space", fontsize=10, fontweight='bold')

                # =================================================================
                # ROW 1: DISTORTED DOMAIN B SPACE (ADDED)
                # =================================================================
                ax_B = fig.add_subplot(gs[1, col])
                ax_B.set_facecolor('#1e1e1e')

                y_dense_B = self._reconstruct_curve(params, batch_idx, stream_type='B')
                ax_B.plot(self.x_dense_np, y_dense_B, color='red', linestyle='--',
                          linewidth=1.5, alpha=0.7, label='True B Curve', zorder=5)

                X_B = self._to_np(batch_data['train']['X_B'][:, batch_idx]).flatten()
                Y_B = self._to_np(batch_data['train']['Y_B'][:, batch_idx]).flatten()
                ax_B.scatter(X_B, Y_B, c='red', s=35, edgecolors='black',
                             linewidth=0.6, zorder=10, label='B_train Support Pts')

                X_AB_gt = self._to_np(batch_data['train']['X_A_in_B'][:, batch_idx]).flatten()
                Y_AB_gt = self._to_np(batch_data['train']['Y_A_in_B'][:, batch_idx]).flatten()
                ax_B.scatter(X_AB_gt, Y_AB_gt, c='orange', s=45, edgecolors='black',
                             linewidth=0.6, zorder=15, label='A_in_B Target (GT)')

                X_mod_AB = self._to_np(X_hat_AB[:, batch_idx]).flatten()
                Y_mod_AB = self._to_np(Y_hat_AB[:, batch_idx]).flatten()
                ax_B.scatter(X_mod_AB, Y_mod_AB, c='lime', s=65, marker='*',
                             edgecolors='black', linewidth=0.5, zorder=20, label='Model Probed (A->B)')

                for xt, yt, xm, ym in zip(X_AB_gt, Y_AB_gt, X_mod_AB, Y_mod_AB):
                    ax_B.plot([xt, xm], [yt, ym], color='lime', linestyle=':',
                              linewidth=0.8, alpha=0.5, zorder=12)

                ax_B.set_xlim(self.min_x - 2, self.max_x + 2)
                ax_B.grid(True, color='#333333', linestyle=':', alpha=0.7)
                ax_B.legend(loc='upper right', fontsize=8, facecolor='#2b2b2b', edgecolor='none', framealpha=0.85)
                ax_B.set_title(f"BATCH ITEM {batch_idx}: Domain B Alignment", fontweight='bold', fontsize=11, pad=10)
                if col == 0:
                    ax_B.set_ylabel("Domain B Space", fontsize=10, fontweight='bold')
                ax_B.set_xlabel("x-coordinate", fontsize=9)

                # =================================================================
                # ROW 2: SPATIALLY-ORDERED GRAM MATRIX HEATMAPS (B->A)
                # =================================================================
                ax_gram = fig.add_subplot(gs[2, col])

                G_pred = self._compute_sorted_gram_matrix(Z_BA_pred, X_BA_gt, batch_idx)
                G_gt = self._compute_sorted_gram_matrix(Z_BA_gt, X_BA_gt, batch_idx)

                np.fill_diagonal(G_pred, np.nan)
                np.fill_diagonal(G_gt, np.nan)

                composite = np.hstack([G_pred, np.full((G_pred.shape[0], 2), np.nan), G_gt])

                im = ax_gram.imshow(composite, cmap='magma', vmin=-1.0, vmax=1.0, aspect='auto')
                ax_gram.set_title("Gram Matrix B->A (Sorted by x): Pred | GT", fontsize=10, fontweight='bold')
                ax_gram.set_xlabel("Sorted Sample Index (Left -> Right along x-axis)", fontsize=9)
                if col == 0:
                    ax_gram.set_ylabel("Sorted Sample Index", fontsize=9)

                fig.colorbar(im, ax=ax_gram, fraction=0.046, pad=0.04)

            return fig

    def run_training_pipeline(
            num_iterations: int = 500,
            batch_size: int = 32,
            log_interval: int = 25,
            lr: float = 3e-4
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{device.type.upper()}] Initializing Student Manifold Aligner Pipeline...")

        # 1. Initialize Prior & Dataset Stream
        prior = HarmonicMixturePrior(
            num_components=4,
            noise_std=0.05,
            share_unrelated=0.2,
            scale=True,
            shift=True,
            warp=True
        )
        dataset = InfiniteHarmonicsStream(
            prior=prior,
            batch_size=batch_size,
            n_A=10,
            n_B=50,
            n_test=200
        )
        dataloader = DataLoader(dataset, batch_size=None)
        data_iterator = iter(dataloader)

        # 2. Initialize Model & Optimizer
        model = StudentManifoldAligner(
            d_model=128,
            n_heads=4,
            n_layers=3,
            dropout=0.1
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        # 3. Tracking metrics
        running_loss = 0.0
        running_loss_BA = 0.0
        running_loss_AB = 0.0

        print("=" * 72)
        print(f"{'ITERATION':<12} | {'TOTAL LOSS (GW)':<18} | {'LOSS B->A':<15} | {'LOSS A->B':<15}")
        print("=" * 72)

        for step in tqdm(range(1, num_iterations + 1)):
            batch_dict = next(data_iterator)

            total_loss, loss_BA, loss_AB = train_step_aligner(
                model=model,
                optimizer=optimizer,
                batch_dict=batch_dict,
                device=device
            )

            # Exponential moving average tracking
            alpha = 0.1 if step > 1 else 1.0
            running_loss = (1 - alpha) * running_loss + alpha * total_loss
            running_loss_BA = (1 - alpha) * running_loss_BA + alpha * loss_BA
            running_loss_AB = (1 - alpha) * running_loss_AB + alpha * loss_AB

            if step % log_interval == 0 or step == 1:
                print(
                    f"{step:<12} | "
                    f"{running_loss:<18.6f} | "
                    f"{running_loss_BA:<15.6f} | "
                    f"{running_loss_AB:<15.6f}"
                )

        print("=" * 72)
        print("Training Complete. Aligner weights are ready for downstream PFN predictor evaluation.")

        # =====================================================================
        # POST-TRAINING VISUALIZATION SANITY CHECK
        # =====================================================================
        # 1. Fit the stop-gradient visualization probe
        # =====================================================================
        # 1. Fit the honest pointwise probe
        shared_probe = PointwiseReconstructionProbe(d_model=model.d_model).to(device)
        optimizer = torch.optim.AdamW(shared_probe.parameters(), lr=1e-3)

        model.eval()
        shared_probe.train()

        for _ in range(1000):
            batch_dict = next(data_iterator)
            X_BA_gt = batch_dict['train']['X_B_in_A'].to(device)
            Y_BA_gt = batch_dict['train']['Y_B_in_A'].to(device)
            X_AB_gt = batch_dict['train']['X_A_in_B'].to(device)
            Y_AB_gt = batch_dict['train']['Y_A_in_B'].to(device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    Z_pred = model(
                        batch_dict['train']['X_A'].to(device),
                        batch_dict['train']['Y_A'].to(device),
                        batch_dict['train']['X_B'].to(device),
                        batch_dict['train']['Y_B'].to(device)
                    )

            # Decode directly from the tokens (no h_A or h_B anchors!)
            X_BA_hat, Y_BA_hat = shared_probe(Z_pred[0].float())
            X_AB_hat, Y_AB_hat = shared_probe(Z_pred[1].float())

            def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # divide by the target's own variance so no single term (e.g. a
                # large-amplitude batch item's Y) dominates the combined loss purely
                # due to scale, not due to being harder to predict
                return F.mse_loss(pred, target) / (target.var() + 1e-8)

            loss = (
                    normalized_mse(X_BA_hat, X_BA_gt)
                    + normalized_mse(Y_BA_hat, Y_BA_gt)
                    + normalized_mse(X_AB_hat, X_AB_gt)
                    + normalized_mse(Y_AB_hat, Y_AB_gt)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # =====================================================================
        # 2. Extract one fresh validation batch
        # =====================================================================
        # 2. Extract one fresh validation batch & compute predictions for BOTH directions
        model.eval()
        shared_probe.eval()

        with torch.no_grad():
            val_batch = next(data_iterator)
            X_A, Y_A = val_batch['train']['X_A'].to(device), val_batch['train']['Y_A'].to(device)
            X_B, Y_B = val_batch['train']['X_B'].to(device), val_batch['train']['Y_B'].to(device)

            Z_pred = model(X_A, Y_A, X_B, Y_B)
            # h_A = model.tokenizer(X_A, Y_A)
            # h_B = model.tokenizer(X_B, Y_B)

            # Compute predictions for both B->A and A->B using the shared probe
            X_hat_BA, Y_hat_BA = shared_probe(Z=Z_pred[0].float())
            X_hat_AB, Y_hat_AB = shared_probe(Z=Z_pred[1].float())

            Z_BA_gt = model.tokenizer(
                val_batch['train']['X_B_in_A'].to(device),
                val_batch['train']['Y_B_in_A'].to(device)
            )

        # 3. Render and save the figure
        visualizer = UnifiedManifoldVisualizer(x_range=(-5.0, 5.0))
        fig = visualizer.generate_figure(
            batch_data=val_batch,
            Z_BA_pred=Z_pred[0],
            Z_BA_gt=Z_BA_gt,
            probe_preds_BA=(X_hat_BA, Y_hat_BA),
            probe_preds_AB=(X_hat_AB, Y_hat_AB),  # <-- ADDED HERE
            batch_indices=(0, 1)
        )
        plot_dir = "./plots"
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, "aligner_post_training_batch.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Visualization successfully saved to: {save_path}")
        print("=" * 72)

        return model


    run_training_pipeline(num_iterations=1000, batch_size=32, log_interval=1000)
