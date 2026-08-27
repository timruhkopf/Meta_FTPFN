"""
Cross-Table Aligner, built on TabPFN v2.5's actual attention mechanics.

Ported from https://github.com/PriorLabs/TabPFN (tabpfn_v2_5.py), simplified:
  - no KV caching, no "thinking rows", no chunked memory-saving eval, no NaN
    imputation / standard-scaler preprocessing -- all production inference
    optimizations, irrelevant to this prototype.
  - no train/test masking inside row-attention, because test rows are NEVER
    mixed into the same tensor as context rows in this design (unlike real
    TabPFN, which packs train+test into one sequence and masks). Test rows
    only ever appear as the query side of a separate cross-attention step,
    so there is nothing for them to leak into.

Kept, faithfully:
  - feature attention: fold (Batch, Row) -> batch, attend across Columns.
  - row attention:      fold (Batch, Col) -> batch, attend across Rows.
  - post-norm residual ordering (attn -> residual -> norm).
  - zero-initialized attention/MLP output projections (blocks start as
    identity pass-through).
  - y as its OWN column, concatenated after the feature columns -- NOT an
    additive bias smeared into every cell. This is a correction relative to
    every earlier design in this line of work.

Extended with the cross-table off-diagonal block construction:
  - column-signature pooling using each column's own persistent identity
    embedding as its pooling query (not a generic shared query -- this is
    what makes the resulting signature actually column-specific).
  - cheap F_A x F_B column-correspondence cross-attention, applied via
    einsum to the SOURCE domain's real per-row cell content (not to the
    pooled signature) -- cell-level output, no O(N_A * N_B) cost anywhere
    in this step.
  - row-grounding cross-attention to anchor translated cells into the
    target domain's real row manifold.
  - asymmetric vertical test-row inference: A_test / B_test never see each
    other, never see raw off-diagonal padding, and only cross-attend
    vertically into their own domain's [real | imputed] row context.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Ported attention primitives
# --------------------------------------------------------------------------- #

def _zero_init_(module: nn.Linear) -> None:
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FeatureAttention(nn.Module):
    """Ported from AlongRowAttention: attention BETWEEN COLUMNS of a single row.

    Fold (Batch, Row) into the batch dimension; attend across the Column axis.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        _zero_init_(self.attn.out_proj)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (Batch, Row, Col, E) -> fold (Batch, Row) -> attend across Col
        Batch, Row, Col, E = h.shape
        flat = h.reshape(Batch * Row, Col, E)
        out, _ = self.attn(flat, flat, flat, need_weights=False)
        return out.reshape(Batch, Row, Col, E)


class RowAttention(nn.Module):
    """Ported from AlongColumnAttention: attention BETWEEN ROWS of a single column.

    Fold (Batch, Col) into the batch dimension; attend across the Row axis.
    No train/test masking here -- see module docstring for why.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        _zero_init_(self.attn.out_proj)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (Batch, Row, Col, E) -> transpose -> fold (Batch, Col) -> attend across Row
        Batch, Row, Col, E = h.shape
        h_t = h.transpose(1, 2)  # (Batch, Col, Row, E)
        flat = h_t.reshape(Batch * Col, Row, E)
        out, _ = self.attn(flat, flat, flat, need_weights=False)
        out = out.reshape(Batch, Col, Row, E)
        return out.transpose(1, 2)  # back to (Batch, Row, Col, E)


class TabPFNStyleBlock(nn.Module):
    """Feature attention -> row attention -> MLP, post-norm, zero-init.

    A single shared instance is applied independently to domain A and domain
    B -- the block is column-count-agnostic, so no domain-specific weights
    are needed for the intra-domain processing.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.feature_attn = FeatureAttention(d_model, n_heads, dropout)
        self.row_attn = RowAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model, bias=False),
        )
        _zero_init_(self.mlp[2])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm1(h + self.feature_attn(h))
        h = self.norm2(h + self.row_attn(h))
        h = self.norm3(h + self.mlp(h))
        return h


# --------------------------------------------------------------------------- #
# Cell encoder: y is its OWN column, appended after the feature columns.
# --------------------------------------------------------------------------- #

class CellEncoder(nn.Module):
    """
    Encodes a table's raw (X, y) into (Batch, Row, F+1, E): F feature columns
    plus ONE trailing target column. y is never broadcast into feature cells.

    Persistent per-column identity embeddings are added here (needed because
    the off-diagonal imputation logic requires stable column identity across
    every use -- unlike vanilla TabPFN, which is deliberately column-identity
    agnostic).
    """

    def __init__(self, d_model: int, max_features: int):
        super().__init__()
        self.d_model = d_model
        self.value_proj = nn.Linear(1, d_model)
        self.target_proj = nn.Linear(1, d_model)
        self.col_pos_emb = nn.Parameter(torch.randn(max_features, d_model) * 0.02)
        self.target_col_tag = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.no_label_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        X: (Batch, Row, F)
        y: (Batch, Row) or None (None => query/test row, target unknown)
        returns: (Batch, Row, F + 1, E)
        """
        Batch, Row, Feat = X.shape
        feat_cols = self.value_proj(X.unsqueeze(-1))              # (Batch, Row, F, E)
        feat_cols = feat_cols + self.col_pos_emb[:Feat][None, None, :, :]

        if y is not None:
            y_col = self.target_proj(y.unsqueeze(-1))             # (Batch, Row, E)
        else:
            y_col = self.no_label_emb.expand(Batch, Row, self.d_model)
        y_col = (y_col + self.target_col_tag).unsqueeze(2)        # (Batch, Row, 1, E)

        h = torch.cat([feat_cols, y_col], dim=2)                  # (Batch, Row, F+1, E)
        return self.norm(h)


# --------------------------------------------------------------------------- #
# Column-signature pooling: each column's OWN identity embedding is its query.
# --------------------------------------------------------------------------- #

class ColumnSignaturePool(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h_feat_only: torch.Tensor, col_pos_emb: torch.Tensor) -> torch.Tensor:
        """
        h_feat_only: (Batch, Row, F, E)  -- feature columns only, no y-column
        col_pos_emb: (F, E)              -- the SAME persistent embeddings used
                                             at encoding time, reused as queries
        returns: (Batch, F, E) -- one signature vector per real column
        """
        Batch, Row, Feat, E = h_feat_only.shape
        h_t = h_feat_only.transpose(1, 2).reshape(Batch * Feat, Row, E)   # (Batch*F, Row, E)
        q = col_pos_emb[None, :, :].expand(Batch, -1, -1).reshape(Batch * Feat, 1, E)
        pooled, _ = self.attn(q, h_t, h_t, need_weights=False)           # (Batch*F, 1, E)
        return pooled.reshape(Batch, Feat, E)


# --------------------------------------------------------------------------- #
# Off-diagonal block construction
# --------------------------------------------------------------------------- #

class CrossFeatureTranslator(nn.Module):
    """
    Learns a D_target x D_source column correspondence from column signatures,
    then applies it to the SOURCE domain's real per-row feature content
    (einsum -- cell-level, cheap, O(R_source * D_target * D_source)).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        col_sig_target: torch.Tensor,   # (Batch, D_target, E)
        col_sig_source: torch.Tensor,   # (Batch, D_source, E)
        h_source_feat_only: torch.Tensor,  # (Batch, R_source, D_source, E)
    ) -> torch.Tensor:
        _, attn_weights = self.cross_attn(
            query=col_sig_target,
            key=col_sig_source,
            value=col_sig_source,
            need_weights=True,
            average_attn_weights=True,
        )
        # attn_weights: (Batch, D_target, D_source)
        cell_translated = torch.einsum("bjk,brkm->brjm", attn_weights, h_source_feat_only)
        return cell_translated  # (Batch, R_source, D_target, E)


class RowGrounding(nn.Module):
    """Cross-attends translated cells into the target domain's REAL rows."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        _zero_init_(self.attn.out_proj)

    def forward(self, translated: torch.Tensor, h_target_feat_only: torch.Tensor) -> torch.Tensor:
        # translated:        (Batch, R_source, D_target, E)
        # h_target_feat_only:(Batch, R_target, D_target, E)
        Batch, R_source, D, E = translated.shape
        R_target = h_target_feat_only.shape[1]

        q = translated.transpose(1, 2).reshape(Batch * D, R_source, E)
        kv = h_target_feat_only.transpose(1, 2).reshape(Batch * D, R_target, E)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out.reshape(Batch, D, R_source, E).transpose(1, 2)  # (Batch, R_source, D, E)


# --------------------------------------------------------------------------- #
# Test-row vertical cross-attention (asymmetric split)
# --------------------------------------------------------------------------- #

class TestRowCrossAttention(nn.Module):
    """
    Test rows cross-attend vertically into [real | imputed] context rows of
    their OWN domain's column space only. No horizontal mixing at this stage,
    no self-attention among test rows -- matches vanilla TabPFN's own
    behavior of never letting test rows attend to each other.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        _zero_init_(self.attn.out_proj)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model, bias=False),
        )
        _zero_init_(self.mlp[2])

    def forward(self, h_test: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # h_test:  (Batch, R_test, C, E)   C = F + 1 (includes the y-column)
        # context: (Batch, R_ctx,  C, E)
        Batch, R_test, C, E = h_test.shape
        R_ctx = context.shape[1]

        q = h_test.transpose(1, 2).reshape(Batch * C, R_test, E)
        kv = context.transpose(1, 2).reshape(Batch * C, R_ctx, E)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out.reshape(Batch, C, R_test, E).transpose(1, 2)  # (Batch, R_test, C, E)

        h_test = self.norm1(h_test + out)
        h_test = self.norm2(h_test + self.mlp(h_test))
        return h_test


# --------------------------------------------------------------------------- #
# Simple Gaussian output head (stand-in for TabPFN's real bar-distribution
# decoder -- swap this out for the real thing when integrating downstream)
# --------------------------------------------------------------------------- #

class GaussianHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, y_col_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # y_col_embedding: (Batch, R, E) -- the y-COLUMN's final embedding only
        out = self.net(y_col_embedding)
        mu, log_sigma = out.unbind(dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma

    @staticmethod
    def nll(mu: torch.Tensor, sigma: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(y_true).mean()


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class CrossTableAligner(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        max_features: int = 32,
        use_source_indicators: bool = False,  # kept optional, per discussion
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_source_indicators = use_source_indicators

        self.encoder_A = CellEncoder(d_model, max_features)
        self.encoder_B = CellEncoder(d_model, max_features)

        self.shared_block = TabPFNStyleBlock(d_model, n_heads, dropout=dropout)  # shared A/B weights
        self.col_pool = ColumnSignaturePool(d_model, n_heads, dropout)
        self.translator = CrossFeatureTranslator(d_model, n_heads, dropout)
        self.grounding = RowGrounding(d_model, n_heads, dropout)

        self.n_layers = n_layers
        self.init_bias_BA = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.init_bias_AB = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)

        if use_source_indicators:
            self.e_real = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
            self.e_synth = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)

        self.test_cross_attn = TestRowCrossAttention(d_model, n_heads, dropout=dropout)
        self.head = GaussianHead(d_model)

    def _maybe_tag_source(self, h: torch.Tensor, real: bool) -> torch.Tensor:
        if not self.use_source_indicators:
            return h
        return h + (self.e_real if real else self.e_synth)

    def forward(
        self,
        X_A: torch.Tensor, Y_A: torch.Tensor,
        X_B: torch.Tensor, Y_B: torch.Tensor,
        X_A_test: torch.Tensor, X_B_test: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        F_A, F_B = X_A.shape[-1], X_B.shape[-1]

        h_A = self.encoder_A(X_A, Y_A)   # (Batch, R_A, F_A+1, E)
        h_B = self.encoder_B(X_B, Y_B)   # (Batch, R_B, F_B+1, E)

        Batch = X_A.shape[0]
        R_A, R_B = X_A.shape[1], X_B.shape[1]
        h_B_in_A = self.init_bias_BA.expand(Batch, R_B, F_A + 1, self.d_model).clone()
        h_A_in_B = self.init_bias_AB.expand(Batch, R_A, F_B + 1, self.d_model).clone()

        for _ in range(self.n_layers):
            h_A = self.shared_block(h_A)
            h_B = self.shared_block(h_B)

            h_A_feat = h_A[:, :, :F_A, :]   # exclude y-column from translation
            h_B_feat = h_B[:, :, :F_B, :]

            col_sig_A = self.col_pool(h_A_feat, self.encoder_A.col_pos_emb[:F_A])
            col_sig_B = self.col_pool(h_B_feat, self.encoder_B.col_pos_emb[:F_B])

            # B -> A
            cell_translated_BA = self.translator(col_sig_A, col_sig_B, h_B_feat)      # (Batch, R_B, F_A, E)
            grounded_BA = self.grounding(cell_translated_BA, h_A_feat)                 # (Batch, R_B, F_A, E)
            update_BA_feat = cell_translated_BA + grounded_BA
            y_col_B = h_B[:, :, F_B:F_B + 1, :]                                         # carry y through unchanged
            update_BA = torch.cat([update_BA_feat, y_col_B], dim=2)                     # (Batch, R_B, F_A+1, E)
            h_B_in_A = h_B_in_A + update_BA

            if self.training():
                # A -> B
                cell_translated_AB = self.translator(col_sig_B, col_sig_A, h_A_feat)
                grounded_AB = self.grounding(cell_translated_AB, h_B_feat)
                update_AB_feat = cell_translated_AB + grounded_AB
                y_col_A = h_A[:, :, F_A:F_A + 1, :]
                update_AB = torch.cat([update_AB_feat, y_col_A], dim=2)
                h_A_in_B = h_A_in_B + update_AB
            else:
                raise NotImplementedError('subsequent operations depend on this branch currently and need to be escaped as well')

        h_A_tagged = self._maybe_tag_source(h_A, real=True)
        h_B_in_A_tagged = self._maybe_tag_source(h_B_in_A, real=False)
        h_B_tagged = self._maybe_tag_source(h_B, real=True)
        h_A_in_B_tagged = self._maybe_tag_source(h_A_in_B, real=False)

        context_for_A = torch.cat([h_A_tagged, h_B_in_A_tagged], dim=1)   # (Batch, R_A+R_B, F_A+1, E)
        context_for_B = torch.cat([h_A_in_B_tagged, h_B_tagged], dim=1)   # (Batch, R_A+R_B, F_B+1, E)

        h_A_test = self.encoder_A(X_A_test, y=None)   # (Batch, R_test_A, F_A+1, E)
        h_B_test = self.encoder_B(X_B_test, y=None)

        h_A_test = self.shared_block.feature_attn(h_A_test)  # intra-row mixing before vertical attn
        h_B_test = self.shared_block.feature_attn(h_B_test)

        out_A = self.test_cross_attn(h_A_test, context_for_A)   # (Batch, R_test_A, F_A+1, E)
        out_B = self.test_cross_attn(h_B_test, context_for_B)

        mu_A, sigma_A = self.head(out_A[:, :, -1, :])   # last column = y-column
        mu_B, sigma_B = self.head(out_B[:, :, -1, :])

        return {
            "mu_A": mu_A, "sigma_A": sigma_A,
            "mu_B": mu_B, "sigma_B": sigma_B,
            "h_B_in_A": h_B_in_A, "h_A_in_B": h_A_in_B,
        }


# --------------------------------------------------------------------------- #
# Shape-only smoke test -- NO TRAINING
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.manual_seed(0)

    Batch, R_A, R_B, R_test_A, R_test_B, F_A, F_B, d_model = 2, 12, 20, 5, 7, 3, 5, 32

    model = CrossTableAligner(d_model=d_model, n_heads=4, n_layers=2, use_source_indicators=True)

    X_A = torch.randn(Batch, R_A, F_A)
    Y_A = torch.randn(Batch, R_A)
    X_B = torch.randn(Batch, R_B, F_B)
    Y_B = torch.randn(Batch, R_B)
    X_A_test = torch.randn(Batch, R_test_A, F_A)
    X_B_test = torch.randn(Batch, R_test_B, F_B)
    Y_A_test = torch.randn(Batch, R_test_A)
    Y_B_test = torch.randn(Batch, R_test_B)

    out = model(X_A, Y_A, X_B, Y_B, X_A_test, X_B_test)

    print("mu_A:", tuple(out["mu_A"].shape), " (expect", (Batch, R_test_A), ")")
    print("sigma_A:", tuple(out["sigma_A"].shape))
    print("mu_B:", tuple(out["mu_B"].shape), " (expect", (Batch, R_test_B), ")")
    print("sigma_B:", tuple(out["sigma_B"].shape))
    print("h_B_in_A:", tuple(out["h_B_in_A"].shape), " (expect", (Batch, R_B, F_A + 1, d_model), ")")
    print("h_A_in_B:", tuple(out["h_A_in_B"].shape), " (expect", (Batch, R_A, F_B + 1, d_model), ")")

    nll_A = GaussianHead.nll(out["mu_A"], out["sigma_A"], Y_A_test)
    nll_B = GaussianHead.nll(out["mu_B"], out["sigma_B"], Y_B_test)
    total = nll_A + nll_B
    print(f"NLL_A={nll_A.item():.4f}  NLL_B={nll_B.item():.4f}  total={total.item():.4f}")

    total.backward()
    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(f"total params: {n_params:,}  |  params that received a gradient: {n_grad:,}")
    print("Backward pass completed successfully.")


    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from ppfn.prior.harmonics.harmonic_mixture_prior import HarmonicMixturePrior
    from ppfn.prior.harmonics.stream_dataset import InfiniteHarmonicsStream

    # =====================================================================
    # Example Training Epoch Orchestrator
    # =====================================================================
    def run_training_epoch(model, optimizer, scaler, dataloader, device, steps=1000):
        total_loss = 0.0
        model.train()

        for step in tqdm(range(steps)):
            batch = next(dataloader)
            loss_val, _ = train_dual_perceiver_pfn_step(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                batch=batch,
                device=device,
                lambda_impute=0.5,
                dtype=torch.float16  # Use torch.bfloat16 if training on Ampere/Hopper GPUs
            )
            total_loss += loss_val

        return total_loss / len(dataloader)


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
    batch_size = 32
    lr = 0.003
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
    model = DualPerceiverImputationPFN(num_features=1, d_model=128, num_bars=100).to(device)
    scaler = GradScaler(enabled=True)  # Enable AMP gradient scaling for mixed precision
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    run_training_epoch(
        model=model, optimizer=optimizer, scaler=scaler, dataloader=data_iterator, device=device)