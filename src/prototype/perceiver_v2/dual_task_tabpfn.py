"""
Dual-task TabPFN: Perceiver-style feature-space alignment between a
"related" task A and a "target" task B that are distorted versions of
each other (affine shifts/scales, monotonic warps, column permutations).

This is a from-scratch reimplementation, architecturally aligned with
PriorLabs/TabPFN's tabpfn_v2_6.py (row/column axial attention + MLP,
post-norm, shared column encoder, NaN-indicator target masking), NOT a
patch against the real file. It's a faithful-enough scaffold to prototype
the alignment idea against; swapping in the real TabPFNBlock later is a
drop-in change because the tensor conventions match.

Shapes follow TabPFN's own convention: B=batch, R=rows (items),
C=columns (feature groups, +1 for the appended target column), E=embed dim.

Author's note on what changed vs. the first draft, and why:
  1. Shared column encoder for task A and task B (was: two independent
     nn.Linear encoders). The two tasks are the *same* underlying process
     under different coordinates -- giving them separate learned encoders
     lets the model cheat by absorbing part of the distortion into
     encoder weights instead of learning to align in the latent space,
     which defeats the point of the exercise.
  2. Real row/column axial attention with a train/test causal mask on
     the column (item) axis, instead of a single MultiheadAttention over
     mean-pooled vectors. Mean pooling over samples throws away the
     per-column distribution before the aligner ever sees it.
  3. Attention-pooling (Set-Transformer/PMA style, K pooling queries per
     column) replaces plain mean-pooling for the tokens fed into the
     Perceiver latents. This keeps compute bounded while giving the
     latents something closer to a mixture-of-moments summary of each
     column's distribution, rather than a single point estimate --
     closer in spirit to the Wasserstein-distance idea from the original
     discussion, without the O(rows_A * rows_B) cost of pairwise OT.
  4. Alignment is applied every K backbone layers, not once at the very
     end, so row/column attention and cross-task alignment can refine
     each other iteratively across depth.
  5. Target y is masked with a NaN-indicator channel (as in the real
     encoder) rather than zero-filled, since zero is a valid value.
  6. Prior distortions (affine / log-warp / tanh-warp / permutation) are
     sampled independently per batch item, not shared across the batch,
     so the aligner can't shortcut on a fixed transform.

Revision of the draft:

Honestly? It's a reasonable scaffold, but I'd call it "plausible," not "well-motivated" yet. The biggest issues aren't bugs — they're in what the design *lets the model get away with*. In priority order:

**1. There's no negative control in the prior — this is the most important gap.**
Every single training example has Task A as a genuine distorted view of Task B. The model never sees a pair where A is unrelated. So it can never learn the "graceful degradation" behavior the original idea was built around — there's no gradient signal telling it *when to ignore A*, because ignoring A is never the right answer during training. Right now you'd likely get a model that always trusts A, confidently, even on garbage. Fix: sample a `related` flag per batch item; when false, generate A from an independent harmonic (different `A, W, P` sampled fresh, no distortion of B's function at all). That single change turns this from "feature alignment network" into something that actually exercises the uncertainty story.

**2. The prior may not force reliance on Task A at all.**
Harmonics with 3 components and 40 context points is an easy regression problem for a competent single-task PFN. If Task B alone is already solvable, gradients into the aligner path are weak — nothing pushes the model to use it. You want a curriculum where Task B's context is deliberately scarce/ambiguous (e.g. n_context small relative to num_harmonics, or context points clustered so parts of the function are unobserved) so that Task A is sometimes *necessary*, not just nice-to-have. Otherwise you risk training a model where the whole aligner is dead weight and you won't be able to tell from the loss curve.

**3. The alignment is asked to discover transform-inversion from nothing but attention.**
Attention-pooling is better than mean-pooling, but it still gives the latents zero inductive bias about *what kind of thing* they're supposed to be recovering (a shift, a scale, a monotonic warp). A hybrid design would likely be far more sample-efficient: have a small head predict explicit summary statistics (per-column mean/std/quantiles) for both tasks, feed the *discrepancy* between those statistics into the latents as an additional signal alongside the learned tokens, and optionally have a side-head regress explicit affine parameters as an auxiliary loss when the distortion is affine. Let attention handle the residual, nonlinear part; let arithmetic handle the part that's just arithmetic. This is the "relative coordinate encoding" idea from the very first message, and I under-weighted it in favor of the flashier Perceiver-only design.

**4. Position embeddings might leak identity into pooling.**
I added per-task column positional embeddings before the backbone runs, which is necessary for row-attention to break symmetry — but by the time `AttentionPool` reads the columns for the aligner, that positional signal has been mixed in through several attention layers. If the model finds it easier to exploit residual positional structure (e.g. "column 3 in A tends to correlate with column 1 in B" as a dataset-level regularity, if your prior isn't careful) than to actually solve the general alignment problem, you'd get something that looks like it's aligning but is partly cheating on the *specific* prior's statistics. Worth an ablation: shuffle column order of A and B independently right before pooling and confirm performance doesn't degrade.

**5. No diagnostic to check if A is actually being used.**
Right now you'd have to trust the loss curve. Cheap fix, high value: log a paired eval every N steps — same batch, once with Task A intact, once with Task A's features replaced by pure noise (same shape) — and compare NLL. If the gap is ~0, the aligner isn't contributing and no amount of further tuning matters until #1/#2 are fixed.

**6. Scalability is a real ceiling, not just a toy-code shortcut.**
Treating every feature as its own column means row-attention cost grows with feature count, and TabPFN's real encoder groups several raw features into one "column" for exactly this reason (`features_per_group`). Fine for a 4-feature prototype; you'll want feature grouping before trying anything with realistic column counts.

If I were sequencing this: fix #1 and #2 first (they determine whether there's anything to learn), add #5 immediately so you can tell whether the network is using Task A at all, and only then invest in #3's more structured aligner — no point making the alignment mechanism smarter if the training signal for using it doesn't exist yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Small building blocks
# --------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Matches TabPFN v2.6's move from LayerNorm to RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class ColumnEncoder(nn.Module):
    """Shared per-column encoder: [value, nan_indicator] -> embed_dim.

    Applied identically to every feature column AND the target column, and
    (crucially) shared between task A and task B -- see note (1) above.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, values: torch.Tensor, nan_mask: torch.Tensor) -> torch.Tensor:
        # values, nan_mask: (B, R, C) -> (B, R, C, embed_dim)
        indicator = nan_mask.float() * 2.0 - 1.0  # {-1, +1} like TabPFN's NAN_INDICATOR scheme
        safe_values = torch.where(nan_mask, torch.zeros_like(values), values)
        stacked = torch.stack([safe_values, indicator], dim=-1)
        return self.proj(stacked)


class ColumnPositionalEmbedding(nn.Module):
    """Fixed-but-learned per-column embedding, to break the permutation
    symmetry of AlongRowAttention *within* a task.

    Deliberately NOT shared between task A and task B: if both tasks used
    the same column-identity embedding, the aligner could match columns by
    identity instead of by content, which is exactly the shortcut we need
    to avoid since columns may be permuted between A and B.
    """

    def __init__(self, max_columns: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(max_columns, embed_dim) * 0.02)

    def forward(self, num_columns: int) -> torch.Tensor:
        return self.embed[:num_columns]


class AlongRowAttention(nn.Module):
    """Features of a single row attend to each other. No masking needed."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x_BRCE: torch.Tensor) -> torch.Tensor:
        B, R, C, E = x_BRCE.shape
        x_flat = x_BRCE.reshape(B * R, C, E)
        out, _ = self.mha(x_flat, x_flat, x_flat)
        return out.reshape(B, R, C, E)


class AlongColumnAttention(nn.Module):
    """Cells of a single column attend across rows, with a causal
    train/test split: train rows attend to train rows only, test rows
    attend to train rows only (never to themselves or each other).
    Mirrors TabPFN's AlongColumnAttention masking, minus the KV-cache
    optimization (not needed for a training-time prototype).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x_BRCE: torch.Tensor, num_train_rows: int | None) -> torch.Tensor:
        B, R, C, E = x_BRCE.shape
        x_BCRE = x_BRCE.transpose(1, 2).reshape(B * C, R, E)

        if num_train_rows is None or num_train_rows == R:
            out, _ = self.mha(x_BCRE, x_BCRE, x_BCRE)
        else:
            train = x_BCRE[:, :num_train_rows]
            test = x_BCRE[:, num_train_rows:]
            train_out, _ = self.mha(train, train, train)
            test_out, _ = self.mha(test, train, train)
            out = torch.cat([train_out, test_out], dim=1)

        return out.reshape(B, C, R, E).transpose(1, 2)


class TabPFNAxialBlock(nn.Module):
    """One row-attention + one column-attention + one MLP layer, post-norm."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.row_attn = AlongRowAttention(embed_dim, num_heads)
        self.col_attn = AlongColumnAttention(embed_dim, num_heads)
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.norm3 = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim, bias=False),
        )
        nn.init.zeros_(self.mlp[-1].weight)  # start as identity, as in TabPFN

    def forward(self, x_BRCE: torch.Tensor, num_train_rows: int | None) -> torch.Tensor:
        x_BRCE = self.norm1(x_BRCE + self.row_attn(x_BRCE))
        x_BRCE = self.norm2(x_BRCE + self.col_attn(x_BRCE, num_train_rows))
        x_BRCE = self.norm3(x_BRCE + self.mlp(x_BRCE))
        return x_BRCE


# --------------------------------------------------------------------------
# Attention-pooling (replaces naive mean-pooling for the aligner's input)
# --------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """K learned pooling queries attend over the row axis of each column,
    producing K summary vectors per column instead of one mean vector.
    Set-Transformer / Perceiver "PMA" pattern.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_pool_queries: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_pool_queries, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.num_pool_queries = num_pool_queries

    def forward(self, x_BRCE: torch.Tensor) -> torch.Tensor:
        # (B, R, C, E) -> per-column pooled summaries (B, C, K, E)
        B, R, C, E = x_BRCE.shape
        x_BCRE = x_BRCE.transpose(1, 2).reshape(B * C, R, E)
        q = self.queries.expand(B * C, -1, -1)
        pooled, _ = self.mha(q, x_BCRE, x_BCRE)  # (B*C, K, E)
        return pooled.reshape(B, C, self.num_pool_queries, E)


# --------------------------------------------------------------------------
# Perceiver alignment bottleneck
# --------------------------------------------------------------------------

class PerceiverFeatureAligner(nn.Module):
    """Latents read the target task's column summaries, then the related
    task's column summaries, reason globally about the implied distortion,
    then write aligned information back into the target task's row/column
    grid. Operates on attention-pooled column tokens (see AttentionPool),
    not on a single mean-pooled vector per column, so it has access to
    distributional shape rather than just a first moment.
    """

    def __init__(self, embed_dim: int, num_latents: int, num_heads: int, num_pool_queries: int = 4):
        super().__init__()
        self.pool = AttentionPool(embed_dim, num_heads, num_pool_queries)
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        self.latent_reads_B = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.latent_reads_A = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.latent_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.write_back = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.n1 = RMSNorm(embed_dim)
        self.n2 = RMSNorm(embed_dim)
        self.n3 = RMSNorm(embed_dim)
        self.n4 = RMSNorm(embed_dim)

    def forward(self, x_A_BRCE: torch.Tensor, x_B_BRCE: torch.Tensor) -> torch.Tensor:
        B = x_A_BRCE.shape[0]

        tokens_A = self.pool(x_A_BRCE).flatten(1, 2)  # (B, C_A*K, E)
        tokens_B = self.pool(x_B_BRCE).flatten(1, 2)  # (B, C_B*K, E)

        L = self.latents.expand(B, -1, -1)

        out, _ = self.latent_reads_B(L, tokens_B, tokens_B)
        L = self.n1(L + out)

        out, _ = self.latent_reads_A(L, tokens_A, tokens_A)
        L = self.n2(L + out)

        out, _ = self.latent_self_attn(L, L, L)
        L = self.n3(L + out)

        # Write aligned signal back into every (row, column) cell of task B.
        Bb, R, C, E = x_B_BRCE.shape
        flat_B = x_B_BRCE.reshape(Bb, R * C, E)
        out, _ = self.write_back(flat_B, L, L)
        x_B_aligned = self.n4(flat_B + out).reshape(Bb, R, C, E)
        return x_B_aligned


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------

@dataclass
class DualTaskConfig:
    embed_dim: int = 128
    num_heads: int = 4
    ff_mult: int = 2
    num_layers: int = 8
    align_every: int = 2      # inject the aligner every K backbone layers
    num_latents: int = 16
    num_pool_queries: int = 4
    max_columns: int = 64


class DualTaskTabPFN(nn.Module):
    """
    Task A ("related"): fully observed context, no train/test split of its
    own -- it only ever plays the role of side information.
    Task B ("target"): context rows (X, Y known) + target rows (X known,
    Y masked). The model predicts a Gaussian (mean, log-variance) for each
    target row's Y, informed by the aligned signal from Task A.
    """

    def __init__(self, cfg: DualTaskConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.embed_dim

        self.column_encoder = ColumnEncoder(E)  # shared across A and B
        self.pos_embed_A = ColumnPositionalEmbedding(cfg.max_columns, E)
        self.pos_embed_B = ColumnPositionalEmbedding(cfg.max_columns, E)

        self.blocks_A = nn.ModuleList(
            TabPFNAxialBlock(E, cfg.num_heads, E * cfg.ff_mult) for _ in range(cfg.num_layers)
        )
        self.blocks_B = nn.ModuleList(
            TabPFNAxialBlock(E, cfg.num_heads, E * cfg.ff_mult) for _ in range(cfg.num_layers)
        )
        self.aligners = nn.ModuleList(
            PerceiverFeatureAligner(E, cfg.num_latents, cfg.num_heads, cfg.num_pool_queries)
            for _ in range(cfg.num_layers // cfg.align_every)
        )

        self.output_head = nn.Sequential(
            nn.Linear(E, E * 2), nn.GELU(), nn.Linear(E * 2, 2)  # mean, log_var
        )

    @staticmethod
    def _build_grid(X: torch.Tensor, Y: torch.Tensor, Y_mask: torch.Tensor, encoder: ColumnEncoder) -> torch.Tensor:
        """X: (B, R, F), Y: (B, R, 1), Y_mask: (B, R, 1) True where Y is unknown.
        Returns (B, R, F+1, E) with the target appended as the last column."""
        B, R, Fdim = X.shape
        x_nan = torch.zeros_like(X, dtype=torch.bool)
        feat_tokens = encoder(X, x_nan)                      # (B, R, F, E)
        y_tokens = encoder(Y, Y_mask)                         # (B, R, 1, E)
        return torch.cat([feat_tokens, y_tokens], dim=2)      # (B, R, F+1, E)

    def forward(
        self,
        X_A: torch.Tensor, Y_A: torch.Tensor,                 # related task, fully observed
        X_B_context: torch.Tensor, Y_B_context: torch.Tensor,  # target task context
        X_B_target: torch.Tensor,                               # target task rows to predict
    ):
        B, n_ctx, _ = X_B_context.shape
        n_tgt = X_B_target.shape[1]

        X_B = torch.cat([X_B_context, X_B_target], dim=1)
        Y_B = torch.cat([Y_B_context, torch.zeros(B, n_tgt, 1, device=X_B.device)], dim=1)
        Y_B_mask = torch.cat(
            [torch.zeros(B, n_ctx, 1, dtype=torch.bool, device=X_B.device),
             torch.ones(B, n_tgt, 1, dtype=torch.bool, device=X_B.device)],
            dim=1,
        )

        grid_A = self._build_grid(X_A, Y_A, torch.zeros_like(Y_A, dtype=torch.bool), self.column_encoder)
        grid_B = self._build_grid(X_B, Y_B, Y_B_mask, self.column_encoder)

        grid_A = grid_A + self.pos_embed_A(grid_A.shape[2])[None, None]
        grid_B = grid_B + self.pos_embed_B(grid_B.shape[2])[None, None]

        align_idx = 0
        for i, (block_a, block_b) in enumerate(zip(self.blocks_A, self.blocks_B)):
            grid_A = block_a(grid_A, num_train_rows=None)          # task A has no train/test split
            grid_B = block_b(grid_B, num_train_rows=n_ctx)         # task B: causal train/test mask

            if (i + 1) % self.cfg.align_every == 0 and align_idx < len(self.aligners):
                grid_B = self.aligners[align_idx](grid_A, grid_B)
                align_idx += 1

        target_y_embeddings = grid_B[:, n_ctx:, -1, :]  # (B, n_tgt, E) -- the target column, target rows
        mean_logvar = self.output_head(target_y_embeddings)
        mean = mean_logvar[..., 0]
        log_var = mean_logvar[..., 1].clamp(-10.0, 10.0)  # stability
        return mean, log_var


# --------------------------------------------------------------------------
# Random-harmonics prior with per-item sampled distortions
# --------------------------------------------------------------------------

def sample_harmonics(batch_size, num_features, num_harmonics, device):
    A = torch.randn(batch_size, num_features, num_harmonics, device=device)
    W = torch.randn(batch_size, num_features, num_harmonics, device=device) * 2.0
    P = torch.rand(batch_size, num_features, num_harmonics, device=device) * 2 * math.pi
    return A, W, P


def eval_harmonics(x, A, W, P):
    # x: (b, s, f) -> (b, s, f, 1); W,P,A: (b, f, h) -> (b, 1, f, h)
    xb = x.unsqueeze(-1)
    Wb, Pb, Ab = W.unsqueeze(1), P.unsqueeze(1), A.unsqueeze(1)
    terms = Ab * torch.sin(Wb * xb + Pb)          # (b, s, f, h)
    return terms.sum(dim=(-1, -2), keepdim=False).unsqueeze(-1)  # (b, s, 1)


def apply_distortion(x: torch.Tensor, kind: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """kind: (batch,) int in {0: affine, 1: log-warp, 2: tanh-warp}, applied per batch item."""
    affine = x * scale + shift
    log_warp = torch.sign(x) * torch.log1p(x.abs() * scale.abs()) + shift
    tanh_warp = torch.tanh(x * scale) * 3.0 + shift

    kind = kind.view(-1, 1, 1)
    out = torch.where(kind == 0, affine, torch.where(kind == 1, log_warp, tanh_warp))
    return out


def generate_distorted_harmonics_prior(
    batch_size: int, num_context: int, num_target: int, num_features: int,
    num_related_samples: int | None = None, device: str = "cpu",
):
    num_related_samples = num_related_samples or (num_context * 2)
    num_harmonics = 3

    A, W, P = sample_harmonics(batch_size, num_features, num_harmonics, device)

    # Task B: original coordinates
    X_B_context = torch.randn(batch_size, num_context, num_features, device=device)
    X_B_target = torch.randn(batch_size, num_target, num_features, device=device)
    Y_B_context = eval_harmonics(X_B_context, A, W, P)
    Y_B_target = eval_harmonics(X_B_target, A, W, P)

    # Task A: same underlying function, evaluated on base coordinates, but the
    # FEATURES shown to the model are a distorted view of those coordinates --
    # this is what the aligner has to invert.
    X_A_base = torch.randn(batch_size, num_related_samples, num_features, device=device)
    Y_A = eval_harmonics(X_A_base, A, W, P)

    distortion_kind = torch.randint(0, 3, (batch_size,), device=device)
    scale = torch.rand(batch_size, 1, num_features, device=device) * 2.5 + 0.2
    shift = torch.randn(batch_size, 1, num_features, device=device) * 3.0
    X_A_distorted = apply_distortion(X_A_base, distortion_kind, scale, shift)

    # Per-batch-item column permutation (fixed bug: was shared across the batch).
    perms = torch.stack([torch.randperm(num_features, device=device) for _ in range(batch_size)])
    X_A_distorted = torch.gather(
        X_A_distorted, dim=2, index=perms.unsqueeze(1).expand(-1, num_related_samples, -1)
    )

    return (X_A_distorted, Y_A), (X_B_context, Y_B_context), (X_B_target, Y_B_target)


# --------------------------------------------------------------------------
# Training pipeline
# --------------------------------------------------------------------------

def train_dual_task_pfn(
    epochs: int = 1000,
    batch_size: int = 32,
    num_features: int = 5,
    num_context: int = 40,
    num_target: int = 20,
    lr: float = 3e-4,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DualTaskConfig(embed_dim=128, num_layers=8, align_every=2, num_latents=16)
    model = DualTaskTabPFN(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        (X_A, Y_A), (X_B_c, Y_B_c), (X_B_t, Y_B_t) = generate_distorted_harmonics_prior(
            batch_size, num_context, num_target, num_features, device=device
        )

        # FIXME: why return mean and log_var here?
        mean, log_var = model(X_A, Y_A, X_B_c, Y_B_c, X_B_t)

        target = Y_B_t.squeeze(-1)
        var = log_var.exp()
        nll = 0.5 * log_var + 0.5 * (target - mean) ** 2 / var
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                rmse = (target - mean).pow(2).mean().sqrt().item()
            print(f"epoch {epoch:4d} | NLL {loss.item():.4f} | RMSE {rmse:.4f} | "
                  f"mean std {log_var.exp().sqrt().mean().item():.4f}")

    return model


if __name__ == "__main__":
    train_dual_task_pfn(epochs=10000, batch_size=32, num_features=4, num_context=32, num_target=8)
