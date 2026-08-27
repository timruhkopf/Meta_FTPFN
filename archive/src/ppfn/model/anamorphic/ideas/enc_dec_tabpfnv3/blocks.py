"""Aligner blocks: B-stream transformation conditioned on A_train.

Everything attention-shaped is imported from TabPFN. The only thing this module
adds is the wiring: which stream is the query, which is the key/value, and in
what order.

Verified against tabpfn==8.3.0.
"""

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import override

# --- v2.5 primitives (post-norm convention, out_projection zero-init) ---------
from tabpfn.architectures.tabpfn_v2_5 import (
    AlongColumnAttention,  # cells of one column attend across rows
    AlongRowAttention,  # feature-group tokens of one row attend to each other
    LowerPrecisionLayerNorm,
)

# --- v3 primitive: bare multi-head cross-attention, out_projection zero-init --
# This is the module underneath v3's ColumnAggregator. We take the attention
# only (not CrossAttentionBlock, which is pre-norm) so we can keep the v2.5
# post-norm convention throughout and stay warm-start compatible.
from tabpfn.architectures.tabpfn_v3 import CrossAttention


class AlignerBlock(nn.Module):
    """One aligner block. Queries are B cells; keys/values come from A_train.

    Order inside the block (this ordering is the design claim -- see README):

        1. self-attention over B's features            [AlongRowAttention]
        2. cross-attention: B columns -> A column summary   [CrossAttention]
        3. self-attention over B's rows                [AlongColumnAttention]
        4. cross-attention: B rows -> A_train rows, per column  [CrossAttention]
        5. MLP

    Feature-cross precedes row-cross so that row-level dot products are computed
    after the column spaces have been recalibrated, rather than under the
    distorted metric.

    Both cross branches inherit ``zeros_(out_projection.weight)`` from TabPFN, so
    a freshly constructed AlignerBlock is an exact identity on the B stream apart
    from its own self-attention -- you can warm-start the self-attention halves
    from a pretrained v2.5 checkpoint and start training from stock behaviour.
    """

    def __init__(
        self,
        *,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        use_cross_feat: bool = True,
        use_cross_row: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        assert emsize % nhead == 0
        head_dim = emsize // nhead
        kw = {"device": device, "dtype": dtype}
        attn_kw = {
            "embedding_size": emsize,
            "num_heads": nhead,
            "head_dim": head_dim,
            **kw,
        }

        self.self_feat = AlongRowAttention(**attn_kw)
        self.cross_feat = CrossAttention(**attn_kw) if use_cross_feat else None
        self.self_row = AlongColumnAttention(**attn_kw)
        self.cross_row = CrossAttention(**attn_kw) if use_cross_row else None

        self.mlp = nn.Sequential(
            nn.Linear(emsize, dim_feedforward, bias=False, **kw),
            nn.GELU(),
            nn.Linear(dim_feedforward, emsize, bias=False, **kw),
        )
        torch.nn.init.zeros_(self.mlp[2].weight)  # matches TabPFNBlock

        ln_kw = {**kw, "elementwise_affine": False}
        self.ln_self_feat = LowerPrecisionLayerNorm(emsize, **ln_kw)
        self.ln_cross_feat = LowerPrecisionLayerNorm(emsize, **ln_kw)
        self.ln_self_row = LowerPrecisionLayerNorm(emsize, **ln_kw)
        self.ln_cross_row = LowerPrecisionLayerNorm(emsize, **ln_kw)
        self.ln_mlp = LowerPrecisionLayerNorm(emsize, **ln_kw)

    @override
    def forward(
        self,
        b_BRCE: torch.Tensor,
        a_train_BNCE: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the B stream one step towards A's domain.

        Args:
            b_BRCE: B cells, shape (batch, R_B rows, C columns, emsize).
            a_train_BNCE: contextualised A_train cells, shape (batch, N, C, E).
                Same C as B (schemas are aligned and we enforce a shared
                constant-column mask upstream).

        Returns:
            The transformed B stream, same shape as ``b_BRCE``.
        """
        B, R, C, E = b_BRCE.shape
        N = a_train_BNCE.shape[1]
        assert a_train_BNCE.shape[2] == C, "A and B must share the column axis"

        # -- 1. B feature self-attention. Rows fold into the batch.
        f_BrCE = b_BRCE.reshape(B * R, C, E)
        f_BrCE = self.ln_self_feat(f_BrCE + self.self_feat(f_BrCE))

        # -- 2. Cross-feature: each B column token reads A's summary of the
        # corresponding (and every other) column. This is the column-space
        # recalibration; it is the part that survives a sparse A_train, because
        # pooling over N rows is far more stable than matching against them.
        if self.cross_feat is not None:
            col_summary_BCE = a_train_BNCE.mean(dim=1)
            kv_BrCE = col_summary_BCE[:, None].expand(B, R, C, E).reshape(B * R, C, E)
            f_BrCE = self.ln_cross_feat(f_BrCE + self.cross_feat(f_BrCE, kv_BrCE))

        b_BRCE = f_BrCE.view(B, R, C, E)

        # -- 3. B row self-attention, per column. single_eval_pos == R selects
        # AlongColumnAttention's unmasked branch (all of B is "training").
        r_BcRE = b_BRCE.transpose(1, 2).contiguous().reshape(B * C, R, E)
        attn_out, _ = self.self_row(r_BcRE, single_eval_pos=R)
        r_BcRE = self.ln_self_row(r_BcRE + attn_out)

        # -- 4. Cross-row: B rows retrieve from A_train rows within the same
        # column. No coupling constraint, no Sinkhorn: each B query has its own
        # softmax over A_train, so |B| != |A_train| is a non-issue.
        if self.cross_row is not None:
            akv_BcNE = a_train_BNCE.transpose(1, 2).contiguous().reshape(B * C, N, E)
            r_BcRE = self.ln_cross_row(r_BcRE + self.cross_row(r_BcRE, akv_BcNE))

        b_BRCE = r_BcRE.view(B, C, R, E).transpose(1, 2).contiguous()

        # -- 5. MLP
        return self.ln_mlp(b_BRCE + self.mlp(b_BRCE))
