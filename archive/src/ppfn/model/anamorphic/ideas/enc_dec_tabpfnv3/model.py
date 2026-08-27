"""WarpAlignPFN: TabPFN v2.5 with a distortion-aligned auxiliary table.

Stacks:
    S0  shared cell embedding (stock TabPFN stem, separate scaler fits per table)
    S1  A prologue        -- stock TabPFNBlocks, train/test mask
    S1' B encoder         -- stock TabPFNBlocks, unmasked
    S2  aligner           -- AlignerBlock, B queries / A_train keys  -> B_hat
    S3  decoder           -- stock TabPFNBlocks over [think, A_train, B_hat, A_test]
    head                  -- FullSupportBarDistribution logits on A_test's y cell

The load-bearing choice is in S3: B_hat rows are *concatenated into the row axis*
rather than exposed through a separate cross-attention branch. A_test then reads
[think, A_train, B_hat] under one shared softmax, so A_train and B_hat compete
for the same probability mass and unrelated B is starved automatically. A
separate branch would normalise over B alone and would have to spend its mass
there regardless.

Verified against tabpfn==8.3.0.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution
from tabpfn.architectures.tabpfn_v2_5 import (
    ENCODING_SIZE_MULTIPLIER,
    TabPFNBlock,
    TabPFNV2p5,
    TabPFNV2p5Config,
)

from .blocks import AlignerBlock


def make_borders(n_bins: int = 256, lo: float = -4.0, hi: float = 4.0) -> torch.Tensor:
    """Fixed bar-distribution borders on standardised-y space.

    Deliberately *not* derived from A_train quantiles. A_train is sparse, so
    quantile borders would be unstable across meta-tasks, and unstable borders
    make the teacher KL meaningless (student and teacher must share bins).
    Standardise y with A_train's mean/std upstream and the fixed grid is fine.
    """
    return torch.linspace(lo, hi, n_bins + 1)


@dataclasses.dataclass
class WarpAlignConfig:
    emsize: int = 192
    nhead: int = 3
    features_per_group: int = 3
    num_thinking_rows: int = 64
    n_bins: int = 256

    n_prologue_A: int = 4
    n_encoder_B: int = 6
    n_aligner: int = 5
    n_decoder: int = 14

    # First aligner block is feature-cross only: recalibrate column spaces
    # before any row matching happens.
    aligner_row_cross_from: int = 1

    @property
    def dim_feedforward(self) -> int:
        return self.emsize * 2  # TabPFNV2p5 uses hidden_size = emsize * 2


@dataclasses.dataclass
class AuxOutputs:
    """Everything the loss module and the diagnostics need."""

    b_hat_y_logits_RBK: torch.Tensor  # B_hat's y cell -> A-domain bar logits
    b_hat_rows_BRE: torch.Tensor  # column-pooled B_hat rows, for anti-collapse
    severity_logit_B1: torch.Tensor  # relatedness head output


def _shared_column_mask(x_A_RBC: torch.Tensor, x_B_RBC: torch.Tensor) -> torch.Tensor:
    """Columns that are non-constant in *both* tables, for every batch element.

    TabPFN drops constant columns per table inside
    ``_preprocess_and_embed_features``. Letting it do that independently for A
    and B would silently break the schema alignment that the whole design rests
    on, so we intersect the masks and pre-select here; the internal removal then
    becomes a no-op.
    """

    def nonconstant(x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.ones(x.shape[1:], dtype=torch.bool, device=x.device)
        x = torch.nan_to_num(x)
        return ~(x[1:] == x[0]).all(0)  # (B, C)

    keep_BC = nonconstant(x_A_RBC) & nonconstant(x_B_RBC)
    return keep_BC.all(dim=0)  # (C,) -- conservative across the batch


class WarpAlignPFN(nn.Module):
    def __init__(
        self,
        config: WarpAlignConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        cfg = config or WarpAlignConfig()
        self.cfg = cfg
        kw = {"device": device, "dtype": dtype}

        # Stem: a stock TabPFNV2p5 with zero blocks. We use it purely for its
        # embedding machinery -- feature grouping, NaN/inf indicators, mean
        # imputation, TorchStandardScaler, feature-group normalisation, the
        # deterministic column embeddings, the target embedder (which carries
        # the NaN token in its indicator weight column), and AddThinkingRows.
        # Reusing it means A and B get *identical* column embeddings, which is
        # the free schema anchor: column j of B lands in the same positional
        # subspace as column j of A.
        self.stem = TabPFNV2p5(
            config=TabPFNV2p5Config(
                emsize=cfg.emsize,
                nlayers=0,
                nhead=cfg.nhead,
                features_per_group=cfg.features_per_group,
                num_thinking_rows=cfg.num_thinking_rows,
            ),
            task_type="regression",
            n_out=cfg.n_bins,
            **kw,
        )

        block_kw = {
            "emsize": cfg.emsize,
            "nhead": cfg.nhead,
            "dim_feedforward": cfg.dim_feedforward,
            **kw,
        }
        self.prologue_A = nn.ModuleList(
            TabPFNBlock(**block_kw) for _ in range(cfg.n_prologue_A)
        )
        self.encoder_B = nn.ModuleList(
            TabPFNBlock(**block_kw) for _ in range(cfg.n_encoder_B)
        )
        self.aligner = nn.ModuleList(
            AlignerBlock(
                use_cross_feat=True,
                use_cross_row=(i >= cfg.aligner_row_cross_from),
                **block_kw,
            )
            for i in range(cfg.n_aligner)
        )
        self.decoder = nn.ModuleList(
            TabPFNBlock(**block_kw) for _ in range(cfg.n_decoder)
        )

        # B gets its own target embedder: B's y lives in a different scale and
        # a different semantic. Pushing it through A's target_embedder is the
        # false-similarity problem at the label level.
        self.b_target_embedder = nn.Linear(ENCODING_SIZE_MULTIPLIER, cfg.emsize, **kw)

        # Domain embeddings (the multilingual-MT language token).
        self.domain_A = nn.Parameter(torch.zeros(cfg.emsize))
        self.domain_B = nn.Parameter(torch.zeros(cfg.emsize))
        nn.init.normal_(self.domain_A, std=0.02)
        nn.init.normal_(self.domain_B, std=0.02)

        # Relatedness head: predicts distortion severity from (A_train, B_hat).
        # Supervised directly, since we know s. Its output modulates B_hat via a
        # learned direction, giving the shared softmax an explicit, inspectable
        # global "how much should I trust B at all" knob.
        self.relatedness_head = nn.Sequential(
            nn.Linear(2 * cfg.emsize, cfg.emsize, **kw),
            nn.GELU(),
            nn.Linear(cfg.emsize, 1, **kw),
        )
        self.severity_token = nn.Parameter(torch.zeros(cfg.emsize))

        # Heads. Mirrors TabPFNV2p5.output_projection.
        def head() -> nn.Module:
            return nn.Sequential(
                nn.Linear(cfg.emsize, cfg.dim_feedforward, **kw),
                nn.GELU(),
                nn.Linear(cfg.dim_feedforward, cfg.n_bins, **kw),
            )

        self.output_projection = head()
        # y-only translation head: B_hat's y cell -> B_inA's y, in A's bins.
        # Far cheaper than full-row reconstruction and it targets exactly the
        # value vectors that carry the function's shape into A_test.
        self.translation_head = head()

        self.criterion = FullSupportBarDistribution(make_borders(cfg.n_bins))

    # ------------------------------------------------------------------ embed

    def _embed_A(
        self, x_A_RBC: torch.Tensor, y_A_NB: torch.Tensor, n_A_train: int
    ) -> torch.Tensor:
        R_A, batch_size, _ = x_A_RBC.shape
        emb_x_BRGE, _, _ = self.stem._preprocess_and_embed_features(
            x_RiBC=x_A_RBC,
            num_train_labels=n_A_train,  # scaler fits on A_train only
            batch_size=batch_size,
        )
        emb_y_BRE = self.stem._preprocess_and_embed_targets(
            y=y_A_NB,
            num_train_rows=R_A,  # NaN-pads y over the test rows
            num_train_labels=n_A_train,
            batch_size=batch_size,
        )
        a = torch.cat([emb_x_BRGE, emb_y_BRE[:, :, None]], dim=2)
        return a + self.domain_A

    def _embed_B(self, x_B_RBC: torch.Tensor, y_B_RB: torch.Tensor) -> torch.Tensor:
        R_B, batch_size, _ = x_B_RBC.shape
        # num_train_labels=R_B -> the scaler fits on all of B. This separate fit
        # is deliberate: it removes the affine part of the warp before the
        # network sees it, leaving the aligner only the residual nonlinearity.
        emb_x_BRGE, _, _ = self.stem._preprocess_and_embed_features(
            x_RiBC=x_B_RBC,
            num_train_labels=R_B,
            batch_size=batch_size,
        )
        y = y_B_RB[..., None]  # (R_B, B, 1)
        mean = y.mean(dim=0, keepdim=True)
        std = y.std(dim=0, keepdim=True).clamp_min(1e-6)
        y_std = (y - mean) / std
        emb_y_RBE = self.b_target_embedder(
            torch.cat([y_std, torch.zeros_like(y_std)], dim=-1)  # observed => no NaN
        )
        b = torch.cat([emb_x_BRGE, emb_y_RBE.transpose(0, 1)[:, :, None]], dim=2)
        return b + self.domain_B

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x_A_RBC: torch.Tensor,
        y_A_NB: torch.Tensor,
        x_B_RBC: torch.Tensor | None,
        y_B_RB: torch.Tensor | None,
        n_A_train: int,
        *,
        use_B: bool = True,
    ) -> tuple[torch.Tensor, AuxOutputs | None]:
        """Predict A_test.

        Args:
            x_A_RBC: (R_A, batch, C). TabPFN's row-major layout.
            y_A_NB: (n_A_train, batch). Standardised with A_train stats.
            x_B_RBC: (R_B, batch, C) or None.
            y_B_RB: (R_B, batch) or None.
            n_A_train: rows of A before the test split.
            use_B: set False to ablate B entirely (the "floor" reference, and
                the mechanism behind the delta-NLL diagnostic).

        Returns:
            ``(logits_MBK, aux)`` with M = R_A - n_A_train test rows and
            K = n_bins. ``aux`` is None when ``use_B`` is False.
        """
        have_B = use_B and x_B_RBC is not None
        if have_B:
            keep_C = _shared_column_mask(x_A_RBC, x_B_RBC)
            x_A_RBC = x_A_RBC[:, :, keep_C]
            x_B_RBC = x_B_RBC[:, :, keep_C]

        a = self._embed_A(x_A_RBC, y_A_NB, n_A_train)

        # -- S1: A prologue. Contextualise A_train before it serves as keys.
        for blk in self.prologue_A:
            a, _ = blk([a], n_A_train, None)  # Wrap 'a' inside the call

        aux: AuxOutputs | None = None
        if have_B:
            b = self._embed_B(x_B_RBC, y_B_RB)
            R_B = b.shape[1]

            # -- S1': B encoder, unmasked (single_eval_pos == R).
            for blk in self.encoder_B:
                b, _ = blk([b], R_B, None)  # Wrap 'b' inside the call

            # -- S2: aligner. B queries, A_train keys.
            a_train = a[:, :n_A_train]
            for blk in self.aligner:
                b = blk(b, a_train)

            # -- relatedness gate
            pooled = torch.cat([a_train.mean(dim=(1, 2)), b.mean(dim=(1, 2))], dim=-1)
            severity_logit = self.relatedness_head(pooled)  # (batch, 1)
            b = b + torch.sigmoid(severity_logit)[:, :, None, None] * self.severity_token

            aux = AuxOutputs(
                b_hat_y_logits_RBK=self.translation_head(b[:, :, -1]).transpose(0, 1),
                b_hat_rows_BRE=b.mean(dim=2),
                severity_logit_B1=severity_logit,
            )

            ctx = torch.cat([a[:, :n_A_train], b, a[:, n_A_train:]], dim=1)
            sep = n_A_train + R_B
        else:
            ctx = a
            sep = n_A_train

        # -- S3: decoder over [think, A_train, B_hat, A_test].
        x, sep = self.stem.add_thinking_rows(ctx, single_eval_pos=sep)
        for blk in self.decoder:
            x, _ = blk([x], sep, None)  # Wrap 'x' inside the call

        # Readout mirrors TabPFNV2p5._decode: y column of the test rows only.
        test_emb_MBE = x[:, sep:, -1].transpose(0, 1)
        return self.output_projection(test_emb_MBE), aux

    # ------------------------------------------------------------- warm start

    def load_tabpfn_weights(self, state_dict: dict, *, verbose: bool = True) -> None:
        """Warm-start from a stock TabPFN v2.5 checkpoint.

        Embedder/thinking-row weights go to the stem; block weights are tiled
        over the four stacks in depth order. Because every new residual branch
        (both cross-attentions, all MLP second layers) is zero-initialised, the
        model right after this call reproduces stock TabPFN behaviour on
        ``[A_train, B]``.
        """
        stem_sd = {
            k: v for k, v in state_dict.items() if not k.startswith("blocks.")
        }
        missing = self.stem.load_state_dict(stem_sd, strict=False)
        if verbose:
            print(f"stem: {missing}")

        src_blocks: dict[int, dict] = {}
        for k, v in state_dict.items():
            if k.startswith("blocks."):
                idx, rest = k[len("blocks.") :].split(".", 1)
                src_blocks.setdefault(int(idx), {})[rest] = v
        order = sorted(src_blocks)

        cursor = 0
        for stack in (self.prologue_A, self.encoder_B, self.aligner, self.decoder):
            for blk in stack:
                sd = src_blocks[order[cursor % len(order)]]
                if isinstance(blk, AlignerBlock):
                    # Map TabPFNBlock's two self-attentions onto the aligner's.
                    sd = {
                        k.replace("per_sample_attention_between_features", "self_feat")
                        .replace("per_column_attention_between_cells", "self_row")
                        .replace("layernorm_mha1", "ln_self_feat")
                        .replace("layernorm_mha2", "ln_self_row")
                        .replace("layernorm_mlp", "ln_mlp"): v
                        for k, v in sd.items()
                    }
                blk.load_state_dict(sd, strict=False)
                cursor += 1
