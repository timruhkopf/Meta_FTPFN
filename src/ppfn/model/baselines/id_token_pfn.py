"""Additive-ID-token single-stream PFN -- ROADMAP.md §3.3 / §8.1 / §8.4,
ARCHITECTURE.md §5.5's "Additive-ID-token single-stream PFN": naive pooling
of A's context and B into ONE undifferentiated train set, each token tagged
with an additive per-cloud embedding (id=0 for A, id=1 for B) rather than
`ppfn.model.registration.embeddings.InputEmbeddings`'s separate-projection
design (that module's own docstring: "the additive-tag cross-term problem
is avoided by construction" -- this baseline is exactly the configuration
that avoids). Scored with a single bar-distribution NLL on A's query
tokens only (`ppfn.loss.id_token_loss.IDTokenLoss`) -- no transport,
coupling, affine, or distillation terms; this is deliberately NOT trying to
register anything, just testing whether an ID tag alone lets a pooled
single-stream PFN exploit B for A's benefit.

Consolidated (2026-09-15, at the user's direction, coordinated with the
`ppfn.prior.lupi`/`ppfn.model.lupi` session) onto `ppfn.prior.lupi`'s
richer prior -- NOT the plain `ppfn.prior.registration` prior this module
was originally built against -- because the plain prior has no y-distortion
`h` and no acquisition-biased A design: it can't test the actual question
("does a decoder-only model resolve T *and* h from a realistically-biased
A on its own"), only a strict subset of it.

`LUPIBatch`'s `z` fields were `[0,1]`-quantile-normalized via each cloud's
own ECDF as of the first version of this consolidation; that normalization
was itself scrapped shortly after (2026-09-15, same session, at the user's
direction) -- A's own ECDF was fit from a small, acquisition-BIASED sample
that can't self-diagnose its own bias
(docs/labbook/2026-09-15-quantile-normalization-bottleneck-and-decoder-only-proposal.md).
A B-referenced z-scoring replacement was tried next, then itself shelved
(same day, same direction) -- it leaked B's privileged clean-value mean/std
(the student never has access to it) and still produced heavy tails.
`ppfn.prior.lupi.sampler.sample_pair` now returns fully raw, unnormalized
targets: `h(f(z)) + noise`, no standardization at all. The predictive head
is accordingly `ppfn.model.pfn.bar_distribution.FullSupportBarDistribution`
with `quantile_bin_borders` fit to an actual target sample
(`ppfn.model.baselines.calibration.sample_calibration_borders`) rather than
a fixed range -- reassessed 2026-09-16 after a first attempt
(`ppfn.model.registration.heads.TailBarDistribution`'s fixed `[-4,4]` body)
turned out to be a poor match for this raw target's actual scale (most
draws' informative range covered only a handful of that head's 64 bins).

Reuses `ppfn.model.pfn.pfn`'s `MaskedMHA`/`PFNBlock` -- the generic
train/test masked-attention transformer block (bidirectional train-train
self-attention, test-to-train-only cross-attention, no test-test leakage)
is architecture-agnostic; nothing about it is specific to that module's own
`PFN` class or its single-domain `BNNPrior`. Deliberately NOT using a
coordinate-free/y-only first layer (`ppfn.model.registration`'s own
invariant) -- the user explicitly ruled that out for this variant: feed x,
z, and the domain tag together from layer 1, no special-casing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.baselines.calibration import sample_calibration_borders
from ppfn.model.pfn.bar_distribution import FullSupportBarDistribution
from ppfn.model.pfn.pfn import PFNBlock
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch


class IDTokenPFN(nn.Module):
    def __init__(
        self,
        d_max: int = D_MAX,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        n_bins_predictive: int = 64,
        dropout: float = 0.0,
        bounded01: bool = False,
    ):
        super().__init__()
        self.d_max = d_max
        # Matches PFNs4BO's transformer.py exactly (2026-09-15, at the
        # user's request -- "why a different weight matrix at all?"): ONE
        # shared x-encoder for train AND test tokens (`self.encoder` there,
        # `x_embed` here), plus a SEPARATE y-encoder whose output is added
        # ONLY to train tokens (`y_src` summed into `train_x` there, never
        # into test positions). No learned placeholder for the missing
        # test-side y -- PFNs4BO doesn't have one either; the missing
        # y-term (zero contribution, not a learned "unknown" vector) is
        # itself the only signal a test token differs from a train token.
        # An earlier version of this class used a SEPARATE, independently-
        # learned `test_x_embed` plus an additive `test_placeholder` --
        # redundant on reflection, since a separate x-encoder for test
        # tokens already lets the model distinguish train/test through
        # which weights processed them, without needing a second,
        # explicit signal on top. Cloud identity (A vs B, ppfn.prior.lupi
        # has no train/test-side distinction there) is carried entirely by
        # domain_embed below, orthogonal to this x/y split.
        self.x_embed = nn.Linear(d_max, d_model)
        self.y_embed = nn.Linear(1, d_model)
        # id=0 -> A (decoder cloud), id=1 -> B (encoder cloud). Added to
        # every train token; also added to every test token (always A, id=0)
        # so both token types go through the exact same "embedding + tag"
        # construction rather than test tokens silently skipping the tag --
        # verified 2026-09-15 at the user's explicit request (A_test must
        # carry A's own id, not a separate one or none at all).
        self.domain_embed = nn.Embedding(2, d_model)
        # Orthogonal init (2026-09-16, at the user's request): with only 2
        # rows, PyTorch's default N(0,1) init is already NEARLY orthogonal
        # at this d_model (expected cosine similarity ~1/sqrt(d_model)
        # =0.06 at d_model=256), so this mostly trades "almost surely
        # orthogonal" for "exactly orthogonal" rather than changing learning
        # dynamics much -- cheap and removes that one random draw either
        # way. `gain=sqrt(d_model)` matches `nn.Embedding`'s own default
        # N(0,1)-per-element init's expected row norm (`orthogonal_`'s
        # default gain=1 would give unit-norm rows instead, ~16x smaller at
        # d_model=256 -- a much weaker tag signal than intended, not a
        # neutral change).
        nn.init.orthogonal_(self.domain_embed.weight, gain=d_model**0.5)

        self.blocks = nn.ModuleList(
            [PFNBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.out_ln = nn.LayerNorm(d_model)
        # FullSupportBarDistribution (quantile-fit borders), not
        # TailBarDistribution's fixed [-4,4] body -- see
        # ppfn.model.pfn.bar_distribution's module docstring (2026-09-16):
        # the fixed body was a poor match for this prior's raw, unnormalized
        # target scale, giving coarse, blocky predictive densities.
        # bounded01 MUST match whatever ppfn.prior.lupi.sampler.sample_pair
        # was actually called with -- see calibration.py's own docstring
        # and docs/labbook/2026-09-17-lupi-bounded01-prior.md.
        borders = sample_calibration_borders(n_bins_predictive, bounded01=bounded01)
        self.predictive_dist = FullSupportBarDistribution(borders)
        self.predictive_head = nn.Linear(d_model, self.predictive_dist.num_bars)

    def forward(self, batch: LUPIBatch, domain_ids_override: torch.Tensor | None = None) -> dict:
        """Pools `[A_context ; B]` as one train set (additive domain tag),
        cross-attends A's query points against it, scores nothing here --
        `IDTokenLoss` reads `predictive_logits` and masks to `dec_qry_mask`.
        -> {"predictive_logits": [B, n_qry, n_bins_predictive]}.

        `domain_ids_override`, if given, replaces the auto-computed
        `[0]*n_ctx + [1]*n_enc` train-token tags (same shape as
        `pooled_mask`) -- for the notebook ablation showing what the tag
        contributes; never used during training."""
        pooled_x = torch.cat([batch.dec_ctx_x, batch.enc_x], dim=1)
        pooled_z = torch.cat([batch.dec_ctx_z, batch.enc_z], dim=1)
        pooled_mask = torch.cat([batch.dec_ctx_mask, batch.enc_mask], dim=1)
        n_ctx = batch.dec_ctx_x.shape[1]
        if domain_ids_override is not None:
            domain_ids = domain_ids_override
        else:
            domain_ids = pooled_mask.new_zeros(pooled_mask.shape, dtype=torch.long)
            domain_ids[:, n_ctx:] = 1  # B tokens (padding rows get id=1 too, harmless -- masked out)

        train_tok = (
            self.x_embed(pooled_x)
            + self.y_embed(pooled_z.unsqueeze(-1))
            + self.domain_embed(domain_ids)
        )

        # Test tokens: SAME x_embed as train, no y term at all (not even a
        # placeholder -- see __init__'s docstring), domain_embed(0) since
        # every query token is A.
        test_tok = self.x_embed(batch.dec_qry_x) + self.domain_embed.weight[0].view(1, 1, -1)

        for block in self.blocks:
            train_tok, test_tok = block(
                train_tok, test_tok, train_key_padding_mask=pooled_mask
            )

        test_tok = self.out_ln(test_tok)
        logits = self.predictive_head(test_tok)
        return {"predictive_logits": logits}


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run it on a real batch from LUPIStreamDataset, print shapes, and
    confirm the loss/backward path works end to end."""
    import torch

    from ppfn.loss.id_token_loss import IDTokenLoss
    from ppfn.prior.lupi.dataset import LUPIStreamDataset, collate_lupi_batch

    torch.manual_seed(0)
    dataset = LUPIStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_lupi_batch(items)

    model = IDTokenPFN(d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins_predictive=16)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("predictive_logits:", out["predictive_logits"].shape)

    criterion = IDTokenLoss()
    loss, metrics = criterion(model, batch, out)
    print("loss/metrics:", metrics)

    model.zero_grad()
    loss.backward()
    n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
    print(f"params with grad=None: {n_none}")

    # Padding-invisibility check: pad batch item 0's B cloud with one extra
    # all-zero, masked-out point and confirm logits don't change.
    padded_enc_x = torch.cat([batch.enc_x, torch.zeros_like(batch.enc_x[:, :1])], dim=1)
    padded_enc_z = torch.cat([batch.enc_z, torch.zeros_like(batch.enc_z[:, :1])], dim=1)
    padded_enc_mask = torch.cat(
        [batch.enc_mask, torch.zeros_like(batch.enc_mask[:, :1])], dim=1
    )
    import dataclasses

    padded_batch = dataclasses.replace(
        batch, enc_x=padded_enc_x, enc_z=padded_enc_z, enc_mask=padded_enc_mask
    )
    out_padded = model(padded_batch)
    diff = (out["predictive_logits"] - out_padded["predictive_logits"]).abs().max().item()
    print("max diff from one masked-out padding point in B (~0 expected):", diff)
