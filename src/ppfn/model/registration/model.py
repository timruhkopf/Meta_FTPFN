"""Top-level `RegistrationPFN` -- wires embeddings + encoder + decoder +
output heads into one module, matching `ppfn.prior.registration.dataset`'s
`RegistrationBatch` on the input side. ARCHITECTURE.md §2 end to end.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.registration.decoder import Decoder
from ppfn.model.registration.embeddings import InputEmbeddings
from ppfn.model.registration.encoder import Encoder
from ppfn.model.registration.heads import TailBarDistribution, TransportHead
from ppfn.prior.registration.dataset import D_MAX, RegistrationBatch


class RegistrationPFN(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        n_layers_enc: int = 6,
        n_layers_dec: int = 8,
        n_fourier_frequencies: int = 8,
        n_bins_predictive: int = 64,
        n_bins_transport: int = 64,
        dropout: float = 0.0,
        d_max: int = D_MAX,
    ):
        super().__init__()
        self.d_max = d_max
        self.embeddings = InputEmbeddings(d_max, d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers_enc, dropout)
        self.decoder = Decoder(
            d_model, d_max, n_heads, d_ff, n_layers_dec, n_fourier_frequencies, dropout
        )
        self.transport_head = TransportHead(d_model, d_max, n_bins_transport)
        self.predictive_dist = TailBarDistribution(n_bins_predictive)
        self.predictive_head = nn.Linear(d_model, self.predictive_dist.num_logits)

    def dim_mask(self, d_real: torch.Tensor, n: int) -> torch.Tensor:
        """d_real: [B] long -> [B,n,d_max] bool, True on the first d_real[b]
        coordinate channels -- masks the loss on the zero-padded channels."""
        idx = torch.arange(self.d_max, device=d_real.device).view(1, 1, -1)
        return idx < d_real.view(-1, 1, 1)

    def forward(
        self,
        batch: RegistrationBatch,
        transport_override: tuple[torch.Tensor, torch.Tensor] | None = None,
        severed_mask: torch.Tensor | None = None,
    ) -> dict:
        """`severed_mask` defaults to `batch.severed` (the per-item coin
        flip drawn at data-assembly time, ARCHITECTURE.md §4.4); callers
        that need a forced mode (e.g. the `lower`/`upper-2` evaluation
        bounds, §5.1) pass an explicit override."""
        if severed_mask is None:
            severed_mask = batch.severed

        enc_tok = self.embeddings.encoder_tokens(batch.enc_x, batch.enc_y)
        m, g_b = self.encoder(enc_tok, batch.enc_mask)

        h_ctx = self.embeddings.decoder_context_tokens(batch.dec_ctx_x, batch.dec_ctx_y)
        h_qry = self.embeddings.decoder_query_tokens(batch.dec_qry_x)

        dec_out = self.decoder(
            h_ctx=h_ctx,
            h_qry=h_qry,
            x_ctx_raw=batch.dec_ctx_x,
            x_qry_raw=batch.dec_qry_x,
            y_ctx=batch.dec_ctx_y,
            ctx_mask=batch.dec_ctx_mask,
            qry_mask=batch.dec_qry_mask,
            m=m,
            x_enc=batch.enc_x,
            y_enc=batch.enc_y,
            enc_mask=batch.enc_mask,
            g_b=g_b,
            severed_mask=severed_mask,
            transport_override=transport_override,
        )

        # Deep-supervised transport head, every layer (see decoder.py's
        # module docstring / model docstring above for the §2.5-vs-§3.2
        # resolution).
        transport_logits_ctx = [
            self.transport_head(h, teacher_targets=batch.transport_ctx)
            for h in dec_out["h_ctx_layers"]
        ]
        transport_logits_qry = [
            self.transport_head(h, teacher_targets=batch.transport_qry)
            for h in dec_out["h_qry_layers"]
        ]

        # Predictive head: final layer, query tokens only (ARCHITECTURE.md §3.1).
        predictive_logits = self.predictive_head(dec_out["h_qry_layers"][-1])

        return {
            "predictive_logits": predictive_logits,  # [B, n_qry, num_pred_logits]
            "transport_logits_ctx": transport_logits_ctx,  # list[L] of [B,n_ctx,d_max,n_bins]
            "transport_logits_qry": transport_logits_qry,  # list[L] of [B,n_qry,d_max,n_bins]
            "bary_ctx_layers": dec_out["bary_ctx_layers"],  # list[L] of [B,n_ctx,d_max]
            "bary_qry_layers": dec_out["bary_qry_layers"],  # list[L] of [B,n_qry,d_max]
            "a_g": dec_out["a_g"],
            "b_g": dec_out["b_g"],
            "gates": dec_out["gates"],  # [L]
            "severed_mask": severed_mask,
        }


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: build a small model,
    run a forward pass on a real batch from `RegistrationStreamDataset`, and
    print shapes -- plus the same query-mask-isolation check the decoder's
    own demo runs, at the whole-model level (embeddings + heads included)."""
    import torch

    from ppfn.prior.registration.dataset import (
        RegistrationStreamDataset,
        collate_registration_batch,
    )

    torch.manual_seed(0)
    dataset = RegistrationStreamDataset(seed=0, s_max=0.1)
    items = [next(iter(dataset)) for _ in range(4)]
    batch = collate_registration_batch(items)

    model = RegistrationPFN(
        d_model=32, n_heads=4, d_ff=64, n_layers_enc=2, n_layers_dec=3
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}")

    out = model(batch)
    print("predictive_logits:", out["predictive_logits"].shape)
    print(
        "transport_logits_ctx: L=",
        len(out["transport_logits_ctx"]),
        out["transport_logits_ctx"][0].shape,
    )
    print("gates:", out["gates"])

    dim_mask_qry = model.dim_mask(batch.d_real, batch.dec_qry_x.shape[1])
    print(
        "dim_mask_qry:", dim_mask_qry.shape, "sum per item:", dim_mask_qry[:, 0].sum(-1)
    )
