"""Decoder stack -- ARCHITECTURE.md §2.3. Each layer: (a) PFN self-attention
over decoder tokens, (b) transport update, (c) gated cross-attention to the
encoder memory M, (d) FFN. Orchestration of the layer-1 special case (no
transport update, coordinate-free cross-attention) and the global affine
head (computed once, right after layer 1) lives in `Decoder.forward` below,
since it spans what would otherwise be two different "layer" behaviors --
see ARCHITECTURE.md §2.3(b)/§2.4 and the module-level notes on invariant #3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppfn.model.registration.fourier import FourierFeatures
from ppfn.model.registration.heads import GlobalAffineHead


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B,N,D], mask: [B,N] bool (True=real) -> [B,D]."""
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


class PFNSelfAttention(nn.Module):
    """(a) -- context tokens attend bidirectionally to context; query tokens
    attend to context only, never to each other or themselves
    (ARCHITECTURE.md invariant #2's prerequisite: query tokens must stay
    mask-isolated). One shared set of Q/K/V/out weights for both calls,
    matching the (unrelated, dead) `ppfn.model.pfn.pfn.MaskedMHA`'s
    documented rationale for weight sharing between the two physically
    separate attention calls.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        h_ctx: torch.Tensor,
        h_qry: torch.Tensor,
        ctx_mask: torch.Tensor,
        qry_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kpm = ~ctx_mask
        ctx_out, _ = self.mha(
            h_ctx, h_ctx, h_ctx, key_padding_mask=kpm, need_weights=False
        )
        qry_out, _ = self.mha(
            h_qry, h_ctx, h_ctx, key_padding_mask=kpm, need_weights=False
        )
        return h_ctx + ctx_out, h_qry + qry_out


class GatedCrossAttention(nn.Module):
    """(c) -- ARCHITECTURE.md §2.3(c). Head 0's value projection is frozen
    to the identity on x-tilde^B: instead of a learned V, head 0's "value"
    IS the raw (padded) encoder coordinates, concatenated in (not projected
    through a learned d_head-wide slot) alongside the other heads' learned
    values before `out_proj` -- see the module-level design note below for
    why this is the natural reading of "frozen to the identity" and how it
    stays compatible with the standard multi-head concatenation.

    Design note: heads 1..n_heads-1 use ordinary learned Q/K/V, each
    d_head = d_model // n_heads wide. Head 0 uses the SAME learned Q/K (the
    "frozen identity" restriction in ARCHITECTURE.md §2.3 is explicitly on
    the *value* projection only) but its value is x-tilde^B directly
    (d_max-wide, not d_head-wide) -- so the concatenation before `out_proj`
    is [head_1_out, ..., head_{H-1}_out, head_0_bary] with total width
    (n_heads-1)*d_head + d_max, and `out_proj` is a learned map from that
    width into d_model. `head_0_bary` (pre-out_proj) IS `T_bary^(l)`
    (ARCHITECTURE.md §2.3: "T_bary,i^(l) = sum_j pi_ij^(l,head0) x_j^B"),
    read out here and returned separately for the coupling loss.
    """

    def __init__(
        self,
        d_model: int,
        d_max: int,
        n_heads: int,
        fourier: FourierFeatures,
        coordinate_free: bool,
    ):
        super().__init__()
        assert n_heads >= 2, "head 0 (coupling) + at least one learned head"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_max = d_max
        self.fourier = fourier
        self.coordinate_free = coordinate_free

        side_dim = d_model + (0 if coordinate_free else fourier.out_dim) + 1
        self.q_proj = nn.Linear(side_dim, n_heads * self.d_head)
        self.k_proj = nn.Linear(side_dim, n_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, (n_heads - 1) * self.d_head)
        self.out_proj = nn.Linear((n_heads - 1) * self.d_head + d_max, d_model)
        self.temp_proj = nn.Linear(2 * d_model, 1)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        h: torch.Tensor,
        y_side: torch.Tensor,
        t_or_none: torch.Tensor | None,
        m: torch.Tensor,
        x_enc: torch.Tensor,
        y_enc: torch.Tensor,
        enc_mask: torch.Tensor,
        g_b: torch.Tensor,
        severed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [B,Nq,d_model]. Returns (h_new [B,Nq,d_model], t_bary [B,Nq,d_max])."""
        b, nq, _ = h.shape
        nk = m.shape[1]

        if self.coordinate_free:
            q_in = torch.cat([h, y_side.unsqueeze(-1)], dim=-1)
            k_in = torch.cat([m, y_enc.unsqueeze(-1)], dim=-1)
        else:
            q_in = torch.cat([h, self.fourier(t_or_none), y_side.unsqueeze(-1)], dim=-1)
            k_in = torch.cat([m, self.fourier(x_enc), y_enc.unsqueeze(-1)], dim=-1)

        q = (
            self.q_proj(q_in).view(b, nq, self.n_heads, self.d_head).transpose(1, 2)
        )  # [B,H,Nq,dh]
        k = (
            self.k_proj(k_in).view(b, nk, self.n_heads, self.d_head).transpose(1, 2)
        )  # [B,H,Nk,dh]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,Nq,Nk]

        temp_in = torch.cat([g_b.unsqueeze(1).expand(-1, nq, -1), h], dim=-1)
        tau = F.softplus(self.temp_proj(temp_in)).squeeze(-1).clamp_min(1e-3)  # [B,Nq]
        scores = scores / tau.view(b, 1, nq, 1)
        scores = scores.masked_fill(
            ~enc_mask.view(b, 1, 1, nk), torch.finfo(scores.dtype).min
        )
        attn = torch.softmax(scores, dim=-1)  # [B,H,Nq,Nk]

        t_bary = attn[:, 0] @ x_enc  # [B,Nq,d_max] -- head 0, frozen identity value

        v = (
            self.v_proj(m).view(b, nk, self.n_heads - 1, self.d_head).transpose(1, 2)
        )  # [B,H-1,Nk,dh]
        out_rest = (
            (attn[:, 1:] @ v)
            .transpose(1, 2)
            .reshape(b, nq, (self.n_heads - 1) * self.d_head)
        )

        cross_out = self.out_proj(torch.cat([out_rest, t_bary], dim=-1))
        gate = self.gate * (~severed_mask).to(h.dtype).view(b, 1, 1)
        return h + gate * cross_out, t_bary


class DecoderLayer(nn.Module):
    """One block: self-attn, [transport update], gated cross-attn, FFN.
    `has_transport_update=False` for layer 1 (ARCHITECTURE.md §2.3(b): the
    layer-1 cross-attention is coordinate-free and t^(0) is computed
    separately, AFTER this layer, by the global affine head -- see
    `Decoder.forward`)."""

    def __init__(
        self,
        d_model: int,
        d_max: int,
        n_heads: int,
        d_ff: int,
        fourier: FourierFeatures,
        coordinate_free: bool,
        has_transport_update: bool,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.has_transport_update = has_transport_update
        self.self_attn = PFNSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = GatedCrossAttention(
            d_model, d_max, n_heads, fourier, coordinate_free
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        if has_transport_update:
            self.delta_ctx = nn.Linear(d_model, d_max)
            self.delta_qry = nn.Linear(d_model, d_max)
            nn.init.zeros_(self.delta_ctx.weight)
            nn.init.zeros_(self.delta_ctx.bias)
            nn.init.zeros_(self.delta_qry.weight)
            nn.init.zeros_(self.delta_qry.bias)

    def forward(self, state: dict) -> dict:
        h_ctx, h_qry = self.self_attn(
            state["h_ctx"], state["h_qry"], state["ctx_mask"], state["qry_mask"]
        )

        override = state.get("transport_override")
        if override is not None:
            # Oracle teacher-forcing (ARCHITECTURE.md §3.4: "t_i^(l) :=
            # A_inB_target(i), all l, all tokens") -- bypasses the affine +
            # Delta_l computation entirely, at every transport-updating
            # layer, so the cross-attention queries below are actually built
            # from the ground truth, not merely logged as if they were.
            t_ctx, t_qry = override
        elif self.has_transport_update:
            base_ctx = GlobalAffineHead.apply(
                state["a_g"], state["b_g"], state["x_ctx_raw"]
            )
            base_qry = GlobalAffineHead.apply(
                state["a_g"], state["b_g"], state["x_qry_raw"]
            )
            t_ctx = (base_ctx + self.delta_ctx(h_ctx)).clamp(0.0, 1.0)
            t_qry = (base_qry + self.delta_qry(h_qry)).clamp(0.0, 1.0)
        else:
            t_ctx = t_qry = None

        y_zero_qry = torch.zeros_like(state["y_qry_placeholder"])
        h_ctx, bary_ctx = self.cross_attn(
            h_ctx,
            state["y_ctx"],
            t_ctx,
            state["m"],
            state["x_enc"],
            state["y_enc"],
            state["enc_mask"],
            state["g_b"],
            state["severed_mask"],
        )
        h_qry, bary_qry = self.cross_attn(
            h_qry,
            y_zero_qry,
            t_qry,
            state["m"],
            state["x_enc"],
            state["y_enc"],
            state["enc_mask"],
            state["g_b"],
            state["severed_mask"],
        )

        h_ctx = h_ctx + self.ff(h_ctx)
        h_qry = h_qry + self.ff(h_qry)

        state = dict(state)
        state["h_ctx"], state["h_qry"] = h_ctx, h_qry
        state["t_ctx"], state["t_qry"] = t_ctx, t_qry
        state["bary_ctx"], state["bary_qry"] = bary_ctx, bary_qry
        return state


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_max: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        n_fourier_frequencies: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert n_layers >= 2, (
            "layer 1 (coordinate-free) + at least one transport-updating layer"
        )
        self.d_max = d_max
        self.fourier = FourierFeatures(d_max, n_fourier_frequencies)
        self.affine_head = GlobalAffineHead(d_model, d_max)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_max,
                    n_heads,
                    d_ff,
                    self.fourier,
                    coordinate_free=(i == 0),
                    has_transport_update=(i > 0),
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        h_ctx: torch.Tensor,
        h_qry: torch.Tensor,
        x_ctx_raw: torch.Tensor,
        x_qry_raw: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        qry_mask: torch.Tensor,
        m: torch.Tensor,
        x_enc: torch.Tensor,
        y_enc: torch.Tensor,
        enc_mask: torch.Tensor,
        g_b: torch.Tensor,
        severed_mask: torch.Tensor,
        transport_override: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> dict:
        """Runs the full L_dec stack. Returns a dict with, per layer:
        `h_ctx_layers`/`h_qry_layers` (for deep-supervised heads),
        `t_ctx_layers`/`t_qry_layers` (None for layer 1),
        `bary_ctx_layers`/`bary_qry_layers` (coupling target, every layer),
        `a_g`/`b_g` (the global affine, computed once after layer 1),
        `gates` (per-layer gamma_l, for the collapse monitor).

        `transport_override`: (t_ctx, t_qry), both [B,N,d_max] in [0,1] --
        when given, EVERY layer's transport update is bypassed and this
        fixed value is used instead (ARCHITECTURE.md §3.4's oracle
        distillation teacher: "t_i^(l) := A_inB_target(i), all l, all
        tokens" -- and §2.4's build-order go/no-go, "Delta_l disabled,
        t_i = x-tilde_i^A"). Layer 1 stays coordinate-free regardless (it
        structurally never reads transport), matching "gates active" being
        the only thing that changes under teacher-forcing.
        """
        state = {
            "h_ctx": h_ctx,
            "h_qry": h_qry,
            "ctx_mask": ctx_mask,
            "qry_mask": qry_mask,
            "x_ctx_raw": x_ctx_raw,
            "x_qry_raw": x_qry_raw,
            "y_ctx": y_ctx,
            "y_qry_placeholder": qry_mask,  # only used for its shape/dtype/device in DecoderLayer
            "m": m,
            "x_enc": x_enc,
            "y_enc": y_enc,
            "enc_mask": enc_mask,
            "g_b": g_b,
            "severed_mask": severed_mask,
            "a_g": None,
            "b_g": None,
            "transport_override": transport_override,
        }

        h_ctx_layers, h_qry_layers = [], []
        t_ctx_layers, t_qry_layers = [], []
        bary_ctx_layers, bary_qry_layers = [], []
        gates = []

        # Layer 1: coordinate-free, no transport update.
        state = self.layers[0](state)
        h_ctx_layers.append(state["h_ctx"])
        h_qry_layers.append(state["h_qry"])
        t_ctx_layers.append(None)
        t_qry_layers.append(None)
        bary_ctx_layers.append(state["bary_ctx"])
        bary_qry_layers.append(state["bary_qry"])
        gates.append(self.layers[0].cross_attn.gate.detach())

        # Global affine head, from post-layer-1 representations -- ARCHITECTURE.md §2.4.
        mean_h1_ctx = _masked_mean(state["h_ctx"], ctx_mask)
        a_g, b_g = self.affine_head(g_b, mean_h1_ctx)
        state["a_g"], state["b_g"] = a_g, b_g

        for layer in self.layers[1:]:
            state = layer(state)
            h_ctx_layers.append(state["h_ctx"])
            h_qry_layers.append(state["h_qry"])
            t_ctx_layers.append(state["t_ctx"])
            t_qry_layers.append(state["t_qry"])
            bary_ctx_layers.append(state["bary_ctx"])
            bary_qry_layers.append(state["bary_qry"])
            gates.append(layer.cross_attn.gate.detach())

        return {
            "h_ctx_layers": [self.out_ln(h) for h in h_ctx_layers],
            "h_qry_layers": [self.out_ln(h) for h in h_qry_layers],
            "t_ctx_layers": t_ctx_layers,
            "t_qry_layers": t_qry_layers,
            "bary_ctx_layers": bary_ctx_layers,
            "bary_qry_layers": bary_qry_layers,
            "a_g": a_g,
            "b_g": b_g,
            "gates": torch.stack(gates),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, d_max, n_heads, d_ff, n_layers = 16, 5, 4, 32, 3
    b, n_ctx, n_qry, n_enc = 2, 6, 4, 9

    decoder = Decoder(d_model, d_max, n_heads, d_ff, n_layers)
    h_ctx = torch.rand(b, n_ctx, d_model)
    h_qry = torch.rand(b, n_qry, d_model)
    x_ctx_raw = torch.rand(b, n_ctx, d_max)
    x_qry_raw = torch.rand(b, n_qry, d_max)
    y_ctx = torch.rand(b, n_ctx)
    ctx_mask = torch.ones(b, n_ctx, dtype=torch.bool)
    qry_mask = torch.ones(b, n_qry, dtype=torch.bool)
    m = torch.rand(b, n_enc, d_model)
    x_enc = torch.rand(b, n_enc, d_max)
    y_enc = torch.rand(b, n_enc)
    enc_mask = torch.ones(b, n_enc, dtype=torch.bool)
    g_b = torch.rand(b, d_model)
    severed_mask = torch.zeros(b, dtype=torch.bool)

    out = decoder(
        h_ctx,
        h_qry,
        x_ctx_raw,
        x_qry_raw,
        y_ctx,
        ctx_mask,
        qry_mask,
        m,
        x_enc,
        y_enc,
        enc_mask,
        g_b,
        severed_mask,
    )
    print("h_ctx_layers:", len(out["h_ctx_layers"]), out["h_ctx_layers"][0].shape)
    print("t_ctx_layers[0] (layer1, expect None):", out["t_ctx_layers"][0])
    print("t_ctx_layers[1] shape:", out["t_ctx_layers"][1].shape)
    print("bary_ctx_layers[0] shape:", out["bary_ctx_layers"][0].shape)
    print("a_g:", out["a_g"].shape, "b_g:", out["b_g"].shape)
    print("gates:", out["gates"])

    # Query-isolation check: perturbing one query point must not change any
    # OTHER query point's final hidden state (mask isolation, ARCHITECTURE.md
    # invariant #2's prerequisite).
    h_qry_pert = h_qry.clone()
    h_qry_pert[:, 0, :] += 1.0
    out_pert = decoder(
        h_ctx,
        h_qry_pert,
        x_ctx_raw,
        x_qry_raw,
        y_ctx,
        ctx_mask,
        qry_mask,
        m,
        x_enc,
        y_enc,
        enc_mask,
        g_b,
        severed_mask,
    )
    other_diff = (
        (out["h_qry_layers"][-1][:, 1:] - out_pert["h_qry_layers"][-1][:, 1:])
        .abs()
        .max()
        .item()
    )
    print(
        "max diff at OTHER query points after perturbing query 0 (~0 expected):",
        other_diff,
    )

    # severed mode: gate contribution must vanish.
    severed_all = torch.ones(b, dtype=torch.bool)
    out_severed = decoder(
        h_ctx,
        h_qry,
        x_ctx_raw,
        x_qry_raw,
        y_ctx,
        ctx_mask,
        qry_mask,
        m,
        x_enc,
        y_enc,
        enc_mask,
        g_b,
        severed_all,
    )
    print(
        "severed run produced finite output:",
        torch.isfinite(out_severed["h_qry_layers"][-1]).all().item(),
    )
