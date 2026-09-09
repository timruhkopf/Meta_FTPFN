"""Output heads -- ARCHITECTURE.md §2.4 (global affine) and §2.5 (predictive,
transport). Reuses `ppfn.model.pfn.bar_distribution.BarDistribution` (a
genuinely reusable, self-contained bar-distribution implementation; see that
module's docstring) for the transport head's per-axis distribution and as
the building block for `TailBarDistribution` below.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppfn.model.pfn.bar_distribution import BarDistribution, uniform_bin_borders


def _zero_init_(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class GlobalAffineHead(nn.Module):
    """A_g = I + MLP_aff([g_B, mean_i h_i^(1)]), b_g = MLP_aff_b(...) --
    ARCHITECTURE.md §2.4. Both MLPs' final layers are zero-init, so A_g
    starts at the identity and b_g at 0 -- "exactly correct at rho=0" per
    the build-order go/no-go (CLAUDE.md step 4)."""

    def __init__(self, d_model: int, d_max: int, hidden: int | None = None):
        super().__init__()
        self.d_max = d_max
        hidden = hidden or d_model
        self.mlp_a = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_max * d_max)
        )
        self.mlp_b = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, d_max)
        )
        _zero_init_(self.mlp_a[-1])
        _zero_init_(self.mlp_b[-1])

    def forward(
        self, g_b: torch.Tensor, mean_h1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """g_b, mean_h1: [B, d_model] -> (A_g [B,d_max,d_max], b_g [B,d_max])."""
        pooled = torch.cat([g_b, mean_h1], dim=-1)
        delta_a = self.mlp_a(pooled).view(-1, self.d_max, self.d_max)
        eye = torch.eye(self.d_max, device=pooled.device, dtype=pooled.dtype).unsqueeze(
            0
        )
        a_g = eye + delta_a
        b_g = self.mlp_b(pooled)
        return a_g, b_g

    @staticmethod
    def apply(a_g: torch.Tensor, b_g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """x: [B,N,d_max] -> A_g x + b_g: [B,N,d_max]."""
        return torch.einsum("bij,bnj->bni", a_g, x) + b_g.unsqueeze(1)


class TransportHead(nn.Module):
    """Autoregressive-over-axes bar distribution -- ARCHITECTURE.md §2.5.
    Teacher-forced during training (the true per-axis target is available
    from the prior); self-conditioned on the running bar-distribution mean
    at inference, when `teacher_targets` is None. One independent per-axis
    MLP (`axis_heads[k]`), not weight-shared across axes -- axis k's MLP
    additionally sees which axis it is via the fixed input layout (axes
    >=k are exactly 0 in `known`), so no separate positional embedding is
    needed for that; sharing weights across axes would need one anyway to
    break the symmetry.
    """

    def __init__(
        self, d_model: int, d_max: int, n_bins: int = 64, hidden: int | None = None
    ):
        super().__init__()
        self.d_max = d_max
        hidden = hidden or d_model
        self.axis_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model + d_max, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, n_bins),
                )
                for _ in range(d_max)
            ]
        )
        self.bar_dist = BarDistribution(uniform_bin_borders(n_bins, 0.0, 1.0))

    def forward(
        self, h: torch.Tensor, teacher_targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        """h: [...,d_model]. teacher_targets: [...,d_max] or None.
        -> logits [...,d_max,n_bins]."""
        known = h.new_zeros(*h.shape[:-1], self.d_max)
        logits_list = []
        for k, head in enumerate(self.axis_heads):
            logits_k = head(torch.cat([h, known], dim=-1))
            logits_list.append(logits_k)
            next_known = known.clone()
            if teacher_targets is not None:
                next_known[..., k] = teacher_targets[..., k]
            else:
                next_known[..., k] = self.bar_dist.mean(logits_k)
            known = next_known
        return torch.stack(logits_list, dim=-2)

    def nll(
        self, logits: torch.Tensor, target: torch.Tensor, dim_mask: torch.Tensor
    ) -> torch.Tensor:
        """logits: [...,d_max,n_bins]  target,dim_mask: [...,d_max] ->
        per-token NLL [...], averaged over the real axes only (dim_mask)."""
        per_axis_nll = self.bar_dist(logits, target)  # [...,d_max]
        mask_f = dim_mask.to(per_axis_nll.dtype)
        return (per_axis_nll * mask_f).sum(-1) / mask_f.sum(-1).clamp_min(1.0)


class TailBarDistribution(nn.Module):
    """Predictive head's output distribution -- ARCHITECTURE.md §2.5: "64
    bins over [-4, 4] plus two half-open tail bins." The two tail bins use
    an exponential (rate `tail_rate`) density anchored at the body's border,
    decaying away from it -- a proper distribution over all of R (finite
    total mass), reusing the same softmax as the 64 body bins so the tail
    and body probabilities are directly comparable. `tail_rate=1.0` (nats
    per unit of standardized y-tilde) is a deliberate simplification: the
    reference `FullSupportBarDistribution` this project's own
    `BarDistribution` was ported from (see that module's docstring) isn't
    vendored in this repo to copy the exact functional form from.
    """

    def __init__(
        self,
        n_bins: int = 64,
        lo: float = -4.0,
        hi: float = 4.0,
        tail_rate: float = 1.0,
    ):
        super().__init__()
        self.body = BarDistribution(uniform_bin_borders(n_bins, lo, hi))
        self.lo, self.hi = lo, hi
        self.tail_rate = tail_rate
        self.num_logits = n_bins + 2  # [left_tail, body x n_bins, right_tail]

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """logits: [..., n_bins+2]  y: [...] -> NLL [...]."""
        log_probs = torch.log_softmax(logits, dim=-1)
        left_logp, body_logp, right_logp = (
            log_probs[..., 0],
            log_probs[..., 1:-1],
            log_probs[..., -1],
        )

        y_clamped = y.clamp(self.lo, self.hi)
        idx = self.body.map_to_bucket_idx(y_clamped)
        body_nll = -(
            body_logp.gather(-1, idx[..., None]).squeeze(-1)
            - torch.log(self.body.bucket_widths[idx])
        )

        log_rate = torch.log(
            torch.as_tensor(self.tail_rate, device=logits.device, dtype=logits.dtype)
        )
        left_dist = (self.lo - y).clamp_min(0)
        left_nll = -(left_logp + log_rate - self.tail_rate * left_dist)
        right_dist = (y - self.hi).clamp_min(0)
        right_nll = -(right_logp + log_rate - self.tail_rate * right_dist)

        return torch.where(
            y < self.lo, left_nll, torch.where(y > self.hi, right_nll, body_nll)
        )

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """E[y] = P(left)*E[y|left] + P(body)*E[y|body] + P(right)*E[y|right],
        where E[y|body] = `self.body.mean` of the body-only softmax (a
        conditional mean, since `BarDistribution.mean` renormalizes over just
        the logits it's given)."""
        probs = torch.softmax(logits, dim=-1)
        left_p, body_p, right_p = (
            probs[..., 0],
            probs[..., 1:-1].sum(-1),
            probs[..., -1],
        )
        body_mean = self.body.mean(logits[..., 1:-1])
        left_mean = self.lo - 1.0 / self.tail_rate
        right_mean = self.hi + 1.0 / self.tail_rate
        return left_p * left_mean + body_p * body_mean + right_p * right_mean


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, d_max = 16, 5

    affine = GlobalAffineHead(d_model, d_max)
    g_b, mean_h1 = torch.rand(3, d_model), torch.rand(3, d_model)
    a_g, b_g = affine(g_b, mean_h1)
    x = torch.rand(3, 7, d_max)
    t0 = GlobalAffineHead.apply(a_g, b_g, x)
    print("A_g:", a_g.shape, "b_g:", b_g.shape, "t0:", t0.shape)
    print(
        "at init, A_g == I and b_g == 0, so t0 == x (expect ~0):",
        (t0 - x).abs().max().item(),
    )

    transport = TransportHead(d_model, d_max, n_bins=8)
    h = torch.rand(3, 7, d_model)
    target = torch.rand(3, 7, d_max)
    logits = transport(h, teacher_targets=target)
    print("transport logits:", logits.shape)
    dim_mask = torch.ones(3, 7, d_max, dtype=torch.bool)
    dim_mask[..., 3:] = False
    print("transport nll:", transport.nll(logits, target, dim_mask).shape)

    pred = TailBarDistribution(n_bins=8)
    logits_pred = torch.zeros(5, pred.num_logits)
    y = torch.tensor([-10.0, -4.0, 0.0, 4.0, 10.0])
    print("predictive NLL at extreme + in-range y:", pred(logits_pred, y))
