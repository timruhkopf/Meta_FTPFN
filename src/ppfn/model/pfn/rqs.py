"""Forward-only rational-quadratic spline (RQS) -- Durkan et al. 2019,
"Neural Spline Flows," the monotone building block their coupling layers use,
stripped down to just the piece this project needs.

NSF's own construction is built to be an *invertible* bijection with a
tractable Jacobian, because a normalizing flow needs exact densities under
change-of-variables. Nothing here inverts this map or needs a log-det term --
it's used purely as a flexible, structurally-monotone regression function
(the value-recalibration head's mean, `ppfn.model.baselines.
iterative_registration_pfn`), so this file keeps only the forward
piecewise-rational-quadratic evaluation and drops the inverse/log-det
machinery entirely.

Monotone by construction: bin widths and heights are softmax-normalized
(positive, sum to `2*tail_bound`), and knot derivatives are softplus'd
(positive) -- Durkan et al.'s own sufficient condition for the spline to be
strictly increasing. Outside `[-tail_bound, tail_bound]` the map is the
identity (NSF's own "linear tails" choice at slope 1) -- `RQSHead` below
wraps this in a learned positive affine (`s > 0, b`) specifically so the
*head* isn't stuck at the identity in the tails, only the raw spline is;
composing an increasing affine with an increasing spline is still increasing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MIN_BIN_WIDTH = 1e-3
_MIN_BIN_HEIGHT = 1e-3
_MIN_DERIVATIVE = 1e-3


def rational_quadratic_spline_forward(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    tail_bound: float = 5.0,
) -> torch.Tensor:
    """inputs: [...]. unnormalized_{widths,heights}: [..., n_bins].
    unnormalized_derivatives: [..., n_bins - 1] (internal knots only --
    boundary derivatives are fixed to 1, matching the identity tails, so the
    map is continuous and C1 at +-tail_bound). -> outputs: [...], same shape
    as `inputs`, monotone increasing in `inputs`.

    Everything broadcasts against `inputs`' shape -- pass per-item
    width/height/derivative params (from a hypernetwork) with a matching
    leading shape, or a single shared set with `inputs`-compatible
    broadcasting."""
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = _MIN_BIN_WIDTH + (1 - _MIN_BIN_WIDTH * widths.shape[-1]) * widths
    # cumsum's own endpoints are already exactly 0 and 1 by construction
    # (widths/heights are renormalized to sum to exactly 1 above), so after
    # the affine map below they land on -tail_bound/+tail_bound up to fp
    # rounding -- no explicit endpoint overwrite needed (avoids an in-place
    # indexed assignment on a tensor autograd still needs for backward
    # through cumsum).
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = (2 * tail_bound) * cumwidths - tail_bound

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = _MIN_BIN_HEIGHT + (1 - _MIN_BIN_HEIGHT * heights.shape[-1]) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = (2 * tail_bound) * cumheights - tail_bound

    derivatives = _MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)
    ones = derivatives.new_ones(*derivatives.shape[:-1], 1)  # boundary slope = 1, ties into the identity tails
    derivatives = torch.cat([ones, derivatives, ones], dim=-1)

    clamped = inputs.clamp(-tail_bound, tail_bound)
    bin_idx = torch.searchsorted(cumwidths.detach(), clamped.unsqueeze(-1).detach()).squeeze(-1) - 1
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    def _gather(t):
        return t.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    x_k = _gather(cumwidths[..., :-1])
    bin_width = _gather(widths) * (2 * tail_bound)
    y_k = _gather(cumheights[..., :-1])
    bin_height = _gather(heights) * (2 * tail_bound)
    d_k = _gather(derivatives[..., :-1])
    d_kp1 = _gather(derivatives[..., 1:])

    s_k = bin_height / bin_width.clamp_min(1e-8)
    xi = (clamped - x_k) / bin_width.clamp_min(1e-8)
    xi = xi.clamp(0.0, 1.0)

    numerator = bin_height * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
    denominator = s_k + (d_kp1 + d_k - 2 * s_k) * xi * (1 - xi)
    spline_out = y_k + numerator / denominator.clamp_min(1e-8)

    return torch.where(inside, spline_out, inputs)


if __name__ == "__main__":
    """Diagnostic per .claude/rules/research-demos.md: confirm the spline is
    monotone increasing and continuous at the tail boundary, and that
    gradients flow into all three parameter tensors."""
    torch.manual_seed(0)
    n_bins = 8
    n = 4001
    uw = torch.randn(n_bins, requires_grad=True)
    uh = torch.randn(n_bins, requires_grad=True)
    ud = torch.randn(n_bins - 1, requires_grad=True)

    x = torch.linspace(-8.0, 8.0, n)
    # Broadcast the (shared, not per-item) params to match x's batch shape --
    # real usage has a hypernetwork produce genuinely per-item params with a
    # matching leading shape already; this is just this standalone test's
    # own setup, not a feature of the function itself.
    y = rational_quadratic_spline_forward(
        x, uw.expand(n, -1), uh.expand(n, -1), ud.expand(n, -1), tail_bound=5.0
    )

    diffs = y[1:] - y[:-1]
    print("monotone increasing everywhere (expect True):", bool((diffs >= -1e-5).all()))
    print(f"min diff: {diffs.min().item():.6f} (expect >= ~0)")

    # Continuity at the tail boundary: y should match the identity there.
    boundary_idx = (x - (-5.0)).abs().argmin()
    print(f"y(-tail_bound)={y[boundary_idx].item():.4f} vs -tail_bound=-5.0 (expect close)")
    boundary_idx = (x - 5.0).abs().argmin()
    print(f"y(+tail_bound)={y[boundary_idx].item():.4f} vs +tail_bound=5.0 (expect close)")

    y.sum().backward()
    print("grad norms (uw,uh,ud), expect all > 0:", uw.grad.norm().item(), uh.grad.norm().item(), ud.grad.norm().item())
