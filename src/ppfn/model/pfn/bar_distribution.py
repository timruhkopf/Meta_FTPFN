"""Bar (Riemann) distribution output head — ported from PFNs4BO's
`pfns4bo/bar_distribution.py` (Müller et al., ICML 2023; vendored reference
at `archive/src/utils/bar_distribution.py`). NLL, mean/median/mode/variance,
`quantile`, `ucb`, closed-form `ei`/`pi` (2026-09-08: moved onto this class
from `models/baselines/pfn_acquisition.py`/`models/surrogates/pfn_surrogate.py`
— an earlier version of this docstring said EI/PI/UCB were deliberately
dropped here ("belongs to the classical baselines instead"); reversed by
user request, they live here now, matching PFNs4BO's own layout) plus
`entropy()`, which the original doesn't have but M5's explore-branch search
needs (closed-form, no Monte Carlo — see the design doc). `ei`/`pi`/`ucb`
are all mirrored for this project's minimize convention — the reference
assumes maximization throughout; see each method's own docstring for the
exact mirroring. Not ported: `smoothing`/`mean_prediction_logits` (both
`forward()`-only training-loss features, unused so far — see
`archive`'s own `forward()` for the shape if ever needed).

`BarDistribution` above: fixed `[0, 1]` borders (bounded) — matches M1's
ECDF-normalized-to-[0,1] prior output. See `docs/OPEN_QUESTIONS.md` #8 for
the full reasoning.

`FullSupportBarDistribution` below: half-normal tail extrapolation beyond
the outermost borders, PORTED (2026-09-16, reassessing a call originally
deferred in `ppfn.model.registration.heads.TailBarDistribution`'s own
docstring) after empirical evidence that a FIXED, symmetric body range
(that class's own `[-4, 4]`) is a poor match for `ppfn.prior.lupi`'s raw,
unnormalized targets: most drawn pairs' informative range covers only a
handful of a 64-bin body sized for a much larger assumed scale, giving
visibly blocky, low-resolution predictive densities regardless of training
quality (docs/labbook/, `notebooks/id_token_lupi_1d_comparison.ipynb`).
Pair with `quantile_bin_borders` (fit to an actual sample of prior targets)
rather than `uniform_bin_borders` — bins go where the data is, and the
tails absorb whatever the fitted range didn't catch.
"""
import torch
from torch import nn


def uniform_bin_borders(n_bins: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    return torch.linspace(lo, hi, n_bins + 1)


def quantile_bin_borders(ys: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Borders spaced by EQUAL PROBABILITY MASS under a sample `ys` of
    actual target draws, not equal width -- ported from PFNs4BO's
    `get_bucket_limits` (`ys` branch only; the fixed-range branch is
    `uniform_bin_borders` above). Puts fine resolution where the data
    actually lives instead of assuming a scale up front -- see
    `FullSupportBarDistribution`'s docstring for why this matters when the
    target has no fixed, known range."""
    ys = ys.flatten()
    ys = ys[~torch.isnan(ys)]
    if len(ys) % n_bins:
        ys = ys[: -(len(ys) % n_bins)]
    ys_per_bucket = len(ys) // n_bins
    ys_sorted, _ = ys.sort(0)
    inner = (ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1] + ys_sorted[ys_per_bucket::ys_per_bucket]) / 2
    return torch.cat([ys_sorted[:1], inner, ys_sorted[-1:]], dim=0)


class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor):
        """borders: 1D, sorted, ascending — bin edges over the support."""
        super().__init__()
        assert borders.dim() == 1 and (borders[1:] >= borders[:-1]).all(), "borders must be sorted"
        self.register_buffer("borders", borders)
        self.register_buffer("bucket_widths", borders[1:] - borders[:-1])
        self.num_bars = len(borders) - 1

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.borders, y.contiguous()) - 1
        idx[y == self.borders[0]] = 0
        idx[y == self.borders[-1]] = self.num_bars - 1
        return idx.clamp(0, self.num_bars - 1)

    def compute_scaled_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """log p(y) of the piecewise-constant density, per bucket."""
        bucket_log_probs = torch.log_softmax(logits, -1)
        return bucket_log_probs - torch.log(self.bucket_widths)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """NLL loss. logits: [..., num_bars], y: [...] (same leading shape)."""
        idx = self.map_to_bucket_idx(y)
        scaled_log_probs = self.compute_scaled_log_probs(logits)
        return -scaled_log_probs.gather(-1, idx[..., None]).squeeze(-1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Closed-form differential entropy H = -sum_i p_i * log(p_i/width_i),
        no Monte Carlo — this is what makes the explore branch's
        entropy-gradient search (M5) a cheap gradient step."""
        p = torch.softmax(logits, -1)
        scaled_log_probs = self.compute_scaled_log_probs(logits)
        return -(p * scaled_log_probs).sum(-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return torch.softmax(logits, -1) @ bucket_means

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        lo, hi = self.borders[:-1], self.borders[1:]
        bucket_mean_sq = (lo.square() + hi.square() + lo * hi) / 3.0
        return torch.softmax(logits, -1) @ bucket_mean_sq

    def variance(self, logits: torch.Tensor) -> torch.Tensor:
        return self.mean_of_square(logits) - self.mean(logits).square()

    def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
        probs = torch.softmax(logits, -1)
        cumprobs = torch.cumsum(probs, -1)
        target = left_prob * torch.ones(*cumprobs.shape[:-1], 1, device=logits.device)
        idx = torch.searchsorted(cumprobs, target).squeeze(-1).clamp(0, self.num_bars - 1)
        cumprobs_padded = torch.cat([torch.zeros(*cumprobs.shape[:-1], 1, device=logits.device), cumprobs], -1)
        rest_prob = left_prob - cumprobs_padded.gather(-1, idx[..., None]).squeeze(-1)
        lo, hi = self.borders[idx], self.borders[idx + 1]
        return lo + (hi - lo) * rest_prob / probs.gather(-1, idx[..., None]).squeeze(-1)

    def median(self, logits: torch.Tensor) -> torch.Tensor:
        return self.icdf(logits, 0.5)

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        return bucket_means[logits.argmax(-1)]

    def quantile(self, logits: torch.Tensor, center_prob: float = 0.682) -> torch.Tensor:
        """[lo, hi] interval containing `center_prob` probability mass
        around the median. Ported from PFNs4BO's own `BarDistribution.quantile`
        (`archive/src/utils/bar_distribution.py`) — direction-agnostic (an
        interval, not an improvement), no min/max mirroring needed.
        -> [..., 2] (logits.shape[:-1] + [2])."""
        side_prob = (1.0 - center_prob) / 2
        return torch.stack((self.icdf(logits, side_prob), self.icdf(logits, 1.0 - side_prob)), dim=-1)

    def ucb(self, logits: torch.Tensor, rest_prob: float = (1 - 0.682) / 2) -> torch.Tensor:
        """Optimistic-for-minimization confidence bound: the `rest_prob`
        lower quantile (the project minimizes throughout — this is the
        mirror of PFNs4BO's own `ucb`, which for *maximization* takes the
        `1 - rest_prob` *upper* quantile via the same `icdf`; swapping which
        tail is "optimistic" is the same mirroring `ei`/`pi` below use).
        `rest_prob=(1-0.682)/2` (the default, matching the reference) is the
        amount of density beyond the confidence bound being ignored,
        equivalent to GP-UCB/LCB with `beta=1`."""
        return self.icdf(logits, rest_prob)

    def ei(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        """Closed-form `E[max(best_f - Y, 0)]` under the piecewise-uniform
        bar density (this project minimizes) — ported from PFNs4BO's own
        `BarDistribution.ei` (`archive/src/utils/bar_distribution.py`,
        which assumes maximization: `E[max(Y - best_f, 0)]`), algebraically
        mirrored for minimization (swap which border plays the "active"
        role: the reference's `borders[1:]` becomes `borders[:-1]` here).
        Cross-checked against a Monte Carlo estimate in
        `tests/test_bar_distribution.py` — not trusted on the algebra
        alone. logits: [..., n_bins]  best_f: caller must already have
        unsqueezed a trailing singleton dim onto any axis `best_f` doesn't
        vary over (e.g. `incumbent.unsqueeze(-1)` for one threshold shared
        across a grid/query axis) -- `best_f.dim()` must equal
        `logits.dim() - 1` on entry, asserted below. Silently mis-broadcasts
        instead of raising if a caller passes `best_f` one dimension short
        (e.g. `[B]` against `[B, n_query, n_bins]` logits) AND `B` happens
        to equal `n_query` -- discovered 2026-09-08 via exactly that
        coincidence in `notebooks/vla_readout_ei_probe.ipynb`, silently
        computing every candidate's EI against a DIFFERENT, arbitrary
        environment's incumbent instead of its own. The assert below turns
        that into an immediate crash instead of a silent, hard-to-diagnose
        correctness bug."""
        assert best_f.dim() == logits.dim() - 1, (
            f"best_f must have one fewer dim than logits ({logits.dim() - 1}), got {best_f.dim()} "
            f"(best_f.shape={tuple(best_f.shape)}, logits.shape={tuple(logits.shape)}) -- "
            "unsqueeze a trailing singleton dim onto any axis best_f doesn't vary over, "
            "e.g. best_f.unsqueeze(-1) for one threshold shared across a grid/query axis"
        )
        lo, hi = self.borders[:-1], self.borders[1:]
        inc = best_f.unsqueeze(-1)  # [..., 1], broadcasts against the n_bins axis
        clamped = inc.clamp(lo, hi)  # [..., n_bins]
        bucket_contributions = (inc * (clamped - lo) - (clamped**2 - lo**2) / 2) / self.bucket_widths
        p = torch.softmax(logits, -1)
        return (p * bucket_contributions).sum(-1)

    def pi(self, logits: torch.Tensor, best_f: torch.Tensor) -> torch.Tensor:
        """Closed-form `P(Y < best_f)` under the piecewise-uniform bar
        density (this project minimizes) — ported from PFNs4BO's own
        `BarDistribution.pi` (assumes maximization: `P(Y > best_f)`),
        mirrored the same way `ei` above is. Same clamped-bucket trick.
        logits: [..., n_bins]  best_f: same calling convention as `ei`
        above (`best_f.dim()` must equal `logits.dim() - 1`) -- see that
        method's docstring for why this is asserted rather than assumed."""
        assert best_f.dim() == logits.dim() - 1, (
            f"best_f must have one fewer dim than logits ({logits.dim() - 1}), got {best_f.dim()} "
            f"(best_f.shape={tuple(best_f.shape)}, logits.shape={tuple(logits.shape)}) -- "
            "unsqueeze a trailing singleton dim onto any axis best_f doesn't vary over"
        )
        lo, hi = self.borders[:-1], self.borders[1:]
        thr = best_f.unsqueeze(-1)
        clamped = thr.clamp(lo, hi)
        bucket_cdf = (clamped - lo) / self.bucket_widths
        p = torch.softmax(logits, -1)
        return (p * bucket_cdf).sum(-1)


class FullSupportBarDistribution(BarDistribution):
    """Same piecewise-uniform body as `BarDistribution`, but the two
    outermost buckets are replaced by half-normal tails extrapolating to
    +-infinity instead of hard-clipping density at the outer borders --
    ported from PFNs4BO's own `FullSupportBarDistribution`
    (`pfns4bo/bar_distribution.py`). `halfnormal_with_p_weight_before(w, p)`
    picks the half-normal's scale so exactly `p` of ITS OWN mass falls
    within one more bucket-width `w` past the border (`p=0.5`, matching the
    reference) -- the softmax's own outer-bucket probability then becomes
    that tail's TOTAL mass, body and tail sharing one normalizer.

    See this module's own docstring for why this pairs with
    `quantile_bin_borders` rather than `uniform_bin_borders`.

    `mean_of_square`'s outer-bucket correction fixes a units bug in the
    reference (`side_normals[1].variance + (side_normals[1].variance +
    borders[-2]).square()` — squaring a variance, dimensionally
    inconsistent with the `[0]`-bucket branch two lines above it, which
    correctly squares a MEAN); this uses `.mean` there instead, matching
    the `[0]` branch's own structure. Not ported: `icdf`/`quantile`/`ucb`/
    `ei`/`pi` (inherited from `BarDistribution` unchanged) treat every
    bucket as piecewise-uniform, including the outermost two -- for those
    inherited methods this is an approximation in the tail region (probability
    mass is right, but the within-bucket answer isn't the continuous
    half-normal shape). Not needed anywhere this class is currently used
    (mean + NLL only); reintroduce the reference's continuous-tail
    versions if that changes."""

    @staticmethod
    def halfnormal_with_p_weight_before(range_max: torch.Tensor, p: float = 0.5):
        s = range_max / torch.distributions.HalfNormal(torch.tensor(1.0)).icdf(torch.tensor(p))
        return torch.distributions.HalfNormal(s)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """NLL loss -- same contract as `BarDistribution.forward`, except
        the two outermost buckets' log-density is corrected by the
        half-normal tail's own log-density beyond the border (plain
        piecewise-uniform density would otherwise hard-clip mass at the
        border, which is exactly what makes `BarDistribution` bounded)."""
        idx = self.map_to_bucket_idx(y)
        log_probs = self.compute_scaled_log_probs(logits).gather(-1, idx[..., None]).squeeze(-1)

        left_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[0])
        right_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[-1])
        left_mask = idx == 0
        right_mask = idx == self.num_bars - 1

        log_probs = log_probs.clone()
        log_probs[left_mask] = log_probs[left_mask] + left_normal.log_prob(
            (self.borders[1] - y[left_mask]).clamp_min(1e-8)
        ) + torch.log(self.bucket_widths[0])
        log_probs[right_mask] = log_probs[right_mask] + right_normal.log_prob(
            (y[right_mask] - self.borders[-2]).clamp_min(1e-8)
        ) + torch.log(self.bucket_widths[-1])
        return -log_probs

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        left_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[0])
        right_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[-1])
        bucket_means[0] = self.borders[1] - left_normal.mean
        bucket_means[-1] = self.borders[-2] + right_normal.mean
        return torch.softmax(logits, -1) @ bucket_means

    def mean_of_square(self, logits: torch.Tensor) -> torch.Tensor:
        lo, hi = self.borders[:-1], self.borders[1:]
        bucket_mean_sq = (lo.square() + hi.square() + lo * hi) / 3.0
        left_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[0])
        right_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[-1])
        bucket_mean_sq[0] = left_normal.variance + (self.borders[1] - left_normal.mean).square()
        bucket_mean_sq[-1] = right_normal.variance + (self.borders[-2] + right_normal.mean).square()
        return torch.softmax(logits, -1) @ bucket_mean_sq


if __name__ == "__main__":
    torch.manual_seed(0)
    bd = BarDistribution(uniform_bin_borders(n_bins=64))

    # Analytic check: uniform logits -> uniform density over [0,1] ->
    # entropy should match the analytic differential entropy of U(0,1), 0.0.
    uniform_logits = torch.zeros(5, 64)
    h = bd.entropy(uniform_logits)
    print("entropy of a uniform bar distribution (expect ~0.0):", h.tolist())

    # A confident (near-delta) distribution should have much lower (very
    # negative) entropy than the uniform case.
    confident_logits = torch.full((1, 64), -10.0)
    confident_logits[0, 5] = 20.0
    print("entropy of a confident bar distribution (expect << 0):", bd.entropy(confident_logits).item())

    y = torch.rand(5)
    nll = bd(uniform_logits, y)
    print("NLL under uniform logits (expect ~0, since density=1 everywhere):", nll.tolist())

    print("mean under uniform logits (expect ~0.5):", bd.mean(uniform_logits).tolist())

    # ei/pi/ucb/quantile sanity, under a confident distribution centered
    # near bucket 5's midpoint (~0.5/64*5+... roughly 0.086) -- best_f well
    # above that point should have EI close to (best_f - mode) and PI close
    # to 1 (the distribution's mass is almost certainly below best_f);
    # best_f well below it should have both close to 0.
    mode_val = bd.mode(confident_logits).item()
    print(f"mode of the confident distribution: {mode_val:.4f}")
    print("EI at best_f=0.9 (expect ~0.9 - mode, large):", bd.ei(confident_logits, torch.tensor([0.9])).item())
    print("EI at best_f=0.01 (expect ~0.0, mode is above it):", bd.ei(confident_logits, torch.tensor([0.01])).item())
    print("PI at best_f=0.9 (expect ~1.0):", bd.pi(confident_logits, torch.tensor([0.9])).item())
    print("PI at best_f=0.01 (expect ~0.0):", bd.pi(confident_logits, torch.tensor([0.01])).item())
    print("68.2% quantile interval under uniform logits (expect ~[0.159, 0.841]):",
          bd.quantile(uniform_logits[:1]).tolist())
    print("ucb (optimistic-for-min lower quantile) under uniform logits (expect ~0.159):",
          bd.ucb(uniform_logits[:1]).tolist())

    # --- FullSupportBarDistribution / quantile_bin_borders ---
    ys = torch.cat([torch.randn(9990) * 0.4 + 0.2, torch.randn(10) * 0.4 + 20.0])  # a few outliers
    borders = quantile_bin_borders(ys, n_bins=16)
    print("\nquantile-fit borders (expect dense near 0.2, sparse near the outliers):", borders.tolist())

    # NOTE: uniform logits do NOT reproduce ys.mean() here -- bucket widths
    # are unequal by construction (quantile-fit), and the outer bucket's
    # mean is the half-normal tail mean (pulled well past its naive
    # midpoint by the 10 outliers widening that bucket), not a plain
    # average -- this is the correctly weighted mean of a MODEL that
    # (wrongly) predicts every bucket equally likely, not a check that the
    # borders reproduce the sample mean.
    full_bd = FullSupportBarDistribution(borders)
    uniform_logits_16 = torch.zeros(3, 16)
    print("FullSupportBarDistribution mean under uniform (mis-specified) logits:",
          full_bd.mean(uniform_logits_16).tolist(), " (for reference, ys.mean():", ys.mean().item(), ")")

    # A y far beyond the fitted range must still get FINITE NLL (the whole
    # point of the half-normal tail vs. BarDistribution's hard clip).
    y_far = torch.tensor([1000.0, -1000.0, ys.mean().item()])
    confident_16 = torch.full((3, 16), -10.0)
    confident_16[:, 8] = 20.0  # a confident mid-body bucket
    nll_far = full_bd(confident_16, y_far)
    print("NLL at y=[1000, -1000, near-mode] under a confident mid-body distribution (expect finite, large, large, small):",
          nll_far.tolist())
    assert torch.isfinite(nll_far).all(), "FullSupportBarDistribution must give finite NLL everywhere on R"
