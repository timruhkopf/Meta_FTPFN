"""Fit y-correction + x-warp ladder between two tasks' config-space surfaces,
at a single fidelity slice (the "classic BO" analysis -- use the max-fidelity
slice; see `fidelity_warp.py` for the multi-fidelity axis itself).

Method, per `docs/experiments/hpo-warp-complexity.md`:

1. y-distortion first: `h` fit on the *identity* correspondence (the grid
   is shared across tasks -- ARCHITECTURE.md invariant #8 style declared-
   space alignment, no warp needed to compare at the same x), held fixed.
2. x-distortion: fit `T` up the capacity ladder (affine -> per-axis monotone
   -> velocity field) against that fixed `h`, by treating B's surface as a
   continuously-interpolated "image" (`interp.py`) and warping A's sampling
   coordinates into it -- this is exactly 2D diffeomorphic image
   registration (cf. LDDMM/free-form-deformation registration), with the
   "image intensity" being validation loss instead of pixel brightness.
3. Two complexity numbers, kept separate per ROADMAP.md §4.2: severity
   (realized log|det J| band of the elbow-capacity fit) and shape-complexity
   (the DOF at which held-out R² stops improving).
4. Diagnose absence of a diffeomorphism: compare the best achievable
   held-out residual to the seed-replicate noise floor (residual ~ floor =>
   fully explained; residual >> floor => genuine heterogeneity beyond any
   warp), and report the fold fraction of the elbow-capacity fit (a fold
   means the fitted map isn't a diffeomorphism there at all).

**`h` is a 3-parameter Yeo-Johnson-plus-affine transform, not a fully
flexible isotonic fit.** This replaced an isotonic `h` after two rounds of
finding out the hard way that "constrain almost nothing, refit more" wasn't
converging on the right answer:

- *First attempt*: isotonic `h` fit once, frozen, `T` fit once. Looked
  reasonable, wasn't: on a synthetic pair with a KNOWN true warp (severity
  2.78), this recovered severity as low as **0.36** -- off by ~8x. Cause: an
  unconstrained monotone `h`, fit at the *raw* identity correspondence, can
  absorb a large share of what a real spatial warp should explain --
  especially for a smooth function, where "warped inputs" and "rescaled
  outputs" can look similar from the y-only vantage point (the same
  phenomenon `docs/ROADMAP.md` §3.3 names for the dataset-ID-token result).
- *Second attempt*: keep isotonic `h`, but alternate -- refit `h` against
  the current `T`'s pushforward, refit `T`, repeat (Alternating Conditional
  Expectations, Breiman & Friedman 1985). This measurably helped (mean
  recovered severity rose from ~0.4 to ~1.8 over a few rounds) but did not
  fix the underlying problem, only reduced it: `h` still had far more
  representational freedom than any real y-distortion phenomenon needs, so
  it could still partially re-absorb `T`'s signal each round, and results
  stayed noisy round to round.
- *What actually fixes it*: constrain `h`'s **family**, not just how many
  times it gets refit. Real y-distortion between two HPO tasks' surfaces is,
  as far as anyone has a mechanism for, a global shift/scale (different
  metric ranges, different Bayes-error floors, sign conventions) plus at
  most one global shape parameter (boundedness/skew -- the same reason
  Box-Cox/Yeo-Johnson transforms are standard for GP-based BO surrogates,
  e.g. HEBO). A 3-parameter family has no mechanism to encode a
  *spatially-varying* pattern the way a many-knot isotonic fit can, so it
  structurally cannot re-absorb `T`'s job. Checked: with this `h`, a single
  pass (no alternation) recovers severity within the same order of
  magnitude as the truth across multiple synthetic draws (e.g. true 2.78,
  recovered 3.34; true 1.87, recovered 1.23; true 2.97, recovered 2.89) --
  no alternation needed, and it's cheaper than the alternation it replaced.

`h(y) = a * YeoJohnson(y; lambda) + b` -- `YeoJohnson(y; 1) = y` for every
`y` (verified), so `lambda=1` is pure affine, the "must-have" floor (see
the docstring on `yeo_johnson_np`).

**Multiple restarts remain, for a separate reason.** Zero-initialized
velocity fields can get stuck very close to the identity depending on
random center placement, independent of the `h`/`T` identifiability issue
above -- checked: 3000 optimization steps from one bad init converged to a
*worse* fit (and near-zero severity) than 300 steps from a good one, so
"train longer" doesn't fix it. `n_restarts` freshly-initialized fits,
selected by *training* loss (not held-out, to avoid using the test set for
model selection on top of evaluation), does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from ppfn.experiments.hpo_warp.artifacts import FitArtifacts
from ppfn.experiments.hpo_warp.interp import clamp01
from ppfn.experiments.hpo_warp.surface import build_config_reader
from ppfn.experiments.hpo_warp.task_grid import TaskGrid
from ppfn.experiments.hpo_warp.warps import build_warp, displacement_volume, logdet_jacobian_band

RUNG_SPLINE_K = (2, 4, 8, 16)
RUNG_VF_M = (2, 4, 8, 16)
ELBOW_NOISE_MARGIN = 0.25  # elbow = smallest capacity within 25% of the noise floor
ELBOW_FALLBACK_MARGIN = 0.05  # ...or, if the floor is unreachable, within 5% of the best MSE seen
YJ_LAMBDA_GRID: tuple[float, ...] = tuple(np.linspace(-2.0, 3.0, 51))  # scan range for fitting h's shape parameter
BOUNDARY_PENALTY_WEIGHT = 1.0  # penalizes raw warp output outside [0,1]^d; see _train_once


@dataclass
class FitResult:
    model: str
    task_a: str
    task_b: str
    n_grid: int
    y_only_r2: float
    h_lambda: float = 1.0  # fitted Yeo-Johnson shape parameter; 1.0 = pure affine, no shape distortion
    h_scale: float = 1.0
    h_shift: float = 0.0
    spearman_pearson_gap: float = float("nan")  # rho - r on raw (y_a, y_b); cheap nonlinearity screen
    ladder_r2: dict[str, float] = field(default_factory=dict)  # rung name -> held-out R^2
    severity_logdet_band: float = float("nan")  # local-distortion severity; 0 for any global affine
    displacement_volume: float = float("nan")  # mean |T(x)-x|; complementary to the band, see warps.py
    bending_energy: float = float("nan")  # closed-form RKHS norm of the elbow velocity field
    shape_complexity_m: int = 0  # elbow M for the velocity field
    shape_complexity_k: int = 0  # elbow K for the per-axis spline
    fold_fraction: float = 0.0  # at the elbow-capacity velocity field
    residual_to_noise_floor: float = float("nan")  # best-fit MSE / seed-noise floor
    noise_floor: float = float("nan")


def _held_out_split(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(n * frac))
    return idx[n_test:], idx[:n_test]


def yeo_johnson_np(y: np.ndarray, lam: float) -> np.ndarray:
    """Yeo & Johnson (2000)'s power transform, extending Box-Cox to the
    whole real line by applying it separately to y>=0 and y<0 with a
    mirrored exponent, chosen so BOTH branches reduce to the exact identity
    at lambda=1 (checked: `yeo_johnson_np([-3,-0.5,0,0.5,3], 1.0)` returns
    the input unchanged) -- that nesting is what makes "no shape distortion,
    just affine" one exactly-reachable point in this family rather than an
    edge case. lambda<1 compresses large values (fixes a long right tail,
    e.g. occasional very bad configs); lambda>1 the reverse."""
    y = np.asarray(y, dtype=np.float64)
    out = np.empty_like(y)
    pos = y >= 0
    if abs(lam) > 1e-6:
        out[pos] = ((y[pos] + 1.0) ** lam - 1.0) / lam
    else:
        out[pos] = np.log1p(y[pos])
    neg = ~pos
    if abs(lam - 2.0) > 1e-6:
        out[neg] = -((-y[neg] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)
    else:
        out[neg] = -np.log1p(-y[neg])
    return out


def yeo_johnson_torch(y: torch.Tensor, lam: float) -> torch.Tensor:
    """Same transform as `yeo_johnson_np`, in torch, for use inside the
    differentiable warp-fitting loop (`lam` is fixed here, not learned --
    only `T`'s parameters get gradients through this). Both branches are
    evaluated on inputs clamped to be always-valid (>=1) before selecting
    via `torch.where`, so no NaN/inf from the unused branch can appear or
    poison gradients -- `torch.where` alone does not protect against that,
    since it still evaluates both arguments."""
    pos_mask = y >= 0
    yp = torch.clamp(y, min=0.0) + 1.0
    yn = torch.clamp(-y, min=0.0) + 1.0
    pos_val = (yp.pow(lam) - 1.0) / lam if abs(lam) > 1e-6 else torch.log(yp)
    neg_val = -(yn.pow(2.0 - lam) - 1.0) / (2.0 - lam) if abs(lam - 2.0) > 1e-6 else -torch.log(yn)
    return torch.where(pos_mask, pos_val, neg_val)


def _apply_h(y: torch.Tensor, h_lambda: float, h_scale: float, h_shift: float) -> torch.Tensor:
    return h_scale * yeo_johnson_torch(y, h_lambda) + h_shift


def _fit_yeojohnson_h(
    y_target_matched: np.ndarray, y_source_matched: np.ndarray, h_train_idx: np.ndarray, h_test_idx: np.ndarray
) -> tuple[float, float, float, float]:
    """Fit `h(y) = a*YeoJohnson(y;lambda) + b` by a grid scan over `lambda`
    (`YJ_LAMBDA_GRID`) with closed-form ordinary-least-squares for `(a,b)`
    at each candidate -- for fixed `lambda`, `y_target ~ a*YeoJohnson(y_source;
    lambda) + b` is plain linear regression, solved exactly, so the whole
    fit has no local-optima risk and needs no gradient descent (the same
    grid-scan-plus-closed-form-inner-solve shape as `fidelity_warp.py`'s
    `tau` search). Returns `(lambda, a, b, held_out_r2)`."""
    best = None
    y_tr, y_te = y_target_matched[h_train_idx], y_target_matched[h_test_idx]
    x_tr, x_te = y_source_matched[h_train_idx], y_source_matched[h_test_idx]
    for lam in YJ_LAMBDA_GRID:
        z_tr = yeo_johnson_np(x_tr, lam)
        design = np.stack([z_tr, np.ones_like(z_tr)], axis=1)
        coef, *_ = np.linalg.lstsq(design, y_tr, rcond=None)
        sse = float(np.sum((y_tr - design @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, lam, float(coef[0]), float(coef[1]))
    _, lam, a, b = best

    z_te = yeo_johnson_np(x_te, lam)
    pred_te = a * z_te + b
    ss_res = np.sum((y_te - pred_te) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(lam), float(a), float(b), float(r2)


def _train_once(
    warp: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    b_reader,
    h_lambda: float,
    h_scale: float,
    h_shift: float,
    steps: int,
    lr: float,
    max_batch: int,
    restart_seed: int,
) -> float:
    """One optimization run from `warp`'s current (fresh) initialization;
    mutates `warp` in place and returns its final **full-train-set** MSE
    (not the last minibatch's, which is a noisy proxy) -- the criterion
    `_fit_warp_module` uses to pick the best of several restarts."""
    opt = torch.optim.Adam(warp.parameters(), lr=lr)
    batch_size = min(max_batch, x_train.shape[0])
    rng = np.random.default_rng(restart_seed)

    for _ in range(steps):
        opt.zero_grad()
        batch_idx = rng.choice(x_train.shape[0], size=batch_size, replace=False)
        x_batch, y_batch = x_train[batch_idx], y_train[batch_idx]
        raw_t_x = warp(x_batch)
        t_x = clamp01(raw_t_x)
        y_b_at_t = b_reader(t_x)
        pred = _apply_h(y_b_at_t, h_lambda, h_scale, h_shift)
        fit_loss = ((y_batch - pred) ** 2).mean()
        # `clamp01` makes the loss exactly flat for any raw output already
        # outside [0,1] -- moving further out changes nothing it's graded
        # on, so nothing shapes what the raw field does past the edge. Found
        # by inspecting the synthetic case study's warp-displacement plot,
        # which shows the RAW (unclamped) output and visibly spikes near
        # boundaries; checked, not just eyeballed: 6.6% of a 3000-point
        # random probe have raw output outside [0,1]^2, and near-boundary
        # points show ~2x the average displacement of interior ones --
        # exactly what inflates `severity_logdet_band` (a max-min range,
        # so a few boundary outliers move it disproportionately). This term
        # gives the raw field a reason to stay in-bounds even though
        # `clamp01` would otherwise mask the alternative.
        boundary_loss = ((raw_t_x - t_x) ** 2).mean()
        loss = fit_loss + BOUNDARY_PENALTY_WEIGHT * boundary_loss
        loss.backward()
        opt.step()

    with torch.no_grad():
        t_x_train = clamp01(warp(x_train))
        pred_train = _apply_h(b_reader(t_x_train), h_lambda, h_scale, h_shift)
        train_mse = ((y_train - pred_train) ** 2).mean().item()
    return train_mse


def _fit_warp_module(
    build_warp_fn: Callable[[], torch.nn.Module],
    x_a: torch.Tensor,
    y_a: torch.Tensor,
    b_reader,
    h_lambda: float,
    h_scale: float,
    h_shift: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    steps: int = 300,
    lr: float = 0.05,
    max_batch: int = 512,
    n_restarts: int = 3,
) -> tuple[torch.nn.Module, float, float]:
    """Fit `n_restarts` freshly-initialized copies of `build_warp_fn()`
    (h held fixed), keep the one with the lowest **full-train-set** MSE, and
    return (best module, held-out R^2, held-out MSE) for it. MSE (not just
    R^2) is needed so the elbow can be chosen relative to the seed-replicate
    noise floor, which lives on the same MSE scale.

    Multiple restarts, not just more steps: see the module docstring's
    "Multiple restarts remain, for a separate reason" -- a zero-initialized
    field can get stuck very close to the identity depending on random
    center placement, and more steps alone does not reliably fix it (can
    make it worse). Selection is by *training* loss, not held-out,
    specifically to avoid using the test set for model selection on top of
    evaluation -- checked that a collapsed-to-identity fit is clearly worse
    on training data too, not only held out, so this doesn't just relabel
    the same problem.

    Minibatched (`max_batch` random training points per step, resampled each
    step) rather than full-batch: `lr`'s 2D grid has 625 points and full-batch
    is fine, but `rf`/`xgb`'s 4D grid has ~9000 -- full-batch there would
    make per-step cost scale with grid size for no fitting-quality reason
    (all warp families here have O(10-100) parameters), stretching a sweep
    from minutes to an estimated 10+ hours. Every point is still used, just
    not in every step."""
    x_train, y_train = x_a[train_idx], y_a[train_idx]

    best_warp, best_train_mse = None, float("inf")
    for restart in range(n_restarts):
        torch.manual_seed(restart)
        warp = build_warp_fn()
        train_mse = _train_once(
            warp, x_train, y_train, b_reader, h_lambda, h_scale, h_shift, steps, lr, max_batch, restart
        )
        if train_mse < best_train_mse:
            best_warp, best_train_mse = warp, train_mse
    warp = best_warp

    with torch.no_grad():
        t_x_test = clamp01(warp(x_a[test_idx]))
        y_b_test = b_reader(t_x_test)
        pred_test = _apply_h(y_b_test, h_lambda, h_scale, h_shift)
        y_test = y_a[test_idx]
        mse = ((y_test - pred_test) ** 2).mean().item()
        ss_res = ((y_test - pred_test) ** 2).sum().item()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return warp, r2, mse


def _pick_elbow(capacities: tuple[int, ...], mse_by_cap: dict[int, float], noise_floor: float) -> int:
    """Smallest capacity within `ELBOW_NOISE_MARGIN` of the seed-replicate
    noise floor -- i.e. "good enough that the residual looks like noise" --
    falling back to smallest capacity within `ELBOW_FALLBACK_MARGIN` of the
    best MSE observed in the sweep if the floor is never reached (the
    structural-mismatch case: more capacity stops helping well above the
    noise floor, which is itself evidence for `residual_to_noise_floor`)."""
    best_mse = min(mse_by_cap.values())
    threshold = max(noise_floor * (1.0 + ELBOW_NOISE_MARGIN), best_mse * (1.0 + ELBOW_FALLBACK_MARGIN))
    return next(c for c in capacities if mse_by_cap[c] <= threshold)


def _fit_ladder(
    x_a: torch.Tensor,
    y_a_t: torch.Tensor,
    b_reader,
    h_lambda: float,
    h_scale: float,
    h_shift: float,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    d: int,
    noise_floor: float,
) -> tuple[dict[str, float], dict[str, torch.nn.Module], int, int, str]:
    """Fit the full affine -> spline(K) -> velocity(M) capacity ladder given
    a FIXED `h`. Returns (ladder_r2, all_modules, shape_complexity_k,
    shape_complexity_m, elbow_rung) -- the elbow velocity field itself is
    `all_modules[elbow_rung]`."""
    all_modules: dict[str, torch.nn.Module] = {}
    ladder_r2: dict[str, float] = {}

    def run(name: str) -> tuple[float, float]:
        warp, r2, mse = _fit_warp_module(
            lambda: build_warp(name, d), x_a, y_a_t, b_reader, h_lambda, h_scale, h_shift, train_idx, test_idx
        )
        all_modules[name] = warp
        return r2, mse

    r2_aff, _ = run("affine")
    ladder_r2["affine"] = float(r2_aff)

    spline_fits = {k: run(f"spline_k{k}") for k in RUNG_SPLINE_K}
    for k, (r2, _) in spline_fits.items():
        ladder_r2[f"spline_k{k}"] = float(r2)
    shape_k = _pick_elbow(RUNG_SPLINE_K, {k: mse for k, (_, mse) in spline_fits.items()}, noise_floor)

    vf_fits = {m: run(f"velocity_m{m}") for m in RUNG_VF_M}
    for m, (r2, _) in vf_fits.items():
        ladder_r2[f"velocity_m{m}"] = float(r2)
    shape_m = _pick_elbow(RUNG_VF_M, {m: mse for m, (_, mse) in vf_fits.items()}, noise_floor)

    elbow_rung = f"velocity_m{shape_m}"
    return ladder_r2, all_modules, shape_k, shape_m, elbow_rung


def fit_config_warp(
    grid_a: TaskGrid, grid_b: TaskGrid, iter_idx: int = -1, held_out_frac: float = 0.2, seed: int = 0
) -> tuple[FitResult, FitArtifacts, torch.nn.Module]:
    """Returns (result, artifacts, elbow-capacity velocity field module).
    `artifacts` holds every rung's fitted state_dict plus `h`'s three
    parameters and the exact train/test split -- see `artifacts.py`'s
    module docstring for why: it lets a later session recompute a
    *different* metric on this same fit without re-optimizing. `elbow_vf`
    is returned separately (a live object, not just its state_dict) purely
    as a same-process convenience for `fidelity_warp.py`, which reuses this
    fit rather than refitting the config-space warp from scratch."""
    d = grid_a.x.shape[1]
    n = grid_a.x.shape[0]
    y_a = grid_a.y_mean_by_iter[iter_idx]
    y_b = grid_b.y_mean_by_iter[iter_idx]
    train_idx, test_idx = _held_out_split(n, held_out_frac, seed)
    # y_mean_by_iter (what's being fit) is a MEAN over n_seeds replicate
    # observations, so the sampling floor for it is Var(single obs) /
    # n_seeds, not Var(single obs) itself -- dividing by n_seeds here is
    # what makes `residual_to_noise_floor` centered near 1 for a
    # well-fit pair rather than systematically appearing "better than
    # noise" by a factor of n_seeds. Caught by noise-floor ratios coming
    # out suspiciously < 1 on the first real run -- see docs/labbook/ for
    # the corrected-vs-first-attempt record.
    noise_floor = float(np.mean(grid_a.y_var_by_iter[iter_idx])) / grid_a.n_seeds

    # The identity/h step needs y_a(x_i) and y_b(x_i) at the literal same x_i
    # -- true by construction for HPOBench's exact grid and for LCBench, but
    # NOT for TaskSet: two tasks under "the same" 1000-config wide-grid
    # design can each have dropped a different subset of configs whose
    # curve was incomplete (divergence), so `grid_a.x`/`grid_b.x` can differ
    # in length and order. `ids` (the design's own identifiers, see
    # `task_grid.py`) is what finds the actual overlap -- caught by
    # `spearmanr` crashing on mismatched array lengths on a real TaskSet
    # pair (1000 vs 555 surviving configs) before this existed.
    common_ids, idx_a, idx_b = np.intersect1d(grid_a.ids, grid_b.ids, return_indices=True)
    if len(common_ids) < 20:
        raise ValueError(
            f"only {len(common_ids)} matched configs between {grid_a.task_id!r} and {grid_b.task_id!r}"
        )
    y_a_matched, y_b_matched = y_a[idx_a], y_b[idx_b]
    h_train_idx, h_test_idx = _held_out_split(len(common_ids), held_out_frac, seed)

    # Cheap nonlinearity screen, no fitting: rho - r on the raw (matched-x)
    # pair. Pearson is already scale/shift-invariant, so a gap here can only
    # come from a genuinely nonlinear monotone relationship, not from a
    # rescaling h could absorb trivially -- see the assessment in
    # docs/experiments/hpo-warp-complexity.md.
    rho = float(spearmanr(y_a_matched, y_b_matched)[0])
    r_lin = float(pearsonr(y_a_matched, y_b_matched)[0])
    spearman_pearson_gap = rho - r_lin

    # y-only correction: h fit ONCE, at the identity correspondence, then
    # held fixed for the rest of this function -- see the module docstring
    # for why a single pass with this constrained h family replaced the
    # isotonic-plus-alternation approach tried first.
    h_lambda, h_scale, h_shift, y_only_r2 = _fit_yeojohnson_h(y_a_matched, y_b_matched, h_train_idx, h_test_idx)

    result = FitResult(
        model=grid_a.model,
        task_a=grid_a.task_id,
        task_b=grid_b.task_id,
        n_grid=n,
        y_only_r2=float(y_only_r2),
        h_lambda=h_lambda,
        h_scale=h_scale,
        h_shift=h_shift,
        spearman_pearson_gap=spearman_pearson_gap,
        noise_floor=noise_floor,
    )

    x_a = torch.as_tensor(grid_a.x, dtype=torch.float32)
    y_a_t = torch.as_tensor(y_a, dtype=torch.float32)
    b_reader = build_config_reader(grid_b, iter_idx)

    ladder_r2, all_modules, shape_k, shape_m, elbow_rung = _fit_ladder(
        x_a, y_a_t, b_reader, h_lambda, h_scale, h_shift, train_idx, test_idx, d, noise_floor
    )
    result.ladder_r2 = {"identity": float(y_only_r2), **ladder_r2}
    result.shape_complexity_k = shape_k
    result.shape_complexity_m = shape_m

    elbow_vf = all_modules[elbow_rung]
    band, fold_frac = logdet_jacobian_band(elbow_vf, d)
    result.severity_logdet_band = band
    result.fold_fraction = fold_frac
    result.displacement_volume = displacement_volume(elbow_vf, d)
    result.bending_energy = elbow_vf.bending_energy()

    # Step 4: diagnose lack of fit against the seed-replicate noise floor
    # -- undefined (NaN), not "infinitely bad", when no replicate seeds
    # exist at all (LCBench/PD1/TaskSet: n_seeds=1, noise_floor=0).
    with torch.no_grad():
        t_x_test = clamp01(elbow_vf(x_a[test_idx]))
        y_b_test = b_reader(t_x_test)
        best_pred = _apply_h(y_b_test, h_lambda, h_scale, h_shift).numpy()
    best_mse = float(np.mean((y_a[test_idx] - best_pred) ** 2))
    result.residual_to_noise_floor = best_mse / noise_floor if noise_floor > 0 else float("nan")

    artifacts = FitArtifacts(
        d=d,
        h_lambda=h_lambda,
        h_scale=h_scale,
        h_shift=h_shift,
        train_idx=train_idx,
        test_idx=test_idx,
        state_dicts={name: m.state_dict() for name, m in all_modules.items()},
        elbow_rung=elbow_rung,
    )

    return result, artifacts, elbow_vf


def result_to_dict(r: FitResult) -> dict[str, Any]:
    return {
        "model": r.model,
        "task_a": r.task_a,
        "task_b": r.task_b,
        "n_grid": r.n_grid,
        "y_only_r2": r.y_only_r2,
        "h_lambda": r.h_lambda,
        "h_scale": r.h_scale,
        "h_shift": r.h_shift,
        "spearman_pearson_gap": r.spearman_pearson_gap,
        "ladder_r2": r.ladder_r2,
        "severity_logdet_band": r.severity_logdet_band,
        "displacement_volume": r.displacement_volume,
        "bending_energy": r.bending_energy,
        "shape_complexity_m": r.shape_complexity_m,
        "shape_complexity_k": r.shape_complexity_k,
        "fold_fraction": r.fold_fraction,
        "residual_to_noise_floor": r.residual_to_noise_floor,
        "noise_floor": r.noise_floor,
    }
