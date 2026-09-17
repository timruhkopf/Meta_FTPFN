"""1D comparison plot: severed lower bound / pooled oracle upper bound
(both from a `BoundsPFN` checkpoint, `ppfn.model.baselines.lupi_bounds_pfn`)
against this branch's own `FlowMatchingRegistrationPFN` student pathway
(the deployable one, registering B via v_phi's own SDE-integrated estimate
rather than the true privileged `B_inA`) -- the 3-column comparison
deferred in docs/labbook/2026-09-15-lupi-bounds-and-stratified-diagnostics.md
("needs the bounds-PFN checkpoint to exist for columns 1 and 4"), now with
this branch's own model standing in for the LUPIPFN student/oracle columns
that entry originally planned.

Both checkpoints must have been trained with the SAME `bounded01` prior
setting the draw below uses (checked against each checkpoint's saved
metrics where available) -- loading a pre-bounded01 checkpoint here would
silently score real [0,1]-scale grid points against a model whose bar
distribution was calibrated for a completely different scale
(.claude/rules/checkpoints.md's own "prior/checkpoint provenance" concern).

Usage:
    uv run python scripts/plot_bounds_vs_flow_matching_registration_1d.py \
        [--bounds-ckpt PATH] [--registration-ckpt PATH] [--out PATH] [--seed N]

Either checkpoint path may be omitted, in which case that model is used
FRESH (randomly initialized) -- useless as a research result, but lets the
plotting mechanics themselves be smoke-tested before any real checkpoint
exists (this repo's own "verify the diagnostic before trusting it on the
real result" convention, applied to a script rather than a training run).
"""

from __future__ import annotations

import argparse
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import torch

from ppfn.model.baselines.lupi_bounds_pfn import BoundsPFN
from ppfn.model.baselines.lupi_flow_matching_pfn import FlowMatchingRegistrationPFN
from ppfn.model.pfn.bar_distribution import FullSupportBarDistribution
from ppfn.prior.lupi.dataset import D_MAX, LUPIBatch, collate_lupi_batch
from ppfn.prior.lupi.sampler import sample_pair

# Matches configs/model/lupi_bounds_pfn.yaml / lupi_flow_matching_registration.yaml's
# real-run sizing -- hardcoded here rather than composed via Hydra, same
# convention as scripts/lupi_eval_checkpoint.py's own model reconstruction.
BOUNDS_KW = dict(d_max=D_MAX, d_model=256, n_heads=8, n_layers=6, d_ff=512, n_bins_predictive=64, bounded01=True)
REGISTRATION_KW = dict(
    d_max=D_MAX, d_model=256, n_heads=8, n_layers=6, d_ff=512, n_bins_predictive=64,
    fm_d_model=256, fm_n_heads=8, fm_n_layers=6, fm_d_ff=512,
    n_transport_samples=1, sde_sigma=0.0, n_integration_steps=10, bounded01=True,
)


def _load(model: torch.nn.Module, ckpt_path: str | None, device: torch.device) -> torch.nn.Module:
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded {ckpt_path}: epoch={ckpt.get('epoch')} best_score={ckpt.get('best_score')}")
    else:
        print("no checkpoint given -- using FRESH (untrained) weights, smoke-test only")
    return model.to(device).eval()


def _build_grid_batch(batch: LUPIBatch, x_grid: np.ndarray, device: torch.device) -> LUPIBatch:
    """Replaces dec_qry_x/dec_qry_mask with a dense grid over A's domain --
    dec_qry_z is untouched (unused by any model's forward(), only by a
    loss), every other field is untouched. See ppfn.model.baselines.lupi_bounds_pfn.BoundsPFN.forward
    and FlowMatchingRegistrationPFN.forward: both read predictions ONLY off
    dec_qry_x, so this is the one substitution needed to evaluate on a
    dense regular grid instead of the prior's own irregular query points
    (.claude/rules/research-demos.md's own convention)."""
    n_grid = x_grid.shape[0]
    grid_x = torch.zeros(1, n_grid, D_MAX, dtype=torch.float32, device=device)
    grid_x[0, :, 0] = torch.from_numpy(x_grid.astype(np.float32))
    grid_mask = torch.ones(1, n_grid, dtype=torch.bool, device=device)
    return dataclasses.replace(batch, dec_qry_x=grid_x, dec_qry_mask=grid_mask)


def _heatmap(ax, bar_dist: FullSupportBarDistribution, x_grid: np.ndarray, logits: torch.Tensor, y_lo: float, y_hi: float, n_y: int = 200):
    """logits: [n_grid, n_bins] -> density heatmap over (x_grid, a dense
    value grid), imshow-style. Evaluates the SAME bar_dist's scaled log
    density at n_y dense value points per grid position (not just the
    bucket midpoints) so the half-normal tails render smoothly too."""
    y_vals = torch.linspace(y_lo, y_hi, n_y)
    idx = bar_dist.map_to_bucket_idx(y_vals.unsqueeze(0).expand(logits.shape[0], -1))  # [n_grid, n_y]
    scaled_log_probs = bar_dist.compute_scaled_log_probs(logits)  # [n_grid, n_bins]
    density = scaled_log_probs.gather(-1, idx).exp()  # [n_grid, n_y]
    ax.imshow(
        density.T.numpy(), origin="lower", aspect="auto", cmap="viridis",
        extent=[x_grid.min(), x_grid.max(), y_lo, y_hi],
    )


def main(bounds_ckpt: str | None, registration_ckpt: str | None, out_path: str, seed: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bounds_model = _load(BoundsPFN(**BOUNDS_KW), bounds_ckpt, device)
    registration_model = _load(FlowMatchingRegistrationPFN(**REGISTRATION_KW), registration_ckpt, device)

    # One d=1 draw, matching configs/experiment/lupi_flow_matching_registration.yaml's
    # real-run prior ranges (n_a_range/n_b_range/n_qry_range/warp_grid_n),
    # bounded01=True (must match both checkpoints' own training prior).
    rng = np.random.default_rng(seed)
    pair, internals = sample_pair(
        rng, rho=float(rng.uniform(0.3, 0.9)), d=1, n_a_range=(8, 100), n_b_range=(8, 100),
        n_qry_range=(8, 64), warp_grid_n=4, bounded01=True, return_internals=True,
    )
    item = {
        "d_real": pair.d, "rho": np.float32(pair.rho), "beta": np.float32(pair.beta),
        "enc_x": pair.x_b.astype(np.float32), "enc_z": pair.z_b.astype(np.float32),
        "enc_x_inA": pair.x_b_inA.astype(np.float32), "enc_z_inA": pair.z_b_inA.astype(np.float32),
        "dec_ctx_x": pair.x_a_ctx.astype(np.float32), "dec_ctx_z": pair.z_a_ctx.astype(np.float32),
        "dec_ctx_oracle_bpos": pair.oracle_bpos_a_ctx.astype(np.float32),
        "dec_qry_x": pair.x_a_qry.astype(np.float32), "dec_qry_z": pair.z_a_qry.astype(np.float32),
        "dec_qry_oracle_bpos": pair.oracle_bpos_a_qry.astype(np.float32),
        "dec_qry_source": pair.qry_source,
        "region_type": pair.meta["region_type"], "volume_fraction": pair.meta["volume_fraction"],
    }
    batch = collate_lupi_batch([item])
    batch = dataclasses.replace(batch, **{
        f.name: (getattr(batch, f.name).to(device) if isinstance(getattr(batch, f.name), torch.Tensor) else getattr(batch, f.name))
        for f in dataclasses.fields(batch)
    })

    z_grid = np.linspace(0.0, 1.0, 300).reshape(-1, 1)
    x_grid = internals.to_a(z_grid)[:, 0]
    order = np.argsort(x_grid)
    x_grid, z_grid = x_grid[order], z_grid[order]
    true_y = internals.true_value_as_a(z_grid)

    grid_batch = _build_grid_batch(batch, x_grid, device)

    with torch.no_grad():
        lower_logits = bounds_model(grid_batch, severed=True)["predictive_logits"][0].cpu()
        upper_logits = bounds_model(grid_batch, severed=False)["predictive_logits"][0].cpu()
        model_logits = registration_model(grid_batch)["student_logits"][0].cpu()

    y_lo, y_hi = -0.05, 1.05  # bounded01 -> everything genuinely lives in [0,1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        (axes[0], "lower bound (A alone, severed)", bounds_model.bar_dist, lower_logits, None),
        (axes[1], "upper bound (oracle, true B_inA)", bounds_model.bar_dist, upper_logits,
         (pair.x_b_inA[:, 0], pair.z_b_inA, "B_inA (oracle)")),
        (axes[2], "this thread's model (student, raw B)", registration_model.bar_dist, model_logits,
         (pair.x_b[:, 0], pair.z_b, "B (raw)")),
    ]
    for ax, title, bar_dist, logits, b_scatter in panels:
        _heatmap(ax, bar_dist, x_grid, logits, y_lo, y_hi)
        ax.plot(x_grid, true_y, color="white", lw=1.5, ls="--", label="true f")
        ax.scatter(pair.x_a_ctx[:, 0], pair.z_a_ctx, s=30, color="red", marker="x", label="A context")
        if b_scatter is not None:
            bx, bz, blabel = b_scatter
            ax.scatter(bx, bz, s=6, alpha=0.4, color="orange", label=blabel)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (A's frame)")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("y")
    fig.suptitle(f"d=1, rho={pair.rho:.2f}, beta={pair.beta:.2f}, seed={seed}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bounds-ckpt", default=None)
    parser.add_argument("--registration-ckpt", default=None)
    parser.add_argument("--out", default="/tmp/bounds_vs_flow_matching_registration_1d.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.bounds_ckpt, args.registration_ckpt, args.out, args.seed)
