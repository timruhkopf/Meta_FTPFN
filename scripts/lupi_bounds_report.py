"""The "standing view": lower/student/oracle/upper NLL together, from two
independently-trained checkpoints (LUPIPFN + BoundsPFN) evaluated on ONE
shared validation batch -- ppfn.monitor.lupi.compute_bounds_report. Prints
the bracket to console and logs it as a small, separate MLflow run in the
lupi-alignment-pfn experiment (not attached to either training run's own
history, since it depends on BOTH checkpoints and is a point-in-time
comparison, not a training-time metric) so it shows up in the MLflow UI
next to the training runs it brackets.

Usage:
    uv run python scripts/lupi_bounds_report.py \
        <lupi_checkpoint.pt> <bounds_checkpoint.pt> [--tracking-uri URI]
"""

from __future__ import annotations

import argparse
import dataclasses

import mlflow
import numpy as np
import torch

from ppfn.model.baselines.lupi_bounds_pfn import BoundsPFN
from ppfn.model.lupi.model import LUPIPFN
from ppfn.monitor.lupi import compute_bounds_report
from ppfn.prior.lupi.dataset import LUPIBatch, build_training_item, collate_lupi_batch


def _move_batch(batch: LUPIBatch, device: torch.device) -> LUPIBatch:
    moved = {
        f.name: (
            getattr(batch, f.name).to(device)
            if isinstance(getattr(batch, f.name), torch.Tensor)
            else getattr(batch, f.name)
        )
        for f in dataclasses.fields(batch)
    }
    return LUPIBatch(**moved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lupi_checkpoint")
    parser.add_argument("bounds_checkpoint")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="lupi-alignment-pfn")
    parser.add_argument("--run-name", default="standing-view-bounds-report")
    parser.add_argument("--val-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Real-run defaults, matching configs/model/lupi_pfn.yaml / lupi_bounds_pfn.yaml.
    lupi_model = LUPIPFN(
        d_model=256, n_heads=8, n_layers_enc_b=3, n_layers_align=6,
        d_ff=512, n_bins_predictive=64, dropout=0.0,
    ).to(device)
    lupi_ckpt = torch.load(args.lupi_checkpoint, map_location=device, weights_only=False)
    lupi_model.load_state_dict(lupi_ckpt["model_state_dict"])
    lupi_model.eval()

    bounds_model = BoundsPFN(
        d_model=256, n_heads=8, n_layers=6, d_ff=512, n_bins_predictive=64, dropout=0.0,
    ).to(device)
    bounds_ckpt = torch.load(args.bounds_checkpoint, map_location=device, weights_only=False)
    bounds_model.load_state_dict(bounds_ckpt["model_state_dict"])
    bounds_model.eval()

    print(f"lupi checkpoint: epoch={lupi_ckpt.get('epoch')} best_score={lupi_ckpt.get('best_score')}")
    print(f"bounds checkpoint: epoch={bounds_ckpt.get('epoch')} best_score={bounds_ckpt.get('best_score')}")

    # Matches configs/experiment/lupi_baseline.yaml / lupi_bounds.yaml's
    # prior ranges -- one shared batch, so all four numbers are on
    # identical data.
    val_rng = np.random.default_rng(999_999)
    val_items = [
        build_training_item(
            val_rng, progress=1.0, s_max=0.1, force_rho_zero=False,
            n_a_range=(8, 100), n_b_range=(8, 100), n_qry_range=(8, 64),
        )
        for _ in range(args.val_size)
    ]
    batch = _move_batch(collate_lupi_batch(val_items), device)

    with torch.no_grad():
        report = compute_bounds_report(lupi_model, bounds_model, batch)

    print(f"\n{'metric':40s} value")
    for k, v in report.items():
        print(f"{k:40s} {v}")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "lupi_checkpoint": args.lupi_checkpoint,
            "bounds_checkpoint": args.bounds_checkpoint,
            "lupi_epoch": lupi_ckpt.get("epoch"),
            "bounds_epoch": bounds_ckpt.get("epoch"),
            "val_size": args.val_size,
        })
        mlflow.log_metrics({k: v for k, v in report.items() if v is not None})
    print(f"\nlogged to MLflow experiment {args.experiment_name!r}, run {args.run_name!r}")


if __name__ == "__main__":
    main()
