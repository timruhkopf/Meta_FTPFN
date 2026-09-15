"""Ad-hoc diagnostic: load a LUPIPFN checkpoint, rebuild the exact same
fixed validation batch LUPITrainer would build (same seed/ranges as
configs/experiment/lupi_baseline.yaml), and run
ppfn.monitor.lupi's query-source and rho-stratified breakdown against it.

Not wired into training -- this is a point-in-time checkpoint inspection,
run manually. Usage:
    uv run python scripts/lupi_eval_checkpoint.py <checkpoint.pt>
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import torch

from ppfn.model.lupi.model import LUPIPFN
from ppfn.monitor.lupi import LUPIMonitorContext, compute_all_lupi_monitors
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


def main(ckpt_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Matches configs/model/lupi_pfn.yaml's real-run defaults.
    model = LUPIPFN(
        d_model=256, n_heads=8, n_layers_enc_b=3, n_layers_align=6,
        d_ff=512, n_bins_predictive=64, dropout=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: epoch={ckpt.get('epoch')} best_score={ckpt.get('best_score')}")

    # Matches configs/experiment/lupi_baseline.yaml + configs/trainer/lupi.yaml's
    # val_seed/val_size/val_s_max/val_force_rho_zero and prior.n_a_range/
    # n_b_range/n_qry_range -- byte-for-byte the same batch LUPITrainer built
    # at training time (build_training_item's RNG draw order is unaffected by
    # the qry_source field added since -- it was already computed, just not
    # exposed until now).
    val_rng = np.random.default_rng(999_999)
    val_items = [
        build_training_item(
            val_rng, progress=1.0, s_max=0.1, force_rho_zero=False,
            n_a_range=(8, 100), n_b_range=(8, 100), n_qry_range=(8, 64),
        )
        for _ in range(256)  # oversample vs. training's val_size=64 for less-noisy per-bucket stats
    ]
    batch = _move_batch(collate_lupi_batch(val_items), device)

    ctx = LUPIMonitorContext(model=model, val_batch=batch)
    with torch.no_grad():
        out = compute_all_lupi_monitors(ctx)

    print(f"\n{'metric':40s} value")
    for k, v in sorted(out.items()):
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main(sys.argv[1])
