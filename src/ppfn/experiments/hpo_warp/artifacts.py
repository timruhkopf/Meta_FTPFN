"""Persisting *fitted models*, not just scalar summaries.

The scalar `FitResult` JSON (see `fit.py`) is one reading off the fit under
one choice of metric. The point of this module is that the metric menu is
still being actively assessed (severity vs. shape-complexity vs. bending
energy vs. whatever comes out of the next round of design discussion), so
every rung's fitted warp (not only the elbow one), `h`'s three parameters,
and the exact train/test split are saved alongside the JSON -- letting a
*new* metric be computed later by reloading the fit, with no re-optimization
needed. Only `torch.save`-safe plain data (state_dicts, numpy arrays,
floats) goes in here; no live `nn.Module` object, so this doesn't rot across
a pytorch version bump the way pickling one would.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ppfn.experiments.hpo_warp.warps import build_warp


@dataclass
class FitArtifacts:
    d: int
    h_lambda: float
    h_scale: float
    h_shift: float
    train_idx: np.ndarray
    test_idx: np.ndarray
    state_dicts: dict[str, dict]  # rung name -> module.state_dict()
    elbow_rung: str

    def to_payload(self) -> dict:
        return {
            "d": self.d,
            "h_lambda": self.h_lambda,
            "h_scale": self.h_scale,
            "h_shift": self.h_shift,
            "train_idx": self.train_idx,
            "test_idx": self.test_idx,
            "state_dicts": self.state_dicts,
            "elbow_rung": self.elbow_rung,
        }


def save_artifacts(artifacts: FitArtifacts, path: str | Path) -> None:
    torch.save(artifacts.to_payload(), path)


def load_artifacts(path: str | Path) -> tuple[FitArtifacts, dict[str, nn.Module]]:
    """Returns (artifacts, {rung_name: reconstructed and loaded module}) --
    every rung that was fit, not only the elbow one, so a re-analysis can
    e.g. recompute severity at max capacity instead of at the elbow."""
    payload = torch.load(path, weights_only=False)
    modules = {}
    for name, sd in payload["state_dicts"].items():
        module = build_warp(name, payload["d"])
        module.load_state_dict(sd)
        module.eval()
        modules[name] = module
    artifacts = FitArtifacts(
        d=payload["d"],
        h_lambda=payload["h_lambda"],
        h_scale=payload["h_scale"],
        h_shift=payload["h_shift"],
        train_idx=payload["train_idx"],
        test_idx=payload["test_idx"],
        state_dicts=payload["state_dicts"],
        elbow_rung=payload["elbow_rung"],
    )
    return artifacts, modules
