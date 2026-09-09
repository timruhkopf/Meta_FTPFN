"""Declared-box coordinate normalization -- ARCHITECTURE.md §1.3/§1.5,
invariant #8. Boxes come from `warp.declared_box` (the grid-estimated image
of the domain under a warp, padded), never from a sampled point's empirical
bounding box -- the empirical box is design-dependent (a restricted A-cloud
has a smaller empirical box than its declared one) and reintroducing that
bias is exactly what this project exists to avoid.
"""

from __future__ import annotations

import numpy as np


def normalize(points: np.ndarray, box: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """points: [N, d]. box: (lo [d], hi [d]) from `warp.declared_box`. Maps
    the box to [0,1]^d; does NOT clip -- a point can legitimately land
    slightly outside [0,1] despite the box's padding (padding reduces this,
    doesn't guarantee it away), and downstream bar-distribution heads already
    clamp out-of-range targets into their extreme bin (see
    `BarDistribution.map_to_bucket_idx`), so silently clipping here would
    just duplicate that logic in the wrong place and hide how often it fires."""
    lo, hi = box
    span = np.maximum(hi - lo, 1e-8)
    return (points - lo) / span
