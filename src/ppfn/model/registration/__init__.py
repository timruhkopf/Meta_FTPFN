"""Encoder-decoder registration PFN -- ARCHITECTURE.md §2.

Distinct from `ppfn.model.pfn` (a single-stream, causal-capable PFN for an
unrelated BO explore/exploit project -- imports from an uninstalled
`anytimeacquisition` package and is dead code in this repo). Nothing here
depends on that module except `ppfn.model.pfn.bar_distribution.BarDistribution`,
which is genuinely reusable (see `heads.py`).
"""
