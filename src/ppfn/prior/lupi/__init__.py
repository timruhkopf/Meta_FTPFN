"""Generative prior for the LUPI-distilled alignment-aware multi-task PFN --
`docs/labbook/`'s 2026-09-14 LUPI build spec (originally
`2026-09-14-lupi-aligned-multitask-pfn-build-spec.md`).

Adds a y-distortion `h` (`monotone.py`) and a simulated-acquisition A-design
(`acquisition.py`) on top of `ppfn.prior.registration`'s existing T-warp
machinery (`warp.py`, reused directly, not duplicated) and shared-latent
function prior (`function_prior.py`, reused directly). Both this package and
`ppfn.prior.registration` derive x_A/x_B from the SAME shared latent z via
independently sampled velocity-field warps; this package additionally
distorts A's y-values through h and biases A's design toward good regions of
f, which `ppfn.prior.registration` deliberately does not do (CLAUDE.md's
gauge there is f_A = f_B o T, no h at all).
"""
