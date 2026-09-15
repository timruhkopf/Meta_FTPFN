# Reference-measure mismatch: why naive pooling can fail completely, not just a little

*Log, 2026-09-15. Follow-up to
[B_inA visualization check, and a naive-pooling failure mode it surfaced](2026-09-15-lupi-visualization-b-inA-check.md).
User's read of the notebook plot: the reused PFN's row 4 (`[A_ctx ; B_inA]`
pooled) fits B_inA's own data reasonably well ("a bit wiggly, but quite a
good fit"); the large gap to the true quantized function is because A's
context points sit at low likelihood/low quantiles of that fitted
predictive distribution, not because the fit itself is bad. Asked to
verify this mechanistically rather than take it on faith.*

## What was checked

Added `LUPIPairInternals.value_under_b_ecdf(z)`
(`ppfn.prior.lupi.sampler`): what a point's raw `h(f(z))` value quantile-
normalizes to under B's own (large, `n_b~124`, uniform) ECDF instead of
A's own (small, `n_a~23`, acquisition-biased) one. For the notebook's own
draw (`s_max=0.2, seed=15, rho=0.9`):

```
raw h(f(.)) over A context:  min=-3.0007  max=-2.6507  mean=-2.9073
raw   f(.)  over B sample :  min=-0.6455  max=-0.5292  mean=-0.6189
```

**Every single A context point lands at quantile 0 under B's ECDF.** Not a
tail effect -- complete non-overlap: A's entire observed raw-value range
sits about 2 full units below B's entire observed range.

## Mechanism, and a correction to the previous entry's framing

`region_type=full` (no support restriction active) and `beta=0.59` (mild
acquisition bias, `sample_beta`'s own range is [0.5, 8], so this is near
the floor) -- neither factor alone looks severe. The dominant driver is
**`h`**: A's raw values go through `h(y) = a*y + b + c*asinh(d*y)`
(`ppfn.prior.lupi.monotone`), B's don't (by design -- B is "used in place,
untouched", spec's own framing). `h` mapped B's own narrow raw range
(width 0.12) into a shifted, ~3x-amplified range (width 0.35, centered
~2.8 lower). Given A's small, acquisition-biased sample, it never lands
back in B's range.

The previous entry characterized this as "A's small-sample and B's
large-sample quantile scales disagree" -- true, but understated: they don't
just disagree, they can estimate quantile functions over **ranges with
zero overlap**. Pooling `[A_ctx ; B_inA]` then hands a naive model two
clouds that are each individually well-calibrated to `[0,1]` (that's what
per-task quantile normalization is supposed to buy) but anchored to
completely disjoint underlying value ranges, with **nothing in the pooled
representation signaling that they're on different scales at all** -- the
model has no way to know unless it's specifically learned to expect this
pattern. It fits mostly to B_inA's dominant mass (124 vs. 23 points) --
consistent with the user's "quite a good fit to B_inA" read -- and A's own
points end up scored in the tail, explaining row 4's worse-than-row-1 NLL.

## Why this isn't a bug

This is exactly `ppfn.prior.lupi`'s option (b) pathology (spec §3.2(b)):
"Generate A's context via a simulated acquisition process, compute F_hat_A
from those biased observations... The network learns to correct a bias
whose distribution it has seen." The severity found here (complete range
non-overlap from a *mild* beta) is a useful calibration data point for how
aggressive this pathology can get in practice, not evidence of a
mis-implementation.

## Open question

Does LUPIPFN (rows 2-3, no spurious bump in the earlier visualization)
actually handle this specific pattern -- distinguishing which of two
pooled value ranges a point belongs to -- or is something else going on
(e.g. cross-attention simply weighting B's contribution down generally,
independent of this specific mismatch)? Not yet tested directly; would
need an attention-weight readout at A's context/query positions,
conditioned on cases with and without a large `h`-induced range gap.

commit: pending
