# B_inA visualization check, and a naive-pooling failure mode it surfaced

*Log, 2026-09-15. Follow-up to
[LUPI bounds PFN, query-source/rho stratification, and a val-batch bug](2026-09-15-lupi-bounds-and-stratified-diagnostics.md).
User looked at `notebooks/lupi_1d_visualization.ipynb`'s first rendering and
raised two things: get a more interesting draw, and a correctness worry --
B's points looked identical across rows 2-4, and row 4 (the pooled
[A_ctx;B_inA] upper bound) looked like it was fitting B in B's own frame
rather than B_inA.*

## The correctness worry: checked, not a bug

Verified numerically before touching any code: for the original draw
(seed=7, rho=0.6, default `s_max=0.1`), `|x_b_inA - x_b|` had mean 0.032,
max 0.06 -- genuinely nonzero, but small enough on a [0,1] axis to be
essentially invisible by eye at that plot's resolution. Re-read both the
plotting cell and the model-feeding cell: both already used
`pair.x_b_inA`/`batch.enc_x_inA` consistently, `pair.x_b`/`batch.enc_x`
(B's own frame) was never plotted or fed to any model in that notebook. No
bug -- an unlucky, visually uninformative draw.

Fixed properly rather than just re-asserting this: added an explicit
"B, own frame" gray reference layer to *every* row that shows B at all
(previously only B_inA was drawn), so the transport is directly visible by
eye instead of asserted in a caption. Picked a new draw via a small local
scan over `(s_max, seed)` for a genuinely non-monotone true function AND a
visually large `|x_b_inA - x_b|`: `s_max=0.2, seed=15, rho=0.9` ->
mean gap 0.171, max 0.279, a clear V-shaped true function. Re-rendered:
gray (B, own frame) and blue/white (B_inA) are now unambiguously offset in
every row.

## An unplanned finding: naive pooling can make the "upper bound" worse

With this new draw, the off-the-shelf reused-PFN's "upper bound" (row 4,
`[A_ctx ; B_inA]` pooled) scored **worse** than its own "lower bound" (row
1, `A_ctx` alone): NLL -0.74 vs. -1.63. Visually, row 4's posterior mean
develops a spurious bump around x=0.3-0.4 that row 1 doesn't have -- the
model gets pulled off the true (white, staircase) curve by the pooled B_inA
points.

Working explanation, consistent with the architecture: `A_ctx`'s value
channel is quantile-normalized against **A's own small, acquisition-biased
sample** (n_a=23 here) while `B_inA`'s value channel is quantile-normalized
against **B's own large, uniform sample** (n_b=124) -- two different
reference measures for the "same" [0,1] scale, by the design spec's own
construction (`ppfn.prior.lupi.ecdf`'s whole premise: quantile-normalize
each task against its own evidence, then treat the result as comparable).
For a small `n_a`, A's estimate is noisy enough that the two scales
genuinely disagree at points, and an off-the-shelf PFN with no exposure to
this specific mismatch during its own training (`ppfn.prior.bnn.BNNPrior`
never produces two clouds with different reference measures) has no reason
to discount it -- it just gets pulled toward whichever pooled points are
locally denser. **LUPIPFN (rows 2-3), by contrast, tracks the true curve
tightly with no such bump**, despite having access to the exact same B
data (just read via cross-attention in B's own frame, not pooled
positionally) -- consistent with it having actually learned to handle this
mismatch, which is the thing this whole architecture exists to do.

This is a plausible illustration of the core premise, not a proven one --
one draw. Worth checking whether it reproduces: (a) across more draws at
similar `n_a`/`n_b` imbalance, (b) whether the properly-trained `BoundsPFN`
(still training on this exact prior, unlike the reused checkpoint) avoids
the spurious bump where the off-the-shelf PFN doesn't -- direct evidence
that prior-matched training, not just correct positional registration,
matters here.

commit: pending
