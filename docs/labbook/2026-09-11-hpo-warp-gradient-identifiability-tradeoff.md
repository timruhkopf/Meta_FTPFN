# |∇f|-identifiability of T: a real but partial explanation, with both an upside and a downside for HPO relatedness

commit: b968452

## What was investigated

Follow-up to the boundary-penalty fix (see the same day's entry on the
synthetic case study's true-vs-recovered warp plot). The lower-right
quadrant of that plot visibly under-registered the true warp relative to
the other three quadrants. `docs/ROADMAP.md`'s Fisher/identifiability
argument says `T` is only identified where `|∇f|` is appreciable — checked
whether that explains it, and thought through what the argument implies for
the HPO-relatedness measurement specifically, not just for the registration
model it was originally written about.

## What was found, quantitatively

Computed `|∇f|` by quadrant on the synthetic ground-truth pair: lower-right
was lowest (1.17) vs 1.35-1.96 elsewhere (~13-40% lower). Recovered
displacement in that quadrant was 0.021 vs 0.025-0.052 elsewhere — a ~2.5x
gap. Directionally consistent with the identifiability argument, but the
gradient contrast alone doesn't account for the size of the displacement
gap; most of the remainder is attributable to where the fixed-capacity
(M=16) velocity-field fit happened to place its RBF kernels, not to
identifiability alone. **Conclusion: real but partial** — worth a caveat in
the notebook, not a full explanation of that artifact.

## The trade-off this implies for measuring HPO relatedness (not just for registration)

This argument was originally about when the *registration model* can learn
`T` at all. Applied to the relatedness-measurement use case here, it cuts
both ways:

**Upside — plausible alignment with what HPO transfer actually needs.**
BO's decisions are driven by where the surface is steep (near optima,
along ridges); flat regions are configs nobody would try or need to
distinguish. If `T` is weakly identified exactly where `|∇f|≈0`, that blind
spot mostly coincides with practically irrelevant regions — a registration
that "gives up" on plateaus isn't losing much decision-relevant signal.

**Downside 1 — the blind spot is correlated with the function's own
geometry, so it biases the relatedness metric rather than just adding
noise.** Severity/displacement will be systematically under-read wherever
both tasks happen to be flat, making two tasks look more related (less
force needed) than warranted, specifically in the regions least covered by
evidence — not a random underestimate.

**Downside 2 — flat-for-one-task regions are often exactly where real HPO
surfaces diverge most across datasets.** Boundary/degenerate regimes (very
high LR, near-zero regularization) are simultaneously the flattest/
least-sampled *and* where dataset-dependent blowup behavior differs most.
The confound can hide precisely the cross-task differences the whole
experiment exists to measure.

## Disposition

Not a code change. Recorded as an interpretation caveat for the notebook's
"how to read this" section and for judging any pair whose surfaces have
large low-gradient regions in common — treat a low severity/displacement
reading there as "insufficiently evidenced," not "confirmed related."
