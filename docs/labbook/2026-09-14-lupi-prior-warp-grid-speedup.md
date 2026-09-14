# LUPI prior bottleneck: warp rejection-sampling grid, not the GPU

*Log, 2026-09-14. Follow-up to
[LUPI-distilled alignment-aware multi-task PFN: first build](2026-09-14-lupi-alignment-pfn-build.md)
-- the real-sized `lupi_baseline` run launched on ulysses hadn't logged a
first epoch after ~25 minutes; user asked whether the prior could be made
more efficient.*

## What was investigated

Whether the slow first epoch on ulysses was a hang, an oversubscription
problem (`prior.num_workers: 8`), or a genuine CPU bottleneck in prior
generation.

## What was found

`nvidia-smi` on ulysses during the run: 0% GPU utilization, ~287-1637 MiB
used (model/data resident, not computing). All 8 dataloader worker
processes at ~180-220% CPU each (`ps`), i.e. the job was already using
essentially the whole 16-core box (8 workers x ~2 active cores) -- not
oversubscribed, not stalled, genuinely CPU-bound on prior generation.

`cProfile` on `ppfn.prior.lupi.dataset.build_training_item` (matching
`lupi_baseline.yaml`'s `n_a_range`/`n_b_range` = (8,100), d up to 5)
confirmed and localized it: of 344 ms/item (single draw, single thread),
**97% is inside `flow_rk4`**, and **84% specifically inside
`ppfn.prior.registration.warp.sample_warp_pair`'s accept/reject check**
(`logdet_jacobian_grid`), which estimates the warp's log-Jacobian band by
finite differences on a `grid_n^d` regular grid (`grid_n=5` by default --
3125 points at `d=5`), evaluated twice per pair (`v_A`, `v_B`), each
requiring `2*d` `flow_rk4` calls (a plus/minus perturbation per axis).
`declared_box` (also called twice per pair, for `box_A`/`box_E`) uses the
same `grid_n^d` grid and adds another ~12%. The actual point-flows used to
build tokens (`to_a`/`to_b` in `ppfn.prior.lupi.sampler`) are cheap by
comparison (~4% combined) -- the cost is almost entirely in the warp
*acceptance check*, not in using the accepted warp.

This machinery is `ppfn.prior.registration.warp`, reused verbatim (shared
with `ppfn.prior.registration`'s own prior) -- not a bug introduced by the
new LUPI prior, just a cost profile that this prior's real-sized `n_a`/`n_b`
ranges (small clouds, so the warp-sampling overhead is a much larger
fraction of total per-item cost than it is for the registration prior's
default, larger `n_b_range`) exposes more sharply.

## Fix and evidence it helped

`sample_warp_pair`/`declared_box` already expose `grid_n` as a keyword
argument (default 5), so `ppfn.prior.lupi.sampler.sample_pair` now passes
its own `warp_grid_n` (new parameter, default 4, threaded through
`ppfn.prior.lupi.dataset.build_training_item`/`LUPIStreamDataset` and
exposed as `prior.warp_grid_n` in `configs/prior/lupi.yaml`) instead of
inheriting the 5 default -- a LUPI-local override that never touches
`ppfn.prior.registration.warp`'s default or `configs/prior/registration.yaml`.

Measured (`d=5`, `s_max=0.1`, 300 trials, isolated `sample_warp_pair`):

| grid_n | ms/pair (warp+box only) | reject rate | log\|det J\| band mean | speedup |
|---|---|---|---|---|
| 5 (registration default, unchanged) | 407 | 1.0% | 0.843 | 1x |
| 4 (new LUPI default) | 131 | -- | -- | 3.1x |
| 3 (considered, rejected) | 33 | 0.0% | 0.509 (40% low) | 12.3x |

grid_n=3 was rejected: it systematically underestimates the true
log-Jacobian band (mean 0.509 vs. 0.843 nats, ~40% low), which would let
some near-fold warps through the accept/reject check that grid_n=5 would
reject. grid_n=4 was chosen as the safer middle ground (user decision).

End-to-end on `build_training_item` (full item, not just the warp step):
344 -> 125 ms/item (2.75x). Debug-config epoch time (2 epochs x 5 steps,
tiny `n_a_range`/`n_b_range`=(8,32)): 11.5s/7.0s -> 5.3s/5.1s, roughly 2x
even at small scale where warp-sampling is a smaller fraction of the total.
`rho=0` invariant re-verified exact after the change (grid_n only affects
the rejection-band estimate and the declared-box padding target, not the
point-flow computation the invariant checks).

Not re-measured: actual wall-clock on the real-sized ulysses run with this
fix applied (the original run was killed and relaunched with it -- see
commit for the restart).

commit: pending
