# Overnight check-in: curriculum fix helped nothing, `upper1_gap` still plateaus around ~0.4

`commit: pending`

## Status

`01-pretraining-lupi-flow-matching-registration` (relaunched 2026-09-17
17:55 with the student-weight/CE-temperature curriculum fix, see
[2026-09-17-first-run-killed-diagnosis-and-relaunch.md](2026-09-17-first-run-killed-diagnosis-and-relaunch.md))
ran unattended overnight, alive and healthy the whole time (no crashes, no
OOM, ~700-715s/epoch), at **epoch 58 of 100** (`train/progress=0.59`) as of
this check-in, ~05:25 the next morning. No further intervention needed
overnight -- the background watcher confirmed continuous progress every 5
minutes, log attached in the previous entry's git history.

## The honest finding: the curriculum fix did NOT close `upper1_gap`

Pulled the full metric history (59 logged epochs) via MLflow. The
diagnosis from the killed first run (`ce_distil` dominating `loss/total`)
was real and the fix visibly worked as INTENDED
(`train/student_weight`/`train/ce_temperature` ramp correctly, confirmed
in the logs), but **it did not solve the actual problem it was meant to
fix**:

```
epochs 20-58 (post-curriculum-ramp, student_weight=1.0 throughout):
  upper1_gap (train): mean 0.404, std 0.122
  upper1_gap (train), epochs 40-58 only: mean 0.376, std 0.105
  upper1_gap (val):   mean 0.533, epochs 40-58: 0.523 -- essentially flat
```

The train-side mean drifted down slightly (0.404 -> 0.376) but well within
one standard deviation -- not a defensible "it's shrinking" claim, and the
validation-side mean shows literally no change across that window. `both
nll_student` and `nll_teacher` keep improving together, in near lockstep,
over the whole run (`nll_teacher`: -0.24 at epoch 0 -> -2.19 at epoch 57;
`nll_student`: -0.20 -> -1.95) -- the MODEL is getting better at the task
in general, but the STUDENT never meaningfully closes the distance to the
TEACHER. `loss/ce_distil` continues a real, steady decline (4.1 -> ~2.2-2.4)
and `flow/total` stays low and controlled (0.002-0.016, noisy but not
diverging) -- so by both of those measures the fix "worked," it just
wasn't the thing actually gating `upper1_gap`.

**Revised diagnosis**: `ce_distil` dominance was a real, worth-fixing
issue (confirmed: `train/student_weight`/`temperature` now correctly
ramp, and both loss terms behave exactly as designed), but it was NOT the
dominant cause of the persistent gap. The more likely remaining
candidates, given what's now ruled out:

1. **`n_integration_steps=16` may still be too coarse.** A low
   `flow/total` (the ONE-STEP velocity-regression loss) does not
   guarantee an accurate INTEGRATED trajectory over 16 discretization
   steps, especially for harder (larger `rho`) draws where the true
   transport path is more curved -- Euler-Maruyama's local truncation
   error compounds over steps in a way `flow/total` alone doesn't surface.
2. **`n_transport_samples=1`, `sde_sigma=0.0` (deterministic mean
   transport) may be the actual structural ceiling.** This was always
   flagged as the "first cut, isolate before trusting coupled" choice
   (see the original labbook design entry's own "where uncertainty comes
   from" section) -- if registration genuinely has multiple plausible
   resolutions for a meaningful fraction of draws, a single deterministic
   trajectory necessarily picks one (or blends incompatible modes into an
   uninformative average), and no amount of loss-curriculum tuning can fix
   a capacity/expressiveness gap. This maps directly onto the reason the
   SDE/stochastic-sampling design was chosen over a point estimate in the
   first place.

**A useful external cross-check**: the peer session's `01-pretraining-lupi-
bounds` run (same prior, same-day relaunch with their own curriculum fix)
shows `bounds_gap≈0.39` (val 0.445) at its own epoch 22 -- the SAME order
of magnitude as this run's `upper1_gap`, despite being a structurally
simpler bare-attention architecture with no explicit registration
mechanism at all. This doesn't prove the ~0.4 gap is an irreducible
property of this prior/task at this point in training, but it's a
material data point against "something is specifically broken in the
flow-matching mechanism" and toward "this is a hard task and ~0.4 nats may
just be where BOTH architectures currently sit" -- worth remembering
before over-interpreting either gap in isolation.

## Comparison plot against near-final checkpoints

See `notebooks/flow_matching_registration_analysis.ipynb`'s own updated
output (re-executed against `epoch 58` (mine) / `epoch 22` (peer bounds,
their own post-relaunch checkpoint) -- both models now further trained
than the very-early-epoch comparison in the previous check-in).

## What's NOT done autonomously, and why

Did not enable `n_transport_samples>1`/`sde_sigma>0` (the natural next
thing to test, per the diagnosis above) without checking in first --
unlike the loss-curriculum fix (cheap, well-precedented, low-risk to
apply on my own judgment), the stochastic-SDE extension multiplies
per-step compute by `K` on an already-longer-than-baseline run, is the
ORIGINAL, more speculative design choice this whole branch exists to test
(not a small parameter tweak), and running it well means a real design
decision (how large a `K`, whether to also open up `sigma` gradually or
fix it, whether to keep running the CURRENT run in parallel as a
K=1 reference or replace it) that's worth a second opinion before
committing more overnight GPU-hours to it.

commit: pending
