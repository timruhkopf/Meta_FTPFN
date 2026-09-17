# First run killed by a peer session's broad pkill at epoch 20; full-arc diagnosis and a curriculum fix

`commit: pending`

## What happened

`01-pretraining-lupi-flow-matching-registration` was killed on ulysses at
epoch 20 (`train/progress=0.21`) by the peer session's
`pkill -9 -f "ppfn.pipelines.train"`, run to kill their OWN two jobs before
relaunching with an unrelated curriculum fix of their own -- that pattern
matches any process with `ppfn.pipelines.train` in its command line, not
just theirs. No malice, an honest mistake, promptly flagged and
acknowledged (see the cross-session messages around 17:3x). Nothing lost:
a checkpoint through epoch 19 was already on disk, and 20 epochs' worth of
real training curves were enough to do the actual diagnosis this entry
covers -- so this incident didn't cost the analysis pass, just moved it
earlier than the originally planned ~epoch-30 checkpoint.

## Full-arc diagnosis, epochs 0-20 (`train/progress` 0.01-0.21)

Pulled the complete metric history directly from MLflow (the zombie-run
lesson from `notebooks/flow_matching_registration_analysis.ipynb`'s
`latest_run` docstring applies here too -- this run is now itself a
"RUNNING-forever" zombie in MLflow's own bookkeeping, since a `kill -9`
never calls `on_train_end`).

```
epoch   flow/total   ce_distil   upper1_gap(train)   upper1_gap(val)
0       0.0116       4.026       0.027                0.029
5       0.0048       3.144       0.334                0.432
9       0.0183       3.776       1.321  <- outlier    1.014  <- outlier
10      0.0065       2.835       0.386                0.496
14      0.0094       2.522       0.419                0.725
18      0.0042       2.797       0.335                0.492
20      0.0012       2.647       0.426                0.459
```

Three findings:

1. **`flow/total` is genuinely declining** (0.012 -> 0.001, noisy but a real
   downward trend) -- `v_φ` IS learning the velocity regression across the
   curriculum ramp, not stuck. This rules out "the SDE mechanism itself
   isn't learning" as the dominant failure mode, at least this early.

2. **`loss/ce_distil` (2.5-4.0) dwarfs both NLL terms (-2 to 0) in
   `loss/total`'s own magnitude for the ENTIRE observed window.** With
   `lambda_ce=1.0` fixed and no student-weight curriculum (the first cut's
   deliberate "isolate before tuning a schedule" choice), the CE
   distillation term -- which pulls the shared backbone toward matching
   the teacher's distribution SHAPE -- had every opportunity to dominate
   the gradient over `nll_student`'s own proper-scoring-rule signal, the
   one that actually reflects the student's real predictive quality.

3. **`upper1_gap` grew fast (0.03 -> ~0.4-0.5) over epochs 0-10, then
   plateaued noisily in that band through epoch 20** rather than
   continuing to climb OR starting to shrink. Read together with finding 2:
   the gap isn't exploding (encoding mechanism isn't obviously broken), but
   it also isn't closing -- consistent with the CE term keeping the shared
   backbone "good enough at matching the teacher's shape" without strongly
   pressuring the student pathway's OWN predictions to actually improve.
   Epoch 9's outlier (`upper1_gap` train=1.32, val=1.01) is a single bad
   epoch, not a sustained regression (epoch 10 recovers immediately) --
   plausibly a hard-batch cluster rather than a systemic issue, not chased
   further.

**Diagnosis**: the most likely single failure mode is `LUPIIDTokenLoss`'s
own already-solved problem, reoccurring here because the first cut
deliberately didn't port that solution over yet -- `ce_distil` dominating
training before `nll_student`'s own signal gets a real chance to shape the
student pathway.

## The fix: port `LUPIIDTokenLoss`'s student-weight/temperature curriculum

`FlowMatchingRegistrationLoss` (`src/ppfn/loss/lupi_flow_matching_registration_loss.py`)
now ramps `nll_student`'s weight from a floor (`0.15`) to `1.0`, and anneals
the CE term's temperature from `2.0` to `1.0`, both over the first 20% of
training (`ramp_frac=0.2`) -- values reused VERBATIM from
`LUPIIDTokenLoss`'s own defaults, not re-derived, for direct comparability
against that already-validated precedent rather than introducing a second,
independently-tuned schedule. Verified: module diagnostic and a debug
Hydra run both still pass, `train/student_weight`/`train/ce_temperature`
now logged and confirmed moving correctly.

**Two secondary changes, same relaunch, smaller confidence each on its
own but cheap to bundle:**
- `n_integration_steps`: 10 -> 16. Coarse SDE discretization was never
  ruled out as a contributing source of registration noise; 16 is a
  moderate step up (not yet the model's own default of 20), affordable
  given the observed ~530-560s/epoch wall-clock had headroom.
- `batch_size`: 24 -> 32, matching `lupi_id_token_baseline`'s own 32
  (also 2x-stacked) -- the first cut's conservatism about `integrate()`'s
  extra cost wasn't actually necessary at the observed wall-clock.

Both configs updated: `configs/loss/lupi_flow_matching_registration.yaml`,
`configs/experiment/lupi_flow_matching_registration.yaml`.

## Relaunch

Fresh run (not resumed from the epoch-19 checkpoint -- the loss function
itself changed, and this repo's own convention favors a fully-attributable
fresh run over a checkpoint whose optimizer/loss state predates a
hyperparameter change). New GPU headroom check performed given the peer's
own jobs were also just relaunched. This is the run intended to actually
occupy the overnight training slot the user asked for.

commit: pending
