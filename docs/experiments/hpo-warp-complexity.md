# Real-benchmark warp assessment: how much diffeomorphism does cross-dataset HPO transfer actually need?

Status: implemented and running. `src/ppfn/experiments/hpo_warp/` fits the
ladder below against real HPOBench data on ulysses; see "Implementation notes"
(bottom of this doc) for what actually got built, where it differs from the
original design below, and where results/notebook live. Read this section
first if you're picking this back up.

## Motivation

The project's premise (`docs/ROADMAP.md` §1.1, `CLAUDE.md` "The problem") is that
the same algorithm recorded on two different datasets gives two related-but-warped
response surfaces `f_A = f_B ∘ T`, and the whole architecture exists to infer `T`
in-context with no correspondence. That premise has so far only been validated
against the synthetic velocity-field prior. This experiment asks the question on
real HPO and multi-fidelity-HPO benchmark data instead: **for real (dataset,
algorithm) pairs that share a hyperparameter search space, how much of a
diffeomorphism is actually needed to align their response surfaces, and how
complex does it need to be?**

This is deliberately *not* a test of the trained model. It never touches the
registration architecture. It is a measurement study that stands on its own,
reported per benchmark, and its main outputs (realized severity, realized
complexity) are the honest floor/ceiling this project should be checking its own
synthetic-prior calibration against.

### Why this is a different (easier) problem than the one the model solves

`docs/ROADMAP.md`'s own "Open items" flags the relevant asymmetry directly: *"the
x-side is usually assumed shared because the hyperparameter space is common across
tasks, which is what makes the deformed-x case here distinct."* In essentially
every real HPO benchmark, two tasks (two datasets, one algorithm) share the exact
same config space `X` — there is no point-cloud correspondence problem. We can
query both tasks' response surfaces at the *same* `x`. That means `T` can be
**fit directly by optimization** against known correspondence — this is exactly
`ROADMAP.md` §8.2's "non-amortized reference" estimator, just pointed at real data
instead of the synthetic prior. Nothing here needs in-context inference.

### The two distortions, kept separate

Per `docs/ARCHITECTURE.md`/`docs/ROADMAP.md` §1.5's x/y split:

- **y-distortion** — different accuracy ceilings, noise floors, task difficulty.
  Absorbed by an unknown monotone `h`: `g_A(x) ≈ h(g_B(x))`, no reparametrization
  of `x` at all.
- **x-distortion** — the location of the good region actually moves in a way no
  rescaling of `y` can explain (e.g. optimal LR/batch-size shifting with dataset
  size). This is the genuine `T`.

Account for y-distortion first, on every pair, before fitting `T` — otherwise a
pure noise-floor difference gets misattributed to a spurious `x`-warp.

## What this experiment measures (and what it deliberately does not)

**In scope:**

1. y-distortion correction (monotone `h`) between every task pair, on the full
   available data for that benchmark — no scarce/abundant split. This project has
   no interest in an n_A-vs-n_B scarcity study here; that question is what the
   trained model itself is for. Use everything the benchmark has.
2. x-distortion (`T`) fit against the project's own warp ladder, on the
   y-corrected surfaces, again using the full data.
3. A complexity measure for the fitted `T`, reported **as two separate numbers**,
   per `docs/ROADMAP.md` §4.2: *"severity and complexity are separate axes and
   conflating them will muddy the central experiment."* Same discipline applies
   here:
   - **severity** — the realized `log|det J|` band of the fitted map (reuses
     `src/ppfn/prior/registration/warp.py`'s `logdet_jacobian_grid` machinery
     directly against the *fitted*, not sampled, field) — how much local stretch
     the warp applies.
   - **shape-complexity** — the minimal degrees of freedom (spline knots, or
     velocity-field kernel centers `M`) at which additional capacity buys
     negligible held-out fit-quality gain. A DOF count, independent of magnitude:
     a large pure translation has near-zero shape-complexity but nonzero
     severity; a small highly local wiggle has the reverse profile.
4. Fidelity-axis warping, as its own sub-study, for multi-fidelity benchmarks
   only: does a per-task monotone reparametrization of the epoch/step axis
   improve learning-curve alignment beyond what the config-space warp + y-h
   already explain? Kept structurally separate from (1)-(3) so a result doesn't
   get attributed to the wrong axis.

**Explicitly out of scope for this pass** (cut per discussion — not because they
lack value in general, but because they're a different question than "how complex
is the warp"):

- Scarce-A / abundant-B sampling variants (uniform vs. restricted-region vs. a
  real BO trace). That's the estimator-degradation question the trained model
  addresses; this experiment characterizes the target it would need to hit, using
  all available data to do so as precisely as possible.
- The fold-fraction diagnostic (`ARCHITECTURE.md` §5.2/§5.4). Meaningful once
  something is inferring `T` from limited context; not needed when `T` is fit by
  direct optimization against the full grid.
- **Gromov-Wasserstein.** Cut, and worth being precise about why, since it's easy
  to think GW measures "the warp's complexity" — it doesn't. GW estimates a
  correspondence between two point clouds using *only* intra-cloud pairwise
  distance geometry, with no use of the shared `x`-coordinates or of `y` at all.
  It's the tool for exactly the situation this experiment does *not* have: no
  correspondence, no shared frame (`docs/decisions.md` D13: *"we have exact
  targets from the prior; GW is for when you have neither correspondence nor
  labels"*). Here the correspondence is free (shared config space), so GW would
  be solving a harder, different problem and — per D4/D13 — is specifically
  biased under the non-uniform-density conditions that are common in these
  benchmarks. It is not a measure of the fitted `T`'s complexity; it would be an
  alternative (and, per the project's own prior finding, worse) way to *estimate*
  a `T` we don't need an alternative estimator for.

## Method, per task pair within a benchmark family

Given a benchmark family with shared config space `X` (and fidelity axis `Z`
where applicable) and task instances `{t_1, ..., t_K}`, for each ordered pair
`(A, B)`:

1. Pull the full available table (tabular benchmarks) or a dense grid over `X`
   (surrogate benchmarks) for both `t_A` and `t_B`, at matched `x`.
2. Fit isotonic `h`: `g_A(x) ≈ h(g_B(x))`. Report the R² this alone captures.
3. On the residual, fit `T` up the ladder already specified in the project's own
   warp family, each optimized directly (torch, gradient descent — the family
   code is reusable, see "Reuse" below):
   - identity (`T = id`, the null baseline)
   - global affine (`A_g x + b_g`)
   - P1: elementwise monotone RQ-spline per axis, sweep knot count `K`
   - P2: stationary velocity field, sweep kernel-center count `M`
4. Hold out a fraction of `x` purely so the reported fit-quality number is honest
   (not train-set R²) — a hygiene detail, not a scarcity experiment: the fit
   itself still uses the great majority of available data.
5. Report severity (log|det J| band of the best-fitting `T`) and shape-complexity
   (the `K`/`M` elbow — the point on the fit-quality-vs-DOF curve where
   additional capacity stops paying) as two separate numbers per pair.
6. For multi-fidelity families, repeat steps 2-5 with `X` replaced by the 1-D
   fidelity axis `Z`, on residuals after the config-space correction — does the
   epoch/step axis itself need a monotone per-task reparametrization.
7. Fitting per *ordered* pair is cheap (an optimization against already-materialized
   data, no real benchmark evaluations), so direction doesn't need to be chosen a
   priori — fit both `T_{A→B}` and `T_{B→A}` and report their agreement as a free
   sanity check (roughly symmetric severity/complexity is the expected, unremarkable
   case; a sharp asymmetry is itself worth flagging).

### Reuse from the existing codebase

`src/ppfn/prior/registration/warp.py`'s `VelocityField`, `flow_rk4`, and
`logdet_jacobian_grid` are written for *sampling* a warp, not fitting one to
data — fitting needs a differentiable (torch) port of the same family
(`v(u) = s · Σ_m w_m exp(...)`, integrated by the same RK4 scheme) with `centers`,
`weights`, `lengthscale` as optimizable parameters instead of RNG draws. This is
new code, but it's a reparametrization of an existing, already-validated family —
not a new warp design. The P1 elementwise RQ-spline is specified in
`docs/ROADMAP.md` §4.1 but not yet implemented anywhere in `src/`; it needs to be
written from scratch (nests the identity, analytic inverse, a `K`-bin complexity
knob — straightforward as a `torch` module).

## Aggregation and reporting, per benchmark

For a family with `K` tasks: report severity and shape-complexity as
**distributions over pairs** (median ± IQR), not single numbers — `K(K-1)`
directed pairs for small `K`, a representative random subsample for large `K`
(see prioritization below). Per benchmark, report:

- The y-only R² distribution (how much is pure rescaling before any `x`-warp is
  even considered).
- The severity distribution (log|det J| band) — comparable, in principle, to the
  band the synthetic prior's own rejection criterion targets (`log(9)`,
  `ARCHITECTURE.md` §1.3), so this doubles as a calibration check on whether
  `s_max` in the synthetic prior brackets real difficulty.
- The shape-complexity distribution (elbow `K`/`M`).
- Severity/complexity regressed or scattered against per-task meta-features
  (n_classes, n_samples, n_features, class imbalance — via the already-installed
  `openml` package's metadata, or the benchmark's own per-task descriptors where
  available) — this is the part that actually answers "does dataset complexity
  predict how much warping is needed."
- For MF families, the fidelity-warp severity/complexity, reported separately,
  regressed against the same meta-features (does "needs more epochs" track
  dataset size, as the freeze-thaw/learning-curve-extrapolation literature would
  predict).
- One qualitative visualization per family: the config-space warp displacement
  (same style as `warp.py`'s own `__main__` demo) for the most- and
  least-severe pair, as a human sanity check before trusting the numbers.
- A one-paragraph verdict per benchmark: does this family show real x-distortion
  worth an architecture, or does y-only/affine already explain nearly everything?

## Benchmark universe

Ordered as requested: classic (single-fidelity) HPO first, multi-fidelity HPO
second. `*` marks the ones worth investigating first within their group —
picked for shared-space breadth (large `K`, real statistical power on the
meta-feature regression) and low integration friction (no extra binary
container/surrogate downloads beyond a `pip install` and a data pull).

### Classic (single-fidelity) HPO — HPOBench

[HPOBench](https://github.com/automl/HPOBench) (Eggensperger, Müller, Hutter et
al., NeurIPS Datasets & Benchmarks 2021) is not currently installed
(`pip show hpobench` — not found) or vendored anywhere in this repo, unlike
`mfpbench` (see below). It needs to be added before any of this runs. From its
published structure (exact class/task names should be re-verified once
installed — not confirmed against the installed package the way the `mfpbench`
list below is):

- **`*` Tabular ML benchmarks** (`hpobench.benchmarks.ml`) — SVM, XGBoost, random
  forest, logistic regression, and a small NN, each with one shared config space
  evaluated (typically exhaustively or via a table) across a common list of
  OpenML classification tasks. This is the most direct real-world analogue of
  the synthetic prior's "many datasets, one algorithm, one config space" setup,
  and the natural family to pilot on for classic BO.
- **`*` The classic tabular NN benchmark** (originally Klein & Hutter's "Tabular
  Benchmarks for HPO", wrapped into HPOBench as the FCNet/NAS-HPO-bench tables) —
  4 UCI regression datasets (protein structure, slice localization, naval
  propulsion, parkinsons telemonitoring), one 9D discretized config space,
  exhaustive grid. Only 4 tasks (`K=4`, 12 directed pairs) so weak for the
  meta-feature regression, but it's the cleanest, smallest, most exhaustively
  tabulated resource available — good as the very first correctness check on the
  fitting pipeline before scaling up to a bigger `K`.
- **ParamNet surrogate benchmarks** — a random-forest-interpolated continuous
  surrogate over several datasets (adult, higgs, letter, mnist, optdigits,
  poker), single-fidelity variant. Useful as a second, independent family once
  the tabular-ML pilot works, since it's continuously queryable rather than
  gridded.
- **NAS-Bench-101/201 (single-fidelity slice)** — categorical cell search space,
  useful mainly as a boundary case (does "diffeomorphism" even make sense on a
  largely categorical space) rather than a primary target; defer.

### Multi-fidelity HPO — mfpbench

Already vendored (`external/ifbo_icml2024/src/mf-prior-bench/`) and already
`pip`-installed in this venv (editable, pointing at a sibling checkout — see
`pip show mfpbench`). `yahpo_gym` itself is **not** installed, so only the
`*_tabular` variants (no ONNX-surrogate dependency, no metadata download beyond
the tabular data pull) are immediately runnable; the YAHPO-surrogate paths need
`pip install yahpo_gym` plus its surrogate/metadata download first.

| Family | Space | Fidelity | `K` (tasks) | Needs |
|---|---|---|---|---|
| **`*` LCBenchTabular** | 7D continuous+int, log-scaled (batch_size, lr, momentum, weight_decay, num_layers, max_units, max_dropout) | epoch, 1–52 | ~35 OpenML datasets | tabular only |
| LCBench (YAHPO surrogate) | same 7D | epoch, 1–52 | same ~35 | `yahpo_gym` + surrogate download |
| PD1Tabular | per-workload (optimizer×arch), shared within a workload group | step | several dataset×model workloads | tabular only |
| PD1 (surrogate) | same, continuous | step | same workloads | own PD1 data download |
| NB201Tabular | categorical cell + fidelity | epoch | 3 (CIFAR-10/100, ImageNet16-120) | tabular only — but `K=3`, weak for meta-feature regression, and largely categorical |
| **`*` RBV2 / IAML** (glmnet, ranger, rpart, xgboost, svm, aknn) | shared per model class | trainsize fraction | 100+ OpenML datasets each | `yahpo_gym` + surrogate download |
| TaskSetTabular | optimizer hyperparams | training step | many tasks | tabular only |
| JAHSBenchmark | joint arch+HP | epoch | 3 image datasets | own surrogate download, `K=3` |

`*` picks, and why: **LCBenchTabular** for the same reasons as the classic-HPO
pilot — no extra downloads beyond the tabular pull, moderate `K` (35, so 1190
directed pairs — already enough to need the subsampling note below), a genuine
continuous config space, and a real fidelity axis for the sub-study in step 6.
**RBV2/IAML** as the second target once the pipeline is validated on
LCBenchTabular — `K > 100` is where the meta-feature regression actually gets
statistical power, at the cost of needing `yahpo_gym` installed and its
surrogate/metadata downloaded first.

### Pair-count blowup

`K(K-1)` directed pairs grows fast: 35 tasks → 1,190 pairs; 100+ tasks → 10,000+.
Since fitting is cheap (optimization against already-materialized data, not real
benchmark evaluations), the practical limit is more about how many pairs are
worth *inspecting* than compute. For `K` above roughly 20-30, subsample a
representative set of pairs (e.g. stratified across a meta-feature like dataset
size, rather than uniform-random, so the reported distribution isn't dominated by
whichever size band happens to be over-represented in the benchmark's own task
list) instead of computing all `K(K-1)`.

## Build order

1. **Dependency/data check.** `pip install hpobench`; confirm the tabular ML
   benchmarks and the FCNet/NAS-HPO-bench tables load with no container
   machinery. Confirm `mfpbench`'s `lcbench_tabular` loads (already installed).
2. **Pilot: the 4-dataset tabular NN benchmark (HPOBench).** Smallest possible
   correctness check — implement the fitting pipeline (y-h, identity, affine, P1,
   P2, severity, shape-complexity) end to end on `K=4`, eyeball the warp
   visualizer, before trusting it on anything bigger.
3. **HPOBench tabular ML benchmarks** (classic BO, larger `K`) — first real
   per-benchmark report with a meta-feature regression.
4. **LCBenchTabular** (multi-fidelity) — same pipeline plus the fidelity-axis
   sub-study.
5. **RBV2/IAML** once `yahpo_gym` is installed and its data downloaded — the
   large-`K` family for the meta-feature regression's statistical power.
6. Remaining families (PD1, NB201Tabular, TaskSetTabular, JAHS) as time permits,
   lower priority per the table above.

Each stage produces its own per-benchmark report section in this doc (or a
follow-on doc under `docs/experiments/`); a `docs/labbook/` entry once a stage
produces a finding worth remembering, per `.claude/rules/labbook.md`.

---

## Implementation notes (read this first if resuming)

What actually got built in `src/ppfn/experiments/hpo_warp/`, and where it
differs from the plan above — scope was narrowed once during a follow-up
discussion; see that discussion's assessment for the reasoning:

- **Scarce-A/abundant-B sampling variants, the fold-fraction test, and GW are
  cut from this pass** (not merely deprioritized) — the ask became "fit the
  diffeomorphism using all available data and report its complexity," not an
  estimator-degradation study. `fold_fraction` is still logged as a free
  byproduct of the severity computation, just not treated as a headline
  metric.
- **y-correction and x-warp are fit sequentially, never jointly**: `h` first
  (on the identity correspondence, before any `x`-warp exists to collude
  with it), held fixed while `T` is fit on the residual. This is a
  deliberate identifiability choice — a joint `min_{T,h}` objective is
  ill-posed (a flexible `T` can collapse the domain and let `h` fake a
  perfect score) — not an oversight.
- **`h` is `a*YeoJohnson(y;lambda)+b` (3 parameters), not an isotonic fit —
  this took two revisions to get right, and both are worth knowing about.**
  Isotonic `h` (any monotone function) was the original choice; a
  ground-truth check found it recovering severity ~8x too low, because an
  unconstrained `h` can absorb a real spatial warp's effect rather than
  leaving it for `T` to explain. First fix: alternate refitting `h` against
  `T`'s current pushforward (ACE-style) — a real, measured improvement, but
  still treating the symptom, since nothing stops a freshly-refit isotonic
  `h` from re-absorbing the same signal. Actual fix: constrain `h`'s family
  to what real HPO y-distortion needs and no more — a mandatory affine part
  (different metrics have genuinely different ranges/floors/signs) plus at
  most one global shape parameter (boundedness/skew), which structurally
  cannot encode a spatially-varying pattern the way a many-knot isotonic fit
  can. Fit by a closed-form grid scan over the shape parameter (no gradient
  descent, no local optima). Single pass, no alternation needed; see
  `docs/labbook/2026-09-11-hpo-warp-h-family-constrained-to-yeojohnson.md`
  for the full reasoning and the two prior attempts.
- **Two complexity numbers were extended to include displacement volume and
  bending energy** (a closed-form RKHS-norm read off the fitted velocity
  field), alongside the original severity (`log|det J|` band) and
  shape-complexity (DOF elbow) pair — each answers a different question
  (magnitude of movement vs. local distortion vs. smoothness cost); none
  subsumes another. A cheap Spearman-Pearson gap is also logged per pair as
  a zero-fitting-cost nonlinearity screen, at the requester's shrugging
  "some BO people will want it."
- **The shape-complexity elbow is tied to the seed-replicate noise floor**
  (smallest capacity within 25% of it, falling back to within 5% of the best
  MSE seen if the floor is unreachable) rather than an arbitrary "95% of best
  observed R²" — see `docs/labbook/2026-09-10-hpo-warp-noise-floor-off-by-n-seeds.md`
  for a real bug this caught (the floor was first computed from the wrong
  variance — single-observation instead of the mean-of-`n_seeds` actually
  being fit — silently biasing elbow selection toward the smallest rung in
  every sweep).
- **Data source: HPOBench's `TabularBenchmark`**, not `mfpbench`, for the
  first pass — `hpobench.util.data_manager.TabularDataManager(model, task_id)`
  gives a fully precomputed, shared grid (e.g. `lr`: 25x25 config grid x 5
  successive-halving fidelities x 5 replicate seeds, identical across ~29
  OpenML task IDs) with **zero interpolation/matching needed for
  correspondence** and, as a bonus neither `mfpbench` family offered for
  free, real seed replicates to use as a noise floor. `hpobench` isn't a
  project dependency (its own dependency tree — `oslo-*`, a pinned
  `ConfigSpace`, etc. — isn't worth adding to the main `uv.lock`) — it's
  installed in an isolated venv on ulysses, `~/hpobench_venv/.venv`
  (`pip install git+https://github.com/automl/HPOBench.git` plus
  `scikit-learn`/`pandas`/`openml`/`xgboost`/`torch`/`omegaconf`, the last two
  needed only because importing `ppfn.experiments.hpo_warp.*` still runs
  through `ppfn/__init__.py`). Model families available:
  `lr`/`svm`/`rf`/`xgb`/`nn`, each Blackbox(single-fidelity)/MF, downloaded
  once to `data/hpobench-tabular/<model>/<task_id>/` on ulysses (a few
  hundred MB per model; `lr` first as the smallest/cheapest correctness
  pilot, matching the "start small" build-order intent below even though the
  concrete benchmark changed).
- **Registration-as-image-registration**: since correspondence is free, `T`
  is fit by treating B's surface as a continuously-interpolated "image"
  (`interp.py`'s differentiable multilinear interpolation) and warping A's
  coordinates into it via gradient descent — literally 2D diffeomorphic
  image registration, with validation loss standing in for pixel intensity.
  No finite-difference surrogate anywhere; interpolation and the isotonic
  readout are both differentiable, so the whole ladder is one
  `loss.backward()`-able graph per rung.
- **Every rung's fitted state is persisted**, not just the elbow one's
  summary — `<A>__<B>.artifacts.pt` (`artifacts.py`) holds all state_dicts,
  `h`'s breakpoints, and the exact train/test split, specifically so a later
  session can compute a *different* metric on an existing fit without
  re-optimizing (the metric menu was still being actively assessed at
  implementation time).
- **Orchestration**: `run_pairs.py`, one JSON (+ one `.artifacts.pt`) shard
  per ordered task pair under `data/hpo_warp_results/<model>/`, safe to
  resume/extend (existing shards are skipped) and parallelized with a forked
  `ProcessPoolExecutor` (ulysses: 16 cores, 14 workers, `torch.
  set_num_threads(1)` per worker). `aggregate.py` collects shards into one
  dataframe on demand — reading it mid-sweep is fine, it just reflects
  whatever's landed so far.
- **Report**: `notebooks/hpo_warp_complexity_report.ipynb` — a live notebook
  (re-running it after more shards land picks them up automatically), not a
  frozen snapshot. Executed with the main project's own `.venv` (which has
  `ppfn` + jupyter already), not the isolated hpobench venv — the notebook
  only reads already-computed JSON/`.pt` shards, so it never needs `hpobench`
  itself.
- **First "sting"**: HPOBench `lr` (2D config space, cheapest), full 29-task
  coverage (812 ordered pairs, no subsampling needed — cheap enough at
  ~12s/pair fit time). `svm`/`rf`/`xgb`/`nn` data is already downloaded and
  ready for the same treatment next.

### Scaling up: scattered-design, higher-d benchmarks (LCBench/TaskSet)

Second round, extending beyond HPOBench's exact grids to LCBench-tabular
(7D) and TaskSet-tabular (up to 8D) — both share their random-search design
exactly across tasks (verified empirically, same free-correspondence
property as HPOBench) but as a **scattered** point cloud rather than a
Cartesian grid. `TaskGrid` gained `is_gridded` and `ids` fields; `surface.py`
dispatches config/fidelity reads between `interp.multilinear_interp` (grid)
and the new `interp.kernel_interp` (scattered, Nadaraya-Watson). See
`docs/labbook/2026-09-10-hpo-warp-scaling-to-scattered-higher-d-benchmarks.md`
for the four bugs this surfaced (all fixed) and one deliberately deferred:

- **PD1-tabular excluded**: checked and confirmed its per-workload config
  samples do NOT match across workloads (unlike LCBench/TaskSet) — would
  need a "no free correspondence" extension (evaluate B's reader at A's own
  points for the identity step, rather than a direct same-row comparison)
  not built in this round.
- **TaskSet excluded from the sweep**: its per-task loss scale is heavy-tailed
  enough (>1.6% of cells past 1e6 on a real pair) to break the MSE-based
  warp-fitting loss even after 1st/99th-percentile winsorization, while the
  rank-based isotonic `h` fit survives it fine — a real fix needs the
  warp-fitting loss itself to work in rank/quantile space (or a per-task
  robust rescaling) before any squared-error objective, which is a loss-design
  decision, not a one-line robustification. `TASKSET_WINSORIZE_PCTL` and the
  per-optimizer-family declared bounds in `scattered_data.py` are already
  correct and reusable once that fix lands.
- **LCBench**: works correctly (verified: fixed bandwidth gives a monotone-ish,
  capacity-sensitive R² ladder), 35 tasks, 1190 ordered pairs, ~68s/pair
  (60s config-warp + 8s fidelity-warp) — run via `run_pairs.py --family
  lcbench`, chained on ulysses to start after the HPOBench `svm`/`rf`/`xgb`/`nn`
  sweeps finish (same 16-core box, sequential to avoid oversubscribing).
