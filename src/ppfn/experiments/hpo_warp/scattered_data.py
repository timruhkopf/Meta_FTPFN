"""Loading benchmarks whose shared design is a fixed random/quasi-random
sample rather than a Cartesian grid: LCBench-tabular and TaskSet-tabular
(both already downloaded under `data/`, see `docs/experiments/
hpo-warp-complexity.md`).

Both still give free correspondence between tasks -- confirmed empirically
(see `docs/labbook/`): LCBench's 2000-config sample is byte-identical across
every OpenML dataset file, and TaskSet's 1000-config `*_wide_grid` sample is
identical across every problem file sharing the same optimizer-family
suffix. That's what makes them fair game for the same registration approach
as HPOBench's exact grid (`hpobench_data.py`) -- just read out through a
scattered-point kernel interpolator (`surface.py`/`interp.kernel_interp`)
instead of a Cartesian one, since the shared design isn't laid out on a
regular grid.

**PD1-tabular is deliberately excluded here.** Checked and rejected: unlike
LCBench/TaskSet, its per-workload config samples are NOT the same across
workloads (verified: `cifar10-wide_resnet-256` and `cifar100-wide_resnet-256`
share zero of 388 sampled `lr_initial` values despite having 388 common
`id`s) -- so PD1 needs an "identity" step built on `kernel_interp(x_a)`
against B's surrogate rather than a direct same-row comparison. That's a real
generalization, not just a bigger download, and is scoped as later work
rather than folded in under time pressure here.

Declared bounds are hardcoded from `mfpbench`'s own `ConfigurationSpace`
definitions (`external/ifbo_icml2024/src/mf-prior-bench/src/mfpbench/
lcbench_tabular/benchmark.py`, `taskset_tabular/benchmark.py`) -- per
`ARCHITECTURE.md` invariant #8, normalization uses the *declared* box, not
these tables' own empirical min/max (which fall slightly short of it, as
expected for a finite random sample).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ppfn.experiments.hpo_warp.task_grid import TaskGrid

# name, lower, upper, log
LCBENCH_PARAMS: list[tuple[str, float, float, bool]] = [
    ("batch_size", 16, 512, True),
    ("learning_rate", 1e-4, 0.1, True),
    ("momentum", 0.1, 0.99, False),
    ("weight_decay", 1e-5, 0.1, False),
    ("num_layers", 1, 5, False),
    ("max_units", 64, 1024, True),
    ("max_dropout", 0.0, 1.0, False),
]
LCBENCH_TARGET = "val_cross_entropy"  # minimize

TASKSET_PARAMS_BY_OPTIMIZER: dict[str, list[tuple[str, float, float, bool]]] = {
    "adam1p": [("learning_rate", 1e-9, 10, True)],
    "adam4p": [
        ("learning_rate", 1e-9, 10, True),
        ("beta1", 1e-4, 1, True),
        ("beta2", 1e-3, 1, True),
        ("epsilon", 1e-12, 1000, True),
    ],
    "adam6p": [
        ("learning_rate", 1e-9, 10, True),
        ("beta1", 1e-4, 1, True),
        ("beta2", 1e-3, 1, True),
        ("epsilon", 1e-12, 1000, True),
        ("l1", 1e-9, 10, True),
        ("l2", 1e-9, 10, True),
    ],
    "adam8p": [
        ("learning_rate", 1e-9, 10, True),
        ("beta1", 1e-4, 1, True),
        ("beta2", 1e-3, 1, True),
        ("epsilon", 1e-12, 1000, True),
        ("l1", 1e-9, 10, True),
        ("l2", 1e-9, 10, True),
        ("linear_decay", 1e-8, 1e-4, True),
        ("exponential_decay", 1e-6, 1e-3, True),
    ],
}
TASKSET_TARGET = "valid1_loss"  # minimize
TASKSET_SUFFIX = "_wide_grid_1k_10000_replica5.parquet"
TASKSET_WINSORIZE_PCTL = 1.0  # clip to [p, 100-p] per task -- some optimizees genuinely diverge


def _log_normalize(values: np.ndarray, lower: float, upper: float, log: bool) -> np.ndarray:
    if log:
        lo, hi = np.log(lower), np.log(upper)
        return (np.log(values) - lo) / (hi - lo)
    return (values - lower) / (upper - lower)


def _winsorize(y: np.ndarray, pctl: float) -> np.ndarray:
    lo, hi = np.percentile(y, [pctl, 100 - pctl])
    return np.clip(y, lo, hi)


def _load_scattered(
    df: pd.DataFrame,
    id_col: str,
    fidelity_col: str,
    target_col: str,
    params: list[tuple[str, float, float, bool]],
    family: str,
    task_id: str,
    winsorize_pctl: float | None,
) -> TaskGrid:
    param_names = [p[0] for p in params]
    df = df.astype({target_col: "float64"})  # pandas nullable Float64 -> plain float64 with NaN
    pivot = df.pivot(index=id_col, columns=fidelity_col, values=target_col).sort_index()
    # Fidelity gets log-scaled downstream (`fidelity_warp.py`, `surface.py`'s
    # scattered joint reader) -- a fidelity of 0 (LCBench's "epoch 0", the
    # pre-training state) has no meaningful log and silently produced
    # -inf/NaN cascading through the fidelity fit until caught. Not a
    # meaningful comparison point anyway (nothing has been learned yet).
    pivot = pivot.loc[:, pivot.columns > 0]
    # Some TaskSet configs stop early (divergence) and are simply absent
    # from later-epoch rows rather than NaN within one -- caught as a
    # `pivot.to_numpy()` ValueError on a task with incomplete coverage.
    # Dropped rather than imputed: an incomplete curve isn't a fair
    # comparison point for any of the fidelities in the ladder.
    pivot = pivot.dropna(axis=0, how="any")
    if len(pivot) < 50:
        raise ValueError(f"only {len(pivot)} fully-observed configs left for {family}/{task_id} after dropping incomplete curves")
    configs = df[[id_col, *param_names]].drop_duplicates(subset=id_col).set_index(id_col).loc[pivot.index]

    iters = np.sort(pivot.columns.to_numpy(dtype=np.float64))
    y_mean = pivot.to_numpy(dtype=np.float64).T  # [n_iters, n_configs]
    if winsorize_pctl is not None:
        y_mean = _winsorize(y_mean, winsorize_pctl)

    n_configs = len(configs)
    x = np.empty((n_configs, len(param_names)), dtype=np.float64)
    for j, (name, lo, hi, log) in enumerate(params):
        x[:, j] = _log_normalize(configs[name].to_numpy(dtype=np.float64), lo, hi, log)

    return TaskGrid(
        model=family,
        task_id=task_id,
        param_names=param_names,
        x=x,
        ids=pivot.index.to_numpy(),  # may be a strict subset for tasks with dropped incomplete configs
        grid_shape=(n_configs,),
        iters=iters,
        y_mean_by_iter=y_mean,
        y_var_by_iter=np.zeros_like(y_mean),  # no replicate seeds -- see task_grid.py
        n_seeds=1,
        is_gridded=False,
    )


def list_lcbench_tasks(data_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(data_dir).glob("*.parquet"))


def load_lcbench_grid(dataset: str, data_dir: str | Path) -> TaskGrid:
    df = pd.read_parquet(Path(data_dir) / f"{dataset}.parquet").reset_index()
    return _load_scattered(
        df, id_col="id", fidelity_col="epoch", target_col=LCBENCH_TARGET,
        params=LCBENCH_PARAMS, family="lcbench", task_id=dataset, winsorize_pctl=None,
    )


def list_taskset_tasks(optimizer: str, data_dir: str | Path) -> list[str]:
    suffix = f"-{optimizer}{TASKSET_SUFFIX}"
    return sorted(p.name[: -len(suffix)] for p in Path(data_dir).glob(f"*{suffix}"))


def load_taskset_grid(task: str, optimizer: str, data_dir: str | Path) -> TaskGrid:
    params = TASKSET_PARAMS_BY_OPTIMIZER[optimizer]
    path = Path(data_dir) / f"{task}-{optimizer}{TASKSET_SUFFIX}"
    df = pd.read_parquet(path).reset_index()
    return _load_scattered(
        df, id_col="config_id", fidelity_col="step", target_col=TASKSET_TARGET,
        params=params, family=f"taskset_{optimizer}", task_id=task, winsorize_pctl=TASKSET_WINSORIZE_PCTL,
    )
