"""Loading HPOBench `TabularBenchmark` tables into a shared, gridded form.

Each (model, task_id) table is a full factorial grid: `x_grid_size` configs
(e.g. 25x25=625 for `lr`'s 2D space) times a fixed set of fidelity values
times `n_seeds` replicate seeds -- and, critically, that grid (config values
AND seed values) is IDENTICAL across every task_id for a given model. That
means correspondence between two tasks is free: no interpolation or nearest-
neighbor matching is needed to compare them at "the same x". See
`docs/labbook/` for where this was confirmed
(`data.alpha.unique() == data.alpha.unique()` etc. across task_ids).

Config axes are normalized using the *declared* ConfigSpace bounds (log-space
for log-scaled hyperparameters), never the empirical grid min/max -- matching
`ARCHITECTURE.md` invariant #8, even though for a grid spanning the declared
box exactly the two nearly coincide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ppfn.experiments.hpo_warp.task_grid import TaskGrid


def _log_normalize(values: np.ndarray, lower: float, upper: float, log: bool) -> np.ndarray:
    if log:
        lo, hi = np.log(lower), np.log(upper)
        return (np.log(values) - lo) / (hi - lo)
    return (values - lower) / (upper - lower)


def load_task_grid(model: str, task_id: str, data_dir: str | Path) -> TaskGrid:
    """Load one HPOBench `TabularBenchmark` (model, task_id) table.

    `data_dir` is the `data_dir` passed to `hpobench.util.data_manager.
    TabularDataManager` when the table was downloaded (see
    `docs/experiments/hpo-warp-complexity.md`'s ulysses setup) --
    `<data_dir>/<model>/<task_id>/{model}_{task_id}_data.parquet.gzip` plus
    the sibling `_metadata.json`.
    """
    task_id = str(task_id)
    base = Path(data_dir) / model / task_id
    df = pd.read_parquet(base / f"{model}_{task_id}_data.parquet.gzip")
    meta = json.loads((base / f"{model}_{task_id}_metadata.json").read_text())

    cs = json.loads(meta["config_spaces"]["x"]) if isinstance(meta["config_spaces"]["x"], str) else meta["config_spaces"]["x"]
    param_names = [h["name"] for h in cs["hyperparameters"] if h["type"] != "constant"]

    # The fidelity column's NAME varies by model -- "iter" for lr, "subsample"
    # for svm, "n_estimators" for rf/xgb -- so it's read off the declared
    # fidelity space (`z`) rather than hardcoded. Caught by a dry run on
    # svm/rf/xgb before launching their sweeps: all three raised
    # `KeyError: 'iter'` under the lr-only hardcoded version.
    cs_z = json.loads(meta["config_spaces"]["z"]) if isinstance(meta["config_spaces"]["z"], str) else meta["config_spaces"]["z"]
    fidelity_candidates = [h["name"] for h in cs_z["hyperparameters"] if h["type"] != "constant"]
    if len(fidelity_candidates) != 1:
        raise ValueError(f"expected exactly one non-constant fidelity dim for {model!r}, got {fidelity_candidates}")
    fidelity_name = fidelity_candidates[0]

    df = df.copy()
    df["function_value"] = df["result"].apply(lambda d: d["function_value"])

    # Build the config grid: unique combinations of the param columns, sorted
    # so every task_id yields configs in the same order (grid is identical
    # across tasks, so a plain groupby-sort is enough -- no matching needed).
    configs = df[param_names].drop_duplicates().sort_values(param_names).reset_index(drop=True)
    n_per_axis = [configs[p].nunique() for p in param_names]

    x = np.empty((len(configs), len(param_names)), dtype=np.float64)
    for j, hp in enumerate(h for h in cs["hyperparameters"] if h["type"] != "constant"):
        x[:, j] = _log_normalize(
            configs[hp["name"]].to_numpy(dtype=np.float64),
            hp["lower"],
            hp["upper"],
            hp.get("log", False),
        )

    iters = np.sort(df[fidelity_name].unique())
    n_configs = len(configs)
    y_mean = np.empty((len(iters), n_configs))
    y_var = np.empty((len(iters), n_configs))
    config_key = list(zip(*[configs[p] for p in param_names]))
    key_to_row = {k: i for i, k in enumerate(config_key)}

    for k, it in enumerate(iters):
        sub = df[df[fidelity_name] == it]
        grouped = sub.groupby(param_names)["function_value"].agg(["mean", "var"])
        for key, row in grouped.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            i = key_to_row[key]
            y_mean[k, i] = row["mean"]
            y_var[k, i] = row["var"] if not np.isnan(row["var"]) else 0.0

    n_seeds = int(df["seed"].nunique())

    return TaskGrid(
        model=model,
        task_id=task_id,
        param_names=param_names,
        x=x,
        ids=np.arange(len(configs)),  # exact grid, every cell always present
        grid_shape=tuple(n_per_axis),
        iters=iters,
        y_mean_by_iter=y_mean,
        y_var_by_iter=y_var,
        n_seeds=n_seeds,
    )


def list_available_task_ids(model: str, data_dir: str | Path) -> list[str]:
    base = Path(data_dir) / model
    return sorted(p.name for p in base.iterdir() if p.is_dir())
