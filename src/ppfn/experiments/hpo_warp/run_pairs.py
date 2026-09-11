"""CLI: fit the config-space warp ladder + fidelity-axis warp for every
ordered task-id pair within one benchmark family, writing one JSON shard
(+ one `.artifacts.pt`) per pair.

Three families, dispatched by `--family`:
  hpobench  -- HPOBench `TabularBenchmark(model=...)`, exact grid (`--model`)
  lcbench   -- LCBench-tabular, shared scattered design (`hpobench_data.py`
               vs `scattered_data.py` -- see `task_grid.py`'s `is_gridded`)
  taskset   -- TaskSet-tabular, shared scattered design per optimizer family
               (`--optimizer`, e.g. `adam8p` for the biggest, 8D, space)

Shard-per-pair, not one mutable table, so this is safe to run repeatedly /
resume / parallelize across families without write contention --
`aggregate.py` collects the shards into one dataframe afterwards. A pair
already on disk is skipped, so re-running after adding new task_ids only
computes the new pairs ("adding to it as we collect them").

Pairs are independent fits, so this parallelizes with a plain process pool.
Workers are forked (the default `multiprocessing` start method on Linux), so
the pre-loaded `grids` dict is shared via copy-on-write rather than
re-pickled per worker. Each worker pins `torch.set_num_threads(1)` to avoid
16 processes each spawning their own intra-op thread pool and thrashing.

Usage (see docs/experiments/hpo-warp-complexity.md for the ulysses setup
this expects):

    python -m ppfn.experiments.hpo_warp.run_pairs \
        --family hpobench --model lr \
        --data-dir data/hpobench-tabular --out-dir data/hpo_warp_results --workers 12

    python -m ppfn.experiments.hpo_warp.run_pairs \
        --family lcbench \
        --data-dir data/lcbench-tabular --out-dir data/hpo_warp_results --workers 12

    python -m ppfn.experiments.hpo_warp.run_pairs \
        --family taskset --optimizer adam8p \
        --data-dir data/taskset-tabular --out-dir data/hpo_warp_results --workers 12
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path
from typing import Callable

import torch

from ppfn.experiments.hpo_warp.artifacts import save_artifacts
from ppfn.experiments.hpo_warp.fidelity_warp import fit_fidelity_warp
from ppfn.experiments.hpo_warp.fit import fit_config_warp, result_to_dict
from ppfn.experiments.hpo_warp.hpobench_data import list_available_task_ids, load_task_grid
from ppfn.experiments.hpo_warp.scattered_data import (
    list_lcbench_tasks,
    list_taskset_tasks,
    load_lcbench_grid,
    load_taskset_grid,
)
from ppfn.experiments.hpo_warp.task_grid import TaskGrid

_GRIDS: dict[str, TaskGrid] = {}


def _resolve_family(family: str, model: str | None, optimizer: str | None, data_dir: str):
    """Returns (label, list_tasks_fn, load_grid_fn) -- `label` names the
    output subdirectory and matches each loader's own `TaskGrid.model`."""
    if family == "hpobench":
        assert model, "--model required for --family hpobench"
        return model, lambda: list_available_task_ids(model, data_dir), lambda tid: load_task_grid(model, tid, data_dir)
    if family == "lcbench":
        return "lcbench", lambda: list_lcbench_tasks(data_dir), lambda tid: load_lcbench_grid(tid, data_dir)
    if family == "taskset":
        assert optimizer, "--optimizer required for --family taskset"
        label = f"taskset_{optimizer}"
        return (
            label,
            lambda: list_taskset_tasks(optimizer, data_dir),
            lambda tid: load_taskset_grid(tid, optimizer, data_dir),
        )
    raise ValueError(f"unknown family: {family!r}")


def _init_worker(grids: dict[str, TaskGrid]) -> None:
    global _GRIDS
    _GRIDS = grids
    torch.set_num_threads(1)


def _process_pair(args: tuple[str, str, str]) -> str:
    a, b, out_base_str = args
    out_base = Path(out_base_str)
    shard_path = out_base / f"{a}__{b}.json"
    artifacts_path = out_base / f"{a}__{b}.artifacts.pt"
    if shard_path.exists():
        return "skipped"
    try:
        result, artifacts, elbow_vf = fit_config_warp(_GRIDS[a], _GRIDS[b])
        payload = result_to_dict(result)
        if len(_GRIDS[a].iters) > 1:
            fid = fit_fidelity_warp(
                _GRIDS[a], _GRIDS[b], artifacts.h_lambda, artifacts.h_scale, artifacts.h_shift, elbow_vf, artifacts.test_idx
            )
            payload["fidelity"] = {
                "tau": fid.tau,
                "log_tau_severity": fid.log_tau_severity,
                "r2_at_tau1": fid.r2_at_tau1,
                "r2_at_best_tau": fid.r2_at_best_tau,
            }
        save_artifacts(artifacts, artifacts_path)
        shard_path.write_text(json.dumps(payload, indent=2))
        return "done"
    except Exception as exc:  # noqa: BLE001 -- one bad pair shouldn't kill the sweep
        (out_base / f"{a}__{b}.FAILED.txt").write_text(f"{type(exc).__name__}: {exc}")
        return "failed"


def run_all_pairs(
    family: str,
    data_dir: str,
    out_dir: str,
    model: str | None = None,
    optimizer: str | None = None,
    task_ids: list[str] | None = None,
    workers: int = 8,
) -> None:
    label, list_tasks_fn, load_grid_fn = _resolve_family(family, model, optimizer, data_dir)
    task_ids = task_ids or list_tasks_fn()
    out_base = Path(out_dir) / label
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"[{label}] {len(task_ids)} tasks -> {len(task_ids) * (len(task_ids) - 1)} ordered pairs, {workers} workers")
    grids = {tid: load_grid_fn(tid) for tid in task_ids}

    pairs = [(a, b, str(out_base)) for a, b in permutations(task_ids, 2)]
    counts = {"done": 0, "skipped": 0, "failed": 0}
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(grids,)) as pool:
        futures = [pool.submit(_process_pair, p) for p in pairs]
        for i, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if i % 50 == 0:
                elapsed = time.time() - t0
                print(f"  ...{i}/{len(pairs)} processed ({counts}), {elapsed:.0f}s elapsed")

    print(f"[{label}] finished: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["hpobench", "lcbench", "taskset"])
    parser.add_argument("--model", default=None, choices=["lr", "svm", "rf", "xgb", "nn"], help="--family hpobench")
    parser.add_argument("--optimizer", default=None, choices=["adam1p", "adam4p", "adam6p", "adam8p"], help="--family taskset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-ids", nargs="*", default=None, help="Subset of task_ids; default: all available")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_all_pairs(args.family, args.data_dir, args.out_dir, args.model, args.optimizer, args.task_ids, args.workers)
