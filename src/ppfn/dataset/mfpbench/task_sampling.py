# FIXME: remove this
import sys
from typing import Union


# Define the path to the 'src' folder
mfpbench_path = "/home/ruhkopf/VSCode/Meta_FTPFN/external/ifbo_icml2024/src/mf-prior-bench/src"

if mfpbench_path not in sys.path:
    sys.path.insert(0, mfpbench_path)
    print(f"DEBUG: Added {mfpbench_path} to sys.path")

try:
    import mfpbench

    print("✅ Success: mfpbench imported from:", mfpbench.__file__)
except ImportError as e:
    print(f"❌ Failed: {e}")
    print("Current sys.path:", sys.path)

# ------------------------------------------------------------------------------
import mfpbench

from src.mfpbench.taskset_tabular.benchmark import TaskSetTabularConfig_8p

import os
import torch
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed

from pfns4hpo.evaluate import _get_normalized_values
from pfns_hpo.run import process_taskset_mfpbench_with_step_0_prior


def get_benchmark(name, task_id, data_path):
    if name == "lcbench_tabular":
        datadir = os.path.join(data_path, "lcbench-tabular")
        benchmark = mfpbench.get(
            name=name,
            task_id=task_id,
            datadir=datadir,
            preload=True,
            prior=None,
            remove_constants=True,
            seed=True,
            value_metric="val_balanced_accuracy",
            value_metric_test="test_balanced_accuracy",
        )
        output_name = task_id
    elif name == "pd1_tabular":
        output_name = f"{task_id['model']}_{task_id['dataset']}_{task_id['batch_size']}"
        if "coarseness" in task_id:
            output_name = f"{output_name}_{task_id['coarseness']}"
        datadir = os.path.join(data_path, "pd1-tabular")
        benchmark = mfpbench.get(
            name=name,
            datadir=datadir,
            preload=True,
            prior=None,
            remove_constants=True,
            seed=True,
            **task_id,
        )
    elif name == "taskset_tabular":
        datadir = os.path.join(data_path, "taskset-tabular")
        output_name = f"{task_id['task_id']}_{task_id['optimizer']}"
        benchmark = mfpbench.get(
            name=name,
            datadir=datadir,
            preload=True,
            prior=None,
            seed=True,
            **task_id,
        )
        benchmark = process_taskset_mfpbench_with_step_0_prior(
            benchmark=benchmark, drop_step_0=True
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return benchmark, output_name


def generate_tasks(
        benchmark_name, task_id, ntasks_per_dataset, single_eval_pos, data_path, seq_len
):
    """Shitcode from section5.1/generate_tasks.py adapted to generate tasks for multiple datasets and aggregate them."""
    EPS = 10 ** -9
    benchmark, output_name = get_benchmark(benchmark_name, task_id, data_path)

    space = benchmark.space
    max_fidelities = benchmark.end
    ncurves = len(benchmark.configs)
    original_id = np.arange(ncurves)
    offset = min([int(_) for _ in benchmark.configs.keys()])

    allocations = []

    for i in range(ntasks_per_dataset):
        epoch = np.zeros(seq_len)
        id_curve = np.zeros(seq_len)

        ok = False

        # determine # observations/queries per curve
        while not ok:
            n_levels = int(np.round(10 ** np.random.uniform(0, 3)))
            n_levels = min(n_levels, max_fidelities)

            alpha = 10 ** np.random.uniform(-4, -1)
            weights = np.random.gamma(alpha, alpha, min(1000, ncurves)) + EPS
            p = weights / np.sum(weights)
            ids = np.arange(min(1000, ncurves))
            all_levels = np.repeat(ids, n_levels)
            all_p = np.repeat(p, n_levels) / n_levels
            if len(all_levels) > seq_len:
                ok = True
        ordering = np.random.choice(all_levels, p=all_p, size=seq_len, replace=False)

        # calculate the cutoff/samples for each curve
        cutoff_per_curve = np.zeros((seq_len,), dtype=int)
        epochs_per_curve = np.zeros((seq_len,), dtype=int)
        for i in range(seq_len):  # loop over every pos
            cid = ordering[i]
            epochs_per_curve[cid] += 1
            if i < single_eval_pos:
                cutoff_per_curve[cid] += 1

        # determine config, epochs for every curve
        curve_xs = []
        for cid in range(seq_len):  # loop over every curve
            if epochs_per_curve[cid] > 0:
                x_ = np.zeros((epochs_per_curve[cid],))
                if cutoff_per_curve[cid] > 0:  # observations (if any)
                    x_[: cutoff_per_curve[cid]] = np.arange(
                        1, cutoff_per_curve[cid] + 1
                    )
                if cutoff_per_curve[cid] < epochs_per_curve[cid]:  # queries (if any)
                    x_[cutoff_per_curve[cid]:] = np.random.choice(
                        np.arange(cutoff_per_curve[cid] + 1, n_levels + 1),
                        size=epochs_per_curve[cid] - cutoff_per_curve[cid],
                        replace=False,
                    )
                curve_xs.append(x_)
            else:
                curve_xs.append(None)

        # construct the batch data element
        curve_counters = np.zeros(seq_len, dtype=np.int64)
        for i in range(single_eval_pos):
            cid = ordering[i]
            id_curve[i] = cid + 1  # start from 1
            epoch[i] = curve_xs[cid][curve_counters[cid]]
            curve_counters[cid] += 1

        # assign max fidelity to all curves in context
        # specific to the evaluation in 5.1
        unique_curves = np.unique(id_curve[:single_eval_pos])
        nbiud = len(id_curve[single_eval_pos:(single_eval_pos + len(unique_curves))])
        num_unique_curves = len(unique_curves)
        id_curve[single_eval_pos: single_eval_pos + num_unique_curves] = unique_curves[:nbiud]
        end_pos = min(single_eval_pos + num_unique_curves, seq_len)
        epoch[single_eval_pos:end_pos] = max_fidelities

        allocations.append([id_curve, epoch])

    if benchmark_name == "taskset_tabular":
        from copy import deepcopy
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter

        default_space = deepcopy(benchmark.space)
        hp_dims = {
            'l1':UniformFloatHyperparameter(
                "l1",
                lower=1e-9,
                upper=10,
                log=True,
            ),
            'l2': UniformFloatHyperparameter(
                "l2",
                lower=1e-9,
                upper=10,
                log=True,
            ),
            'linear_decay': UniformFloatHyperparameter(
                "linear_decay",
                lower=1e-8,
                upper=0.0001,
                log=True,
            ),
            'exponential_decay': UniformFloatHyperparameter(
                "exponential_decay",
                lower=1e-6,
                upper=1e-3,
                log=True,
            ),
        }
        default_space.add_hyperparameters(
            [v for k, v in hp_dims.items() if k not in space._hyperparameters.keys()],
        )

    all_tasks = []
    for id_curve, epoch in allocations:
        np.random.shuffle(original_id)
        task_data = []
        for ordering, config_id, fidelity in zip(
                id_curve, original_id[id_curve.astype(int) - 1], epoch
        ):
            if ordering == 0:
                tmp = [0] * len(task_data[-1])
            else:
                _config_id = str(config_id + offset)
                tmp = []
                tmp = tmp + [ordering, fidelity]

                if  benchmark_name == "taskset_tabular"  and 'adam4p' in benchmark.name:
                    # adam4p is a subspace of adam8p, so some hp are set to their defaults
                    # implicitly.
                    # we need to figure out, what the missing hyperparameters are set to
                    # after normalization in the hierarchical definition of taskset_tabular
                    # so that we can fill them in the correct spaces of the input vector for the pfn
                    conf = benchmark.configs[_config_id].as_dict()
                    conf.update({
                        'l1': 1e-7,
                        'l2': 1e-7,
                        'linear_decay': 1e-8,
                        'exponential_decay': 1e-6,
                        'id': _config_id
                    })

                    normalized_config = _get_normalized_values(
                        config=TaskSetTabularConfig_8p(**conf), configuration_space=default_space
                    )

                else:
                    # collect the actual config and translate it to PFN bounded space
                    normalized_config = _get_normalized_values(
                        config=benchmark.configs[_config_id], configuration_space=space
                    )

                tmp = tmp + normalized_config
                tmp = tmp + [benchmark.query(config=_config_id, at=fidelity).error] # add y
            task_data.append(tmp)
        all_tasks.append(task_data)

    all_tasks = torch.from_numpy(np.array(all_tasks).astype(np.float32))

    return all_tasks.permute(1, 0, 2)  # [seq_len, nallocations_per_dataset, num_features]


def save_task_batch(
        benchmark_name,
        task_id: Union[str, dict],
        ntasks_per_tensor,
        single_eval_pos,
        data_path,
        target_path,
        seq_len, rep_idx
):
    """
    Calls the original generate_tasks existing function and saves the result to a specific
    sub-directory.
    """
    if isinstance(task_id, dict):
        task_name = "_".join([f"{v}" for  v in task_id.values()])
    else:
        task_name = str(task_id)

    # Create the specific directory for this configuration
    target_dir = Path(target_path) / benchmark_name / f"task_{task_name}" / f"sep_{single_eval_pos}"
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"rep_{rep_idx}.pt"

    # Generate the data using your original function
    # Note: I'm assuming _get_normalized_values and get_benchmark are in scope
    tasks_tensor = generate_tasks(
        benchmark_name, task_id, ntasks_per_tensor, single_eval_pos, data_path, seq_len
    )

    # Save as a standalone file
    torch.save(tasks_tensor, file_path)
    return str(file_path)


def orchestrate_generation(
        benchmarks,
        task_ids:Union[list, dict],
        single_eval_positions,
        num_repetitions,
        ntasks_per_dataset,
        data_path,
        target_path,
        seq_len,
        n_jobs=-1
):
    """
    Parallelizes generation across all combinations.
    """
    tasks = [
        delayed(save_task_batch)(
            b_name, t_id, ntasks_per_dataset, pos, data_path, target_path, seq_len, r_idx
        )
        for b_name in benchmarks
        for t_id in task_ids
        for pos in single_eval_positions
        for r_idx in range(num_repetitions)
    ]

    print(f"Starting parallel generation of {len(tasks)} batches...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)
    return results


if __name__ == '__main__':
    from ppfn.dataset.mfpbench.tasks import LCBENCH_IDS, PD1_IDS, TASKSET_IDS

    datasets = {
        'taskset_tabular': TASKSET_IDS,
        'pd1_tabular': PD1_IDS,
        'lcbench_tabular': LCBENCH_IDS
    }

    for benchmark_name, task_ids in datasets.items():
        print(f"Generating tasks for benchmark: {benchmark_name}")
        # TODO: consider having a new meta-train allocation for every single test allocation
        #  this means we need num_repetitions * len(task_ids) total repetitions and move the
        #  cursor on the repetition during __iter__

        # FIXME: naming with PD1 & taskset is broken?
        # Fixme: during __iter__ we need to harmonize the searchspace dim for taskset -- by setting
        #  missing hyperparameters to the "default" values according to the hierachical definition in
        #  the benchmark
        orchestrate_generation(
            benchmarks=[benchmark_name],
            task_ids=task_ids,  # Just a few for testing
            ntasks_per_dataset=1,  # only one allocation per dataset
            single_eval_positions=[128],
            num_repetitions=3,
            seq_len=1000,
            data_path="/home/ruhkopf/VSCode/Meta_FTPFN/data/",
            target_path="/home/ruhkopf/VSCode/Meta_FTPFN/data/validation/",
            n_jobs=-1
        )

    print("Task generation completed.")
